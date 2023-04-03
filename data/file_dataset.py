# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import os
import torch
import pickle
import logging
import time
from pathlib import Path

import torch.distributed as dist

logger = logging.getLogger(__name__)
is_master_process = (not dist.is_initialized()) or (
    dist.is_initialized() and dist.get_rank() == 0
)

class FileDataset:
    def __init__(self, file_path, selected_col_ids=None, dtypes=None, separator="\t", cached_index=False):
        self.file_path = file_path
        assert os.path.exists(self.file_path), "Error: The local datafile {} not exists!".format(self.file_path)

        self.separator = separator
        if selected_col_ids is None:
            # default to all fields
            self.selected_col_ids = list(
                range(len(open(self.file_path).readline().rstrip("\n").split(self.separator))))
        else:
            self.selected_col_ids = [int(col_id) for col_id in selected_col_ids.split(",")]
        if dtypes is None:
            # default to str
            self.dtypes = [str for col_id in self.selected_col_ids]
        else:
            self.dtypes = [eval(col_dtype) for col_dtype in dtypes.split(",")]
            assert len(self.dtypes) == len(self.selected_col_ids)

        self.data_cnt = 0
        try:
            self.slice_id = torch.distributed.get_rank()
            self.slice_count = torch.distributed.get_world_size()
        except Exception:
            self.slice_id = 0
            self.slice_count = 1
        self.cached_index = True
        self._init_seek_index()
        self._reader = self._get_reader()
        print("file {} slice_id {} row count {} total row count {}".format(
            self.file_path, self.slice_id, self.row_count, self.total_row_count)
        )

    def _init_seek_index(self):
        if self.cached_index:
            cache_path = "{}.index".format(self.file_path)
            while not os.path.exists(cache_path):
                try:
                    if is_master_process:
                        working_flag = Path(f"{cache_path}.working")
                        working_flag.touch()
                        logger.info(f"index cache file {cache_path} not exists!")
                        logger.info(f"initializing a new one...")
                        
                        self._sweep_datafile()
                        with open(working_flag, "wb") as fp:
                            pickle.dump([self.total_row_count, self.lineid_to_offset], fp)
                        working_flag.rename(cache_path)
                except:
                    pass
                time.sleep(1)

            while True:
                try:
                    with open(cache_path, "rb") as fp:
                        self.total_row_count, self.lineid_to_offset = pickle.load(fp)
                    break
                except:
                    time.sleep(1)

        else:
            self._sweep_datafile()
        
        self._compute_start_pos_and_row_count()
        logger.info("local datafile {} slice_id {} finished initializing row_count and line_idx-to-offset mapping".format(self.file_path, self.slice_id))

    def _sweep_datafile(self):
        # make an iteration over the file to get row_count and line_idx-to-offset mapping
        with open(self.file_path, "r") as fp:
            self.total_row_count = 0
            offset = 0
            self.lineid_to_offset = []
            for line in fp:
                self.lineid_to_offset.append(offset)
                self.total_row_count += 1
                offset += len(line.encode('utf-8'))

    def _compute_start_pos_and_row_count(self):
        self.row_count = self.total_row_count // self.slice_count
        if self.slice_id < self.total_row_count - self.row_count * self.slice_count:
            self.row_count += 1
            self.start_pos = self.row_count * self.slice_id
        else:
            self.start_pos = self.row_count * self.slice_id + (self.total_row_count - self.row_count * self.slice_count)

    def _get_reader(self):
        fp = open(self.file_path, "r")
        fp.seek(self.lineid_to_offset[self.start_pos])
        return fp

    def _seek(self, offset=0):
        try:
            print("slice_id {} seek offset {}".format(self.slice_id, self.start_pos + offset))
            self._reader.seek(self.lineid_to_offset[self.start_pos + offset])
            self.data_cnt = offset
        except Exception:
            print("slice_id {} seek offset {}".format(self.slice_id, offset))
            self._reader.seek(self.lineid_to_offset[offset])
            self.data_cnt = offset

    def __del__(self):
        self._reader.close()

    def __len__(self):
        return self.row_count

    def get_total_row_count(self):
        return self.total_row_count

    def __getitem__(self, index):
        if self.data_cnt == self.row_count:
            # print("reach the end of datafile, start a new reader")
            self.data_cnt = 0
            self._reader = self._get_reader()
        column_l = self._reader.readline().rstrip("\n").split(self.separator)
        self.data_cnt += 1
        column_l = [dtype(column_l[col_id]) for col_id, dtype in zip(self.selected_col_ids, self.dtypes)]
        return column_l
