# Modified from OFA

# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import json
import logging
from typing import Optional
from argparse import Namespace
from itertools import zip_longest
from collections import OrderedDict
import torch
import os

from einops import rearrange
from mmseg.ops import resize

import numpy as np
import string
from fairseq import metrics, utils
from fairseq.tasks import register_task
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.data import Dictionary

from tasks.ofa_task import OFATask, OFAConfig
from data.mm_data.segmentation_dataset import SegmentationDataset
from data.file_dataset import FileDataset
from utils.cider.pyciderevalcap.ciderD.ciderD import CiderD
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class SegmentationConfig(OFAConfig):
    eval_acc: bool = field(
        default=True, metadata={"help": "evaluation with accuracy"}
    )
    eval_args: Optional[str] = field(
        default='{}',
        metadata={
            "help": 'generation args, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )
    uses_ema: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use ema"},
    )
    add_object: bool = field(
        default=False,
        metadata={"help": "add object to encoder"},
    )
    max_object_length: int = field(
        default=30, metadata={"help": "the maximum object sequence length"}
    )    
    valid_batch_size: int = field(
        default=1,
        metadata={"help": "valid batch size per step"},
    )
    scst: bool = field(
        default=False, metadata={"help": "Self-critical sequence training"}
    )
    scst_args: str = field(
        default='{}',
        metadata={
            "help": 'generation args for Self-critical sequence training, as JSON string'
        },
    )
    artificial_image_type: str = field(
        default='random',
        metadata={
            "help": 'random | gt_seg | upsampling | none'
        },
    )
    prompt_prefix: str = field(
        default='',
        metadata={
            "help": 'could be "what is the segmentation map of the image? object:"'
        },
    )
    num_seg_tokens: int = field(
        default=150,
        metadata={"help": "number of seg tokens"},
    )
    category_list: str = field(
        default="",
        metadata={"help": "list of semantic category words (comma separated)"},
    )
    epoch_row_count: int = field(
        default=-1,
        metadata={"help": "if -1, disabled."},
    )

@register_task("segmentation", dataclass=SegmentationConfig)
class SegmentationTask(OFATask):
    def __init__(self, cfg: SegmentationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        self.uses_ema = self.cfg.uses_ema
        self.num_seg_tokens = cfg.num_seg_tokens
        self.category_list = cfg.category_list

    @classmethod
    def setup_task(cls, cfg: DictConfig, **kwargs):
        """Setup the task."""
        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(cfg.bpe_dir, "dict.txt")
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(cfg.bpe_dir, "dict.txt")
        )
        src_dict.add_symbol("<mask>")
        tgt_dict.add_symbol("<mask>")
        for i in range(cfg.code_dict_size):
            src_dict.add_symbol("<code_{}>".format(i))
            tgt_dict.add_symbol("<code_{}>".format(i))

        for i in range(cfg.num_bins):
            src_dict.add_symbol("<bin_{}>".format(i))
            tgt_dict.add_symbol("<bin_{}>".format(i))

        num_segs = cfg.num_seg_tokens + 1
        for i in range(num_segs):
            src_dict.add_symbol("<seg_{}>".format(i))
            tgt_dict.add_symbol("<seg_{}>".format(i))

        logger.info("source dictionary: {} types".format(len(src_dict)))
        logger.info("target dictionary: {} types".format(len(tgt_dict)))
        return cls(cfg, src_dict, tgt_dict)


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            table_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            table_path = paths[-1]
        
        # assert self.cfg.selected_cols == '0,1,2'
        dataset = FileDataset(table_path, self.cfg.selected_cols)
        if split == 'train' and self.cfg.epoch_row_count > -1:
            logger.info(f"Setting epoch row count to {self.cfg.epoch_row_count}")
            dataset.total_row_count = self.cfg.epoch_row_count
            dataset._compute_start_pos_and_row_count()

        self.datasets[split] = SegmentationDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            patch_image_size=self.cfg.patch_image_size,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            cfg=self.cfg,
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.eval_acc:
            gen_args = json.loads(self.cfg.eval_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )

        return model

    def _calculate_ap_score(self, hyps, refs, thresh=0.5):
        interacts = torch.cat(
            [torch.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2]),
             torch.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:])],
            dim=1
        )
        area_predictions = (hyps[:, 2] - hyps[:, 0]) * (hyps[:, 3] - hyps[:, 1])
        area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
        interacts_w = interacts[:, 2] - interacts[:, 0]
        interacts_h = interacts[:, 3] - interacts[:, 1]
        area_interacts = interacts_w * interacts_h
        ious = area_interacts / (area_predictions + area_targets - area_interacts + 1e-6)
        return ((ious >= thresh) & (interacts_w > 0) & (interacts_h > 0)).float()

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False, **extra_kwargs
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample, update_num=update_num, ema_model=extra_kwargs.get('ema_model'))
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output


    def valid_step(self, sample, model, criterion, **extra_kwargs):
        model.eval()
        loss, sample_size, logging_output = criterion(model, sample)

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_all_acc(meters):
            all_acc = meters['_area_intersect_infer'].sum.sum() / meters['_area_pred_label_infer'].sum.sum()
            all_acc = all_acc if isinstance(all_acc, float) else all_acc.item()
            return round(all_acc, 4)

        def compute_mean_iou(meters):
            miou = torch.nanmean(meters['_area_intersect_infer'].sum / (meters['_area_union_infer'].sum))
            miou = miou if isinstance(miou, float) else miou.item()
            return round(miou, 4)

        def compute_mean_acc(meters):
            macc = torch.nanmean(meters['_area_intersect_infer'].sum / (meters['_area_label_infer'].sum))
            macc = macc if isinstance(macc, float) else macc.item()
            return round(macc, 4)

        if "_area_union_infer" in logging_outputs[0]: # check if valid
            metrics.log_scalar_sum("_area_intersect_infer", sum_logs("_area_intersect_infer"))
            metrics.log_scalar_sum("_area_pred_label_infer", sum_logs("_area_pred_label_infer"))
            metrics.log_scalar_sum("_area_label_infer", sum_logs("_area_label_infer"))
            metrics.log_scalar_sum("_area_union_infer", sum_logs("_area_union_infer"))

            metrics.log_derived("infer_aAcc", compute_all_acc)
            metrics.log_derived("infer_mIoU", compute_mean_iou)
            metrics.log_derived("infer_mAcc", compute_mean_acc)

    def _inference(self, generator, sample, model):
        gen_out = self.inference_step(generator, [model], sample)
        
        hyps = gen_out
        refs = sample["downsampled_target"][:, :-1]
        
        if self.cfg.eval_print_samples:
            logger.info("example hypothesis: " + hyps)
            logger.info("example reference: " + refs)

        return hyps, refs
