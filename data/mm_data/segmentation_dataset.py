# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from io import BytesIO

import logging
import warnings
import string
import cv2
import random

import numpy as np
import torch
import base64
from torchvision import transforms

from PIL import Image, ImageFile

from mmseg.datasets.pipelines import Resize, RandomCrop, RandomFlip, PhotoMetricDistortion, MultiScaleFlipAug, Normalize, ImageToTensor

from torchvision.utils import save_image

from data import data_utils
from data.ofa_dataset import OFADataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def flatten(l):
    return [item for sublist in l for item in sublist]

def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])
    
    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    if samples[0].get("ori_shape", None) is not None:
        ori_shape = [x["ori_shape"] for x in samples]
        
    if samples[0].get("ori_semantic_seg", None) is not None:
        ori_semantic_seg = [x["ori_semantic_seg"] for x in samples]

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
        ntokens = tgt_lengths.sum().item()

        prev_output_tokens = None
        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")

        downsampled_target = None
        if samples[0].get("downsampled_target", None) is not None:
            downsampled_target = merge("downsampled_target")

    else:
        ntokens = src_lengths.sum().item()

    aux_input = None
    text2seg_target = None
    if samples[0].get("text2seg_source", None) is not None:
        text2seg_src_tokens = merge("text2seg_source")
        text2seg_src_lengths = torch.LongTensor([s["text2seg_source"].ne(pad_idx).long().sum() for s in samples])

        if samples[0].get("text2seg_target", None) is not None:
            text2seg_target = merge("text2seg_target")

        if samples[0].get("text2seg_prev_output_tokens", None) is not None:
            text2seg_prev_output_tokens = merge("text2seg_prev_output_tokens")

        text2seg_patch_images = None
        text2seg_patch_masks = None
        if samples[0].get("text2seg_patch_image", None) is not None:
            text2seg_patch_images = merge("text2seg_patch_image")
            text2seg_patch_masks = torch.cat([sample['text2seg_patch_mask'] for sample in samples])
        
        aux_input = {
            "src_tokens": text2seg_src_tokens,
            "src_lengths": text2seg_src_lengths,
            "patch_images": text2seg_patch_images,
            "patch_masks": text2seg_patch_masks,
            "prev_output_tokens": text2seg_prev_output_tokens,
        }
        
    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "aux_input": aux_input,
        "target": target,
        "downsampled_target": downsampled_target,
        "text2seg_target": text2seg_target,
        "ori_shape": ori_shape,
        "ori_semantic_seg": ori_semantic_seg,
    }

    return batch

class SegmentationDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        patch_image_size=224,
        imagenet_default_mean_and_std=True,
        cfg=None
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.patch_image_size = patch_image_size
        logger.info(f"patch_image_size: {patch_image_size}")
        self.cfg = cfg

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.image_normalize = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=mean, std=std)])
        if self.split == 'train':
            self.image_transform = transforms.Compose([
                Resize(img_scale=(self.patch_image_size*4, self.patch_image_size), ratio_range=(0.5, 2.0), min_size=self.patch_image_size),
                RandomCrop(crop_size=(self.patch_image_size, self.patch_image_size), cat_max_ratio=0.75),
                RandomFlip(prob=0.5),
                PhotoMetricDistortion(),
            ])
            
            self.downsample_gt_seg = transforms.Resize((self.patch_image_size//16, self.patch_image_size//16), transforms.InterpolationMode.NEAREST)


        else:
            self.image_transform = MultiScaleFlipAug(img_scale=(self.patch_image_size*4, self.patch_image_size),
                                                          flip=False,
                                                          transforms=[dict(type='Resize', keep_ratio=True),
                                                                      dict(type='RandomFlip')])
            self.downsample_gt_seg = transforms.Resize((self.patch_image_size//16, self.patch_image_size//16), transforms.InterpolationMode.NEAREST)

        prompt_prefix = cfg.prompt_prefix
        if len(prompt_prefix):
            self.prompt = self.encode_text(f' {prompt_prefix.lstrip()}')
        else:
            self.prompt = None

        self.artificial_image_type = cfg.artificial_image_type
        self.num_seg = cfg.num_seg_tokens # 15 (coco-unseen), 150 (ade), 171 (coco-fine)
        self.id2rawtext = [x.strip() for x in cfg.category_list.split(',')] + ['unknown']
        assert len(self.id2rawtext) == self.num_seg+1

        self.id2text = [self.encode_text(f" {x}") for x in self.id2rawtext]
        self.text_length = torch.tensor([len(x) for x in self.id2text])

        self.id2seg = np.array([f'<seg_{idx}>' for idx in range(self.num_seg + 1)])
        self.seg2code = self.encode_text(" ".join(self.id2seg), use_bpe=False)
        self.upsample_gt_seg = transforms.Resize((self.patch_image_size, self.patch_image_size), transforms.InterpolationMode.NEAREST)

    def encode_text(self, text, length=None, append_bos=False, append_eos=False, use_bpe=True):
        line = [self.bpe.encode(' {}'.format(word.strip())) if not word.startswith('<seg_') else word for word in text.strip().split()]
        line = ' '.join(line)
        
        s = self.tgt_dict.encode_line(
            line=line,
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([self.bos_item, s])
        if append_eos:
            s = torch.cat([s, self.eos_item])
        return s

    def __getitem__(self, index):
        image, segmentation, uniq_id = self.dataset[index]

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        image_arr = np.asarray(image)
        if len(image_arr.shape) < 3:
            image_arr = cv2.cvtColor(image_arr, cv2.COLOR_GRAY2RGB)
        
        image_arr = image_arr[:, :, ::-1].copy() # to BGR
        
        segmentation = Image.open(BytesIO(base64.urlsafe_b64decode(segmentation)))
        segmentation_arr = np.asarray(segmentation).copy()
        
        patch_mask = torch.tensor([True])

        results = {}
        results['img'] = image_arr
        results['img_shape'] = image_arr.shape
        results['scale_factor'] = 1.0

        # avoid using underflow conversion
        segmentation_arr[segmentation_arr == 0] = 255
        segmentation_arr = segmentation_arr - 1
        segmentation_arr[segmentation_arr == 254] = self.num_seg
        results['gt_semantic_seg'] = segmentation_arr
        results['seg_fields'] = ['gt_semantic_seg']

        ori_shape = image_arr.shape
        ori_semantic_seg = segmentation_arr.copy()
        if self.split == 'train':
            aug_dict = self.image_transform(results)
            
            img = aug_dict.pop('img')
            img = img[:, :, ::-1].copy() # to RGB
            img = self.image_normalize(img)
            gt_semantic_seg = aug_dict.pop('gt_semantic_seg')
            gt_semantic_seg = torch.from_numpy(gt_semantic_seg.astype(np.int64))

            gt_semantic_seg_downsampled = self.downsample_gt_seg(gt_semantic_seg.unsqueeze(0)).flatten()
            seg_ids_downsampled = self.seg2code[gt_semantic_seg_downsampled]
            downsampled_target = torch.cat([seg_ids_downsampled, self.eos_item])
            prev_output_item = torch.cat([self.bos_item, seg_ids_downsampled])

        else:
            img_dict = self.image_transform(results)
            img = img_dict.pop('img')[0]
            img = img[:, :, ::-1].copy() # to RGB

            img = self.image_normalize(img)

            gt_semantic_seg = img_dict.pop('gt_semantic_seg')[0]
            gt_semantic_seg = torch.from_numpy(gt_semantic_seg.astype(np.int64))
            
            downsampled_target=None
            gt_semantic_seg_downsampled = self.downsample_gt_seg(gt_semantic_seg.unsqueeze(0)).flatten()
            seg_ids_downsampled = self.seg2code[gt_semantic_seg_downsampled]
            prev_output_item = torch.cat([self.bos_item, seg_ids_downsampled])

        seg_ids = self.seg2code[gt_semantic_seg.flatten()]
        target = torch.cat([seg_ids, self.eos_item])

        # build 
        prompt_ids = [idx for idx in range(len(self.id2text))]
        
        src_text = [self.bos_item]
        if self.prompt is not None:
            src_text += [self.prompt]
        
        src_text += [self.id2text[idx] for idx in prompt_ids]
        src_text += [self.eos_item]

        src_item = torch.cat(src_text)

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": img,
            "patch_mask": patch_mask,
            "target": target,
            "downsampled_target": downsampled_target,
            "prev_output_tokens": prev_output_item,
            "ori_shape": ori_shape,
            "ori_semantic_seg": ori_semantic_seg,
        }

        if self.artificial_image_type == 'none':
            # no fake image
            return example
            
        elif self.artificial_image_type == 'norand_k':
            artificial_image_ids = np.random.choice(self.num_seg, size=1024, replace=True).tolist()
            artificial_image_target = self.seg2code[artificial_image_ids]

        elif self.artificial_image_type.startswith('rand_k'):
            if self.artificial_image_type == 'rand_k':
                l, r = 1, 33
            elif len(self.artificial_image_type.split('-')) == 3:
                l, r = self.artificial_image_type.split('-')[1:3]
                l, r = int(l), int(r)
            else:
                raise NotImplementedError

            sh, sw = torch.randint(l,r,(2,))
            sh, sw = sh.item(), sw.item()
            rand = np.random.choice(self.num_seg, size=sh*sw, replace=True)
            rand = torch.from_numpy(rand).reshape(1, 1, sh, sw)
            artificial_image_ids = self.downsample_gt_seg(rand).reshape(-1).tolist()
            
            upsample_rand = self.upsample_gt_seg(rand).reshape(-1).tolist()
            downsample_rand = self.downsample_gt_seg(rand).reshape(-1).tolist()
            artificial_image_target = self.seg2code[upsample_rand]
            artificial_image_prev = self.seg2code[downsample_rand]
        else:
            raise NotImplementedError

        embedbag_ids = torch.cat([self.id2text[idx] for idx in artificial_image_ids])
        embedbag_offsets = torch.tensor([self.text_length[idx] for idx in artificial_image_ids], dtype=torch.long).cumsum(dim=0)

        target = torch.cat([artificial_image_target, self.eos_item])
        prev_output_tokens = torch.cat([self.bos_item, artificial_image_prev])

        prompt_ids = [idx for idx in range(len(self.id2text))]
        
        src_text = [self.bos_item]
        if self.prompt is not None:
            src_text += [self.prompt]
        
        src_text += [self.id2text[idx] for idx in prompt_ids]
        src_text += [self.eos_item]
        src_item = torch.cat(src_text)
        
        example["text2seg_patch_image"] = embedbag_ids
        example["text2seg_patch_mask"] = embedbag_offsets
        example["text2seg_source"] = src_item
        example["text2seg_target"] = target
        example["text2seg_prev_output_tokens"] = prev_output_tokens

        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)