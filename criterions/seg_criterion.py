# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import math
from dataclasses import dataclass, field
from re import S, X
from typing import Optional
from collections import OrderedDict

import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange
from mmseg.ops import resize

from timm.models.layers import trunc_normal_

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

import logging
logger = logging.getLogger(__name__)

@dataclass
class SegCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    ignore_eos: bool = field(
        default=True,
        metadata={"help": "Ignore eos token"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    drop_worst_ratio: float = field(
        default=0.0,
        metadata={"help": "ratio for discarding bad samples"},
    )
    drop_worst_after: int = field(
        default=0,
        metadata={"help": "steps for discarding bad samples"},
    )
    use_rdrop: bool = field(
        default=False, metadata={"help": "use R-Drop"}
    )
    reg_alpha: float = field(
        default=1.0, metadata={"help": "weight for R-Drop"}
    )
    sample_patch_num: int = field(
        default=196, metadata={"help": "sample patches for v1"}
    )
    constraint_range: Optional[str] = field(
        default=None,
        metadata={"help": "constraint range"}
    )
    upscale_lprobs: str = field(
        default='true',
        metadata={"help": "true | fasle"}
    )
    unsupervised_segmentation: str = field(
        default='true',
        metadata={"help": "true | fasle"}
    )
    criterion_update_freq: int = field(
        default=1, metadata={"help": "update frequency used in this criterion (e.g., for updating logit center)."}
    )
    freeze_embedding_iter: int = field(
        default=-1, metadata={
            "help": "Freeze the token embedding after this iteration (ignored if -1). ``effective iteration'' (1 iter per N update-freq) is used."}
    )
    full_context_alignment: str = field(
        default='false', metadata={"help": "whether to apply full attention in decoder"}
    )
    init_seg_with_text: str = field(
        default='true', metadata={"help": "whether to lazy initialize the segmentation with text embedding bags."}
    )
    resnet_topk: int = field(
        default=3, metadata={"help": "filtering with topk adjacent resnet features"}
    )
    resnet_prob_temperature: float = field(
        default=1.0, metadata={"help": "resnet softmax temperature"}
    )
    resnet_iters: int = field(
        default=0, metadata={"help": "resnet filtering iterations"}
    )

def resolve_str_true_false(x):
    x = x.lower()
    if x == 'true':
        return True
    elif x == 'false':
        return False
    else:
        raise ValueError(f"Unable to recognize string bool input: {x}")

@register_criterion(
    "seg_criterion", dataclass=SegCriterionConfig
)
class SegCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        ignore_eos=True,
        report_accuracy=False,
        drop_worst_ratio=0,
        drop_worst_after=0,
        use_rdrop=False,
        reg_alpha=1.0,
        sample_patch_num=196,
        constraint_range=None,
        upscale_lprobs='true',
        unsupervised_segmentation='true',
        criterion_update_freq=1,
        freeze_embedding_iter=-1,
        full_context_alignment='false',
        init_seg_with_text='true',
        resnet_topk=3,
        resnet_prob_temperature=1.0,
        resnet_iters=0
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.sample_patch_num = sample_patch_num
        
        self.iter = -1
        self.effective_iter = -1 # effective_iter = iter // criterion_update_freq
        self.criterion_update_freq = criterion_update_freq
        
        self.upscale_lprobs = resolve_str_true_false(upscale_lprobs)
        self.unsupervised_segmentation = resolve_str_true_false(unsupervised_segmentation)
        self.full_context_alignment = resolve_str_true_false(full_context_alignment)
        self.init_seg_with_text = resolve_str_true_false(init_seg_with_text)
        
        self.resnet_topk = resnet_topk
        self.resnet_prob_temperature = resnet_prob_temperature
        self.resnet_iters = resnet_iters
        
        self.seg_id_offset = task.target_dictionary.index("<seg_0>")
        self.num_seg = task.cfg.num_seg_tokens
        self.id2rawtext = [x.strip() for x in task.cfg.category_list.split(',')]
        assert len(self.id2rawtext) == self.num_seg
        
        logger.info(f"ResNet iterations {self.resnet_iters}")

    def forward(self, model, sample, update_num=0, reduce=True, ema_model=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.iter == -1:
            self.iter = self.criterion_update_freq * update_num - 1
            self._lazy_initialization(sample, model, ema_model) # Do lazy initializations;
            logger.info(f"Restored iteration counts {self.iter}")
        self._update_iteration(sample, model, ema_model)
        
        if self.unsupervised_segmentation and model.training:
            net_output = model(full_context_alignment=self.full_context_alignment, aux_input=sample["aux_input"])
            imfree_output = net_output[1].get("aux_output")
            imfree_loss = self.compute_imfree_loss(model, imfree_output, sample, update_num, reduce=reduce)
            loss = imfree_loss
            with torch.inference_mode():
                seg_output = model(**sample["net_input"], full_context_alignment=self.full_context_alignment)
                seg_loss, metrics, ntokens = self.compute_loss(model, seg_output, sample, update_num, reduce=reduce, ema_model=ema_model)

        elif model.training:
            net_output = model(**sample["net_input"], full_context_alignment=self.full_context_alignment)
            imfree_loss = net_output[0].new_zeros(size=(1, ))
            seg_loss, metrics, ntokens = self.compute_loss(model, net_output, sample, update_num, reduce=reduce, ema_model=ema_model)
            loss = seg_loss
        
        else:
            net_output = model(**sample["net_input"], full_context_alignment=self.full_context_alignment)
        
            if self.resnet_iters > 0:
                _x, extra = net_output
                
                resnet_feature = extra['encoder_returns']['image_embed_before_proj'][0]
                resnet_feature_norm = F.normalize(resnet_feature, dim=-1)
                cosine_sim = resnet_feature_norm @ resnet_feature_norm.transpose(-1, -2)
                _, topk_ind = torch.topk(cosine_sim, k=self.resnet_topk, dim=-1)
                bsz, seqlen = topk_ind.shape[:2]
                batch_ind = torch.arange(bsz).unsqueeze(-1).unsqueeze(-1).expand(bsz, seqlen, self.resnet_topk)
                
                resnet_prob = (_x / self.resnet_prob_temperature).softmax(-1)
                for _ in range(self.resnet_iters):
                    resnet_prob_topk = resnet_prob[batch_ind, topk_ind]
                    resnet_prob = resnet_prob_topk.mean(dim=-2)
                resnet_prob = torch.cat([resnet_prob, resnet_prob.new_zeros(size=(bsz, 1, resnet_prob.size(-1)))], dim=1) # fake eos token
                
                extra['resnet_postprocess_probability'] = resnet_prob
            
            imfree_loss = torch.zeros(size=(1, ), device='cuda')
            seg_loss, metrics, ntokens = self.compute_loss(model, net_output, sample, update_num, reduce=reduce, ema_model=ema_model)
            loss = seg_loss
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else ntokens
        )
        
        logging_output = {
            "loss": loss.data,
            "imfree_loss": imfree_loss.data,
            "seg_loss": seg_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.data
            logging_output[key] = value

        return loss, sample_size, logging_output
    
    def upsample_logits(self, logits, hp=32, wp=32, h=512, w=512):
        logits_ = logits[:, :-1]
        logits_ = rearrange(logits_, 'b (h w) d -> b d h w', h=hp, w=wp)
        logits_ = resize(logits_, size=(h, w), mode='bilinear', align_corners=False)
        logits_ = rearrange(logits_, 'b d h w -> b (h w) d')
        logits = torch.cat([logits_, logits[:, -1:]], dim=1)
        
        return logits
        
    def compute_imfree_loss(self, model, net_output, sample, update_num, reduce=True):
        logits, extra = net_output
        logits = logits.float()
        logits = self.upsample_logits(logits) # bilinear upsample
        
        target = sample.get('text2seg_target')

        logits = logits[:, :-1]  # remove eos
        target = target[:, :-1]
        
        logits = logits.reshape(-1, logits.size(-1))
        target = target.reshape(-1)

        mask = torch.logical_and(target != self.padding_idx, target != (self.seg_id_offset+self.num_seg))
        logits = logits[mask]
        target = target[mask]

        target = target - self.seg_id_offset

        loss = F.cross_entropy(logits, target.detach(), label_smoothing=self.eps)
            
        return loss

    def compute_loss(self, model, net_output, sample, update_num, reduce=True, ema_model=None):
        classifier_scores_lowres, extra = net_output
        metrics = dict()

        target_lowres = sample.get("downsampled_target")
        if target_lowres is not None:
            classifier_scores_lowres = classifier_scores_lowres.float()
            target_lowres_shape = target_lowres.shape
            assert target_lowres_shape == classifier_scores_lowres.shape[:-1]

            sample_masks_lowres = torch.logical_or(target_lowres == self.padding_idx, target_lowres == (self.seg_id_offset+self.num_seg))
            eos_masks_lowres = target_lowres.eq(self.task.tgt_dict.eos())
            sample_masks_lowres = torch.logical_or(sample_masks_lowres, eos_masks_lowres)

        # calculate upscaled versions
        target = sample.get("target")

        (hp, wp) = extra['encoder_returns']['image_embed_shape'][0]
        (h, w) = sample['net_input']['patch_images'].shape[-2:]

        if not model.training:
            target = sample.get("ori_semantic_seg")[0] 
            target = torch.from_numpy(rearrange(target, 'h w -> () (h w)')).long().to(classifier_scores_lowres.device) + self.seg_id_offset
            ori_shape = sample['ori_shape']
            (h, w) = ori_shape[0][:2]
        classifier_scores = self.upsample_logits(classifier_scores_lowres, hp=hp, wp=wp, h=h, w=w) # bilinear upsample
        resnet_postprocess_probability = extra.get("resnet_postprocess_probability")
        if resnet_postprocess_probability is not None:
            resnet_postprocess_probability = self.upsample_logits(resnet_postprocess_probability, hp=hp, wp=wp, h=h, w=w) # bilinear upsample
        if not model.training:
            classifier_scores = classifier_scores[:,:-1]
            if resnet_postprocess_probability is not None:
                resnet_postprocess_probability = resnet_postprocess_probability[:,:-1]

        target_shape = target.shape
        assert target_shape == classifier_scores.shape[:-1]

        sample_masks = torch.logical_or(target == self.padding_idx, target == (self.seg_id_offset+self.num_seg))
        eos_masks = target.eq(self.task.tgt_dict.eos())
        sample_masks = torch.logical_or(sample_masks, eos_masks)

        # apply masking to targets
        target = target[~sample_masks] - self.seg_id_offset
        classifier_scores = classifier_scores[~sample_masks]

        area_intersect, area_pred_label, area_label, area_union = self.compute_metric(classifier_scores.detach(), target.detach())
        metrics["area_intersect"] = area_intersect
        metrics["area_pred_label"] = area_pred_label
        metrics["area_label"] = area_label
        metrics["area_union"] = area_union

        if target_lowres is not None:
            target_lowres = target_lowres[~sample_masks_lowres] - self.seg_id_offset
            classifier_scores_lowres = classifier_scores_lowres[~sample_masks_lowres]

            area_intersect_lowres, area_pred_label_lowres, area_label_lowres, area_union_lowres = self.compute_metric(classifier_scores_lowres.detach(), target_lowres.detach())
            metrics["area_intersect_lowres"] = area_intersect_lowres
            metrics["area_pred_label_lowres"] = area_pred_label_lowres
            metrics["area_label_lowres"] = area_label_lowres
            metrics["area_union_lowres"] = area_union_lowres

        if resnet_postprocess_probability is not None:
            resnet_postprocess_probability = resnet_postprocess_probability[~sample_masks]

            area_intersect, area_pred_label, area_label, area_union = self.compute_metric(resnet_postprocess_probability.detach(), target.detach())
            metrics["area_intersect_resnet_postprocess"] = area_intersect
            metrics["area_pred_label_resnet_postprocess"] = area_pred_label
            metrics["area_label_resnet_postprocess"] = area_label
            metrics["area_union_resnet_postprocess"] = area_union
            
        if self.upscale_lprobs:
            loss = F.cross_entropy(classifier_scores, target.detach(), label_smoothing=self.eps) # just for display
        else:
            loss = F.cross_entropy(classifier_scores_lowres, target_lowres.detach(), label_smoothing=self.eps) # just for display
        
        metrics["nll_loss"] = loss
        ntokens = 1

        return loss, metrics, ntokens

    def compute_metric(self, lprobs, target):
        num_classes = lprobs.size(-1)
        pred_label = lprobs.argmax(-1)

        intersect = pred_label[pred_label == target]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
        area_label = torch.histc(
            target.float(), bins=(num_classes), min=0, max=num_classes - 1)
        area_union = area_pred_label + area_label - area_intersect
        
        return area_intersect, area_pred_label, area_label, area_union

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    def _lazy_initialization(self, sample, model, ema_model=None):
        if self.init_seg_with_text:
            def encode_text(text):
                line = [self.task.bpe.encode(' {}'.format(word.strip())) for word in text.strip().split()]
                line = ' '.join(line)
                
                s = self.task.tgt_dict.encode_line(
                    line=line,
                    add_if_not_exist=False,
                    append_eos=False
                ).long()
                return s

            with torch.inference_mode():
                id2text = [encode_text(f" {x}") for x in self.id2rawtext]
                id2text_tokens = torch.cat(id2text).cuda()
                
                text_length = torch.tensor([len(x) for x in id2text])
                start_offset = torch.cat([text_length.new_zeros(size=(1, ), dtype=torch.long), text_length.cumsum(dim=0)[:-1]], dim=0).cuda()
                
                avg_embedding = model.encoder.embed_tokens_bag(id2text_tokens, offsets=start_offset).data
            
            avg_embedding = avg_embedding.clone()
            model.encoder.seg_embed_tokens.weight.data = avg_embedding
            model.decoder.seg_embed_tokens.weight.data = avg_embedding
            if ema_model is not None:
                ema_model.encoder.seg_embed_tokens.weight.data = avg_embedding
                ema_model.decoder.seg_embed_tokens.weight.data = avg_embedding
            
            if not model.decoder.tie_seg_projection:
                model.decoder.seg_projection.weight.data = avg_embedding
                if ema_model is not None:
                    ema_model.decoder.seg_projection.weight.data = avg_embedding
                    
            logger.info("Initialized seg tokens with embedding bag.")
    
    def _update_iteration(self, sample, model, ema_model=False):
        # count effective iterations given iter and criterion_update_freq
        # if model.training:
        self.iter += 1
        self.effective_iter = self.iter // self.criterion_update_freq

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        imfree_loss_sum = sum(log.get("imfree_loss", 0) for log in logging_outputs)
        seg_loss_sum = sum(log.get("seg_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "imfree_loss", imfree_loss_sum / sample_size, ntokens, round=3
        )
        metrics.log_scalar(
            "seg_loss", seg_loss_sum / sample_size, ntokens, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / sample_size, ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        metrics.log_scalar(
            "ntokens", ntokens, 1, round=3
        )
        metrics.log_scalar(
            "nsentences", nsentences, 1, round=3
        )
        metrics.log_scalar(
            "sample_size", sample_size, 1, round=3
        )
        
        if "area_intersect_lowres" in logging_outputs[0]:
            area_intersect_lowres_sum = sum(log.get("area_intersect_lowres", 0) for log in logging_outputs)
            area_pred_label_lowres_sum = sum(log.get("area_pred_label_lowres", 0) for log in logging_outputs)
            area_label_lowres_sum = sum(log.get("area_label_lowres", 0) for log in logging_outputs)
            area_union_lowres_sum = sum(log.get("area_union_lowres", 0) for log in logging_outputs)

            metrics.log_scalar_sum(
                "_area_intersect_lowres", area_intersect_lowres_sum, 1
            )
            metrics.log_scalar_sum(
                "_area_pred_label_lowres", area_pred_label_lowres_sum, 1
            )
            metrics.log_scalar_sum(
                "_area_label_lowres", area_label_lowres_sum, 1
            )
            metrics.log_scalar_sum(
                "_area_union_lowres", area_union_lowres_sum, 1
            )
            
            def compute_all_acc_lowres(meters):
                all_acc = meters['_area_intersect_lowres'].sum.sum() / meters['_area_pred_label_lowres'].sum.sum()
                all_acc = all_acc if isinstance(all_acc, float) else all_acc.item()
                
                return round(all_acc, 4)

            def compute_mean_iou_lowres(meters):
                miou = torch.nanmean(meters['_area_intersect_lowres'].sum / (meters['_area_union_lowres'].sum))
                miou = miou if isinstance(miou, float) else miou.item()
                
                return round(miou, 4)

            def compute_mean_acc_lowres(meters):
                macc = torch.nanmean(meters['_area_intersect_lowres'].sum / (meters['_area_label_lowres'].sum))
                macc = macc if isinstance(macc, float) else macc.item()
                
                return round(macc, 4)

            metrics.log_derived("aAcc_lowres", compute_all_acc_lowres)
            metrics.log_derived("mIoU_lowres", compute_mean_iou_lowres)
            metrics.log_derived("mAcc_lowres", compute_mean_acc_lowres)

        if "area_intersect_resnet_postprocess" in logging_outputs[0]:
            area_intersect_resnet_postprocess_sum = sum(log.get("area_intersect_resnet_postprocess", 0) for log in logging_outputs)
            area_pred_label_resnet_postprocess_sum = sum(log.get("area_pred_label_resnet_postprocess", 0) for log in logging_outputs)
            area_label_sum_resnet_postprocess = sum(log.get("area_label_resnet_postprocess", 0) for log in logging_outputs)
            area_union_sum_resnet_postprocess = sum(log.get("area_union_resnet_postprocess", 0) for log in logging_outputs)

            metrics.log_scalar_sum(
                "_area_intersect_resnet_postprocess", area_intersect_resnet_postprocess_sum, 1
            )
            metrics.log_scalar_sum(
                "_area_pred_label_resnet_postprocess", area_pred_label_resnet_postprocess_sum, 1
            )
            metrics.log_scalar_sum(
                "_area_label_resnet_postprocess", area_label_sum_resnet_postprocess, 1
            )
            metrics.log_scalar_sum(
                "_area_union_resnet_postprocess", area_union_sum_resnet_postprocess, 1
            )
            
            def compute_all_acc_resnet_postprocess(meters):
                all_acc = meters['_area_intersect_resnet_postprocess'].sum.sum() / meters['_area_pred_label_resnet_postprocess'].sum.sum()
                all_acc = all_acc if isinstance(all_acc, float) else all_acc.item()
                
                return round(all_acc, 4)

            def compute_mean_iou_resnet_postprocess(meters):
                miou = torch.nanmean(meters['_area_intersect_resnet_postprocess'].sum / (meters['_area_union_resnet_postprocess'].sum))
                miou = miou if isinstance(miou, float) else miou.item()
                
                return round(miou, 4)

            def compute_mean_acc_resnet_postprocess(meters):
                macc = torch.nanmean(meters['_area_intersect_resnet_postprocess'].sum / (meters['_area_label_resnet_postprocess'].sum))
                macc = macc if isinstance(macc, float) else macc.item()
                
                return round(macc, 4)

            metrics.log_derived("aAcc_resnet_postprocess", compute_all_acc_resnet_postprocess)
            metrics.log_derived("mIoU_resnet_postprocess", compute_mean_iou_resnet_postprocess)
            metrics.log_derived("mAcc_resnet_postprocess", compute_mean_acc_resnet_postprocess)

        if "area_intersect" in logging_outputs[0]:
            area_intersect_sum = sum(log.get("area_intersect", 0) for log in logging_outputs)
            area_pred_label_sum = sum(log.get("area_pred_label", 0) for log in logging_outputs)
            area_label_sum = sum(log.get("area_label", 0) for log in logging_outputs)
            area_union_sum = sum(log.get("area_union", 0) for log in logging_outputs)

            metrics.log_scalar_sum(
                "_area_intersect", area_intersect_sum, 1
            )
            metrics.log_scalar_sum(
                "_area_pred_label", area_pred_label_sum, 1
            )
            metrics.log_scalar_sum(
                "_area_label", area_label_sum, 1
            )
            metrics.log_scalar_sum(
                "_area_union", area_union_sum, 1
            )

            def compute_all_acc(meters):
                all_acc = meters['_area_intersect'].sum.sum() / meters['_area_pred_label'].sum.sum()
                all_acc = all_acc if isinstance(all_acc, float) else all_acc.item()
                
                return round(all_acc, 4)

            def compute_mean_iou(meters):
                miou = torch.nanmean(meters['_area_intersect'].sum / (meters['_area_union'].sum))
                miou = miou if isinstance(miou, float) else miou.item()
                
                return round(miou, 4)

            def compute_mean_acc(meters):
                macc = torch.nanmean(meters['_area_intersect'].sum / (meters['_area_label'].sum)) # nanmean
                macc = macc if isinstance(macc, float) else macc.item()
                
                return round(macc, 4)

            metrics.log_derived("aAcc", compute_all_acc)
            metrics.log_derived("mIoU", compute_mean_iou)
            metrics.log_derived("mAcc", compute_mean_acc)

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
