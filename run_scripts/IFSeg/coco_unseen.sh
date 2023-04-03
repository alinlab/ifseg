#!/usr/bin/env
# fuser -k /dev/nvidia*
export MASTER_ADDR=127.0.0.1
export RANK=0
export MASTER_PORT=9999
GPUS_PER_NODE=4

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
bpe_dir=./utils/BPE
user_dir=./ofa_module

data_dir=./dataset/coco
data=${data_dir}/unseen_val2017.tsv,${data_dir}/unseen_val2017.tsv
num_seg_tokens=15
selected_cols=0,1,2
category_list='frisbee, skateboard, cardboard, carrot, scissors, suitcase, giraffe, cow, road, concrete wall, tree, grass, river, clouds, playingfield'

task=segmentation
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_tgt_length=20
num_bins=1000
criterion=seg_criterion

decoder_type=surrogate
freeze_entire_resnet=true
freeze_embedding=true
freeze_embedding_iter=-1 # includes freezing of the seg projection weights

batch_size=4
update_freq=1
criterion_update_freq=${update_freq}
patch_image_size=512
orig_patch_image_size=512 # for evaluation
prompt_prefix='what is the segmentation map of the image? object:'
upscale_lprobs=true
init_seg_with_text=true
tie_seg_projection=true
decoder_input_type=encoder_output
full_context_alignment=false
lr_scheduler=cosine

modelsize=base
restore_file=ofa_${modelsize}.pt
arch=segofa_${modelsize}
log_root=./experiment_outputs

### image-free / supervised option
unsupervised_segmentation=true
artificial_image_type=rand_k-1-33
lr=5.0e-5
wd=0.1
warmup_ratio=0.0
label_smoothing=0.0
max_epoch=20
resnet_topk=3
resnet_iters=25
iters_per_epoch=100
epoch_row_count=$[16*iters_per_epoch]
debugging="--epoch-row-count=${epoch_row_count} --resnet-topk=${resnet_topk} --resnet-iters=${resnet_iters} --num-workers=0"

session_name=coco_unseen

log_file=${log_root}/${session_name}/"logging.log"
save_path=${log_root}/${session_name}

mkdir -p $save_path
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ./train.py \
    $data \
    --selected-cols=${selected_cols} \
    --freeze-encoder-embedding=${freeze_embedding} --freeze-decoder-embedding=${freeze_embedding} --freeze-seg-embedding=${freeze_embedding} --freeze-entire-resnet=${freeze_entire_resnet} \
    --bpe-dir=${bpe_dir} \
    --user-dir=${user_dir} \
    --restore-file=${restore_file} \
    --reset-optimizer --reset-dataloader --reset-meters \
    --save-dir=${save_path} \
    --task=${task} \
    --arch=${arch} \
    --criterion=${criterion} \
    --label-smoothing=${label_smoothing} \
    --batch-size=${batch_size} \
    --batch-size-valid=1 \
    --update-freq=${update_freq} \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --share-decoder-input-output-embed \
    --share-all-embeddings \
    --layernorm-embedding \
    --find-unused-parameters \
    --patch-layernorm-embedding \
    --code-layernorm-embedding \
    --resnet-drop-path-rate=${resnet_drop_path_rate} \
    --encoder-drop-path-rate=${encoder_drop_path_rate} \
    --decoder-drop-path-rate=${decoder_drop_path_rate} \
    --dropout=${dropout} \
    --attention-dropout=${attention_dropout} \
    --weight-decay=${wd} --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=1.0 \
    --lr-scheduler=${lr_scheduler} --lr=${lr} \
    --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
    --log-format=simple --log-interval=10 \
    --fixed-validation-seed=7 \
    --keep-best-checkpoints=1 \
    --keep-last-epochs=1 \
    --save-interval=1 --validate-interval=1 \
    --eval-args='{"beam":5,"max_len":1024,"min_len":1024,"no_repeat_ngram_size":0}' \
    --best-checkpoint-metric=mIoU --maximize-best-checkpoint-metric \
    --max-src-length=${max_src_length} \
    --max-tgt-length=${max_tgt_length} \
    --add-type-embedding \
    --scale-attn \
    --scale-fc \
    --scale-heads \
    --disable-entangle \
    --num-bins=${num_bins} \
    --patch-image-size=${patch_image_size} \
    --orig-patch-image-size=${orig_patch_image_size} \
    --fp16 \
    --fp16-scale-window=512 \
    --upscale-lprobs=${upscale_lprobs} \
    --criterion-update-freq=${criterion_update_freq} \
    --unsupervised-segmentation=${unsupervised_segmentation} \
    --freeze-embedding-iter=${freeze_embedding_iter} \
    --tie-seg-projection=${tie_seg_projection} \
    --init-seg-with-text=${init_seg_with_text} \
    --decoder-type=${decoder_type} \
    --artificial-image-type=${artificial_image_type} \
    --full-context-alignment=${full_context_alignment} \
    --decoder-input-type=${decoder_input_type} \
    --num-seg-tokens=${num_seg_tokens} \
    --prompt-prefix="${prompt_prefix}" \
    --category-list="${category_list}" \
    ${debugging} 2>&1 | tee ${log_file}