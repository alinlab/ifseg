# IFSeg: Image-free Semantic Segmentation via Vision-Language Model
The official codebase for IFSeg: Image-free Semantic Segmentation via Vision-Language Model (CVPR 2023)

<img width="1381" alt="ifseg_method" src="https://user-images.githubusercontent.com/20102/228140458-d36e4050-93a0-4fe3-b508-c4943892893d.png">

This codebase is largely derived from <a href="https://github.com/OFA-Sys/OFA">OFA</a>.
<br></br>

# Requirements
* Python 3.9.15
* PyTorch 1.12.1+cu116
* torchvision 0.13.1+cu116
* mmsegmentation v0.28.0
<br></br>

Install PyTorch and torchvision
```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

Install mmsegmentation
```bash
pip install openmim
mim install mmcv-full==1.6.2
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation && git checkout v0.28.0 && pip install -v -e .
```

Install other dependencies
```bash
pip install -e ./custom_fairseq/
pip install -r requirements.txt
```
<br></br>

# Training and Inference
## Image Processing
To ensure the efficiency of processing data, we did not store images with small files, but instead we encode them to base64 strings, following the procedure described in <a href="https://github.com/OFA-Sys/OFA/blob/main/datasets.md">OFA datasets preparation guide</a>.


<b>1. Prepare the Dataset and download pretrained checkpoint </b>

Download the COCO Stuff images and annotations from <a href="https://cocodataset.org">https://cocodataset.org</a> and the build the <code>unseen_val2017.tsv</code> and <code>fineseg_refined_val2017.tsv</code> with the example notebook <code>convert_segmentation_unseen_split.ipynb</code> and <code>convert_segmentation_coco.ipynb</code>.

For ADE20K, download images and annotations from <a href="https://groups.csail.mit.edu/vision/datasets/ADE20K/">https://groups.csail.mit.edu/vision/datasets/ADE20K/</a> and build "validation.tsv" with the example notebook <code>convert_segmentation_ade.ipynb</code>.

The pretrianed OFA checkpoint is available at <a href="https://github.com/OFA-Sys/OFA/blob/main/checkpoints.md">https://github.com/OFA-Sys/OFA/blob/main/checkpoints.md/</a>. Specifically, we require the OFA-Base model; <code>ofa_base.pt</code>.

We recommend that your workspace directory should be organized like this:
```
IFSeg/
├── criterions/
├── data/
├── dataset/
│   ├── coco/unseen_val2017.tsv; fineseg_refined_val2017.tsv
│   └── ade/ade_valid.tsv
├── custom_fairseq/
├── models/
├── run_scripts/
│   ├── IFSeg/coco_unseen.sh
│   ├── IFSeg/ade.sh
│   └── IFSeg/coco_fine.sh
├── tasks/
├── train.py
├── trainer.py
├── convert_segmentation_coco_unseen_split.ipynb
├── convert_segmentation_coco.ipynb
├── convert_segmentation_ade.ipynb
├── visualize_segmentation_web.ipynb
└── utils/
```
<br></br>
<b>2. Finetuning and Inference Scripts </b>

For running the image-free experiment for 15 unseen COCO categories (Table 1), refer to <code>run_scripts/IFSeg/coco_unseen.sh</code>

For running the image-free experiment for 150 ADE categories (Table 2), refer to <code>run_scripts/IFSeg/ade.sh</code>

For running the image-free experiment for 171 COCO-stuff categories (Table 3), refer to <code>run_scripts/IFSeg/coco_fine.sh</code>
<br></br>

# Visualizing the results
To obtain the web image visualization, follow the directions in <code>visualize_segmentation_web.ipynb</code>

The pre-trained checkpoint for the visualization can be downloaded from https://drive.google.com/file/d/167sIrrSsBTRQlrVHYMKYoWA5A9r04eAD/view?usp=sharing

One may also produce their own checkpoint with novel semantic categories. For example, based on a example script in <code>./run_scripts/IFSeg</code>, modify <code>category_list</code> and <code>num_seg_tokens</code> for your segmentation setting.

<img width="543" alt="ifseg_vis" src="https://user-images.githubusercontent.com/20102/228140726-a683839d-5038-4961-8f94-5c5a9b3dac2c.png">
<br></br>

# Related Codebase
* [OFA](https://github.com/OFA-Sys/OFA)
* [Fairseq](https://github.com/pytorch/fairseq)
* [taming-transformers](https://github.com/CompVis/taming-transformers)
<br></br>

# Citation
Please cite our paper if you find it helpful :)

```
@inproceedings{yun2023ifseg,
  title     = {IFSeg: Image-free Semantic Segmentation via Vision-Language Model},
  author    = {Sukmin Yun and
               Seong Hyeon Park and
               Paul Hongsuck Seo and
               Jinwoo Shin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
<br></br>
