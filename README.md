# PathLDM: Text conditioned Latent Diffusion Model for Histopathology

Official code for our WACV 2024 publication [PathLDM: Text conditioned Latent Diffusion Model for Histopathology.](https://openaccess.thecvf.com/content/WACV2024/papers/Yellapragada_PathLDM_Text_Conditioned_Latent_Diffusion_Model_for_Histopathology_WACV_2024_paper.pdf) This codebase builds heavily on [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)

## Requirements
To install python dependencies, 

```
conda env create -f environment.yaml
conda activate ldm
```

## Downloading + Organizing Data

We obtained machine readable text reports for TCGA from [here](https://github.com/tatonetti-lab/tcga-path-reports), and used GPT-3.5 to summarize them. Summaries of all BRCA reports can be found [here](https://drive.google.com/drive/folders/1it4W4DBN4xFrLFX3nyVGoW0mTVklF6WY?usp=sharing).

We then used the [DSMIL](https://github.com/binli123/dsmil-wsi) repository to extract 256 x 256 patches @ 10x magnification, resulting in 3.2 million patches for TCGA-BRCA. 

## Pretrained models

We provide the following trained models

| Conditioning network |  Conditioning  type  |          Modality         |  FID  | Link |
|:--------------------:|:--------------------:|:-------------------------:|:-----:|:----:|
|    Class embedder    |      Tumor + TIL     |  Class label (4 classes)  | 29.45 | [link](https://drive.google.com/drive/folders/1OzFDEWlqXHUTAG5IGEVviObj10A66mOu?usp=sharing) |
|      OpenAI CLIP     | Report + tumor + TIL | Text caption (154 tokens) | 10.64 | [link](https://drive.google.com/drive/folders/1UTbWXq5wZWdb6_DJpBcygYZvBaPKabHf?usp=sharing) |
|         PLIP         | Report + tumor + TIL | Text caption (154 tokens) |  7.64 | [link](https://drive.google.com/drive/folders/1v3SXkA1D94w7Q1XMPSEA1yrSfpwhXzCr?usp=sharing) |


## Training

To train a diffusion model, create a config file similar to [this](./configs/latent-diffusion/text_cond/plip_imagenet_finetune.yaml) and create / update the corresponding dataloader (ex [this](./ldm/data/text_cond/tumor_til_in_text.py)). To download frozen VAEs, follow instructions in the [original LDM repo](https://github.com/CompVis/latent-diffusion/tree/main).

Example training command :
```
python main.py -t --gpus 0,1 --base configs/latent-diffusion/text_cond/plip_imagenet_finetune.yaml 
```


## BibTeX

```
@InProceedings{Yellapragada_2024_WACV,
    author    = {Yellapragada, Srikar and Graikos, Alexandros and Prasanna, Prateek and Kurc, Tahsin and Saltz, Joel and Samaras, Dimitris},
    title     = {PathLDM: Text Conditioned Latent Diffusion Model for Histopathology},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {5182-5191}
}
```


