# PathLDM: Text conditioned Latent Diffusion Model for Histopathology

Official code for our WACV 2024 publication [PathLDM: Text conditioned Latent Diffusion Model for Histopathology.](https://openaccess.thecvf.com/content/WACV2024/papers/Yellapragada_PathLDM_Text_Conditioned_Latent_Diffusion_Model_for_Histopathology_WACV_2024_paper.pdf) This codebase builds heavily on [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)

## Updates
💥 Check out our CVPR paper - [Learned representation-guided diffusion models for large-image generation
](https://histodiffusion.github.io/docs/publications/cvpr_24), where we train histopathology diffusion models without labeled data.

## Requirements
To install python dependencies, 

```
conda env create -f environment.yaml
conda activate ldm
```

## Downloading + Organizing Data

>**tl;dr** : TCGA-BRCA Image patches, captions and Tumor/TIL probabilities used in our training can be downloaded from [this link](https://drive.google.com/drive/folders/1MPBsVjh7q57DzYJXSLF2wkKjssw3jEtF?usp=sharing). See [this file](https://github.com/cvlab-stonybrook/PathLDM/blob/main/ldm/data/text_cond/tumor_til_in_text.py) for the Dataset class we use during training. 


We obtained machine readable text reports for TCGA from [this repo](https://github.com/tatonetti-lab/tcga-path-reports), and used GPT-3.5 to summarize them. Summaries of all BRCA reports can be found at [this link](https://drive.google.com/drive/folders/1it4W4DBN4xFrLFX3nyVGoW0mTVklF6WY?usp=sharing).

### Obtaining Tumor and TIL probabilities

We used [wsinfer](https://wsinfer.readthedocs.io/en/latest/) to obtain tumor and TIL probabilities. Wsinfer works directly with the WSI files, and outputs a csv with the probabilities for each patch, but the size and magnification might be different from the patches extracted by DSMIL. For each 10x patch, we use the average probabilities of the overlapping patches from wsinfer.  


### Download the WSIs


We used the [DSMIL](https://github.com/binli123/dsmil-wsi) repository to extract 256 x 256 patches @ 10x magnification, resulting in 3.2 million patches for TCGA-BRCA. The following steps are borrowed from the DSMIL repository. 



>**From GDC data portal.** You can use [GDC data portal](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Getting_Started/) with a manifest file and configuration file. The raw WSIs take about 1TB of disc space and may take several days to download. Please check [details](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Getting_Started/) regarding the use of TCGA data portal. Otherwise, individual WSIs can be download manually in GDC data portal [repository](https://portal.gdc.cancer.gov/repository?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22content%22%3A%7B%22field%22%3A%22files.cases.primary_site%22%2C%22value%22%3A%5B%22bronchus%20and%20lung%22%5D%7D%2C%22op%22%3A%22in%22%7D%2C%7B%22content%22%3A%7B%22field%22%3A%22files.data_format%22%2C%22value%22%3A%5B%22svs%22%5D%7D%2C%22op%22%3A%22in%22%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.experimental_strategy%22%2C%22value%22%3A%5B%22Diagnostic%20Slide%22%5D%7D%7D%5D%7D)  

### Prepare the patches

Once you clone the DSMIL repository, you can use the following command to extract patches from the WSIs. 

```
$ python deepzoom_tiler.py -m 0 -b 10
```



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

## Sampling

[This notebook](./example_sampling.ipynb) shows how to sample from the text conditioned diffusion model. 


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


