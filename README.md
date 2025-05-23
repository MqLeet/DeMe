

<div align=center>
  
# **[CVPR2025]** Decouple-Then-Merge: Finetune Diffusion Models as Multi-Task Learning

<p>
<a href='https://arxiv.org/abs/2410.06664'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://mqleet.github.io/DeMe_Project/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
</p>

</div>


## :fire: News


- [2025/02/27] DeMe has been accepted to CVPR 2025! 🤗🤗
- [2025/05/23] Code has been released! 🤗🤗

## :memo: TODO

- [x] Training code
- [x] Inference code
- [x] Project page
- [ ] Journal version


## 🛠️ Getting Started
To setup the environment of DeMe, follow the installation instructions below.

### 1. Clone the github repo
```bash
git clone https://github.com/MqLeet/DeMe.git
```

### 2. Create conda environment
```bash
cd DeMe
conda env create -f environment.yml
conda activate DeMe
```


### 3. Prepare the pretrained model (An example on LSUN-Church)

We use an official DDPM model offered by Google in Huggingface, named [ddpm-ema-church-256](https://huggingface.co/google/ddpm-ema-church-256). You can download the pretrained DDPM model at `ddpm-ema-church-256/` for the following experiments.

The used pretrained models can be found here:

| Model | Link |
|---|---|
| ddpm-cifar10-32 | https://huggingface.co/google/ddpm-cifar10-32 |
| ddpm-ema-church-256 | https://huggingface.co/google/ddpm-ema-church-256 |
| ddpm-ema-bedroom-256 | https://huggingface.co/google/ddpm-ema-bedroom-256 |
|stable-diffusion-v1-4 |https://huggingface.co/CompVis/stable-diffusion-v1-4|



### 4. Dataset preparation (An example on LSUN-Church)

For convenience, we use [LSUN-Church](https://huggingface.co/datasets/tglcourse/lsun_church_train) from Huggingface without downloading data by using the `load_dataset` function offered by the `datasets` library. Additionally, you can also download the dataset yourself.


For evaluation, you can download the `.npz` files at [Google Drive](https://drive.google.com/file/d/1sEzXxkXApO59fEv3jFFjQp4sHEwSujuw/view?usp=drive_link)



## 📘 Usage (Experiments on LSUN-Church as an example)

### *Decouple*: Finetune DDPM at 4 Different Timestep Ranges
Finetune the pretrained DDPM and save finetuned model
```bash
sh lsun_church_deme.sh
```

### *Merge*: Merge Finetuned Diffusion Models into One Unified Model
Compute task vectors and merge finetuned diffusion models into one diffusion model
```python
python merge.py
```

*For the convenience of the users, we have provided the merged model weights on the LSUN-Church, which can be downloaded at [Google Drive](https://drive.google.com/file/d/1v0rnyNyQOgTzvD5RLsBYpDAN1y21c50l/view?usp=sharing). You can download the merged model weights to path 'merged_models/diffusion_pytorch_model.bin' and continue to evaluate the FID in the following steps.*


### *Inference*:
To accelerate sampling process, we use multiple GPUs to sample
```bash
sh merge_run_ddp_single.sh
```

### *Evaluation*: 
We offer a `church_fid.npz` file here to evaluate the generated images quality, use the command line to compute FID
```bash
python -m pytorch_fid "ddim_samples/merge/reproduce_deme" "church_fid.npz"
```
to reproduce the results demonstrated in paper.




<a name="citation_and_acknowledgement"></a>
## :black_nib: Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
   @InProceedings{ma2024decouple,
         title={Decouple-Then-Merge: Towards Better Training for Diffusion Models},
         author={Ma, Qianli and Ning, Xuefei and Liu, Dongrui and Niu, Li and Zhang, Linfeng},
         booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
         year={2025}
         }
   ```


## :heart: Acknowledgement

Our code is built upon [diffusers](https://github.com/huggingface/diffusers) and [task vectors](https://github.com/mlfoundations/task_vectors). We also refer to the [model soups](https://github.com/mlfoundations/model-soups) and [dnn mode connectivity](https://github.com/timgaripov/dnn-mode-connectivity). Thanks to the contributors for their great work!
