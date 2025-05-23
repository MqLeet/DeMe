import os
import torch
import numpy as np
from diffusers import DDPMPipeline, DDIMPipeline
import copy
from task_vectors import TaskVector



def main():

    # the finetuned diffusion models path here
    finetuned_paths = [    
    'ddpm-ema-church_low0_high250_P0.3/checkpoint-200000/unet_ema_lora/diffusion_pytorch_model.bin',
    'ddpm-ema-church_low250_high500_P0.3/checkpoint-200000/unet_ema_lora/diffusion_pytorch_model.bin',
    'ddpm-ema-church_low500_high750_P0.3/checkpoint-200000/unet_ema_lora/diffusion_pytorch_model.bin',
    'ddpm-ema-church_low750_high1000_P0.3/checkpoint-200000/unet_ema_lora/diffusion_pytorch_model.bin',]


    origin_path = 'ddpm-ema-church-256'

    learn_addition(origin_path, finetuned_paths, output_dir='merged_models')


def learn_addition(origin_path, finetuned_paths, output_dir):
    origin_path = 'ddpm-ema-church-256'
    pipeline = DDIMPipeline.from_pretrained(origin_path)
    origin_model = pipeline.unet
    origin_model.add_adapters()

    finetune_models = []

    for ft_path in finetuned_paths:
        finetuned_model = copy.deepcopy(origin_model)
        finetuned_checkpoint = torch.load(ft_path)

        if isinstance(finetuned_checkpoint, torch.nn.Module):
            finetuned_checkpoint = finetuned_checkpoint.state_dict()
        
        finetuned_model.load_state_dict(finetuned_checkpoint)
        finetune_models.append(finetuned_model)

    # Create the task vectors
    task_vectors = [
        TaskVector(pretrained_checkpoint=origin_model, finetuned_checkpoint=finetuned_model)
        for finetuned_model in finetune_models
    ]


    weights = np.array([0.1, 0.1, 0.2, 0.5]) # input your optimal weights combinations here
    task_vector_sum = sum(tv * w for w, tv in zip(weights, task_vectors))
    new_finetune_model = task_vector_sum.apply_to(origin_model)

    os.makedirs(output_dir, exist_ok=True)
    torch.save(new_finetune_model, f'{output_dir}/diffusion_pytorch_model.bin')
    print(f"new finetuned weights saved to {output_dir}")


if __name__ == '__main__':
    main()

