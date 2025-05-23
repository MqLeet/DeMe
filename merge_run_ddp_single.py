import os
import torch
from diffusers import DDPMPipeline, DDIMPipeline
import argparse
import copy
import torch.multiprocessing as mp
from multiprocessing import Process

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--batchsize', type=int, required=True)
parser.add_argument('--num_images', type=int, required=True)
parser.add_argument('--num_inference_steps', type=int, required=True)
parser.add_argument('--path_addition_ckpts', type=str, required=True)
parser.add_argument('--prefix', type=int, default=0)
args = parser.parse_args()


def sample_images(gpu_id, num_gpus, args):
    rank = gpu_id
    prefix = gpu_id
    origin_path = 'ddpm-ema-church-256'
    origin_pipe = DDIMPipeline.from_pretrained(origin_path, torch_dtype=torch.float16)

    origin_model = origin_pipe.unet
    origin_model.add_adapters()
    origin_model.adapter_list.to(torch.float16)

    path_addition_ckpts = args.path_addition_ckpts
    print(path_addition_ckpts)
    finetune_weight = torch.load(path_addition_ckpts)

    origin_model.load_state_dict(finetune_weight.state_dict())
    origin_pipe.unet = origin_model



    # Move the pipeline to the specified GPU
    origin_pipe.to(f"cuda:{gpu_id}")
    generator = torch.Generator(device=origin_pipe.device).manual_seed(prefix)

    # Sample images
    num_batches_per_gpu = args.num_images // (args.batchsize * num_gpus) + 1
    for index in range(num_batches_per_gpu):
        images = origin_pipe(
            generator=generator,
            batch_size=args.batchsize,
            num_inference_steps=args.num_inference_steps,
        ).images
        for i, item in enumerate(images):
            img_index = (rank * num_batches_per_gpu + index) * args.batchsize + i
            item.save(f"{args.output}/{prefix}-fake-{img_index}.png")




if __name__ == "__main__":


    os.makedirs(args.output, exist_ok=True)
    num_gpus = torch.cuda.device_count()

    mp.set_start_method('spawn')
    
    for gpu_id in range(num_gpus):
        p = mp.Process(target=sample_images, args=(gpu_id, num_gpus, args))
        p.start()





