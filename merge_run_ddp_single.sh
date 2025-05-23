#!/bin/bash

EXPNAME=reproduce_deme
SAMPLES=50000
INFERS=50
PATH_ADDITION_CKPTS='merged_models/diffusion_pytorch_model.bin' # your merged model path


python merge_run_ddp_single.py --output "ddim_samples/merge/reproduce_deme" --batchsize 32 --num_images $SAMPLES --num_inference_steps $INFERS --path_addition_ckpts $PATH_ADDITION_CKPTS \
