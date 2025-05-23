LEARNING_RATE=5e-5
MAX_TRAIN_ITERS=200000
CHECKPOINTING_STEPS=5000

KD_LOW=0
KD_HIGH=250
WHICH_UNET=0
P=0.3

accelerate launch train_unconditional_deme.py  \
    --dataset_name="tglcourse/lsun_church_train"  --resolution=256  --output_dir="ddpm-ema-church_low${KD_LOW}_high${KD_HIGH}_P${P}"  --gradient_accumulation_steps=4  --learning_rate=$LEARNING_RATE  --mixed_precision=fp16 --model_config_name_or_path "ddpm-ema-church-256" --train_batch_size 16 --checkpointing_steps=$CHECKPOINTING_STEPS \
    --kd_low=$KD_LOW --kd_high=$KD_HIGH --kd_global_ratio=$P --num_iters=$MAX_TRAIN_ITERS --use_ema \
    --lr_scheduler="constant" --lr_warmup_steps=0 --resume_from_checkpoint 'latest' \
    --add_lora_layer \
