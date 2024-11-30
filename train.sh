# Train the LDM from scracth with a flan-t5-large text encoder
export CUDA_VISIBLE_DEVICES=5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_NET_GDR_LEVEL=5
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_TIMEOUT=7200000
export TORCH_DISTRIBUTED_DEBUG=DETAIL


nohup accelerate launch --num_processes=2 --main_process_port 29501 train.py \
--train_file="/data4/xiongchenxu/tango/tango/data/train_v2.json" --validation_file="/data4/xiongchenxu/tango/tango/data/valid_v2.json" \
--text_encoder_name="path/to/local/google/flan-t5-large" --scheduler_name="stabilityai/stable-diffusion-2-1" \
--unet_model_config="configs/diffusion_model_config.json" --freeze_text_encoder \
--gradient_accumulation_steps 4 --per_device_train_batch_size=2 --per_device_eval_batch_size=2 --augment \
--num_train_epochs 5 --snr_gamma 5 --resume_from_checkpoint "/data4/xiongchenxu/tango/tango/saved/1725195004/best_model.bin" \
--text_column captions --audio_column location --checkpointing_steps="best" > train_output_attn.log 2>&1 &





