export MODEL_NAME="stabilityai/stable-diffusion-2"
export OUTPUT_DIR="./db_out/pig"
export INSTANCE_DIR="./data/pig"
export CLASS_DIR="./data/cartoon_boy/reg"

# --with_prior_preservation --prior_loss_weight=1.0 \
# --class_data_dir=$CLASS_DIR \
# --class_prompt="photo of cartoon character" \
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --instance_data_dir=$INSTANCE_DIR \
  --instance_prompt="photo of zxw cartoon character" \
  --train_batch_size=2 \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --gradient_checkpointing \
  --num_class_images=200 \
  --max_train_steps=450
