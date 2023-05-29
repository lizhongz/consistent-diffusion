export MODEL_NAME="stabilityai/stable-diffusion-2"
export OUTPUT_DIR="./out/robo"
export INSTANCE_DIR="./data/cartoon_boy"

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --concepts_list="./data/cartoon_boy_and_style_1.json" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=1500 \
  --num_class_images=200 \
  --scale_lr --hflip  \
  --mixed_precision=fp16 \
  --modifier_token "<c1>+<s1>" 
