export MODEL_NAME="stabilityai/stable-diffusion-2"
export OUTPUT_DIR="./generated_images/robo"
export INSTANCE_DIR="./data/cartoon_boy"

accelerate launch train.py \
 --pretrained_model_name_or_path=$MODEL_NAME  \
 --instance_data_dir=$INSTANCE_DIR \
 --output_dir=$OUTPUT_DIR \
 --instance_prompt="photo of a <new> cartoon character"  \
 --resolution=1024  \
 --mixed_precision=fp16 \
 --train_batch_size=2  \
 --learning_rate=1e-5  \
 --lr_warmup_steps=0 \
 --max_train_steps=2500 \
 --freeze_model crossattn \
 --scale_lr --hflip  \
 --modifier_token "<new>" \
 --enable_xformers_memory_efficient_attention

python3 inference.py