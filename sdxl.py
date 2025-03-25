import os
import subprocess

def train_sdxl():
    # Set environment variables
    os.environ["MODEL_NAME"] = "stabilityai/stable-diffusion-xl-base-1.0"
    os.environ["VAE_NAME"] = "madebyollin/sdxl-vae-fp16-fix"
    os.environ["DATASET_NAME"] = "lambdalabs/naruto-blip-captions"
    
    # Define the training command
    command = [
        "accelerate", "launch", "train_text_to_image_sdxl.py",
        "--pretrained_model_name_or_path", os.environ["MODEL_NAME"],
        "--pretrained_vae_model_name_or_path", os.environ["VAE_NAME"],
        "--dataset_name", os.environ["DATASET_NAME"],
        "--enable_xformers_memory_efficient_attention",
        "--resolution=512", "--center_crop", "--random_flip",
        "--proportion_empty_prompts=0.2",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4", "--gradient_checkpointing",
        "--max_train_steps=10000",
        "--use_8bit_adam",
        "--learning_rate=1e-06", "--lr_scheduler=constant", "--lr_warmup_steps=0",
        "--mixed_precision=fp16",
        "--report_to=wandb",
        "--validation_prompt=a cute Sundar Pichai creature", "--validation_epochs", "5",
        "--checkpointing_steps=5000",
        "--output_dir=sdxl-naruto-model",
        "--push_to_hub"
    ]
    
    # Run the command
    subprocess.run(command, check=True)

if __name__ == "__main__":
    train_sdxl()