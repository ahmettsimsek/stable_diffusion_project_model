# 🧠 Model Training with train_text_to_image_sdxl.py
The following command was used to train the model using Stable Diffusion XL (SDXL):

 ```
accelerate launch diffusers/examples/text_to_image/train_text_to_image_sdxl.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
--pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
--train_data_dir="dataset" --image_column="image" --caption_column="text" \
--resolution=512 --center_crop --random_flip --train_batch_size=1 \
--gradient_accumulation_steps=4 --learning_rate=1e-06 \
--lr_scheduler="constant" --mixed_precision="fp16" \
--report_to="wandb" --output_dir="sdxl-naruto-model"
 ```

✅ Training Summary
During the training process, we encountered some memory-related issues and missing dependencies. These were addressed by modifying the training configuration and adjusting some hyperparameters to ensure a more stable and efficient training pipeline.

🔧 Updates & Fixes:
Learning rate and batch size were fine-tuned to improve training performance.

max_train_steps was set to 100 to control the training duration and avoid overfitting or instability.
