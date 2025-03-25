from diffusers import StableDiffusionPipeline

# Modeli Hugging Face'ten yükle
model_name = "./output_model"
pipeline = StableDiffusionPipeline.from_pretrained(model_name)

# Prompt'u tanımla
prompt = "a guy driving car"
# Görüntüyü oluştur
image = pipeline(prompt).images[0]

# Görüntüyü kaydet
image.save("output3.png")
print("✅ Görüntü başarıyla oluşturuldu ve kaydedildi: output.png")