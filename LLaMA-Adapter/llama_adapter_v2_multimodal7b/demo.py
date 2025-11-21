import cv2
import llama
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

llama_dir = "/work/pi_mshao_umassd_edu/Iqbal/RadarCLIP/llama_adapter/LLaMA-Adapter/LLaMA2-7B/models--meta-llama--Llama-2-7b-hf/"

# choose from BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
model, preprocess = llama.load("BIAS-7B", llama_dir, device)
model.eval()

prompt = llama.format_prompt('Please introduce this image.')
img = Image.fromarray(cv2.imread("/work/pi_mshao_umassd_edu/Iqbal/RadarCLIP/scenario36/unit1/rgb1/frame_11-46-31.205818.jpg"))
img = preprocess(img).unsqueeze(0).to(device)

result = model.generate(img, [prompt])[0]

print(result)
