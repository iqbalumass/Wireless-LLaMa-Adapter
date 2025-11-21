import os
from llama.llama_adapter import LLaMA_adapter
import util.misc as misc
import util.extract_adapter_from_checkpoint as extract
from PIL import Image
import cv2
import torch
import llama

# ============================================================
# CONFIG
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# path to your LLaMA weights and tokenizer
llama_dir = "/work/pi_mshao_umassd_edu/Iqbal/RadarCLIP/llama_adapter/LLaMA-Adapter/LLaMA2-7B/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
llama_type = "7B"

llama_ckpt_dir = os.path.join(llama_dir, llama_type)
llama_tokenizer_path = os.path.join(llama_dir, "tokenizer.model")

# load your fine-tuned checkpoint (the last one is usually best)
finetuned_ckpt = "/work/pi_mshao_umassd_edu/Iqbal/RadarCLIP/llama_adapter/LLaMA-Adapter/llama_adapter_v2_multimodal7b/output/checkpoint-3.pth"

# where to save the extracted adapter
adapter_save_path = "/work/pi_mshao_umassd_edu/Iqbal/RadarCLIP/llama_adapter/LLaMA-Adapter/llama_adapter_v2_multimodal7b/output/adapter-7B.pth"

# ============================================================
# LOAD MODEL
# ============================================================
model = LLaMA_adapter(llama_ckpt_dir, llama_tokenizer_path)
misc.load_model(model, finetuned_ckpt)
model.eval().to(device)

# ============================================================
# INFERENCE TEST
# ============================================================
prompt = llama.format_prompt("Describe the object in the image.")
img_path = "/work/pi_mshao_umassd_edu/Iqbal/RadarCLIP/scenario36/unit1/rgb1/frame_13-37-29.198563.jpg"  # change this
img = Image.fromarray(cv2.imread(img_path))
img = model.clip_transform(img).unsqueeze(0).to(device)

result = model.generate(img, [prompt])[0]
print("Model output:", result)

# ============================================================
# SAVE ADAPTER
# ============================================================
extract.save(model, adapter_save_path, "BIAS")
print(f"Adapter saved to: {adapter_save_path}")
