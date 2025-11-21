# llama_adapter_v2_multimodal7b/datasets/radar_text_dataset.py
import torch, json
from torch.utils.data import Dataset
from PIL import Image
from transformers import BlipProcessor

class RadarTextDataset(Dataset):
    def __init__(self, json_path, processor=None):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.processor = processor or BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        radar_path, text = sample["image"], sample["text"]
        radar_img = Image.open(radar_path).convert("RGB")  # or radar heatmap
        inputs = self.processor(images=radar_img, text=text, return_tensors="pt", padding="max_length", truncation=True)
        return {k: v.squeeze(0) for k, v in inputs.items()}
