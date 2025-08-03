

import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import requests
import os



class llavaImageCaptioner:
#loading model
    def __init__(self, model_name="llava-hf/llava-v1.6-mistral-7b-hf"):
       
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
        ).to(self.device)

        
#generate captions for images
    def generate_caption(self, image_path: str, question: str = "What is shown in this image?") -> str:
       
        if image_path.startswith("http"):
            image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
        elif os.path.isfile(image_path):
            image = Image.open(image_path).convert("RGB")
        else:
            raise ValueError(f"Invalid image path or URL: {image_path}")

        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, torch.float16)

        
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=520)

        return self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
