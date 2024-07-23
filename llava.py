# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:56:34 2024

@author: Hojun
"""

import requests
from PIL import Image
from transformers import pipeline
import torch
from transformers import BitsAndBytesConfig


# image_url = "https://img.freepik.com/premium-photo/sensual-elegant-young-fashionable-sexy-woman-beautiful-girl-fashion-model-fashion-lady-sexy-girl_882954-26431.jpg"
# image_url = "https://pds.joongang.co.kr/news/component/htmlphoto_mmdata/201701/19/htm_2017011914219153477.jpg"
image_path = '/home/hojun/projects/llm/llava-rag/artifacts/00002.png'
image = Image.open(image_path)
# image = Image.open(requests.get(image_url, stream=True).raw)




## 4bit quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)


## loading a pre-trained checkpoint
model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})  ##LLaVA belongs to image-to-text category


max_new_tokens = 200
# prompt = "USER: <image>\nWhat is unnatural in the photo?\nASSISTANT:"
prompt = "USER: <image>\nDoes this photo have any unnatural artifacts unlike real-world photos? What is unnatural in the photo compared to real-world photos?\nASSISTANT:"
# prompt = "USER: <image>\nDoes this photo have any unnatural artifacts unlike real-world photos? If so, say unnatural. Otherwise, say natural. You can only say either unnatural and natural.\nASSISTANT:"

outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs[0]["generated_text"])