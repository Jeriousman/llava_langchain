# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 05:47:10 2024

@author: Hojun
"""

import base64
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

IMAGE_PATH = r"C:\Users\Hojun\Desktop\DOB\PROJECTs\LLM\LLaVA\images\trump2.jpg"



img = Image.open(IMAGE_PATH)
plt.imshow(img)

# Open the image file and encode it as a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image(IMAGE_PATH)


from openai import OpenAI
## Set the API key
client = OpenAI(api_key="sk-proj-H6lBFUuqY8c9BKH3jwp5T3BlbkFJdsOWOeX8VKt4wi2W4b1h")



MODEL="gpt-4o"
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": 
         '''
         You are a helpful assistant that helps me detect if provided photos are real or fake photos. 
         I want you to tell me 'real photo' even if the photo is digitally manipulated or generated in case the photo is natural and realistic. 
         Here are some fake photo examples.
         1. There are people who have not 5 fingers.
         2. There are people who have unnatural parts in their bodies.
         3. There are people who have unusual artifacts in the photos.
         '''},
        {"role": "user", "content": [
            {"type": "text", "text": "Is this photo a real photo? If it is not a real photo, can you tell me the reasons in detail?"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{base64_image}"}
            }
        ]}
    ],
    temperature=0.5,
)
print(response.choices[0].message.content)

