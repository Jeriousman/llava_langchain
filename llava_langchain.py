'''
Reference:
https://medium.com/@shivansh.kaushik/implementing-large-multimodal-models-lmm-in-few-lines-of-code-using-langchain-and-ollama-6c08b1c25fdd
https://python.langchain.com/v0.2/docs/integrations/llms/ollama/
'''


from langchain_core.prompts import ChatPromptTemplate
import os
# from dotenv import load_dotenv
import shutil
import requests
from PIL import Image
import base64
import io
from io import BytesIO
import gradio as gr
# import chromadb
import numpy as np
from langchain.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
# from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

##Gradio
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
print(f'{os.environ["LANGCHAIN_API_KEY"]}', 'os.environ is working')
##loading LLaVA


# pdf_url = "https://www.getty.edu/publications/resources/virtuallibrary/0892360224.pdf"
# image_url = "https://img.freepik.com/premium-photo/sensual-elegant-young-fashionable-sexy-woman-beautiful-girl-fashion-model-fashion-lady-sexy-girl_882954-26431.jpg"
# image = Image.open(requests.get(image_url, stream=True).raw)


mllm = Ollama(model="llava:34b-v1.6-q4_0")

def convert_to_base64(pil_image: Image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def load_image(image_path: str):
    pil_image = Image.open(image_path)
    image_b64 = convert_to_base64(pil_image)
    # print("Loaded image successfully!")
    return image_b64

# from IPython.display import HTML, display
def plt_img_base64(img_base64):
    """
    Display base64 encoded string as image

    :param img_base64:  Base64 string
    """
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    # Display the image by rendering the HTML
    display(HTML(image_html))

# def take_inputs_then_answer(img_path:str, prompt:str, mllm):
#     # image_b64 = load_image("artifacts/00101.png")
#     # resp = mllm.invoke("Is the photo life-like natural?", images=[image_b64])
#     image_b64 = load_image(img_path)
#     response = mllm.invoke(prompt, images=[image_b64])
#     return response

def take_inputs_then_answer(img, prompt:str):
    # mllm = Ollama(model="llava-llama3:8b-v1.1-q4_0")
    # mllm = Ollama(model="llava:34b-v1.6-q4_0")
    
    image_b64 = load_image(img)
    response = mllm.invoke(prompt, images=[image_b64])
    return response



# prompt = '''
# You are an expert that checks if photos are really natural with real-world notions. 
# For example, people in photos cannot have more than 5 fingers because people only have 5 fingers in real-world.
# \n\n
# Is this photo realistic in real-world situation?
# '''


# prompt = '''
# You are a helpful assistant that helps me detect if provided photos are real or fake photos. 
# I want you to tell me 'real photo' even if the photo is digitally manipulated or generated in case the photo is natural and realistic. 
# Here are some fake photo examples.
# 1. There are people who have not 5 fingers.
# 2. There are people who have unnatural parts in their bodies.
# 3. There are people who have unusual artifacts in the photos.
# \n\n
# Is this photo a real photo? If it is not a real photo, can you tell me the reasons in detail?
# '''


# prompt1 = "Is the photo life-like natural? Is there any pixels with unnatural artifacts?"

if __name__=='__main__':
    gradio_demo = gr.Interface(fn=take_inputs_then_answer, 
                               inputs=[gr.Image(type="filepath"), gr.Textbox('''
You are a helpful assistant that helps me detect if provided photos are real or fake photos. 
I want you to tell me 'real photo' even if the photo is digitally manipulated or generated in case the photo is natural and realistic. 
Here are some fake photo examples.
1. There are people who have not 5 fingers.
2. There are people who have unnatural parts in their bodies.
3. There are people who have unusual artifacts in the photos.
\n\n
Is this photo a real photo? If it is not a real photo, can you tell me the reasons in detail?
''')], 
                               outputs='text',
                               title="LLaVA Image Checker Boom",
                               description="Please write prompt carefully for the MLLM model to work best")
    gradio_demo.launch(share=True)

## https://www.youtube.com/watch?v=tIU2tw3PMUE&list=PLQIgLu3Wf-q_Ne8vv-ZXuJ4mztHJaQb_v&index=5
## 1. document loaders

## 2. text splitters

## 3. vector embedding models

## 4. create vector store

## 5. retrieve relevant documents in vector store by retriever

## Create vector store and retain it

## Create MLLM prompt

## Create chain with MLLM prompt!


## Create retriever 
# retrieval_chain = create_retrieval_chain(retriever, document_chain)
# retrieval_chain.invoke({})

## Gradio connection to showcase!



# # Create chroma
# vectorstore = Chroma(
#     collection_name="photos", embedding_function=OpenCLIPEmbeddings()
# )

# # Add images  ##vectorstore.add_images will store / retrieve images as base64 encoded strings.
# vectorstore.add_images(uris='artifacts/00101.png')

# # Add documents
# # vectorstore.add_texts(texts=texts)

# # retriever
# retriever = vectorstore.as_retriever()




# def resize_base64_image(base64_string, size=(128, 128)):
#     """
#     Resize an image encoded as a Base64 string.

#     Args:
#     base64_string (str): Base64 string of the original image.
#     size (tuple): Desired size of the image as (width, height).

#     Returns:
#     str: Base64 string of the resized image.
#     """
#     # Decode the Base64 string
#     img_data = base64.b64decode(base64_string)
#     img = Image.open(io.BytesIO(img_data))

#     # Resize the image
#     resized_img = img.resize(size, Image.LANCZOS)

#     # Save the resized image to a bytes buffer
#     buffered = io.BytesIO()
#     resized_img.save(buffered, format=img.format)

#     # Encode the resized image to Base64
#     return base64.b64encode(buffered.getvalue()).decode("utf-8")


# def is_base64(s):
#     """Check if a string is Base64 encoded"""
#     try:
#         return base64.b64encode(base64.b64decode(s)) == s.encode()
#     except Exception:
#         return False


# def split_image_text_types(docs):
#     """Split numpy array images and texts"""
#     images = []
#     text = []
#     for doc in docs:
#         doc = doc.page_content  # Extract Document contents
#         if is_base64(doc):
#             # Resize image to avoid OAI server error
#             images.append(
#                 resize_base64_image(doc, size=(250, 250))
#             )  # base64 encoded str
#         else:
#             text.append(doc)
#     return {"images": images, "texts": text}
