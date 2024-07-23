FROM langchain/langchain
# FROM ollama/ollama:0.1.34


# ENV DEBIAN_FRONTEND=noninteractive
##
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC \
 && apt-get -y update \
 && apt-get -y install \
 && apt-get install -y lshw \
    poppler-utils \ 
    vim \
    libx264-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 \
    wget curl cmake build-essential pkg-config \
    libxext6 libxrender-dev git
    
RUN apt-get clean && rm -rf /tmp/* /var/tmp/* \
 && python3 -m pip install --upgrade pip \
 && python3 -m pip install --upgrade setuptools

# RUN python -m pip install --upgrade pip \
#  && python -m pip install --upgrade setuptools  
#  && apt-get install -y python3-pip

# RUN conda remove --force ffmpeg -y
# RUN curl -fsSL https://ollama.com/install.sh | sh ##download ollama but no need to do this when u are doing with docker image?
COPY . /workspace
WORKDIR /workspace
ENV LANGCHAIN_API_KEY=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 \
    OPENAI_API_KEY=sk-proj-vX6Cav2vLu68AFJY1ekIT3BlbkFJaWwzSJ8kRgZj4zAxsU2S


# FROM ollama/ollama
# RUN pip3 install --no-cache-dir -Iv \
#     numpy==1.20.3 opencv-python==4.5.5.64 onnx==1.12.0 \
#     onnxruntime-gpu==1.7.0 mxnet-cu111 mxnet-mkl scikit-image \
#     insightface==0.2.0 requests kornia==0.5.11 dill wandb \
#     notebook ipython ipykernel psutil==5.9.2


RUN curl -fsSL https://ollama.com/install.sh | sh
# RUN ollama serve
# RUN ollama run llava-llama3:8b
RUN pip3 install -r /workspace/requirements_llm_langchain.txt



# RUN pip uninstall -y torch torchvision

# RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
    # timm matplotlib
