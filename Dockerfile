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

# RUN curl -fsSL https://ollama.com/install.sh | sh ##download ollama but no need to do this when u are doing with docker image?
COPY . /workspace
WORKDIR /workspace


RUN curl -fsSL https://ollama.com/install.sh | sh
# RUN ollama serve
# RUN ollama run llava-llama3:8b
RUN pip3 install -r /workspace/requirements_llm_langchain.txt


