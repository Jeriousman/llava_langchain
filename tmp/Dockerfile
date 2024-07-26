FROM langchain/langchain:0.0.344

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



COPY . /workspace
WORKDIR /workspace

RUN curl -fsSL https://ollama.com/install.sh | sh
RUN pip3 install --no-cache-dir -Iv tqdm
RUN pip3 install -r /workspace/requirements_llm_langchain.txt


ENV LANGCHAIN_API_KEY=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 \
    OPENAI_API_KEY=sk-proj-H6lBFUuqY8c9BKH3jwp5T3BlbkFJdsOWOeX8VKt4wi2W4b1h



