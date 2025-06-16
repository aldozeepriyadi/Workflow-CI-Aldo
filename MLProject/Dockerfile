# Step 1: Use official Python image as base
FROM python:3.12.2-slim

# Step 2: Set environment variables
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV CONDA_ENV_PATH=/opt/conda/envs/mlflow-env
ENV PATH=$CONDA_ENV_PATH/bin:$PATH

# Step 3: Install Miniconda (we'll need it to manage dependencies)
RUN apt-get update && apt-get install -y wget bzip2 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Step 4: Copy the environment YAML file and create Conda environment
COPY MLProject/conda.yaml /tmp/conda.yaml
RUN conda env create -f /tmp/conda.yaml && \
    conda clean --all --yes

# Step 5: Set working directory
WORKDIR /workspace

# Step 6: Install mlflow and other required dependencies
COPY . /workspace
RUN pip install mlflow

# Step 7: Set entry point to run the model in the container
CMD ["mlflow", "models", "serve", "-m", "/workspace/mlruns/0/artifacts/model", "--host", "0.0.0.0", "--port", "5000"]
