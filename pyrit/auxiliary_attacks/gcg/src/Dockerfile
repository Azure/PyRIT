FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:7

RUN apt-get update

ENV CONDA_ENVIRONMENT_PATH /azureml-envs/pyrit

# Prepend path to AzureML conda environment
ENV PATH $CONDA_ENVIRONMENT_PATH/bin:$PATH

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y

RUN conda list

RUN git clone -b romanlutz/gcg_azureml https://github.com/blakebullwinkel/PyRIT.git

RUN cd PyRIT && pip install -e .[dev,gcg]

RUN pip freeze
