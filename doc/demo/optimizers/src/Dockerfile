FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20240429.v1

ENV CONDA_ENVIRONMENT_PATH /azureml-envs/pyrit

# Prepend path to AzureML conda environment
ENV PATH $CONDA_ENVIRONMENT_PATH/bin:$PATH

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    conda run -p $CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y

RUN conda list

# TODO fix
RUN git clone -b romanlutz/gcg_azureml https://github.com/romanlutz/PyRIT.git

# TODO upgrade to next FastChat release
RUN cd PyRIT && pip install -e .[dev,torch] && pip install git+https://github.com/lm-sys/FastChat.git@main && pip install accelerate

RUN pip freeze
