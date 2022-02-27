FROM tensorflow/tensorflow:latest-gpu-jupyter

# update apt and get miniconda
RUN apt-get update \
    && apt-get install -y wget libnuma-dev \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    
# install miniconda
ENV PATH="/root/miniconda3/bin:$PATH"
RUN mkdir /root/.conda && bash Miniconda3-latest-Linux-x86_64.sh -b

RUN conda update -n base -c defaults conda

COPY environment.yml .
RUN conda env create -f environment.yml

RUN conda init bash \
    && . ~/.bashrc \
    && source activate rapids_env \
    && pip install -U ipykernel tensorflow cupy-cuda112 cupy-cuda115 \
    && python -m ipykernel install --user --name rapids_env --display-name "Python (rapids_env)" 
    

WORKDIR "/trading"

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
