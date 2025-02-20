# multimodal-captions-2

## Environment setup

### Setup Instructions

1. [Install conda/miniconda](https://docs.anaconda.com/miniconda/install/)
2. Create the environment from `env/environment.yml`
```bash
conda env create -f env/environment.yml -y
conda activate multimodal-captions-2
# install pytorch via pytorch channel so that mps can be used on macos
conda install pytorch torchvision -c pytorch
conda deactivate # to exit
```
