# Abhijeet-LightweightFineTuning-Project

This is an example of using parameter efficient fine-tuning (LoRA) on
a sequence classification model using a downstream dataset.

Low ranked adaptation (LoRA) reduces the number of parameters to train making it
possible to run fine-tuning on commodity hardware with a smaller compute and
memory footprint.

Check out: https://github.com/huggingface/peft

## Prerequisutes

* Python 3 - https://www.python.org/downloads/
* Pycharm (optional) - https://www.jetbrains.com/pycharm/
* Pip (to install dependencies) - https://pypi.org/project/pip/

Dev environment with at least 16 GB RAM.

### Background reading

* Hugging Face: https://huggingface.co/
* Transformers API: https://huggingface.co/docs/transformers/index
* Datasets API: https://huggingface.co/docs/datasets/index
* Evaluate API: https://huggingface.co/docs/evaluate/index
* PEFT API: https://huggingface.co/docs/peft/index
* Auto classes: https://huggingface.co/docs/transformers/en/model_doc/auto
* Trainer: https://huggingface.co/docs/transformers/en/main_classes/trainer

### Foundation model

* GPT2: https://huggingface.co/openai-community/gpt2

### Training dataset

* Dair-AI/emotion: https://huggingface.co/datasets/dair-ai/emotion

### Install dependencies in your virtual environment using:

pip install -r requirements.txt

Checkout: https://stackoverflow.com/questions/41427500/creating-a-virtualenv-with-preinstalled-packages-as-in-requirements-txt

## Run script

The script is included as main method. You can run the script using pycharm or via command line.

## Current Limitation

Currently not using [Quantization](https://huggingface.co/docs/peft/developer_guides/quantization), since 
Quantization library (bitsandbytes) currently only has support for CUDA and lacks support for Metal 
(GPU available for Mac) 