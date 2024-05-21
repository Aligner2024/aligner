<h1 align="center">Achieving Efficient Alignment through Learned Correction  </h1>

With the rapid development of large language models (LLMs) and ever-evolving practical requirements, finding an efficient and effective alignment method has never been more critical. However, the tension between the complexity of current alignment methods and the need for rapid iteration in deployment scenarios necessitates the development of a model-agnostic alignment approach that can operate under these constraints. In this paper, we introduce *Aligner*, a novel and simple alignment paradigm that learns the correctional residuals between preferred and dispreferred answers using a small model.
Designed as a model-agnostic, plug-and-play module, *Aligner* can be directly applied to various open-source and API-based models with only one-off training, making it suitable for rapid iteration.
Notably, *Aligner* can be applied to powerful, large-scale upstream models. 
It can even iteratively bootstrap the upstream models using corrected responses as synthetic human preference data, breaking through the model's performance ceiling.
Our experiments demonstrate performance improvements by deploying the same *Aligner* model across 11 different LLMs, evaluated on the 3H dimensions (helpfulness, harmlessness, and honesty).
Specifically, *Aligner*-7B has achieved an average improvement of **68.9\%** in helpfulness and **23.8\%** in harmlessness across the tested LLMs while also effectively reducing hallucination.
In the Alpaca-Eval leaderboard, stacking *Aligner*-2B on GPT-4 Turbo improved its LC Win Rate from 55.0\% to **58.3\%**, surpassing GPT-4 Omni's 57.5\% Win Rate (community report).

See our website for more details : https://aligner2024.github.io/

### Table of Contents  <!-- omit in toc -->

- [<em>Aligner</em>: Achieving Efficient Alignment through
Weak-to-Strong Correction](#Aligner)
- [Installation](#installation)
- [Training](#training)
- [Dataset & Models](#dataset-models)
- [Acknowledgment](#acknowledgment)


## <em>Aligner</em>: Achieving Efficient Alignment through Weak-to-Strong Correction 

### Architecture of the *Aligner* module.
As a plug-and-play module *Aligner* stack upon an upstream LLM. The *Aligner* redistributes initial answers from the upstream model into more helpful and harmless answers, thus aligning the composed LLM responses with human intentions.

<div align="center">
  <img src="images/main-paradigm.jpg" width="70%"/>
</div>

### Illustration of its behavior in architecture and semantic space.
Like a residual block that adds modifications via a shortcut without altering the base structure, the *Aligner* employs a *copy and correct* method to improve the original answer. 
This analogy highlights the *Aligner*'s dual role in preserving the parameter of the upstream model while enhancing it to align with desired outcomes.

<div align="center">
  <img src="images/semantic_space.png" width="90%"/>
</div>

### Performance of *Aligner* Models
It is shown that *Aligner* achieves significant performances in all the settings. All assessments in this table were conducted based on integrating various models with *Aligner*s to compare with the original models to quantify the percentage increase in the *3H* standard.
When integrated and assessed in conjunction with various upstream models, the *Aligner* requires only a single training session (*i.e.*, the *Aligner* can operate in a zero-shot manner and enhance the performance of all upstream models.)
<div align="center">
  <img src="images/performance.png" width="90%"/>
</div>

### More Details
For more details, please refer to our [website]( https://aligner2024.github.io/) 

## Installation
Clone the source code from GitHub:

```bash
git clone https://github.com/Aligner2024/aligner.git
cd aligner
```

**Native Runner:** Setup a conda environment using [`conda`](https://github.com/conda/conda) / [`mamba`](https://github.com/mamba-org/mamba):

```bash
conda env create --file conda-recipe.yaml  # or `mamba env create --file conda-recipe.yaml`
```

## Training

`aligner` supports a complete pipeline for Aligner <em>residual correction</em> training.

0. Follow the instructions in section [Installation](#installation) to setup the training environment properly.

```bash
conda activate aligner
export WANDB_API_KEY="..."  # your W&B API key here
```

1. Supervised Fine-Tuning (SFT)

```bash
bash scripts/sft-correction.sh \
    --train_datasets <your-correction-dataset> \
    --model_name_or_path <your-model-name-or-checkpoint-path> \
    --output_dir output/sft
```

NOTE: 
1. You may need to update some of the parameters in the script according to your machine setup, such as the number of GPUs for training, the training batch size, etc. 
2. Your dataset format should be consistent with aligner/template-dataset.json


## Dataset & Models
We have open-sourced a 20K [training dataset](https://huggingface.co/datasets/aligner/aligner-20K) and a [7B Aligner model](https://huggingface.co/aligner/aligner-7b-v1.0). Further dataset and models will come soon.

## Acknowledgment

This repository benefits from [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai), [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [DeepSpeed](https://github.com/microsoft/DeepSpeed), [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/HEAD/applications/DeepSpeed-Chat) and [Safe-RLHF](https://github.com/PKU-Alignment/safe-rlhf).

Thanks for their wonderful works and their efforts to further promote LLM research.
Aligner and its related assets are built and open-sourced with love and respect.

