# BioMedLM Fine-tuning for Medical Question Answering

This repository contains code for fine-tuning the BioMedLM language model on medical reasoning data using Parameter-Efficient Fine-Tuning (PEFT) with LoRA/QLoRA techniques. The project focuses on creating a specialized model capable of answering medical questions with detailed reasoning.

## Overview

The project fine-tunes [stanford-crfm/BioMedLM](https://huggingface.co/stanford-crfm/BioMedLM) on the [medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) dataset to develop a model that can:

- Answer complex medical questions
- Provide detailed reasoning for medical diagnoses and treatments
- Generate evidence-based medical responses

## Features

- 🚀 Memory-efficient fine-tuning using 4-bit quantization
- 🧠 Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- 🔍 Complete pipeline from data preprocessing to evaluation
- 💻 Optimized for T4 GPUs on Google Colab

## Requirements

```
transformers>=4.30.0
peft>=0.4.0
bitsandbytes>=0.39.0
torch>=2.0.0
datasets>=2.10.0
scikit-learn>=1.0.0
tensorboard
```

## Getting Started

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

The code uses the FreedomIntelligence/medical-o1-reasoning-SFT dataset from Hugging Face:

```python
from datasets import load_dataset
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT")
```

### 3. Fine-tune the Model

Run the main fine-tuning script:

```bash
python train.py
```

This will:
- Load the BioMedLM model
- Apply 4-bit quantization for memory efficiency
- Configure LoRA for parameter-efficient training
- Fine-tune on the medical dataset
- Save the resulting model and LoRA adapters

### 4. Generate Medical Responses

After training, use the model to generate medical responses:

```python
from inference import generate_medical_response

question = "What are the potential complications of untreated hypertension?"
response = generate_medical_response(question)
print(response)
```

## Model Architecture

The fine-tuning process uses:

- **Base Model**: stanford-crfm/BioMedLM
- **Quantization**: 4-bit quantization using bitsandbytes
- **PEFT Method**: LoRA with rank=16, alpha=32
- **Target Modules**: Attention layers (query_key_value)

## Performance Optimization

The repository includes optimizations for running on constrained hardware:

- 4-bit quantization to reduce memory usage
- Gradient accumulation to simulate larger batch sizes
- Mixed precision training with fp16
- Optional dataset subsetting for faster experimentation

## Evaluation

The repo includes code to evaluate the model on a test set, calculating:

- Response accuracy
- Generation quality metrics
- Comparison with original and generated answers

## Customization

You can customize the fine-tuning process by modifying:

- The LoRA configuration (rank, alpha, etc.)
- Training hyperparameters (learning rate, epochs, etc.)
- Input/output formatting in the preprocessing function
- Generation parameters for inference

## Acknowledgements

- [Stanford CRFM](https://crfm.stanford.edu/) for the BioMedLM model
- [Hugging Face](https://huggingface.co/) for the Transformers library
- [PEFT](https://github.com/huggingface/peft) for efficient fine-tuning methods

## License

[MIT License](LICENSE)

## Citation

```
@software{biomedlm_finetuning,
  author = {Sana Imran},
  title = {BioMedLM Fine-tuning for Medical Question Answering},
  year = {2025},
  url = {https://github.com/SanaImran0/biomedlm-finetuning}
}
```

