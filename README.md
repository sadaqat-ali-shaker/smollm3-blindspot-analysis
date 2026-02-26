
# SmolLM3-3B Base Model Blind Spot Analysis

This repository contains reproducible experiments evaluating blind spots in the base (non-instruction-tuned) model:

HuggingFaceTB/SmolLM3-3B-Base (3B parameters, Oct 2025)

## Objective

To analyze reasoning, instruction-following, and calibration weaknesses in a base language model across structured prompt categories.

## Evaluation Categories

- Arithmetic reasoning
- Logical negation traps
- Constraint-following
- JSON formatting compliance
- Multilingual instruction following
- Hallucination / abstention behavior

## Model Loading

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "HuggingFaceTB/SmolLM3-3B-Base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)




## Dataset

Hugging Face dataset containing all prompts and model outputs:
https://huggingface.co/datasets/faiziiiiii/smollm3-blindspots-10
