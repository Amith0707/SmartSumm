# SmartSumm: Abstractive Text Summarization using FLAN-T5

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Model Evaluation](#model-evaluation)
- [Quantization Details](#quantization-details)
- [Demo](#demo)
- [Tech Stack](#tech-stack)
- [Artifacts](#artifacts)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

---

## Overview
**SmartSumm** is a modular project that demonstrates fine-tuning, exporting, and quantizing a **FLAN-T5** model for **abstractive text summarization**.  
It uses the **CNN/DailyMail** dataset and is optimized for efficient inference through **ONNX Runtime** and **dynamic INT8 quantization**.

The project covers the complete lifecycle of an LLM-based summarization system — from dataset preparation and fine-tuning to quantization and deployment.

---

## Features
- End-to-end modular pipeline:
  - Dataset loading, tokenization, and preprocessing  
  - LoRA-based fine-tuning for parameter-efficient training  
  - ONNX model export for faster inference  
  - Post-training dynamic quantization (INT8)
- Clean and scalable folder structure
- Gradio interface for Hugging Face Spaces
- ROUGE metric-based evaluation and comprehensive logging

---

## Project Structure
```
SmartSumm/
│
├── main.py # Orchestrates the complete pipeline
│
├── preprocessing/
│ ├── data_setup.py # Dataset and model download utilities
│ ├── tokenizer.py # Tokenization logic
│ ├── loader.py # DataLoader creation
│
├── testing/
│ ├── model_training.py # Fine-tuning and ONNX export
│ ├── quantization.py # Quantization and validation
│
├── app/
│ └── app.py # Gradio app for Hugging Face demo
│
├── utils/
│ └── constant.py # Constants and configuration
│
├── artifacts/ # Stored datasets and model files
├── results/ # ONNX and quantized models
├── logs/ # Pipeline logs
└── requirements.txt
```

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Amith0707/SmartSumm.git
   cd SmartSumm
   ```

2. **Create a virtual environment**

    ```bash
    python -m venv smart_summ_env
    smart_summ_env\Scripts\activate   # (Windows)
    ```
    
3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Usage**

    ```bash
    python -m main
    ```

### This will:
1.  Download and store the CNN/DailyMail dataset and base FLAN-T5 model

2.  Tokenize and prepare datasets

3.  Fine-tune the model using LoRA

4.  Evaluate using ROUGE metrics

5.  Export the model to ONNX

6.  Quantize and validate the quantized model

**NOTE-** 
1.  If you want a faster Inference kindly uncomment the lines `39`,`43`,`47` from `loader.py` which will allow you to consider only 8 Data points out of entire dataset. 

2. But do note that by following above procedure model won't be fine-tuned well enough.

---

## Model Evaluation

SmartSumm uses **ROUGE** metrics for evaluation during validation and testing.  
Scores are automatically logged and stored under: **results/stats**


---

## Quantization Details

Quantization is handled using **ONNX Runtime dynamic quantization** to reduce model size and improve inference latency.

### Quantized Models
- `encoder_model.onnx`
- `decoder_model.onnx`
- `decoder_with_past_model.onnx`

Quantized versions are stored in: **results/quantized_model/**


---

## Demo

A Hugging Face Spaces demo showcasing the quantized SmartSumm model is available here:

**[View SmartSumm on Hugging Face Spaces](#)**  
*(Link to be added after deployment)*

---

## Tech Stack

- **Model:** FLAN-T5 (Google)  
- **Frameworks:** PyTorch, Hugging Face Transformers, Optimum, ONNX Runtime  
- **Quantization:** Dynamic INT8 Quantization  
- **Evaluation:** ROUGE  
- **Deployment:** Gradio (Hugging Face Spaces)

---

## Artifacts

Artifacts are preserved for reproducibility and version control.

| Artifact Type     | Location                   | Description                     |
|--------------------|----------------------------|---------------------------------|
| Dataset            | `artifacts/dataset/`       | CNN/DailyMail dataset           |
| Model              | `artifacts/model/`         | Fine-tuned FLAN-T5 model        |
| ONNX Models        | `results/onnx/`            | Exported ONNX versions          |
| Quantized Models   | `results/quantized_model/` | Final compressed models         |

---

## Future Enhancements

- Explore FP8 and mixed-precision quantization  
- Experiment with multi-document summarization  
- Integrate automatic evaluation dashboard for ROUGE and BLEU  
- Extend app functionality for user-defined summarization lengths  

---

## Author

**Amith V**  
Focused on applied machine learning, model optimization, and end-to-end AI system development.  
SmartSumm was created to demonstrate a complete, production-ready summarization pipeline combining fine-tuning, ONNX conversion, and quantization.

