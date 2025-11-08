import os
import json
import shutil
import onnx
import logging
from pathlib import Path
from tqdm import tqdm
from logger import setup_logger
from onnxruntime.quantization import quantize_dynamic, QuantType
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer
from preprocessing.loader import get_dataloaders
import evaluate
import torch

# Silencing logs coming from ONNX quantization logs
import onnxruntime as ort
import warnings
warnings.filterwarnings('ignore')
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("optimum").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
logging.getLogger("onnxruntime.quantization").setLevel(logging.ERROR)
# Suppress ONNX Runtime internal logging (INFO/WARNING/ERROR)
ort.set_default_logger_severity(3)  # 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL
logging.getLogger().setLevel(logging.ERROR)

os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"
logger = setup_logger("quantization")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Directories
ONNX_DIR = Path("results/onnx")
QUANTIZED_DIR = Path("results/quantized_model")
os.makedirs(QUANTIZED_DIR, exist_ok=True)

# Step 1: Quantize ONNX Models
def quantize_onnx_model(model_name: str, quant_type=QuantType.QInt8):
    """
    Quantizes an ONNX model file and saves the quantized version.
    """
    model_path = ONNX_DIR / model_name
    output_path = QUANTIZED_DIR / model_name.replace(".onnx", "_quantized.onnx")

    if not model_path.exists():
        logger.warning(f"[WARN] - Model file not found: {model_path}")
        return None

    logger.info(f"[INFO] - Quantizing {model_name} ...")

    quantize_dynamic(
        model_input=str(model_path),
        model_output=str(output_path),
        weight_type=quant_type
    )

    orig_size = model_path.stat().st_size / (1024 * 1024)
    quant_size = output_path.stat().st_size / (1024 * 1024)
    compression_ratio = (1 - quant_size / orig_size) * 100

    logger.info(f"[SUCCESS] - Quantized {model_name}")
    logger.info(f"[INFO] - Size reduced: {orig_size:.2f}MB → {quant_size:.2f}MB ({compression_ratio:.1f}% smaller)")
    return output_path

# Step 2: Copy Tokenizer and Config Files
def copy_tokenizer_and_configs():
    """
    Copies tokenizer and model config files from ONNX_DIR to QUANTIZED_DIR.
    """
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "config.json"
    ]

    for file in tokenizer_files:
        src = ONNX_DIR / file
        dest = QUANTIZED_DIR / file
        if src.exists():
            shutil.copy(src, dest)
            logger.info(f"[INFO] - Copied {file} to quantized directory.")
        else:
            logger.warning(f"[WARN] - {file} not found in ONNX directory; skipping.")

# Step 3: Validate Quantized Model
def validate_quantized_model(model_dir=QUANTIZED_DIR,val_loader=None,tokenizer=None):
    """
    Runs a small validation on the quantized model using ROUGE.
    """
    logger.info("[INFO] - Validating quantized ONNX model...")
    if val_loader is None or tokenizer is None:
        logger.warning('[WARN]- Validation skipped in quantization..')
        return 

    try:
        model = ORTModelForSeq2SeqLM.from_pretrained(model_dir)
        rouge = evaluate.load("rouge")
        all_preds, all_labels = [], []

        # model.eval() ---> no needed as ONNX is always in inference mode and is not pytorch or tf
        logger.info("[INFO] - Starting evaluation on validation set...")

        with torch.no_grad():
            loop = tqdm(val_loader, desc="Evaluating (Quantized)")
            for batch in loop:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=128,
                    num_beams=4
                )

                preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                labels = torch.where(labels != -100, labels, torch.tensor(tokenizer.pad_token_id).to(labels.device))
                labels_decoded = tokenizer.batch_decode(labels, skip_special_tokens=True)

                all_preds.extend(preds)
                all_labels.extend(labels_decoded)

        rouge_scores = rouge.compute(predictions=all_preds, references=all_labels)
        save_path = QUANTIZED_DIR / "quantized_model_rouge_scores.json"
        with open(save_path, "w") as f:
            json.dump(rouge_scores, f, indent=4)

        logger.info(f"[INFO] - Quantized model ROUGE scores saved to {save_path}")
        for k, v in rouge_scores.items():
            logger.info(f"{k}: {v:.4f}")

    except Exception as e:
        logger.exception("[ERROR] - Quantized model validation failed.")
        raise e

# Step 4: Main Pipeline
def run_quantization(tokenizer,val_loader):
    """
    Main quantization pipeline.
    """
    logger.info("=" * 100)
    logger.info("[START] - SmartSumm Quantization Pipeline Initiated")

    # Quantize encoder/decoder ONNX models
    for name in [
        "encoder_model.onnx",
        "decoder_model.onnx",
        "decoder_with_past_model.onnx"
    ]:
        quantize_onnx_model(name, quant_type=QuantType.QInt8)

    # Copy supporting tokenizer/config files
    copy_tokenizer_and_configs()

    # Validate quantized model
    validate_quantized_model(tokenizer=tokenizer,val_loader=val_loader)

    logger.info("[COMPLETE] - Quantization pipeline completed successfully.")
    logger.info("=" * 100)

