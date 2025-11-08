import os
import torch
import evaluate
import json
import shutil
import matplotlib.pyplot as plt
from torch.optim import AdamW
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from logger import setup_logger
from timeit import default_timer as timer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# Initialize logger and device
logger = setup_logger()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
RESULTS_DIR = "results"
ONNX_SAVE_DIR = os.path.join(RESULTS_DIR, "onnx")
STATS_SAVE_DIR = os.path.join(RESULTS_DIR, "stats")
BUFFER_DIR = "buffer"  # Temporary checkpoint folder
os.makedirs(ONNX_SAVE_DIR, exist_ok=True)
os.makedirs(STATS_SAVE_DIR, exist_ok=True)
os.makedirs(BUFFER_DIR, exist_ok=True)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
logger.info("[INFO] - LoRA configuration set up.")


def train_model(model,optimizer: torch.optim, num_epochs: int = 4, loader=None):
    """Perform PEFT (LoRA fine-tuning)."""
    model = model.to(device)

    # Freeze base model params
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    logger.info("[INFO] - LoRA applied to base model.")
    all_losses = []
    start_time = timer()

    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(loader, leave=True)
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_losses.append(loss.item())

            loop.set_description(f"Epoch {epoch+1}/{num_epochs}")
            loop.set_postfix(loss=loss.item())

    total_time = (timer() - start_time) / 3600
    logger.info(f"[INFO] - Training completed in {total_time:.2f} hours.")

    # IMPORTANT- Fusing the model after training into base model to avoid peft as a wrapper and confusion while saving as ONNX
    model = model.merge_and_unload() #-->IMP
    model.to(device)
    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.plot(all_losses, label="Training Loss", color="blue")
    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(STATS_SAVE_DIR, "training_loss_curve.png"))
    logger.info(f"[INFO] - Training loss curve saved to {STATS_SAVE_DIR}/training_loss_curve.png")

    return model


def save_as_onnx(model, tokenizer, save_dir=ONNX_SAVE_DIR):
    """
    Exports the fine-tuned (LoRA-fused) model to ONNX format.
    Uses a temporary checkpoint buffer to store the model before ONNX conversion.
    Once ONNX is saved, buffer is deleted automatically.
    """
    logger.info("[INFO] - Starting ONNX export...")

    try:
        # Save temporary model checkpoint in 'buffer'
        logger.info(f"[INFO] - Saving checkpoint temporarily to {BUFFER_DIR}")
        model.save_pretrained(BUFFER_DIR)
        tokenizer.save_pretrained(BUFFER_DIR)

        # Convert that saved model to ONNX
        logger.info("[INFO] - Exporting model from buffer to ONNX format...")
        ort_model = ORTModelForSeq2SeqLM.from_pretrained(
            BUFFER_DIR,
            # from_transformers=True,
            export=True
        )
        ort_model.save_pretrained(save_dir)
        logger.info(f"[INFO] - Model successfully exported to ONNX at {save_dir}")

        # copying tokenizer and config files from buffer
        tokenizer_files=[
            "tokenizer.json","tokenizer_config.json","special_tokens_map.json"
        ]

        for file in tokenizer_files:
            src=os.path.join(BUFFER_DIR,file)
            dest=os.path.join(save_dir,file)
            if os.path.exists(src):
                shutil.copy(src,dest)
                logger.info(f"[INFO]- Copied {file} to ONNX directory")
            else:
                logger.warning(f"[WARN]-{file} not found in buffer; skipping")
                
    except Exception as e:
        logger.exception("[ERROR] - ONNX export failed.")
        raise e

    finally:
        # Step 3: Cleanup the temporary buffer
        if os.path.exists(BUFFER_DIR):
            shutil.rmtree(BUFFER_DIR)
            logger.info(f"[INFO] - Temporary buffer {BUFFER_DIR} deleted successfully.")

    return save_dir


def evaluate_dataset(model, loader, tokenizer, split_name="validation"):
    """Evaluate fine-tuned model using ROUGE metrics."""
    model.eval()
    rouge_metric = evaluate.load("rouge")
    all_preds, all_labels = [], []
    logger.info(f"[INFO] - Starting evaluation on {split_name} split")

    with torch.no_grad():
        loop = tqdm(loader, desc=f"Evaluating ({split_name})")
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

    rouge_scores = rouge_metric.compute(predictions=all_preds, references=all_labels)
    logger.info(f"[INFO] - {split_name.capitalize()} ROUGE scores:")
    for key, value in rouge_scores.items():
        logger.info(f"{key}: {value:.4f}")

    save_path = os.path.join(STATS_SAVE_DIR, f"{split_name}_rouge_scores.json")
    with open(save_path, "w") as f:
        json.dump(rouge_scores, f, indent=4)
    logger.info(f"[INFO] - ROUGE metrics saved to {save_path}")
    return rouge_scores


# # Load data
# train_loader, val_loader, test_loader = get_dataloaders()
# print("=" * 100)
# print("STARTING MODEL TRAINING...")

# Train
# trained_model = train_model(optimizer=optimizer, loader=train_loader, num_epochs=1)

# Evaluate
# logger.info("[INFO] - Evaluating model performance...")
# evaluate_dataset(trained_model, val_loader, TOKENIZER, split_name="validation")
# evaluate_dataset(trained_model, test_loader, TOKENIZER, split_name="test")

# Export directly to ONNX (via buffer checkpoint)
# save_as_onnx(trained_model, TOKENIZER)

