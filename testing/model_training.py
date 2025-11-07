import os
import torch
import evaluate
import shutil
import tempfile
import json
import matplotlib.pyplot as plt
from torch.optim import AdamW
from tqdm import tqdm
from peft import LoraConfig,get_peft_model
from logger import setup_logger
from utils.constant import MODEL,TOKENIZER,ZIP
from preprocessing.loader import get_dataloaders
from timeit import default_timer as timer

# from transformers import AutoModelForSeq2SeqLM
logger=setup_logger()
device="cuda" if torch.cuda.is_available() else "cpu"
model=MODEL.to(device)
optimizer=AdamW(model.parameters(),lr=3e-4)

# Setting up output paths
RESULTS_DIR="results"
MODEL_SAVE_DIR=os.path.join(RESULTS_DIR,"model")
STATS_SAVE_DIR=os.path.join(RESULTS_DIR,"stats")
os.makedirs(MODEL_SAVE_DIR,exist_ok=True)
os.makedirs(STATS_SAVE_DIR,exist_ok=True)

# Performing LoRA configuration
lora_config=LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q","v"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

logger.info(f"[INFO]-LORA configuration is setup..")

# SmartSumm Cloud.ipynb
def train_model(optimizer:torch.optim=optimizer,num_epochs:int=4,loader:torch.utils.data=None):
    """
    Perofrming PEFT(LoRA- Fine tuning)
    Subset is in Lora Complete.ipynb
    """
    # Fetching the model
    model=MODEL
     #Freezing the model paramters-- Order matters
    for param in model.parameters():
        param.requires_grad=False
    # Applyting LoRA
    model=get_peft_model(model,lora_config)

    logger.info(f"[INFO]-Model Paramters are Freezed and LoRA is applied..")
    print(model.print_trainable_parameters)
    print("="*100)

    model.to(device)

    all_losses=[] # To store losses
    start_timer=timer()

    for epoch in range(num_epochs):
        model.train()
        loop=tqdm(loader,leave=True)
        for batch in loop:
            # Moving everything to device
            input_ids=batch["input_ids"].to(device)
            attention_mask=batch["attention_mask"].to(device)
            labels=batch["labels"].to(device)

            # Forward Pass
            outputs=model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Calculate the loss
            loss=outputs.loss

            #Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_losses.append(loss.item())

            # Updating progress bar
            loop.set_description(f"Epoch{epoch+1}/{num_epochs}")
            loop.set_postfix(loss=loss.item()) # To display loss values for eatch batch
    
    total_time=(timer()-start_timer)/3600
    logger.info(f"[INFO] - Total Time taken to train the model is: {total_time:.2f} Hours")

    print("="*100)
    logger.info(f"Plotting the graphs:")
    plt.figure(figsize=(12,6))
    plt.plot(all_losses,label="Training Loss",color="blue")
    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    #Saving the graph in artifacts/stats
    plt.savefig(os.path.join(STATS_SAVE_DIR,"training_loss_curve.png"))
    logger.info(f"[INFO]-Training loss curve saved to {STATS_SAVE_DIR}/training_loss_curve.png")

    return model

def save_trained_model(model, tokenizer, save_dir=MODEL_SAVE_DIR, zip_file: bool = ZIP):
    """
    Save the fine-tuned model and tokenizer in zipped format under results/model/
    The final structure will contain ONLY the .zip file, not individual model files.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Always save model + tokenizer first
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"[INFO]-Model and tokenizer saved to {save_dir}")

    # Only zip if user enabled it via utils.constant.ZIP or argument
    if zip_file:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"[INFO]-Model is being saved as zip file in {save_dir} this would take some minutes kindly wait.. :)")

            # Zipping it
            zip_path = os.path.join(save_dir, "smart_summ_model_fused.zip")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            try:
                shutil.make_archive(
                    base_name=zip_path.replace(".zip", ""),
                    format="zip",
                    root_dir=save_dir,
                )
                logger.info(f"[INFO]-Model saved and zipped at :{zip_path}")
            except KeyboardInterrupt:  # For testing purpose
                logger.warning("[WARNING] - Zipping interrupted by user. Cleaning up...")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                    os.remove(save_dir)
                raise
        logger.info(f"[INFO]-Temporary model files cleaned up. Final zip stored at {zip_path}")
        return zip_path

    else:
        logger.info(f"[INFO]-Skipping zip creation (ZIP flag is False). Model saved normally at {save_dir}")
        return save_dir
    
def evaluate_dataset(model,loader,tokenizer,split_name="validation"):
    """
    Evaluating the fine tuned model on train,test as well as validation dataset
    We will be using ROUGE Score as a metric to evaluate the model
    """
    model.eval()
    rouge_metric=evaluate.load("rouge")

    all_preds,all_labels=[],[]
    logger.info(f"[INFO]- Starting evaluation on {split_name} split")

    with torch.no_grad():
        loop=tqdm(loader,desc=f"Evaluating ({split_name})")
        for batch in loop:
            input_ids=batch["input_ids"].to(device)
            attention_mask=batch["attention_mask"].to(device)
            labels=batch["labels"].to(device)

            # Generate summaries (Model generates summaries we check with labels and calc/evaluate the model performance)
            generated_ids=model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4  # Important parameter checks 4 possibilities check HF Documentation
            )

            preds=tokenizer.batch_decode(generated_ids,skip_special_tokens=True)
            # we are running into an issue PAD is replaced with -100 by data collator replacing manually
            labels = torch.where(labels != -100, labels, torch.tensor(tokenizer.pad_token_id).to(labels.device))

            labels_decoded=tokenizer.batch_decode(labels,skip_special_tokens=True)

            all_preds.extend(preds)
            all_labels.extend(labels_decoded)

    rouge_scores=rouge_metric.compute(predictions=all_preds,references=all_labels)
    logger.info(f"[INFO]-{split_name.capitalize()} ROUGE scores:")
    for key,value in rouge_scores.items():
        logger.info(f"{key}:    {value:.4f}")

    # saving the metrics
    save_path = os.path.join("results", "stats", f"{split_name}_rouge_scores.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path,"w") as f:
        json.dump(rouge_scores,f,indent=4)

    logger.info(f"[INFO]-ROUGE Metrics saved to {save_path}")

    return rouge_scores

# Fetching the data loader
train_loader, val_loader, test_loader = get_dataloaders()
#train_loader is of 100 sample size for testing

print("="*100)
print(f"STARTING MODEL TRAINING..")
train_model(optimizer=optimizer,loader=train_loader,num_epochs=1)

# After training:
logger.info(f"[INFO]- Evaluating model performance with ROUGE...")
val_scores = evaluate_dataset(model, val_loader, TOKENIZER, split_name="validation")
test_scores = evaluate_dataset(model, test_loader, TOKENIZER, split_name="test")
# Save trained model
save_trained_model(model, TOKENIZER)

logger.info(f"[INFO]- Entire training pipeline completed successfully.")