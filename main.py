import os
import shutil
from torch.optim import AdamW
from logger import setup_logger
from transformers import DataCollatorForSeq2Seq
from utils.constant import DATASET_NAME,VERSION,MODEL_NAME

from preprocessing.data_setup import download_model,download_dataset
print("Dataset Downloaded...")
from preprocessing.loader import get_dataloaders
print("Data Loaders done")
from preprocessing.tokenizer import feed_dataset
print("Tokenizer module")
from testing.model_training import train_model,evaluate_dataset,save_as_onnx
from testing.quantization import run_quantization
logger=setup_logger()

print("-"*200)
logger.info(f"[INFO]-Intizalizing Dataset and Model download from main.py")

# Downloading Dataset
dataset=download_dataset(dataset_name=DATASET_NAME,version=VERSION,save_dir="artifacts")

tokenizer,model=download_model(model_name=MODEL_NAME)

logger.info(f"[INFO]-Completed downloading Dataset and Model download from main.py")

# Tokenizing the datasets
print('-'*200)
logger.info(f"[INFO]-Tokenizing the dataset..")
feed_dataset()
logger.info(f"[INFO]- Dataset Tokenization complete from main.py ")

# setting up DataLoaders
print('-'*200)
logger.info(f"[INFO]-Initalizing DataLoader...")
# Defining how to collate
data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                     model=model,
                                     padding="longest",
                                     return_tensors="pt") #pt means pytorch
train_loader,validation_loader,test_loader=get_dataloaders(collator=data_collator)
logger.info(f"[INFO]-DataLoader complete...")

# Fine-tuning the model
print('='*200)
print('='*200)
print('-'*200)
logger.info(f"[INFO]- Main.py intializing Model Fine-tuning...")
optimizer = AdamW(model.parameters(), lr=3e-4)
trained_model = train_model(model=model,optimizer=optimizer, loader=train_loader, num_epochs=1) # 1 for sake of simplicity as of now

# Evaluating the fine-tuned model
logger.info("[INFO] - Evaluating model performance...")
evaluate_dataset(trained_model, validation_loader, tokenizer, split_name="validation")
evaluate_dataset(trained_model, test_loader,tokenizer, split_name="test")

# saving the quantized model as ONNX file
save_as_onnx(trained_model,tokenizer)
logger.info("[INFO] - Entire training and ONNX export pipeline completed successfully.")

print('-'*200)
logger.info(f"[INFO]-Initiating Model Quantization from main.py")
run_quantization(tokenizer=tokenizer,val_loader=validation_loader)

print("="*200)
logger.info(f"[INFO]- ENTIRE PROJECT PIPELINE IS COMPLETED...")

