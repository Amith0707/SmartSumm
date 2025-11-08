import os
from logger import setup_logger
from transformers import AutoTokenizer
from datasets import load_from_disk
from dotenv import load_dotenv
from utils.constant import MODEL_NAME

DATASET_DIR = os.path.join("artifacts", "dataset")
TOKENIZED_OUTPUT_DIR = os.path.join("artifacts", "tokenized_data")

def tokenize_dataset_collator(batch, tokenizer, input_col="article", target_col="highlights"):
    inputs = tokenizer(batch[input_col], truncation=True)
    outputs = tokenizer(batch[target_col], truncation=True)
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": outputs["input_ids"]
    }

def feed_dataset():
    """Called from main.py to tokenize and save datasets."""
    load_dotenv()
    os.makedirs(TOKENIZED_OUTPUT_DIR, exist_ok=True)
    logger = setup_logger()

    logger.info(f"[INFO] - Loading dataset from {DATASET_DIR}")
    dataset = load_from_disk(DATASET_DIR)

    logger.info(f"[INFO] - Loading tokenizer for {MODEL_NAME}")
    tokenizer =AutoTokenizer.from_pretrained(MODEL_NAME)

    logger.info("[INFO] - Tokenizing Training split...")
    tokenized_train = dataset["train"].map(
        lambda batch: tokenize_dataset_collator(batch, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    logger.info("[INFO] - Tokenizing Validation split...")
    tokenized_validation = dataset["validation"].map(
        lambda batch: tokenize_dataset_collator(batch, tokenizer),
        batched=True,
        remove_columns=dataset["validation"].column_names
    )

    logger.info("[INFO] - Tokenizing Test split...")
    tokenized_test = dataset["test"].map(
        lambda batch: tokenize_dataset_collator(batch, tokenizer),
        batched=True,
        remove_columns=dataset["test"].column_names
    )

    logger.info(f"[INFO] - Saving tokenized datasets to: {TOKENIZED_OUTPUT_DIR}")
    tokenized_train.save_to_disk(os.path.join(TOKENIZED_OUTPUT_DIR, "train"))
    tokenized_validation.save_to_disk(os.path.join(TOKENIZED_OUTPUT_DIR, "validation"))
    tokenized_test.save_to_disk(os.path.join(TOKENIZED_OUTPUT_DIR, "test"))
    logger.info("[INFO] - Tokenization complete")

# nothing executes here automatically
