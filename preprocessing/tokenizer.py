import os
from logger import setup_logger
logger=setup_logger()
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from datasets import load_from_disk
from dotenv import load_dotenv
from utils.constant import TOKENIZER,MODEL,MODEL_NAME
load_dotenv()

DATASET_DIR=os.path.join("artifacts","dataset")
TOKENIZED_OUTPUT_DIR=os.path.join("artifacts","tokenized_data")

# Creating output folder
os.makedirs(TOKENIZED_OUTPUT_DIR,exist_ok=True)
logger.info("Loaded the Dataset from{DATASET_DIR}")
dataset=load_from_disk(DATASET_DIR)

logger.info(f"Loading tokenizer for {MODEL_NAME}")
tokenizer=TOKENIZER


def tokenize_dataset_collator(batch,input_col="article",target_col="highlights"):
    """
    Tokenize the dataset to make it ready for model to be taken as input.
    Mainly the article(Input) and hightlight(output-Summary of article) will be tokenized across train,test and validation dataset
    """

    inputs=tokenizer(batch[input_col],truncation=True)
    outputs=tokenizer(batch[target_col],truncation=True)

    return{
        "input_ids":inputs["input_ids"],
        "attention_mask":inputs["attention_mask"],
        "labels":outputs["input_ids"] #-->output ids
    }

print("="*100)
logger.info("Tokenizing Training split...")
tokenized_train=dataset["train"].map(
    tokenize_dataset_collator,
    batched=True, #-->Default is 1000 (Check HF Documentation)
    remove_columns=dataset["train"].column_names # All thre train test and val have same cols 
)
print("="*100)
logger.info("Tokenizing Validation split...")
tokenized_train=dataset["validation"].map(
    tokenize_dataset_collator,
    batched=True, #-->Default is 1000 (Check HF Documentation)
    remove_columns=dataset["train"].column_names # All thre train test and val have same cols 
)
print("="*100)
logger.info("Tokenizing test split...")
tokenized_train=dataset["test"].map(
    tokenize_dataset_collator,
    batched=True, #-->Default is 1000 (Check HF Documentation)
    remove_columns=dataset["train"].column_names # All thre train test and val have same cols 
)
print("TYPE OF :",type(tokenized_train))
print("="*100)
logger.info(f"Saving the tokenized datsets to:{TOKENIZED_OUTPUT_DIR}")
tokenized_train.save_to_disk(os.path.join(TOKENIZED_OUTPUT_DIR,"train"))
tokenized_train.save_to_disk(os.path.join(TOKENIZED_OUTPUT_DIR,"validation"))
tokenized_train.save_to_disk(os.path.join(TOKENIZED_OUTPUT_DIR,"test"))

logger.info("[INFO]-Tokenization complete")


