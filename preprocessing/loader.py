import os
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq #->for smart padidng
from utils.constant import TOKENIZER,MODEL,MODEL_NAME
from datasets import Dataset,load_from_disk


from logger import setup_logger
logger=setup_logger()
tokenizer,model=TOKENIZER,MODEL


# Setting up device agnostic code
device="cuda" if torch.cuda.is_available() else "cpu"

# Defining how to collate
data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                     model=model,
                                     padding="longest",
                                     return_tensors="pt") #pt means pytorch

def data_loader(dataset:Dataset,collator:DataCollatorForSeq2Seq=data_collator,task:str="train"):
    """
    Code block for converting your dataset(tokenized datset) into DataLoader(PyTorch) so that Model can iterate and dataset can be batched well.
    """
    print("="*100)
    logger.info(f"[INFO]- Converting {task} dataset into DataLoader..")

    if task=="train":
        train_dataloader=DataLoader(
            collate_fn=collator,
            shuffle=True,
            batch_size=8,
            dataset=dataset 
        )
        logger.info(f"[INFO]- Converting {task} dataset into DataLoader suceessful..")
        return train_dataloader
    elif task=="validation":
        validation_dataloader=DataLoader(
            collate_fn=collator,
            batch_size=8,
            dataset=dataset,
            shuffle=False
        )
        logger.info(f"[INFO]- Converting {task} dataset into DataLoader suceessful..") 
        return validation_dataloader
    else:
        test_dataloader=DataLoader(
            collate_fn=collator,
            batch_size=8,
            dataset=dataset,
            shuffle=False
        )
        logger.info(f"[INFO]- Converting {task} dataset into DataLoader suceessful..") 
        return test_dataloader
        
    
# Extracting the dataset(tyokenized)
tokenized_data_path="artifacts/tokenized_data"

train_dataset=load_from_disk(os.path.join(tokenized_data_path,"train"))
validation_dataset=load_from_disk(os.path.join(tokenized_data_path,"validation"))
test_dataset=load_from_disk(os.path.join(tokenized_data_path,"test"))

#Creating DataLoaders
train_loader=data_loader(dataset=train_dataset,collator=data_collator,task="train")
validation_loader=data_loader(dataset=validation_dataset,collator=data_collator,task="validation")
test_loader=data_loader(dataset=test_dataset,collator=data_collator,task="test")

logger.info("[INFO]- All dataloaders created successfully!")
