import os
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq #->for smart padidng
from datasets import Dataset,load_from_disk

from torch.utils.data import Subset
from logger import setup_logger
logger=setup_logger()

# Setting up device agnostic code
device="cuda" if torch.cuda.is_available() else "cpu"

def data_loader(dataset:Dataset,collator:DataCollatorForSeq2Seq,task:str="train"):
    """
    Code block for converting your dataset(tokenized datset) into DataLoader(PyTorch) so that Model can iterate and dataset can be batched well.
    """
    print("="*100)
    logger.info(f"[INFO]- Converting {task} dataset into DataLoader..")

    #simplifying it for working on model_training.py
    loader=DataLoader(
        collate_fn=collator,
        shuffle=(task=="train"), # shuffle only when it's train dataset
        dataset=dataset,
        batch_size=8
    )
    return loader
    
#Creating DataLoaders
def  get_dataloaders(tokenized_data_path:str="artifacts/tokenized_data",collator:DataCollatorForSeq2Seq=None,task:str="train"):
    """
    Code for fetching DataLoader and transferring to model training
    """
    print("="*100)
    logger.info(f"[INFO]-Loading {task} Tokenized Datasets..")
    # train dataset
    train_dataset=load_from_disk(os.path.join(tokenized_data_path,"train"))
    # train_dataset=Subset(train_dataset,range(8)) # --Uncomment

    # validation dataset
    validation_dataset=load_from_disk(os.path.join(tokenized_data_path,"validation"))
    # validation_dataset=Subset(validation_dataset,range(8)) # --Uncomment

    # test dataset
    test_dataset=load_from_disk(os.path.join(tokenized_data_path,"test"))
    # test_dataset=Subset(test_dataset,range(8))    # --Uncomment

    logger.info(f"[INFO]-Creating DataLoaders..")

    train_loader=data_loader(dataset=train_dataset,collator=collator,task="train")
    validation_loader=data_loader(dataset=validation_dataset,collator=collator,task="validation")
    test_loader=data_loader(dataset=test_dataset,collator=collator,task="test")
    logger.info("[INFO]- All dataloaders created successfully!")

    return train_loader,validation_loader,test_loader
