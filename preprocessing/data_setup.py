# This file is for downloading and storing the model and dataset
import os
from logger import setup_logger
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from datasets import load_dataset
# from dotenv import load_dotenv
logger=setup_logger("data_model_setup")

def download_dataset(dataset_name:str="cnn_dailymail",version="3.0.0",save_dir="artifact"):
    """
    Task of this function is to download the Dataset and store it in artifacts for future reference"""
    print(f"Download_dataset intialized...")
    dataset=load_dataset(dataset_name,version)
    os.makedirs(save_dir,exist_ok=True)

    save_dir="artifacts/dataset"
    os.makedirs(save_dir,exist_ok=True)

    dataset.save_to_disk(save_dir)
    logger.info(f"[INFO]-Dataset is fully saved in {save_dir}")

    # Saving the metadata
    with open(os.path.join(save_dir,"dataset_info.txt"),"w") as f:
        f.write(str(dataset))

    logger.info(f"[INFO]-Dataset(MetaData) is fully saved in {save_dir}")
    return dataset

def download_model(model_name:str="google/flan-t5-small",save_dir="artifacts/model"):
    """
    Task of this function is to save the model.
    Model will be saved in artifacts/models.
    As well as downloads the tokenizer.
    """
    logger.info(f"Downloading the model and Tokenizer:  {model_name}")
    os.makedirs(save_dir,exist_ok=True)

    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # saving it locally
    tokenizer.save_pretrained(save_dir)    
    model.save_pretrained(save_dir)
    
    logger.info(f"[INFO] Model and tokenizer saved to: {save_dir}")

    return tokenizer,model
if __name__=="__main__":
    # try:
        # from huggingface_hub import login
        # hf_token=os.getenv("HF_WRITE")
        # login(token=hf_token)
        # print("Login Sucessful")
    # except Exception as e:
        # print("Login Failed...",e)
    
    print("Started Process...")
    dataset=download_dataset()
    tokenizer,model=download_model()
    print("data_setup.py workinh....")

    

