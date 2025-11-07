from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM

TOKENIZER=AutoTokenizer.from_pretrained("google/flan-t5-small")
MODEL=AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
MODEL_NAME="google/flan-t5-small"
ZIP=False