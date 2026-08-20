"""
获取数据集
"""
import os

# 一定放在 transformers / hugginggingface_hub import 之前
os.environ["HF_HOME"] = r"F:\huggingface"
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, AutoTokenizer

from src.config import BATCH_SIZE, DATA_PATH


def get_dataset(train: bool = True):
    dataset = load_from_disk(str(DATA_PATH / 'processed' / ('train' if train else 'test')))
    return dataset


def get_dataloader(train: bool = True):
    dataset = get_dataset(train)
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
    return DataLoader(dataset=dataset,
                      batch_size=BATCH_SIZE,
                      collate_fn=DataCollatorWithPadding(tokenizer, padding=True, return_tensors='pt'))


if __name__ == "__main__":
    dataset =get_dataset()
    print(dataset)
