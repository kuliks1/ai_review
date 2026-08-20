import os

# 一定放在 transformers / hugginggingface_hub import 之前
os.environ["HF_HOME"] = r"F:\huggingface"

"""
进行数据预处理。保存处理好的数据文件
"""
from datasets import load_dataset
from transformers import AutoTokenizer

from src.config import DATA_PATH


def process():
    ds = load_dataset('csv', data_files=str(DATA_PATH / 'row' / 'online_shopping_10_cats.csv'))['train']

    ds = ds.remove_columns('cat')
    ds = ds.rename_column('label', 'labels')
    ds = ds.filter(lambda x: x['review'] is not None)

    ds_dict = ds.train_test_split(test_size=0.2, shuffle=True)

    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')

    def map_func(batch):
        return tokenizer(batch['review'], padding=False, truncation=True)

    ds_dict = ds_dict.map(map_func, batched=True, remove_columns=['review'])

    ds_dict.save_to_disk(DATA_PATH / 'processed')

    print(ds_dict)


if __name__ == "__main__":
    process()
