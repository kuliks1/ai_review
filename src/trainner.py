# import os
#
# # 一定放在 transformers / hugginggingface_hub import 之前
# os.environ["HF_HOME"] = r"F:\huggingface"
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

from src.config import MODEL_PATH, BATCH_SIZE, LR, WARMUP_STEPS, WEIGHT_DECAY, GRADIENT_ACCUMULATION_STEPS, REPORT_TO, \
    LOGGING_STEPS, EVAL_STRATEGY, EVAL_STEPS, PER_DEVICE_EVAL_BATCH_SIZE, LOAD_BEST_MODEL_AT_END
from src.dataset import get_dataset


def train():
    # 配置trainner参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-chinese").to(device)
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
    train_dataset = get_dataset()
    test_dataset = get_dataset(train=False)

    trainner_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        report_to=REPORT_TO,
        logging_steps=LOGGING_STEPS,
        eval_strategy=EVAL_STRATEGY,
        eval_steps=EVAL_STEPS,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        save_steps=EVAL_STEPS
    )

    trainer = Trainer(model=model,
                      args=trainner_args,
                      train_dataset=train_dataset,
                      eval_dataset=test_dataset,
                      processing_class=tokenizer
                      )

    trainer.train()

    trainer.save_model(str(MODEL_PATH))


if __name__ == "__main__":
    train()
