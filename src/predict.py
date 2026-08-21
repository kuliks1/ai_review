
import os

# 一定放在 transformers / hugginggingface_hub import 之前
os.environ["HF_HOME"] = r"F:\huggingface"
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import MODEL_PATH




def predict(text: str |list[str],tokenizer,model,device)->int|list[int]:
    is_str = isinstance(text,str)
    if is_str:
        text = [text]

    model.eval().to(device)

    inputs = tokenizer(text,padding=True,truncation=True,return_tensors='pt').to(device)
    inputs =  {k: v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
    logits = output.logits
    result =  torch.argmax(logits,dim=-1)
    return result.tolist()[0] if is_str else  result.tolist()


if __name__ == "__main__":
    text =["这个很好","这个什么垃圾","这本书写的真好，下次别写了"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    output = predict(text,tokenizer,model,device)
    print(output)