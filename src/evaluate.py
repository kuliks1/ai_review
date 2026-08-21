import os

from tqdm import tqdm

# 一定放在 transformers / hugginggingface_hub import 之前
os.environ["HF_HOME"] = r"F:\huggingface"
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import MODEL_PATH
from src.dataset import get_dataloader


def evaluate(dl,model,device):
   model.eval()
   all_labels = []
   all_predictions = []
   model.to(device)

   with torch.no_grad():
       for batch in tqdm(dl):
           batch =  {k:v.to(device) for k,v in batch.items()}

           labels =  batch.pop("labels")
           all_labels.extend(labels.tolist())
           output = model(**batch)
           logits = output.logits
           result = torch.argmax(logits,dim=-1)
           all_predictions.extend(result.tolist())

   return  accuracy_score(all_labels,all_predictions)



if __name__ == "__main__":

    dl= get_dataloader(train=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    result =  evaluate(dl,model,device)
    print(result)