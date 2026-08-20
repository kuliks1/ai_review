import os
import time

import torch
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from torch.utils.tensorboard import SummaryWriter
from src.config import LR, EPOCH, LOG_PATH, STEP_SIZE, MODEL_PATH
from src.dataset import get_dataloader

os.environ['HF_HOME'] = r"F:\huggingface"


batch_step = 0
best_loss = float('inf')
total_loss = 0

def train_one_epoch(device,model,dataloader,optimizer,writer):
    global batch_step,best_loss,total_loss

    model.train()

    for batch in tqdm(dataloader):
        batch =  {k:v.to(device) for k,v in batch.items()}

        output = model(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        batch_step += 1

        if batch_step % STEP_SIZE == 0:
            avg_loss = total_loss/STEP_SIZE
            writer.add_scalar('loss',avg_loss,batch_step)
            total_loss = 0
            if avg_loss < best_loss:
                model.save_pretrained(MODEL_PATH)
                best_loss = avg_loss



def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-chinese').to(device)
    dataloader = get_dataloader()
    optimizer = Adam(model.parameters(),lr=LR)
    writer = SummaryWriter(LOG_PATH / time.strftime("%Y-%m-%d_%H-%M-%S"))

    for epoch in range(1,EPOCH+1):
        train_one_epoch(device,model,dataloader,optimizer,writer)

    writer.close()
if __name__ == "__main__":
    train()
