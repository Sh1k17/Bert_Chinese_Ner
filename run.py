from pytorch_pretrained import BertModel,BertTokenizer
# -*- coding: utf-8 -*
import torch
from config import Config
from torch import nn
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import time
import os
from sklearn.metrics import f1_score, classification_report
from utils import built_train_dataset,built_dev_dataset
import random
import operator
from functools import reduce
import torch.optim as optim
from torch.optim import lr_scheduler
from net import BiLSTMCRF
random.seed(1)
config = Config()
best_score = float("inf")
# config = Config()
print("Loading Datas...")
train_dataset = built_train_dataset(config)
dev_dataset = built_dev_dataset(config)
tokenzier = BertTokenizer.from_pretrained(config.bert_path)
bert_model = BertModel.from_pretrained(config.bert_path)
flag = False
model = BiLSTMCRF()

optimizer = optim.SGD(model.parameters(),lr = config.learning_rate,momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
if os.path.exists("./model/params.pkl"):
    model.load_state_dict(torch.load("./model/params.pkl"))
for epoch in range(100):
    scheduler.step()
    total_loss = 0
    batch_count = 0
    start_time = time.time()
    for i,batch in enumerate(train_dataset):
        model.zero_grad()

        input_id, input_mask, label, output_mask = batch
        # print(input_id,input_mask,label,output_mask)
        # print(label)
        loss = model.neg_log_likelihood(input_id,label,input_mask)
        total_loss += loss.tolist()[0]
        batch_count += 1
        loss.backward()
        optimizer.step()
    if best_score > total_loss / batch_count:
        best_score = total_loss/batch_count
        torch.save(model.state_dict(),"./model/params.pkl")
    print("epoch: {}\tloss: {:.2f}\ttime: {:.1f} sec".format(epoch + 1, total_loss / batch_count,
                                                                 time.time() - start_time))
    if (epoch + 1) % 10 == 0:
        y_predicts = []
        y_labels = []
        for i,batch in enumerate(train_dataset):
            input_id, input_mask, label, output_mask = batch
            _,path = model(input_id,input_mask)
            tmp = reduce(operator.add, path)
            y_predicts += tmp
            label = label.view(1,-1)
            label = label[label != -1]
            # print(label)
            y_labels.append(label)
        y_true = torch.cat(y_labels,dim=0)
        y_pred = np.array(y_predicts)
        y_true = y_true.cpu().numpy()
        f1 = f1_score(y_true,y_pred,average="macro")
        print('f1: {}'.format(f1))
        classification = classification_report(y_true,y_pred)
        print(classification)