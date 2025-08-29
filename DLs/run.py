import sys
import time
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import torch.utils.data as Data
from importlib import import_module
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from utils import DataProcessor, BiolmDataSet, ModelEvaluator
import random
from extract_embedding import *
from config import *


def seed_everything(seed=42):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'Total Parameters': total_params,
        'Trainable Parameters': trainable_params,
        'Non-trainable Parameters': total_params - trainable_params
    }

# 设置全局随机种子
GLOBAL_SEED = 42
seed_everything(GLOBAL_SEED)

t5_embedding_config = T5_PLM_CONFIG
calm_embedding_config = CALM_PLM_CONFIG

calm_embedding = EmbeddingExtractor(calm_embedding_config)._run()
t5_embedding = EmbeddingExtractor(t5_embedding_config)._run()

print( calm_embedding.shape )
print( t5_embedding.shape )



# Load data
data_processor =  DataProcessor( )
# basedir = "/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/data/2-8/embeddings"

basedir = "/hpcfs/fhome/yangchh/workdir/self/TLCP-EPE/data/embeddings-4"   

#  /hpcfs/fhome/yangchh/ai/finetuneFMs/data/embedding/

# /hpcfs/fhome/yangchh/ai/finetuneFMs/data/embedding/
# /hpcfs/fhome/yangchh/ai/finetuneFMs/data/embedding/esm2_t30_150M_UR50D.pkl


# codon_features, labels = data_processor.load_data( os.path.join( basedir, "huggingface_calm.pkl" )  ) # CaLM.pkl  huggingface_calm.pkl
# prot_features, labels = data_processor.load_data( os.path.join( basedir, "esm2_t6_8M_UR50D.pkl")  ) # ESM2(8M).pkl               esm2_t6_8M_UR50D.pkl

features, labels = data_processor.load_data( os.path.join( basedir, "Finetune_CaLM+ProtT5.pkl")  )

# features = pd.concat( [codon_features, prot_features], axis=1 )   


X_scaled, y_scaled = data_processor.preprocess_data(features, labels)
df = pd.DataFrame( X_scaled )
df["label"]=y_scaled
X_train, X_test, y_train, y_test = data_processor.split_data(df , test_size=0.2)

# Create datasets
train_dataset = BiolmDataSet(X_train, y_train)
val_dataset = BiolmDataSet(X_test, y_test)

batch_size = 128       
train_iter = Data.DataLoader(train_dataset, batch_size=batch_size)
val_iter = Data.DataLoader(val_dataset, batch_size=batch_size)

# Model setup
model_name = 'Lrtf'    
x = import_module(f'models.{model_name}')
config = x.Config()
model = x.Model(config).to(config.device)
print(model.parameters)

# 打印模型参数量
params_info = count_parameters(model)
print("\nModel Parameters Summary:")
print("-" * 50)
print(f"Total Parameters: {params_info['Total Parameters']:,}")
print(f"Trainable Parameters: {params_info['Trainable Parameters']:,}")
print(f"Non-trainable Parameters: {params_info['Non-trainable Parameters']:,}")
print("-" * 50)


optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=1)
criterion = nn.CrossEntropyLoss( label_smoothing=0.05 ).to(config.device)

train_losses, train_accs = [], []
dev_losses, dev_accs = [], []
np_epoch = 100


# Initialize lists to store metrics for each epoch
test_accs, test_precisions, test_recalls, test_f1s, test_aucs = [], [], [], [], []

all_true = []
all_pred = []
all_probs = []

for epoch in range(np_epoch):
    train_loss, train_acc = 0, 0
    dev_loss, dev_acc = 0, 0

    # Training loop
    model.train()
    for trains, labels in train_iter:
        trains, labels = trains.cuda().float(), labels.long().cuda()
        outputs = model(trains)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        true = labels.data.cpu()
        probabilities = F.softmax(outputs, dim=1)
        predict = torch.max(probabilities, 1)[1].cpu()
        train_acc += metrics.accuracy_score(true, predict)

    train_loss /= len(train_iter)
    train_acc /= len(train_iter)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validation loop
    model.eval()
    with torch.no_grad():
        for dev_trains, dev_labels in val_iter:
            dev_trains, dev_labels = dev_trains.cuda().float(), dev_labels.long().cuda()
            dev_outputs = model(dev_trains)
            dev_loss += criterion(dev_outputs, dev_labels).item()
            dev_true = dev_labels.data.cpu()
            dev_probabilities = F.softmax(dev_outputs, dim=1)
            dev_predict = torch.max(dev_probabilities, 1)[1].cpu()
            dev_acc += metrics.accuracy_score(dev_true, dev_predict)

    dev_loss /= len(val_iter)
    dev_acc /= len(val_iter)
    dev_losses.append(dev_loss)
    dev_accs.append(dev_acc)
    scheduler.step(dev_loss)

    # Print training and validation stats
    print(f"Epoch [{epoch+1}/{np_epoch}]")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.4f}")

