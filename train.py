import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from model import PrivacyModel
#from data import get_labels_types
from dataset import dataset
import joblib

LABELS_PATH = 'dataset/First_Phase_Release(Correction)/answer.txt'
#label_types,_ = get_labels_types(LABELS_PATH)
num_labels = 22# len(label_types)

train_loader = joblib.load('loader.plk')
print(train_loader)

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.01
NUM_EPOCH = 10

OUTPUT_SIZE = num_labels

model = PrivacyModel(OUTPUT_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

total_loss = 0.0

for epoch in range(NUM_EPOCH):
    print(epoch)
    for batch in train_loader:
        inputs, labels = batch['ids'], batch['targets']
        print(inputs.shape)
        print(labels.shape)
        #inputs = inputs.view(64, -1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{NUM_EPOCH} - Loss: {average_loss}")
