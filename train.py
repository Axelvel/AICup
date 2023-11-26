import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from model import PrivacyModel
from data import get_labels_types

LABELS_PATH = 'dataset/First_Phase_Release(Correction)/answer.txt'
label_types = get_labels_types(LABELS_PATH)
num_labels = len(label_types)

# Hyperparameters

BATCH_SIZE = 16
LEARNING_RATE = 0.01
EPOCH = 10

OUTPUT_SIZE = num_labels

model = PrivacyModel(OUTPUT_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)