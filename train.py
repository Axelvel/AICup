import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from model import PrivacyModel
from data import get_labels_types
from dataset import dataset
import joblib
import sys
import curses

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

LABELS_PATH = 'dataset/First_Phase_Release(Correction)/answer.txt'
label_types, _ = get_labels_types(LABELS_PATH)
num_labels = len(label_types)

#train_loader = joblib.load('loader.plk')
tensor_data = joblib.load('tensor_data.plk')
tensor_labels = joblib.load('tensor_labels.plk')
attention_mask = joblib.load('attention_mask.plk')
train_loader = DataLoader(dataset(tensor_data, tensor_labels, attention_mask), shuffle=True, batch_size=8)

# Hyperparameters
LEARNING_RATE = 0.01
NUM_EPOCH = 3

OUTPUT_SIZE = num_labels

model = PrivacyModel(OUTPUT_SIZE).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)


for epoch in range(NUM_EPOCH):
    total_loss = 0.0
    print('Epoch:', epoch+1)
    for num_batch, batch in enumerate(train_loader):
        inputs_r, labels_r, attention_mask_r = batch['ids'], batch['targets'], batch['attention_mask']
        inputs = inputs_r.to(device)
        labels = labels_r.to(device)
        attention_mask = attention_mask_r.to(device)
        labels = labels.view(-1)
        optimizer.zero_grad()
        outputs = model(inputs, attention_mask)
        outputs = outputs.view(-1, num_labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f'{num_batch}/{len(train_loader)}')
    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCH} - Loss: {average_loss}")

# Saving the PyTorch model
MODEL_PATH = 'models/privacy-model.pt'

# Classic model export
torch.save(model, MODEL_PATH)
torch.save(model.state_dict(), MODEL_PATH)

# Exporting model to Torchscript
#model.eval()
#dummy_input = next(iter(train_loader))
#dummy_input = dummy_input['ids'][0]
#traced_model = torch.jit.trace(model, dummy_input)
#torch.jit.save(traced_model, MODEL_PATH)
