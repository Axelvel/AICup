import torch
from model import PrivacyModel
from data import retrieveData, tokenize_and_preserve_labels, get_labels_types
from transformers import BertTokenizer

# Loading the model
model = PrivacyModel(21)
model.load_state_dict(torch.load('models/privacy-model.pt'))
print(model)
