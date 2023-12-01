import pandas as pd
import numpy as np
import data
from sklearn.metrics import accuracy_score
from transformers import pipeline
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForTokenClassification

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

# --------------------------------- CONSTANTS -------------------------------- #

EPOCHS = 20
LEARNING_RATE = 0.001

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertForTokenClassification.from_pretrained('bert-base-uncased', 
                                                   num_labels=len(data.id2label),
                                                   id2label=data.id2label,
                                                   label2id=data.label2id)
model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "My name is Wolfgang and I live in Berlin"

ner_results = nlp(example)
print(ner_results)