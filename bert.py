import pandas as pd
import numpy as np
import data
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForTokenClassification

from torch import cuda
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if cuda.is_available() else 'cpu'

# --------------------------------- CONSTANTS -------------------------------- #

TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

FIRST_DATASET_PATH = 'dataset/First_Phase_Release(Correction)/First_Phase_Text_Dataset/'
LABELS_PATH = 'dataset/First_Phase_Release(Correction)/answer.txt'

first_phase_data = data.retrieveData(FIRST_DATASET_PATH)
train_labels_dict = data.create_labels_dict(LABELS_PATH)
