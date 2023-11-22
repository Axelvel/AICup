import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from data import get_labels_types

LABELS_PATH = 'dataset/First_Phase_Release(Correction)/answer.txt'
label_types = get_labels_types(LABELS_PATH)
num_labels = len(label_types)


model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


class MyModel(nn.Module):

    def __init__(self) -> None:
        super(MyModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, x):
        self.bert(x)
        return x


model = MyModel()