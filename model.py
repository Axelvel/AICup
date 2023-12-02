import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.nn.utils.rnn import pack_padded_sequence


model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


class PrivacyModel(nn.Module):

    def __init__(self, output_size) -> None:
        super(PrivacyModel, self).__init__()

        self.bert_output_size = 768
        self.output_size = output_size

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(self.bert_output_size, self.output_size)

    def forward(self, x):
        x = self.bert(x, return_dict=True)  #TODO: Add attention mask
        x = self.dropout(x)
        x = self.fc(x)
        return x
