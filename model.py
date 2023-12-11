import torch
import torch.nn as nn
from transformers import AutoModel

model_name = "bert-base-uncased"


class PrivacyModel(nn.Module):

    def __init__(self, output_size) -> None:
        super(PrivacyModel, self).__init__()

        self.bert_output_size = 768
        self.output_size = output_size

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(self.bert_output_size, self.output_size)

    def forward(self, x, attention_mask):
        x = self.bert(x, attention_mask=attention_mask, return_dict=True)  #TODO: Add attention mask
        x = self.dropout(x.last_hidden_state)
        x = self.fc(x).view(-1, 512, self.output_size)
        return x
