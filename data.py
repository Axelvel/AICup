
import torch
from os import listdir
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForTokenClassification

# --------------------------------- CONSTANTS -------------------------------- #

FIRST_DATASET_PATH = 'dataset/First_Phase_Release(Correction)/First_Phase_Text_Dataset/'
LABELS_PATH = 'dataset/First_Phase_Release(Correction)/answer.txt'

BATCH_SIZE = 62

# ------------------------------ RETRIEVING DATA ----------------------------- #
def retrieveData(path, dict):
    list_file = listdir(path)
    items = ["[CLS]"]
    labels = ["[CLS]"]


    for file in list_file:
        raw_data = dict[file[:-4]]
        with open(path + file) as f:
            lines = f.readlines()
        for line in lines:
            if line != "\n":
                splitted = line.split()
                for word in splitted:
                    items.append(word)

                    check = False
                    for list in raw_data:
                        if word in list and word != list[0] and check == False:
                            labels.append(list[0])
                            check = True
                    if check == False:
                        labels.append("OTHER")

        items.append("[SEP]")
        labels.append("[SEP]")
    
    return items,labels



# Creating label dict

def create_labels_dict(path):
    labels = {}
    with open(LABELS_PATH, mode='r') as f:
        # Cleaning up the file
        text = f.read()
        text = text.strip("\ufeff").strip()
    for line in text.split('\n'):
        sample = line.split('\t')
        sample[2], sample[3] = (int(sample[2]), int(sample[3])) # Converting 'start' and 'end' to int
        if sample[0] not in labels.keys(): # Create key if not in dict
            labels[sample[0]] = [sample[1:]]
        else:
            labels[sample[0]].append(sample[1:]) # Append if key already exists
    return labels


train_labels_dict = create_labels_dict(LABELS_PATH)

labels_type = list(set( [label[0] for labels in train_labels_dict.values() for label in labels] ))
labels_type = ["OTHER"] + labels_type
labels_num = len(labels_type)

data,labels = retrieveData(FIRST_DATASET_PATH,train_labels_dict)
label2id = {k: v for v, k in enumerate(list(set(labels)))}
id2label = {v: k for v, k in enumerate(list(set(labels)))}



# ------------------------------ DATA PROCESSING ----------------------------- #

#MIGHT DELETE THIS LATER BC WE HAVE A LOT OF NAME AND DATE AND THAT MAY CREATE TROUBLE
def tokenize_and_preserve_labels(sentence, labels, tokenizer):
    """
    This will take each word one by one (to preserve  the correct 
    label) and create token of it
    This is not mandatory.
    """

    tokenized_sentence = []
    tokenized_labels = []

    for word, label in zip(sentence, labels):

        if word == "[CLS]" or word == "[SEP]" or any(letter.isdigit() for letter in word):
            tokenized_sentence.append(word)
            tokenized_labels.append(label)

        else:

            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            tokenized_labels.extend([label] * n_subwords)

    return tokenized_sentence, tokenized_labels

tk_data,tk_labels = tokenize_and_preserve_labels(data, labels, BertTokenizer.from_pretrained('bert-base-uncased'))

# ------------------------------ DATASET OBJECT ------------------------------ #
class dataset(Dataset):
    def __init__(self,data,target):
        self.data = data
        self.target = target
        self.len = len(data)

    def __getitem__(self, index):
        return {
            "ids": self.data,
            "targets": self.target
        }
    
    def __len__(self):
        return self.len
    

# ------------------------- TRANSFORM DATA TO TENSOR ------------------------- #

id_data = BertTokenizer.from_pretrained('bert-base-uncased').convert_tokens_to_ids(tk_data)
id_labels = [label2id[label] for label in tk_labels]

tensor_data = torch.tensor(id_data,dtype=torch.long)
tensor_labels = torch.tensor(id_labels,dtype=torch.long)

params = {
    "batch_size": BATCH_SIZE,
    "shuffle": True,
    "num_workers": 0
}

loader = DataLoader(dataset(tensor_data,tensor_labels),**params)

