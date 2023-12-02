
import torch
from os import listdir
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from torch.nn.utils.rnn import pad_sequence
from dataset import dataset
import joblib

# --------------------------------- CONSTANTS -------------------------------- #

FIRST_DATASET_PATH = 'dataset/First_Phase_Release(Correction)/First_Phase_Text_Dataset/'
LABELS_PATH = 'dataset/First_Phase_Release(Correction)/answer.txt'

BATCH_SIZE = 64
MAX_SEQ_LENGTH = 512


# ------------------------------ RETRIEVING DATA ----------------------------- #
def retrieveData(path, dict):
    list_file = listdir(path)
    items = []
    labels = []
    for file in list_file:
        file_items = ["[CLS]"]
        file_labels = ["[CLS]"]
        raw_data = dict[file[:-4]]
        with open(path + file) as f:
            lines = f.readlines()
        for line in lines:
            if line != "\n":
                splitted = line.split()
                for word in splitted:
                    file_items.append(word)

                    check = False
                    for list in raw_data:
                        if word in list and word != list[0] and check == False:
                            file_labels.append(list[0])
                            check = True
                    if check == False:
                        file_labels.append("OTHER")
                file_items.append("[SEP]")
                file_labels.append("[SEP]")
        items.append(file_items)
        labels.append(file_labels)
    return items,labels

# Creating label dict

def create_labels_dict(path):
    labels = {}
    with open(path, mode='r', encoding='utf-8') as f:
        # Cleaning up the file
        text = f.read()
        text = text.strip("\ufeff").strip()
    for line in text.split('\n'):
        sample = line.split('\t')
        sample[2], sample[3] = (int(sample[2]), int(sample[3]))  # Converting 'start' and 'end' to int
        if sample[0] not in labels.keys():  # Create key if not in dict
            labels[sample[0]] = [sample[1:]]
        else:
            labels[sample[0]].append(sample[1:])  # Append if key already exists
    return labels

def get_labels_types(path):
    labels_dict = create_labels_dict(path)
    labels_types = list(set([label[0] for labels in labels_dict.values() for label in labels]))
    labels_types = ["OTHER"] + labels_types
    return labels_types, labels_dict


# ------------------------------ DATA PROCESSING ----------------------------- #

#MIGHT DELETE THIS LATER BC WE HAVE A LOT OF NAME AND DATE AND THAT MAY CREATE TROUBLE
def tokenize_and_preserve_labels(sentence, labels, tokenizer):
    """
    This will take each word one by one (to preserve  the correct label) and create token of it
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
            tokenized_word = tokenizer.tokenize(word)  #TODO: Add padding?
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            tokenized_labels.extend([label] * n_subwords)

    return tokenized_sentence, tokenized_labels

# Split sequences longer than 512 tokens
def truncate_sequences(data, labels, max_seq_length):
    truncated_data = []
    truncated_labels = []

    for seq, label_seq in zip(data, labels):
        if len(seq) <= max_seq_length:
            truncated_data.append(seq)
            truncated_labels.append(label_seq)
        else:
            for i in range(0, len(seq), max_seq_length):
                end_idx = min(i + max_seq_length, len(seq))
                truncated_data.append(seq[i:end_idx])
                truncated_labels.append(label_seq[i:end_idx])

    return truncated_data, truncated_labels


if __name__ == "__main__":
    
    labels_types, train_labels_dict = get_labels_types(LABELS_PATH)

    data, labels = retrieveData(FIRST_DATASET_PATH,train_labels_dict)

    flat_label = []
    for file in labels:
        for word in file:
            flat_label.append(word)
    label2id = {k: v for v, k in enumerate(list(set(flat_label)))}
    id2label = {v: k for v, k in enumerate(list(set(flat_label)))}

    tk_data, tk_labels = tokenize_and_preserve_labels(data, labels, BertTokenizer.from_pretrained('bert-base-uncased'))

    # ------------------------- TRANSFORM DATA TO TENSOR ------------------------- #

    full_tk_data = []
    full_tk_label = []
    for file_tk_data, file_tk_labels in zip(tk_data, tk_labels):
        full_tk_data.append(BertTokenizer.from_pretrained('bert-base-uncased').convert_tokens_to_ids(file_tk_data))
        full_tk_label.append([label2id[label] for label in file_tk_labels])

    truncated_data, truncated_labels = truncate_sequences(full_tk_data, full_tk_label, MAX_SEQ_LENGTH)
    padded_data = pad_sequence([torch.tensor(seq) for seq in truncated_data], batch_first=True)
    tensor_data = padded_data.type(torch.long)
    print('Padded:', padded_data)
    print('Tensor:', tensor_data)
    print('Shape:', tensor_data.shape)

    padded_labels = pad_sequence([torch.tensor(seq) for seq in truncated_labels], batch_first=True)
    tensor_labels = padded_labels.type(torch.long)

    params = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": 0
    }

    loader = DataLoader(dataset(tensor_data, tensor_labels), **params)
    joblib.dump(loader, 'loader.plk')
    joblib.dump(tensor_data, 'tensor_data.plk')
    joblib.dump(tensor_labels, 'tensor_labels.plk')

