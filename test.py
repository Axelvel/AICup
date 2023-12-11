import torch
from torch.utils.data import TensorDataset, DataLoader
from model import PrivacyModel
from data import retrieveData, tokenize_and_preserve_labels, get_labels_types, truncate_sequences, fetchData
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import joblib
from os import listdir
import numpy as np
from copy import deepcopy


TOKENIZE_TEST_DATA = False
GENERATE_OUTPUT = True

MAX_SEQ_LENGTH = 512

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

# Loading the model
model = PrivacyModel(21).to(device)
model.load_state_dict(torch.load('models/privacy-model.pt'))
print(model)

#VALIDATION_SET = 'dataset/validation_dataset/Validation_Release/'
VALIDATION_SET = 'dataset/Final_Data/opendid_test/'

LABELS_PATH = 'dataset/validation_dataset/answer.txt'
#LABELS_PATH = 'dataset/Final_Data/valid/opendid_valid.tsv'

if TOKENIZE_TEST_DATA:
    labels_types, test_labels_dict = get_labels_types(LABELS_PATH)
    print('Retrieving Data')
    #test_data, test_labels = retrieveData(VALIDATION_SET, test_labels_dict)
    test_data = fetchData(VALIDATION_SET)
    test_labels = deepcopy(test_data)
    print('Tokenizing data')
    tk_test_data, tk_test_labels = tokenize_and_preserve_labels(test_data, test_labels, BertTokenizer.from_pretrained('bert-base-uncased'))
    print('Bert Tokenizing')
    full_tk_data = []
    for file_tk_data in tk_test_data:
        full_tk_data.append(BertTokenizer.from_pretrained('bert-base-uncased').convert_tokens_to_ids(file_tk_data))
    print(full_tk_data)
    print('Truncating data')
    truncated_data, _ = truncate_sequences(full_tk_data, full_tk_data, MAX_SEQ_LENGTH)
    print('Padding data')
    padded_data = pad_sequence([torch.tensor(seq) for seq in truncated_data], batch_first=True)
    tensor_test_data = padded_data.type(torch.long)
    mask = (tensor_test_data != 0)
    print('Padded:', padded_data)
    print('Tensor:', tensor_test_data)
    print('Shape:', tensor_test_data.shape)

    joblib.dump(tensor_test_data, 'tensor_test_data.plk')
    joblib.dump(mask, 'test_attention_mask.plk')

if GENERATE_OUTPUT:
    tensor_test_data = joblib.load('tensor_test_data.plk')
    test_attention_mask = joblib.load('test_attention_mask.plk') 
    id2label = joblib.load('id2label.plk')
    test_dataset = TensorDataset(tensor_test_data.to(device), test_attention_mask.to(device))
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=16)

    full_outputs = []
    labeled_output = []
    for num_batch, (inputs, attention_mask) in enumerate(test_loader):
        print(f'Batch {num_batch+1}/{len(test_loader)}')
        outputs = model(inputs, attention_mask)
        output_labels = torch.argmax(outputs.squeeze(), dim=-1)
        for labeled_seq in output_labels:
            full_outputs.append(labeled_seq)
            #print(labeled_seq)
            # converted_array = [id2label.get(x) for x in labeled_seq.cpu().numpy()]
            # labeled_output.append(converted_array)
            # print(labeled_seq)
            # print(converted_array)
    print(full_outputs)
    joblib.dump(full_outputs, 'output.plk')

print('Loading output tensors')
output = joblib.load('output.plk')
mask = joblib.load('test_attention_mask.plk')

def save_tensors_to_txt(filename, tensors, attention_mask):
    with open(filename, 'w', encoding='utf-8') as f:
        for tensor, mask in zip(tensors, attention_mask):
            array = tensor.cpu().numpy()
            for value, mask_value in zip(array, mask):
                if mask_value:
                    f.write(str(value) + ' ')
            f.write('\n')


#save_tensors_to_txt("output.txt", output, mask)

def load_data_from_txt(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            values = []
            for value_str in line.strip().split(' '):
                value = int(value_str)
                values.append(value)
            data.append(values)
    return data


output_data = load_data_from_txt("output.txt")
print(f"Loaded data: {output_data}")
id2label = joblib.load('id2label.plk')
label2id = joblib.load('label2id.plk')

IGNORE_TOKENS = ['[CLS]', '[SEP]']
# IGNORE_LABELS = ['OTHER', '[CLS]', '[SEP]']

labels_types, test_labels_dict = get_labels_types(LABELS_PATH)
print('Retrieving Data')
test_data, _ = retrieveData(VALIDATION_SET, test_labels_dict)


ANSWER_PATH = 'answer.txt'

def reformat_output(file_data, output):
    reformated_output = []
    squeezed_output = [item for sublist in output for item in sublist]
    pos = 0
    for file in file_data:
        file_length = len(file)
        reformated_output.append(squeezed_output[pos:pos+file_length])
        pos += file_length
    return reformated_output


reformated_output = reformat_output(test_data, output_data)

list_files = listdir(VALIDATION_SET)
file_ids = [filename[:-4] for filename in list_files]

def find_word(file_path, word, start_pos=0):
    positions = []
    file_position = 0
    with open(file_path, "r") as f:
        file = f.read()
        start = file.find(word, start_pos)
        end = start + len(word)
        return start, end

def generate_answer(file_path, output):
    print('Generating answers.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        last_position = 0
        for id_file, (file_data, output_data) in enumerate(zip(test_data, output)):
            position = 0
            for value, output_value in zip(file_data, output_data):
                if value not in IGNORE_TOKENS:
                    file = file_ids[id_file]
                    data = value
                    label = id2label.get(output_value)
                    start, end = find_word(VALIDATION_SET + file + '.txt', data, last_position)
                    last_position = end
                    f.write(file + '\t' + label + '\t' + str(start) + '\t' + str(end) + '\t' + data + '\n')


generate_answer(ANSWER_PATH, reformated_output)
print(label2id)