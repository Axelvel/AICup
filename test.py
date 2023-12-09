import torch
from model import PrivacyModel
from data import retrieveData, tokenize_and_preserve_labels, get_labels_types, pad_sequence, truncate_sequences
from transformers import BertTokenizer
import joblib

TOKENIZE_TEST_DATA = True

MAX_SEQ_LENGTH = 512

# Loading the model
model = PrivacyModel(21)
model.load_state_dict(torch.load('models/privacy-model.pt'))
print(model)

VALIDATION_SET = 'dataset/validation_dataset/Validation_Release/'
LABELS_PATH = 'dataset/validation_dataset/answer.txt'

if TOKENIZE_TEST_DATA:
    labels_types, test_labels_dict = get_labels_types(LABELS_PATH)
    print('Retrieving Data')
    test_data, test_labels = retrieveData(VALIDATION_SET, test_labels_dict)
    print('Tokenizing')
    tk_test_data, tk_test_labels = tokenize_and_preserve_labels(test_data, test_labels, BertTokenizer.from_pretrained('bert-base-uncased'))
    print('Full TK')
    full_tk_data = []
    for file_tk_data in tk_test_data:
        print(file_tk_data)
        if len(file_tk_data) == 0:
            print('EMPTY')
        full_tk_data.append(BertTokenizer.from_pretrained('bert-base-uncased').convert_tokens_to_ids(file_tk_data))
    print('Truncating')
    truncated_data, _ = truncate_sequences(full_tk_data, [], MAX_SEQ_LENGTH)
    print('Padding')
    padded_data = pad_sequence([torch.tensor(seq) for seq in truncated_data], batch_first=True)
    tensor_test_data = padded_data.type(torch.long)
    print('Padded:', padded_data)
    print('Tensor:', tensor_test_data)
    print('Shape:', tensor_test_data.shape)

    joblib.dump(tensor_test_data, 'tensor_test_data.plk')

tensor_test_data = joblib.load('tensor_test_data.plk')

outputs = model(tensor_test_data)
print(outputs)
