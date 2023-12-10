import torch
from torch.utils.data import TensorDataset, DataLoader
from model import PrivacyModel
from data import retrieveData, tokenize_and_preserve_labels, get_labels_types, truncate_sequences
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import joblib

TOKENIZE_TEST_DATA = True

MAX_SEQ_LENGTH = 512

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

# Loading the model
model = PrivacyModel(21).to(device)
model.load_state_dict(torch.load('models/privacy-model.pt'))
print(model)

VALIDATION_SET = 'dataset/validation_dataset/Validation_Release/'
LABELS_PATH = 'dataset/validation_dataset/answer.txt'

if TOKENIZE_TEST_DATA:
    labels_types, test_labels_dict = get_labels_types(LABELS_PATH)
    print('Retrieving Data')
    test_data, test_labels = retrieveData(VALIDATION_SET, test_labels_dict)
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

tensor_test_data = joblib.load('tensor_test_data.plk')
test_attention_mask = joblib.load('test_attention_mask.plk')
test_dataset = TensorDataset(tensor_test_data.to(device), test_attention_mask.to(device))
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=16)

full_outputs = []
for num_batch, (inputs, attention_mask) in enumerate(test_loader):
    print(f'Batch {num_batch+1}/{len(test_loader)}')
    outputs = model(inputs, attention_mask)
    output_labels = torch.argmax(outputs.squeeze(), dim=-1)
    for labeled_seq in output_labels:
        full_outputs.append(labeled_seq)
        print(labeled_seq)
print(full_outputs)
joblib.dump(full_outputs, 'output.plk')