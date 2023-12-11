import torch
from data import retrieveData, create_labels_dict

# --------------------------------- CONSTANTS -------------------------------- #

FIRST_DATASET_PATH = 'dataset/First_Phase_Release(Correction)/First_Phase_Text_Dataset/'
LABELS_PATH = 'dataset/First_Phase_Release(Correction)/answer.txt'

if __name__ == '__main__':
    print('Torch version: ', torch.__version__)
    # Launch data.py to load and tokenize the dataset
    # Launch train.py to train the model
    # Launch test.py to test on the model on eval dataset and generate the answer.txt
