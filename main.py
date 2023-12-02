import torch
from data import retrieveData, create_labels_dict

# --------------------------------- CONSTANTS -------------------------------- #

FIRST_DATASET_PATH = 'dataset/First_Phase_Release(Correction)/First_Phase_Text_Dataset/'
LABELS_PATH = 'dataset/First_Phase_Release(Correction)/answer.txt'

if __name__ == '__main__':
    print('Torch version: ', torch.__version__)
    train_labels_dict = create_labels_dict(LABELS_PATH)
    train_data = retrieveData(FIRST_DATASET_PATH,train_labels_dict)
