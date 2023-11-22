from os import listdir

# --------------------------------- CONSTANTS -------------------------------- #

FIRST_DATASET_PATH = 'dataset/First_Phase_Release(Correction)/First_Phase_Text_Dataset/'
LABELS_PATH = 'dataset/First_Phase_Release(Correction)/answer.txt'


# ------------------------------ RETRIEVING DATA ----------------------------- #
def retrieveData(path):
    list_file = listdir(path)
    dataset = ["[CLS]"]

    for file in list_file:
        with open(path + file, encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            if line != "\n":
                splitted = line.split()
                for word in splitted:
                    dataset.append(word)
        dataset.append("[SEP]")

    return dataset


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
    return labels_types
