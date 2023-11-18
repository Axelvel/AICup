from os import listdir

# --------------------------------- CONSTANTS -------------------------------- #
TRAINING_PATH = "./AICup/dataset/First_Phase_Release(Correction)/First_Phase_Text_Dataset/"


# ------------------------------ RETRIEVING DATA ----------------------------- #
def retrieveData(path):
    list_file = listdir(path)
    dataset = ["[CLS]"]

    for file in list_file:
        with open(path + file) as f:
            lines = f.readlines()
        for line in lines:
            if line != "\n":
                splitted = line.split()
                for word in splitted:
                    dataset.append(word)
        dataset.append("[SEP]")
    
    return dataset



