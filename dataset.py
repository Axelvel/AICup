from torch.utils.data import Dataset

# ------------------------------ DATASET OBJECT ------------------------------ #
class dataset(Dataset):
    def __init__(self, data, target, attention_mask):
        self.data = data
        self.target = target
        self.attention_mask = attention_mask
        self.len = len(data)

    def __getitem__(self, index: int):
        return {
            "ids": self.data[index],
            "targets": self.target[index],
            "attention_mask": self.attention_mask[index]
        }
    
    def __len__(self):
        return self.len
    
    #TODO: Add collate function
    def collate_fn(self, items: list):
        pass