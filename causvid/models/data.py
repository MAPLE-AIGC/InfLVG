from torch.utils.data import Dataset
import numpy as np
import json

class TextDataset(Dataset):
    def __init__(self, data_path):
        self.texts = []
        with open(data_path, "r") as f:
            for line in f:
                self.texts.append(line.strip())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]
    
class EventPromptSetDataset(Dataset):
    def __init__(self, data_path):
        self.eps_sets = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                eps_list = json.loads(line.strip())  # 每行是一个 list[dict]
                self.eps_sets.append(eps_list)

    def __len__(self):
        return len(self.eps_sets)

    def __getitem__(self, idx):
        return self.eps_sets[idx]  # 返回 list[dict]