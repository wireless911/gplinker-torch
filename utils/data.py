from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self.data)
