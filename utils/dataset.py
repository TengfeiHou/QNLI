#coding=utf8
from torch.utils.data import Dataset
from utils.reader import read_dataset

class QNLIDataset(Dataset):

    def __init__(self, choice='train'):
        """
            data is a list of Example
        """
        super(QNLIDataset, self).__init__()
        self.data = read_dataset(choice)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]