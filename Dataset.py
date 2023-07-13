
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset,DataLoader,ConcatDataset

from datetime import datetime,timedelta

class EarthquakeDataset(Dataset):
    def __init__(self,path):
        self.df = pd.read_csv(path)


    def __getitem__(self,idx):
        
        data = self.df.iloc[idx]
        features = eval(data["Features"])
        label = data["Label"]

        return torch.FloatTensor(features),torch.tensor(label)

    def __len__(self):
        return len(self.df)
    


if __name__=="__main__":
    dataset = EarthquakeDataset("trainData_5daysInterval.csv")
    dataLoader = DataLoader(dataset,shuffle=True,batch_size=32)
    
    print("start")
    for features,label in dataLoader:
        print(features.shape)
        print(label.shape)
    print("end")