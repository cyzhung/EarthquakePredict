#Model.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Dataset import EarthquakeDataset

class RNN(nn.Module):
    def __init__(self,in_dim=12,hidden_dim=64,Bidirection=False,num_layers=2):
        super().__init__()
        self.hyperParameter = {"in_dim":in_dim,"hidden_dim":hidden_dim,"Bidirection":Bidirection,"num_layers":num_layers}

        self.name = "RNNModel"
        self.num_layers = num_layers
        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )


        self.Bidirection = 2 if Bidirection==True else 1

        self.fc1 = nn.Linear(in_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,2)

        self.hidden_dim = hidden_dim
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self,x):
        h0 = torch.zeros(self.num_layers , x.size(0), self.hidden_dim).to(next(self.parameters()).device,dtype=torch.float32)
        c0 = torch.zeros(self.num_layers , x.size(0), self.hidden_dim).to(next(self.parameters()).device,dtype=torch.float32)

        out,_ = self.rnn(x,(h0,c0))

        out = self.fc2(out[:,-1,:])

        return out
    
class DNN(nn.Module):
    def __init__(self,in_dim=12,hidden_dim=64):
        super().__init__()
        self.hyperParameter = {"in_dim":in_dim,"hidden_dim":hidden_dim}

        self.name = "DNNModel"
        self.fc1 = nn.Linear(in_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,2)

        self.hidden_dim = hidden_dim
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self,x):

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out)

class CNN(nn.Module):
    def __init__(self,in_dim=12,out_channels=3,kernal_size=(3,3)):
        super().__init__()
        self.hyperParameter = {"in_dim":in_dim,"kernal_size":kernal_size,"out_channels":out_channels}

        self.name = "CNNModel"
        self.cnn1 = nn.Conv2d(in_channels=1,out_channels=out_channels,kernel_size=kernal_size)
        self.cnn2 = nn.Conv2d(in_channels=out_channels,out_channels=5,kernel_size=kernal_size)


        self.fc2 = nn.Linear(1040,2)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(0.1)
    def forward(self,x):

        out = self.cnn1(x)
        out = self.relu(out)
        out = self.cnn2(out)
        out = self.relu(out)
        out = self.flatten(out)
        out = self.fc2(out)
        return (out)
    

if __name__=="__main__":
    inputs = torch.rand(8,1,30,12)
    model = CNN()

    out = model(inputs)
    print(out.shape)