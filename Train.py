#Train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader,ConcatDataset
from Dataset import EarthquakeDataset
from Model import CNN,DNN,RNN
from Test import test


device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model,trainLoader,testLoader,lr,epochs,loss_fn=nn.CrossEntropyLoss()):
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        with tqdm(total=len(trainLoader),desc=f'Epoch:{epoch}') as progress:
            trainLoss = 0
            for features,targets in trainLoader:
                if model.name =="CNNModel":
                    features = features.to(device).unsqueeze(1)
                else:
                    features = features.to(device)
 
                targets = targets.to(device)
 
                out = model(features).to(device)

                loss = loss_fn(out,targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress.set_description(f'Epoch:{epoch},Loss:{loss.item()}')
                progress.update()
                trainLoss += loss.item()
                
            trainLoss /= len(trainLoader)

            testLoss = test(model,testLoader,loss_fn)
            progress.set_description(f'Epoch:{epoch},TrainLoss:{trainLoss},TestLoss:{testLoss}')

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'hyperParameter':model.hyperParameter
                }, f"{model.name}_{epoch}.pt")

if __name__=="__main__":



    model = RNN().to(device)


    days = 5
    trainPath= f"balanced_nonzero_trainData_{days}daysInterval.csv"
    testPath= f"testData_{days}daysInterval.csv"


    trainDataset = EarthquakeDataset(trainPath)
    testDataset = EarthquakeDataset(testPath)

    trainLoader = DataLoader(dataset=trainDataset,batch_size=256,shuffle=True)
    testLoader = DataLoader(dataset=testDataset,batch_size=64)

    loss_fn = nn.CrossEntropyLoss()

    train(model,trainLoader,testLoader,lr=0.0001,epochs=1000,loss_fn=loss_fn)

    


