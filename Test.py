#test.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,ConcatDataset
import pandas as pd
from tqdm import tqdm
from Dataset import EarthquakeDataset
from Model import CNN,RNN,DNN
import random
device = "cuda"

def test(model,testLoader,loss_fn=nn.CrossEntropyLoss()):
    model.eval()

    testLoss = 0

    label = []
    predict = []

    with tqdm(total=len(testLoader)) as progress:
         for features,targets in testLoader:
                if model.name =="CNNModel":
                    features = features.to(device).unsqueeze(1)
                else:
                    features = features.to(device)
                targets = targets.to(device)



                out = model(features).to(device)
                loss = loss_fn(out,targets)
                
                out = torch.softmax(out,dim=1)
                label += targets.cpu().numpy().tolist()     
                predict += torch.argmax(out,dim=1).cpu().numpy().tolist()


                testLoss += loss.item()
                progress.update()

    df = pd.DataFrame({"predict":predict,"label":label})
    df.to_csv("result.csv",index=False)
    return testLoss/len(testLoader)

import math

def calculate_confidence_interval(success_count, total_count, confidence_level):
    success_rate = success_count / total_count
    standard_error = math.sqrt((success_rate * (1 - success_rate)) / total_count)
    z_value = 0.0

    if confidence_level == 0.90:
        z_value = 1.645
    elif confidence_level == 0.95:
        z_value = 1.96
    elif confidence_level == 0.99:
        z_value = 2.576
    else:
        return None

    lower_bound = success_rate - (standard_error * z_value)
    upper_bound = success_rate + (standard_error * z_value)

    return (lower_bound, upper_bound)


def randomModel(testLoader):
    correct_prob = 0
    epochs = 10
    for _,targets in testLoader:
        zeros = torch.zeros(targets.shape)
        indices = random.sample(range(1000), 100)
        zeros[indices] = 1

        targets_index = torch.nonzero(targets).squeeze(1)
        zeros_index = torch.nonzero(zeros).squeeze(1)
        num_same_values = 0
        if(len(targets_index)>0):
            for i in range(len(targets_index)):
                tr = targets_index[i]
                if tr in zeros_index:
                    num_same_values = num_same_values + 1
                
            correct_rate = (num_same_values/len(targets_index))
            if(correct_rate > 0.1):
                correct_prob += 1
    print(correct_prob / (len(testLoader)))
                
    return correct_prob / (len(testLoader))

def randomTest(testLoader):
    correct_prob = 0
    epochs = 10
    for features,targets in testLoader:
        features = features.to(device)
        targets = targets.to(device)
        out = model(features)
        predict = torch.argmax(out,dim=1)

        predict_index = torch.nonzero(predict).squeeze(1)
        targets_index = torch.nonzero(targets).squeeze(1)

        num_same_values = 0
        if(len(targets_index)>0):
            for i in range(len(targets_index)):
                tr = targets_index[i]
                if tr in predict_index:
                    num_same_values = num_same_values + 1
                
            correct_rate = (num_same_values/len(targets_index))
            if(correct_rate > 0.1):
                correct_prob += 1
    print(correct_prob / (len(testLoader)))
    return correct_prob / (len(testLoader))
if __name__=="__main__":

    model = RNN().to(device)
    checkpoint = torch.load("RNNModel_13.pt")
    model.load_state_dict(checkpoint["model_state_dict"])


    testDataset = EarthquakeDataset("testData_5daysInterval.csv")



    testLoader = DataLoader(dataset=testDataset,batch_size=1000,drop_last=True,shuffle=True)


    df = pd.read_csv("testData_5daysInterval.csv")
    num_class0 = (len(df.loc[df["Label"]==0]))
    num_class1 = (len(df.loc[df["Label"]==1]))

    max_num_class = max(num_class0,num_class1)
    weightTensor = torch.FloatTensor([max_num_class/num_class0,max_num_class/num_class1]).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weightTensor)


    #test(model,testLoader,loss_fn=loss_fn)

    df = pd.read_csv("result.csv")
    ##
    correct = df['predict'] == df['label']
    mistake = df['predict'] != df['label']

    predict_0 = df['predict'] == 0
    predict_1 = df['predict'] == 1


    count_pre0 = len(df[correct & predict_0])
    count_pre1 = len(df[correct & predict_1])

    count_label0 = len(df[mistake & predict_1])
    count_label1 = len(df[mistake & predict_0])

    num_0 = count_pre0 + count_label0
    num_1 = count_pre1 + count_label1
    print("0的正確率:", count_pre0/num_0)
    print("1的正確率:", count_pre1/num_1)
    print("總正確率", len(df[correct]) / len(df) )

    prob = 1
    n = 50
    for i in range(n):
        p = randomTest(testLoader)
        if p != 0:
            prob *= p
        else:
            n -= 1

    print(prob**(1/n))