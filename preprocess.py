#preprocess.py

import pandas as pd
import numpy as np
from tqdm import tqdm

def preprocess():

    statDF = pd.read_csv("StatisticIndexEQK\summary_stat.csv")
    pattern = r'^S'
    matching_columns = statDF.filter(regex=pattern).columns

    
    statDF[matching_columns] = statDF[matching_columns].fillna(0)


    pattern = r'^K'
    matching_columns = statDF.filter(regex=pattern).columns
    column_means = statDF[matching_columns].mean()
    statDF[matching_columns] = statDF[matching_columns].fillna(column_means)


    statDF["DateTime"] = pd.to_datetime(statDF["DateTime"], format='%d-%b-%y')

    statDF["DateTime"]  = statDF["DateTime"] .dt.strftime('%d-%b-%Y')
    
    statDF.to_csv("StatisticIndexEQK\summary_stat_processed.csv",index=False)

def splitStation(path):
    if path =="train":
        path = "trainData.csv"
        targetDire = "trainingData/"
    else:
        path = "testData.csv"
        targetDire = "testingData/"

    statDF = pd.read_csv(path)
    stations = ["MS","TW","TT","YL","HC","HL","PT","YH","SL","LY","NC","KM","CS","MT","LN","ZB","XC","SM","KUOL","HUAL","TOCH","ENAN","SIHU","HERM","CHCH","DAHU","KAOH","PULI","SHRL","SHCH","FENL","YULI","RUEY","LIOQ","LISH","DABA","WANL","FENG"]
    
    for stn in stations:
        df = statDF.loc[statDF["stn"]==stn]
        df.to_csv(targetDire+stn+"_"+path)
def split_train_test():
    statDF = pd.read_csv("StatisticIndexEQK\summary_stat_processed.csv")
    eqkDF = pd.read_csv("StatisticIndexEQK\summary_eqk_Rc[50]_ML[5].csv").set_index(['DateTime'])


    testData = statDF.loc[statDF["DateTime"].str.contains("2022")]
    trainData = statDF.drop(statDF[statDF.isin(testData)].dropna().index)

    label = []
    with tqdm(total=len(testData)) as p:
        for i in range(len(testData)):
            stn = testData["stn"].iloc[i]
            date = testData["DateTime"].iloc[i]

            label.append(eqkDF.loc[[date]][stn].values.tolist()[0])
            p.update()

    testData["Earthquake"] = label

    testData.to_csv("testData.csv")

    label = []

    with tqdm(total=len(trainData)) as p:
        for i in range(len(trainData)):
            stn = trainData["stn"].iloc[i]
            date = trainData["DateTime"].iloc[i]

            label.append(eqkDF.loc[[date]][stn].values.tolist()[0])
            p.update()

    trainData["Earthquake"] = label

    trainData.to_csv("trainData.csv")



def makeRNNData(path,daysInterval = 5):
    stations = ["MS","TW","TT","YL","HC","HL","PT","YH","SL","LY","NC","KM","CS","MT","LN","ZB","XC","SM","KUOL","HUAL","TOCH","ENAN","SIHU","HERM","CHCH","DAHU","KAOH","PULI","SHRL","SHCH","FENL","YULI","RUEY","LIOQ","LISH","DABA","WANL","FENG"]
    
    station = []
    datetime = []
    features = []
    label = []

    if path =="train":
        filePath = "trainingData/"
        fileName = "trainData.csv"
        targetFileName = f"trainData_{daysInterval}daysInterval.csv"
    else:
        filePath = "testingData/"
        fileName = "testData.csv"

        targetFileName = f"testData_{daysInterval}daysInterval.csv"
    
    with tqdm(total = len(stations)) as p:
        for stn in stations:
            df = pd.read_csv(filePath+f"{stn}_"+fileName)

            for idx in range(len(df)-daysInterval):
                station.append(stn)
                data = df.iloc[idx:idx+daysInterval]
                feature = np.flip(data[["S_NS","S_EW","K_NS","K_EW","K","K_x","K_y","K_z","S","S_x","S_y","S_z"]].to_numpy())

                features.append(feature.tolist())

                datetime.append(data["DateTime"].values.tolist())

                data = df.iloc[idx+daysInterval]

                label.append(data["Earthquake"])
            p.update()


    pd.DataFrame({"Station":station,"DateTime":datetime,"Features":features,"Label":label}).to_csv(targetFileName)
            
            
def removeAllZeroData(path):
    df = pd.read_csv(path)
    nan_data = [0.0,0.0,43.704803518946086,42.220488514448384,22.494873380532788,18.351998919473683,23.05252797378771,20.176213219809565,0.0,0.0,0.0,0.0]
    nan_data.reverse()

    df = df[df['Features'].apply(lambda x: not all(all(elem == nan_data[i] for i,elem in enumerate(sublist)) for sublist in eval(x)))]
    print(df)
    df.reset_index()
    df.to_csv("nonzero_"+path)

def balancedData(path):
    
    df = pd.read_csv(path)
    is_zero = df['Label'] == 0

    # 生成一个与 DataFrame 行数相同的随机概率向量
    probabilities = np.random.uniform(size=len(df))

    # 根据概率筛选出需要删除的行
    to_delete = is_zero & (probabilities < 0.995)

    # 删除符合条件的行
    df = df[~to_delete]

    # 重新设置索引（可选）
    df = df.reset_index(drop=True)

    df.to_csv("balanced_"+path)
if __name__=="__main__":
    preprocess()
    split_train_test()
    splitStation("train")
    splitStation("test")
    daysInterval = 30
    makeRNNData("train",daysInterval=daysInterval)
    makeRNNData("test",daysInterval=daysInterval)

    removeAllZeroData(f"trainData_{daysInterval}daysInterval.csv")
    balancedData(f"nonzero_trainData_{daysInterval}daysInterval.csv")