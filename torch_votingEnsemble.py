import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from keras.datasets import fashion_mnist
import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
from preprocessing import load_bus, load_delay, load_passenger, load_weather
from function_package import split_xy
from sklearn.metrics import r2_score
from torcheval.metrics import R2Score
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import torch.nn.functional as F
import copy

print(torch.__version__)    # 2.2.0+cu118

# x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]]).reshape(-1,3,1)
# y = np.array([4,5,6,7,8,9,10]).astype(np.float32)
# print(x.shape,y.shape) # (7, 3, 1) (7,)

# bus_csv = load_bus()
# print(bus_csv.shape)
# x, y = split_xy(bus_csv,100)
# print(x.shape, y.shape)
# print(y[:10])

# delay_csv = load_delay()
# x, y = split_xy(delay_csv,100)

# passenger_csv = load_passenger()
# x, y = split_xy(passenger_csv,24)
# print(x.shape,y.shape)

weather_csv = load_weather()
print(weather_csv.head(24))
x, y = split_xy(weather_csv,480)
NAME = 'Weather'
print("Weather")

np.save('./data/temp_x',x)
np.save('./data/temp_y',y)

# x = np.load('./data/temp_x.npy')
# y = np.load('./data/temp_y.npy')

print("x shape, y shape ",x.shape,y.shape)

class CustomImageDataset(Dataset):
    def __init__(self,x_data,y_data,transform=None) -> None:    # 생성자, x,y데이터, 변환함수 설정
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform
        
    def __len__(self):          # 어떤 값을 길이로서 반환할지
        return len(self.y_data)
    
    def __getitem__(self, idx): # 인덱스가 들어왔을 때 어떻게 값을 반환할지 
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = torch.FloatTensor(self.x_data[idx].copy())  # torchTensor 형식으로 변환
        label = self.y_data[idx].copy()                     # 얘는 9를 변환하면 사이즈9짜리 텐서로 만들어버리기에 그냥 이대로 사용
        sample = image, label                               # 반드시 x, y 순으로 반환할 것
        
        if self.transform:  # 전처리 함수 적용
            sample = self.transform(sample)
            
        return sample

import matplotlib.pyplot as plt

training_data = CustomImageDataset(x,y) # 인스턴스 선언 및 데이터 지정

BATCH_SIZE = 128
train_dataloader = DataLoader(training_data,batch_size=BATCH_SIZE, pin_memory=True)  # 만든 커스텀 데이터를 iterator형식으로 변경
test_dataloader = DataLoader(training_data,batch_size=BATCH_SIZE, pin_memory=True)

device = (
    # "cpu"
    "cuda"
    if torch.cuda.is_available() 
    else "mps" 
    if torch.backends.mps.is_available() 
    else "cpu"
)

class EnsembleModel(nn.Module):
    def __init__(self,) -> None:
        super(EnsembleModel,self).__init__()
        self.xgb = XGBRegressor()
        self.rf  = RandomForestRegressor()
        self.dt  = DecisionTreeRegressor()
        
    def forward(self, x):
        pred1 = XGBRegressor(x)
        
        

loss_fn = nn.MSELoss()
# loss_fn = nn.CrossEntropyLoss() # 이 함수에 softmax가 내재되어있기에 모델에서 softmax를 쓰면 안된다
optimizer = torch.optim.Adam(model.parameters())

def train(dataloader, model, loss_fn, optimizer, verbose=True):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # 예측 오류 계산
        pred = model(X)
        pred = pred.to(device)
        loss = loss_fn(pred, y)
        
        # 역전파
        optimizer.zero_grad()   # 역전파하기 전에 기울기를 0으로 만들지 않으면 전의 학습이 영향을 준다
        loss.backward()         # 역전파 계산
        optimizer.step()        # 역전파 계산에 따라 파라미터 업데이트(이 시점에서 가중치가 업데이트 된다)
        
        if (batch % 10 == 0) and verbose:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # print("size check: ",size,num_batches) # size는 테스트 데이터의 개수, num_batches는 batch_size에 의해 몇 바퀴 도는지
    model.eval()    # 모델을 평가 모드로 전환, Dropout이나 BatchNomalization등을 비활성화
    test_loss, correct = 0, 0
    
    last_pred = last_y = None
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            last_y = y
            last_pred = pred = model(X)
            pred = pred.reshape(-1)
            pred = pred.to(device)
            test_loss += loss_fn(pred, y).item()
            metric = R2Score().to(device)
            metric.update(pred,y).to(device)
            r2 = metric.compute()
            
            correct += r2 
    test_loss /= num_batches    
    correct /= num_batches
    print(f"Test Error \n R2: {(correct):>0.4f}, Avg loss: {test_loss:>8f}\n")
    return test_loss, last_pred, last_y
    
if __name__ == '__main__':
    print(f"Using {device} device")
    print(model)
    
    EPOCHS = 1000
    PATIENCE = 1000
    best_loss = 987654321
    best_model = None
    patience_count = 0
    loss_list = []
    last_pred = last_y = None
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n---------------------")
        train(train_dataloader, model, loss_fn, optimizer,verbose=False)
        loss, last_pred, last_y = test(test_dataloader, model, loss_fn)
        loss_list.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_model = copy.deepcopy(model)   # restore best weight 구현
            patience_count = 0
        else:
            patience_count += 1
            
        if patience_count >= PATIENCE:
            print("Early Stopped")
            break

    print("===== best model =====")
    test(test_dataloader,best_model,loss_fn)
    print("Best loss: ",best_loss)
    print("last_pred: ",last_pred, "| last_y: ", last_y)
    print("Done")
    torch.save(best_model.state_dict,f'./data/torch_model_save/{NAME}_loss_{best_loss:.4f}.pth')

    
    '''
    지연 예측을 다중분류로 만들어버린 뒤 sklearn.ensemble.VotingClassifier적용해보기
    
    '''