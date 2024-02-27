import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
from preprocessing import load_bus, load_delay, load_passenger, load_weather
from function_package import split_xy
from sklearn.metrics import r2_score
from torcheval.metrics import R2Score
import copy
from skorch import NeuralNetRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor 
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

print(torch.__version__)    # 2.2.0+cu118

# 변수 설정
LINE_NUM = 8

device = (
    # "cpu"
    "cuda"
    if torch.cuda.is_available()
    else "mps" 
    if torch.backends.mps.is_available() 
    else "cpu"
)

# data
def data_gen(line_num):
    # 데이터 로드
    bus_csv = load_bus()
    passenger_csv = load_passenger()
    weather_csv = load_weather()
    delay_csv, delay_scaler = load_delay(return_scaler=True)

    # 레이블 선택
    bus = bus_csv
    passenger = passenger_csv
    weather = weather_csv
    x1 = weather
    x2 = delay_csv
    line_list = ['1호선지연(분)',
                 '2호선지연(분)',
                 '3호선지연(분)',
                 '4호선지연(분)',
                 '5호선지연(분)',
                 '6호선지연(분)',
                 '7호선지연(분)',
                 '8호선지연(분)']
    
    y = delay_csv[line_list[line_num-1]]
    x = np.concatenate((x1, x2), axis=1)

    # 훈련 및 테스트 데이터 분할(원하는 상황으로 주석처리를 바꾸기)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.9, random_state=100, stratify=y)
    
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    return x_train, y_train, x_test, y_test, delay_scaler

x_train, y_train, x_test, y_test, delay_scaler = data_gen(1)
print(f"{x_train.shape=},{y_train.shape=},{x_test.shape=},{y_test.shape=}")

# model
class TorchLSTM(nn.Module):
    def __init__(self,input_shape,output_shape) -> None:
        super().__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.LSTM(input_shape,256,batch_first=True),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,output_shape)
        )
        
    def forward(self,x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
class TorchDNN(nn.Module):
    def __init__(self,input_shape,output_shape) -> None:
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_shape,256,device=device),
            nn.ReLU(),
            nn.Linear(256,128,device=device),
            nn.ReLU(),
            nn.Linear(128,output_shape,device=device)
        )
        
    def forward(self,x):
        logits = self.linear_relu_stack(x)
        logits = logits.reshape(-1,)
        return logits
    
class MyLSTM(nn.Module):
    def __init__(self, num_classes, input_shape, hidden_size, num_layers) -> None:
        super(MyLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers  = num_layers
        self.input_size  = input_shape[0]
        self.hidden_size = hidden_size
        self.seq_length  = input_shape[1]
        # self.lstm = nn.LSTM(input_size=input_shape[1], hidden_size=hidden_size,
        #                     num_layers=num_layers, batch_first=True)
        self.conv1d = nn.Conv1d(in_channels=self.input_size, out_channels=32, kernel_size=3, stride=1, padding=1, device=device)
        self.lstm = nn.LSTM(input_size=self.seq_length, hidden_size=hidden_size, device=device,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128, device=device),
            nn.ReLU(),
            nn.Linear(128,64, device=device),
            nn.ReLU(),
            nn.BatchNorm1d(64, device=device),
            nn.Dropout(0.01),
            nn.Linear(64,32, device=device),
            nn.ReLU(),
            # nn.Linear(32,16, device=device),
            # nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(32,num_classes)
        )
    
    def forward(self, x):
        # print("x size at first",x.size())
        x = self.conv1d(x)
        # print("x size at second",x.size())
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        ula, (h_out, c_out) = self.lstm(x, (h_0,c_0))
        h_out = h_out.view(-1, self.hidden_size)   
        # print("x size at third",x.size())
        out = self.fc(h_out)
        # print("x size at final",x.size())
        return out
    
my_dnn = NeuralNetRegressor(TorchDNN(input_shape=x_train.shape[1],output_shape=1),
                            max_epochs=1000,
                            device=device,
                            criterion=nn.MSELoss,
                            optimizer=torch.optim.Adam,
                            )

xgb_params = {'learning_rate': 0.2218036245351803,
              'n_estimators': 199,
              'max_depth': 3,
              'min_child_weight': 0.07709868781803283,
              'subsample': 0.80309973945344,
              'colsample_bytree': 0.9254025887963853,
              'gamma': 6.628562492458777e-08,
              'reg_alpha': 0.012998871754325427,
              'reg_lambda': 0.10637051171111844}

model = VotingRegressor([
    ('My_DNN',my_dnn),
    # ('MyLSTM',my_lstm),
    ('RandomForestRegressor',RandomForestRegressor()),
    ('XGBRegressor',XGBRegressor(**xgb_params)),
    # ('CatBoostRegressor',CatBoostRegressor()), # error
    # ('AdaBoostRegressor',AdaBoostRegressor()),
    # ('LGBMRegressor',LGBMRegressor()),
    # ('SVR',SVR()),
    # ('LinearRegression',LinearRegression()),
])
    
# fit & eval
hist = model.fit(x_train,y_train)
r2 = model.score(x_test,y_test)
y_predict = model.predict(x_test)
loss = mean_squared_error(y_predict,y_test)
print("R2:   ",r2)
print("LOSS: ",loss)

# 결과를 파일로 저장해서 확인
y_test_1 = delay_scaler.inverse_transform(np.asarray(y_test).reshape(-1,1))
y_predict_1 = delay_scaler.inverse_transform(y_predict.reshape(-1,1))
y_submit_csv = pd.DataFrame()
y_submit_csv['true'] = y_test_1.reshape(-1)
y_submit_csv['pred'] = np.around(y_predict_1.reshape(-1))
y_submit_csv.to_csv(f'./data/weather_delay_LINE{LINE_NUM}_r2{r2:.8f}.csv')

# 모델 저장
PATH = f'./data/model_save/'
pickle.dump(model,open(PATH+f'weather_delay_ensemble_Line{LINE_NUM}_R2_{r2:.8f}.pkl', 'wb'))
model2 = pickle.load(open(PATH+f'weather_delay_ensemble_Line{LINE_NUM}_R2_{r2:.8f}.pkl', 'rb'))
model2_R2 = model2.score(x_test,y_test)
print("model2_R2: ",model2_R2)


# only my_dnn
# R2:    0.7944192166720178
# LOSS:  0.00019496497

# only XGBRegressor
# R2:    0.9988251359529255
# LOSS:  1.1141963e-06

# only RandomForestRegressor
# R2:    0.9960478186615606
# LOSS:  3.748097635884988e-06

# ensemble 1
# R2:    0.9999233737373666
# LOSS:  8.988690112451798e-07