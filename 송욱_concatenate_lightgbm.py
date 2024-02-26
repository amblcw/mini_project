import pandas as pd
import numpy as np
import pickle
import os.path
import datetime as dt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler,RobustScaler
from preprocessing import load_bus, load_deay, load_passenger, load_weather
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import optuna
import lightgbm as lgb
from keras.layers import concatenate
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
import time
# 데이터 로드
bus_csv = load_bus()
passenger_csv = load_passenger()
weather_csv = load_weather()
delay_csv = load_deay()

# 레이블 선택
bus = bus_csv
passenger = passenger_csv
weather = weather_csv
y = delay_csv['1호선지연(분)']
# y = delay_csv['2호선지연(분)']
# y = delay_csv['3호선지연(분)']
# y = delay_csv['4호선지연(분)']
# y = delay_csv['5호선지연(분)']
# y = delay_csv['6호선지연(분)']
# y = delay_csv['7호선지연(분)']
# y = delay_csv['8호선지연(분)']

# 훈련 및 테스트 데이터 분할(원하는 상황으로 주석처리를 바꾸기)
# x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
#     bus,passenger,weather, y, train_size=0.99, random_state=100, stratify=y)
# #버스데이터로 예측
x_train, x_test, y_train, y_test = train_test_split(bus, y, train_size=0.99, random_state=100, stratify=y)
# #인원데이터로 예측
# x_train, x_test, y_train, y_test = train_test_split(passenger, y, train_size=0.99, random_state=100, stratify=y)
# #날씨데이터로 예측
# x_train, x_test, y_train, y_test = train_test_split(weather, y, train_size=0.99, random_state=100, stratify=y)

# 스케일링(모든 데이터 이용시)
scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
# x1_train_scaled = scaler.fit_transform(x1_train)
# x1_test_scaled = scaler.transform(x1_test)

# x2_train_scaled = scaler.fit_transform(x2_train)
# x2_test_scaled = scaler.transform(x2_test)

# x3_train_scaled = scaler.fit_transform(x3_train)
# x3_test_scaled = scaler.transform(x3_test)
# 스케일링(각각)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# 데이터 연결(모든 데이터 이용시)
# x_train = np.concatenate((x1_train_scaled, x2_train_scaled, x3_train_scaled), axis=1)
# x_test = np.concatenate((x1_test_scaled, x2_test_scaled, x3_test_scaled), axis=1)
s_t = time.time()
import random
def objective(trial):
    rd=random.randint(1,1000)
    params = {
        "metric": "mse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": rd,
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'num_leaves': trial.suggest_int('num_leaves', 10, 50),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-2, 100.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-2, 100.0),
        'min_split_gain': trial.suggest_loguniform('min_split_gain', 1e-8, 100.0),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-8, 100.0),
        'cat_smooth': trial.suggest_int('cat_smooth', 1, 100),
        "early_stopping_rounds": 100,  # 얼리 스탑
    }
    
    model = lgb.LGBMRegressor(**params, device='gpu')
    
    model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
              eval_metric='mse',)
    
    y_pred = model.predict(x_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)
    
    r2 = r2_score(y_test, y_pred)
    print("R^2 Score:", r2)
    
    return r2

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5000)
#n_trials : 최적화를 위해 시도할 하이퍼파라미터 조합의 수
e_t = time.time()
print("Best parameters found: ", study.best_params)
print("r2: ", study.best_value)
print("시간 :", e_t-s_t)
 
#데이터 폴더에 다른 파일이 들어가는걸 방지하기위해 경로를 다른곳으로 설정해둔것
# best_params = study.best_params
# best_model = lgb.LGBMClassifier(**best_params, device='gpu')
# best_model.fit(x_train, y_train)
# best_model.booster_.save_model("c:/_data/_save/project/mini_project_lightbgm.h5")


#############################파라미터 설명#####################################
# metric: 모델의 성능을 측정하는 데 사용할 평가 지표를 지정합니다. 여기서는 평균 제곱 오차(MSE)를 사용하고 있습니다.

# verbosity: LightGBM의 출력을 제어하는 매개변수입니다. -1로 설정하면 출력이 없습니다.

# boosting_type: 부스팅 방법을 지정합니다. 여기서는 gbdt를 사용하고 있습니다. gbdt는 Gradient Boosting Decision Tree의 약자로 일반적인 그래디언트 부스팅을 의미합니다.

# random_state: 모델의 랜덤 시드를 설정합니다.

# learning_rate: 각 트리의 가중치를 줄이는 데 사용되는 학습 속도를 나타냅니다.

# n_estimators: 부스팅 라운드의 수, 즉 트리의 개수를 나타냅니다.

# num_leaves: 각 트리가 가질 수 있는 최대 잎 노드의 수를 나타냅니다.

# max_depth: 트리의 최대 깊이를 제한합니다.

# min_child_samples: 리프 노드에 필요한 최소 샘플 수를 나타냅니다.

# subsample: 트리를 학습하는 데 사용되는 데이터의 일부분을 나타냅니다.

# colsample_bytree: 각 트리를 학습할 때 사용되는 특성의 비율을 나타냅니다.

# reg_alpha: L1 정규화를 제어하는 매개변수입니다.

# reg_lambda: L2 정규화를 제어하는 매개변수입니다.

# min_split_gain: 분할에서 최소한으로 필요한 증가를 제어하는 매개변수입니다.

# min_child_weight: 리프 노드에 필요한 최소 가중치 합을 나타냅니다.

# cat_smooth: 카테고리 데이터를 부드럽게 처리하는 데 사용되는 매개변수입니다.
