import pandas as pd
import numpy as np
import pickle
import os.path
import datetime as dt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from preprocessing import load_bus, load_deay, load_passenger, load_weather
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import optuna
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
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

def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 67,
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-8, 100.0),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 100.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-2, 100.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-2, 100.0),
    }

    model = XGBRegressor(**params, tree_method='gpu_hist')
    model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
              early_stopping_rounds=100,
              verbose=False)
    
    y_pred = model.predict(x_test)
    
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("RMSE:", rmse)
    
    r2 = r2_score(y_test, y_pred)
    print("R2 Score:", r2)
    
    return r2

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best parameters found: ", study.best_params)
print("R2: ", study.best_value)
# 모델 저장이 잘 안되서 사용금지
# best_params = study.best_params
# best_model = xgb.train(best_params, x_train)

# # 모델 저장
# with open("c:/_data/_save/project/mini_project_xgboost.pkl', 'wb') as f:
#     pickle.dump(best_model, f)
    

'''
Trial 99 finished with value: 0.46598770124503597 and parameters: {'learning_rate': 0.2955423310683057, 'n_estimators': 177, 'max_depth': 4, 'min_child_weight': 2.9827350300098127e-05, 'subsample': 0.5287966582381816, 'colsample_bytree': 0.8395279165759366, 'gamma': 1.4013629962131966e-07, 'reg_alpha': 0.718045486089428, 'reg_lambda': 0.016341631050124455}. Best is trial 91 with value: 0.7489799542492495.
Best parameters found:  {'learning_rate': 0.24983278165321376, 'n_estimators': 174, 'max_depth': 4, 'min_child_weight': 0.006505538523243821, 'subsample': 0.5916242355883965, 'colsample_bytree': 0.7840800801351581, 'gamma': 1.5183362947922708e-08, 'reg_alpha': 0.09517805958438799, 'reg_lambda': 0.0248083493094398}
R2:  0.7489799542492495
'''
