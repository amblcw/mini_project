import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing import make_passenger_csv, make_bus_csv, make_delay_csv

def passenger_bus_delay_corr():
    '''
    전철 인원, 버스 인원, 지연시간
    셋 데이터의 상관관계
    '''
    org = make_passenger_csv()
    
    cols = org.columns 
    
    new_passenger = pd.DataFrame()
    
    # temp1 = pd.Series(org[cols[0]]+org[cols[1]])
    # new_passenger['1호선'] = temp1
    
    temp1 = pd.Series(org[cols[0]])
    
    for i in range(10, len(cols)):
        temp1 += org[cols[i]]

    new_passenger['1호선'] = temp1
    
    
    print(cols)
    print(new_passenger)    
    
passenger_bus_delay_corr()    
    
    
    
    
    
"""
    bus_data = make_bus_csv()
    # 첫 번째와 두 번째 열을 삭제하기
    bus_data = bus_data.iloc[:, 2:]
    delay_data = make_delay_csv()
    
    print(passenger_data.head(3))    
    print('='*100)
    print(bus_data.head(3))    
    print('='*100)
    print(delay_data.head(3))    
    print('='*100)
    
    # 전철, 지연시간 합치기
    passenger_delay_data = pd.concat([passenger_data, delay_data], axis=1)
    
    # 버스, 지연시간 합치기
    bus_delay_data = pd.concat([bus_data, delay_data], axis=1)
    
    # 상관관계 계산
    passenger_matrix = passenger_delay_data.corr()
    bus_matrix = bus_delay_data.corr()
    
       
    # 상관관계 행렬 출력
    print("전철 상관관계 상위 5 행 :\n", passenger_matrix.head(5))    
    print("버스 상관관계 상위 5 행 :\n", bus_matrix.head(5))    
    
    return passenger_matrix, bus_matrix

def plot_correlation_heatmap(passenger_matrix, bus_matrix):
    from matplotlib import font_manager, rc
    
    # 한글 폰트 경로 설정
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 사용하고자 하는 한글 폰트 경로로 변경해주세요
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
    
    # 마이너스 부호 깨짐 방지 설정
    plt.rcParams['axes.unicode_minus'] = False
    
    
    plt.figure(figsize=(12, 6))

    # 전철 상관관계 히트맵
    plt.subplot(1, 2, 1)
    sns.heatmap(passenger_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap - Passenger & Delay')
    plt.xlabel('Variables')
    plt.ylabel('Variables')

    # 버스 상관관계 히트맵
    plt.subplot(1, 2, 2)
    sns.heatmap(bus_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap - Bus & Delay')
    plt.xlabel('Variables')
    plt.ylabel('Variables')

    plt.tight_layout()
    plt.show()

# 데이터 준비 및 상관관계 계산
passenger_matrix, bus_matrix = passenger_bus_delay_corr()


# 시각화
plot_correlation_heatmap(passenger_matrix, bus_matrix)
"""
