'''
실제 import되어 사용되는 코드가 아니라 데이터를 분석하기 위해 만든 파일입니다

시각적으로 화려하게 하라고 하셨으니 plt로 분석자료를 많이 띄워야합니다
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def congestion_file_analyze():
    '''
    혼잡도 파일 분석 함수입니다
    congestion : 혼잡도
    '''
    congestion_csv = pd.read_csv("./data/서울교통공사_지하철혼잡도정보_20221231.csv",index_col=0,encoding="EUC-KR")
    print(congestion_csv.info)
    
    ''' 각 호선별 최대 혼잡도 계산 '''
    max_congestion = {}
    for i in range(congestion_csv.shape[0]): # 호선별 최대치 생성
        data = congestion_csv.iloc[i]
        name = int(data['호선'])
        real_data = data.loc['5시30분':'00시30분'].copy()
        if name in max_congestion:
            max_congestion[name] = max(max_congestion[name], real_data.max()) # 기존데이터와 신규 데이터중 큰쪽으로 대체
        else:
            max_congestion[name] = real_data.max()

    print("각 호선별 최대 혼잡도",max_congestion)
    # 각 호선별 최대 혼잡도 {1: 107.8, 2: 172.3, 3: 154.8, 4: 185.5, 5: 140.9, 6: 113.8, 7: 160.6, 8: 136.8}
    
    ''' 각 호선 별 최대 승객수 계산'''
    max_transfer = {}
    for key, value in max_congestion.items():
        if key <= 4:
            max_transfer[key] = round(160 * 10 * value)
        elif key == 8:
            max_transfer[key] = round(160 * 6 * value)
        else:
            max_transfer[key] = round(160 * 8 * value)
    print("각 호선별 최대 승객수",max_transfer)
    # 각 호선 별 최대 승객수 {1: 172480, 2: 275680, 3: 247680, 4: 296800, 5: 180352, 6: 145664, 7: 205568, 8: 131328}
            

if __name__ == '__main__':
    congestion_file_analyze()




def wether_file_analyze():
    '''
    날씨 파일 분석 함수
    '''
    wether_csv = pd.read_csv("./data/SURFACE_ASOS_108_HR_2023_2023_2024.csv", index_col=0, encoding="euc-kr")
    print(wether_csv.info())
    

wether_file_analyze()





def weather_file_analyze():
    '''
    날씨 파일 분석 함수
    '''
    # 원하는 컬럼들을 리스트로 지정합니다.
    desired_columns = ["일시", "기온(°C)", "강수량(mm)", "적설(cm)"]

    # 원하는 컬럼만 선택하여 파일을 읽어옵니다.
    weather_csv = pd.read_csv("./data/SURFACE_ASOS_108_HR_2023_2023_2024.csv",
                                usecols=desired_columns, 
                                parse_dates=['일시'],
                                encoding="euc-kr")
    
    # 2023-08-31까지의 데이터만 필터링합니다.
    weather_csv_filtered = weather_csv[weather_csv['일시'] <= '2023-08-31']
    
    
    '''온도 데이터 선택'''
    temperature_data = weather_csv['기온(°C)']
    
    
    '''읽어온 데이터 정보를 출력합니다.'''
    print(weather_csv_filtered.info())
    #  0   일시       5809 non-null   datetime64[ns]
    #  1   기온(°C)   5809 non-null   float64
    #  2   강수량(mm)  809 non-null    float64
    #  3   적설(cm)   76 non-null     float64
    
    '''데이터를 출력.'''
    print(weather_csv_filtered.head(3))
    #                  일시  기온(°C)  강수량(mm)  적설(cm)
    # 0 2023-01-01 00:00:00     0.9      NaN     NaN
    # 1 2023-01-01 01:00:00     1.5      NaN     NaN
    # 2 2023-01-01 02:00:00     1.5      NaN     NaN


    def outliers(data_out):
        quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])  # 퍼센트 지점
        print('1사분위 : ', quartile_1)
        print('q2 : ', q2)
        print('3사분위 : ', quartile_3)
        iqr = quartile_3 - quartile_1   # 이상치 찾는 인스턴스 정의
        # 최대값이 이상치라면 최대값최소값으로 구하는 이상치는 이상치를 구한다고 할수없다
        print('iqr : ', iqr)
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        # -10 의 1.5 범위만큼 과 50의 1.5 범위만큼을 이상치로 생각을 하고 배제
        # 4~10 까지는 안전빵이라고 정의

        # 조건문(인덱스 반환) 
        return np.where((data_out > upper_bound) | (data_out < lower_bound))


    # 이상치를 탐지할 데이터를 임의로 생성합니다.
    data = np.random.normal(loc=0, scale=1, size=1000)

    # 이상치를 탐지합니다.
    outlier_indices = outliers(data)

    # Seaborn의 박스플롯을 생성합니다.
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data)

    # 이상치의 위치를 표시합니다.
    plt.scatter(outlier_indices, data[outlier_indices], color='red', label='Outliers')

    # 그래프 제목과 축 라벨을 설정합니다.
    plt.title('Boxplot with Outliers')
    plt.xlabel('Data')
    plt.ylabel('Values')

    # 범례를 표시합니다.
    plt.legend()

    # 그래프를 출력합니다.
    plt.show()        
    


