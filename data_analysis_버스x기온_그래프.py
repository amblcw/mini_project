'''
실제 import되어 사용되는 코드가 아니라 데이터를 분석하기 위해 만든 파일입니다

시각적으로 화려하게 하라고 하셨으니 plt로 분석자료를 많이 띄워야합니다
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
            


def bus_user_file_analyze():
    '''
    1월~8월 일별 버스 이용인원 파일 분석 함수
    '''
    # 원하는 컬럼들을 리스트로 지정합니다.
    desired_columns = ["사용일자", "승차총승객수", "하차총승객수",]
                       

    # 원하는 컬럼만 선택하여 파일을 읽어옵니다.
    user_csv = pd.read_csv("./data/2023년1~8월 일별 버스 이용객수2.csv",
                                usecols=desired_columns, 
                                parse_dates=['사용일자'],
                                # encoding="euc-kr",
                                # index_col=0
                                )
    
    
    '''승차 하차 데이터 선택'''
    user_on_data = user_csv['승차총승객수']  # 
    user_off_data = user_csv['하차총승객수']  # 
    
    
    '''읽어온 데이터 정보를 출력합니다.'''
    print(user_csv.info())
    #  #   Column  Non-Null Count  Dtype
    # ---  ------  --------------  -----
    #  0   사용일자    242 non-null    datetime64[ns]
    #  1   승차총승객수  242 non-null    int64
    #  2   하차총승객수  242 non-null    int64    
    
    
    '''데이터를 출력.'''
    print(user_csv.head(3))
    #   사용일자   승차총승객수   하차총승객수
    # 0 2023-01-01  2310197  2257688
    # 1 2023-01-02  4413450  4319065
    # 2 2023-01-03  4625953  4527318    
    

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


    # 승차, 하차 데이터에 대한 이상치를 찾습니다.
    outliers_indices = outliers([user_on_data, user_off_data])
    print("이상치 인덱스:", outliers_indices)


    # 박스 플롯 그리기
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rc

    # 한글 폰트 경로 설정
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 사용하고자 하는 한글 폰트 경로로 변경해주세요
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
    
    # 마이너스 부호 깨짐 방지 설정
    plt.rcParams['axes.unicode_minus'] = False
    

    # 박스 플롯 대신 막대 그래프를 사용하여 버스 이용객 수를 시각화합니다.
    plt.figure(figsize=(10, 6))
    plt.bar(user_csv['사용일자'], user_csv['승차총승객수'], label='승차총승객수', color='skyblue', alpha=0.7)
    plt.bar(user_csv['사용일자'], user_csv['하차총승객수'], label='하차총승객수', color='orange', alpha=0.7)
    plt.xlabel('날짜')
    plt.ylabel('이용객 수')
    plt.title('일별 버스 이용객 수')
    plt.xticks(rotation=45, ha='right')  # x축 라벨 회전
    plt.legend()
    plt.tight_layout()
    plt.show()

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
    temperature_data = weather_csv_filtered['기온(°C)']  # 기온 데이터만 선택
    
    
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


    # 기온 데이터에 대한 이상치를 찾습니다.
    outliers_indices = outliers(temperature_data)
    print("이상치 인덱스:", outliers_indices)

    
    # 박스 플롯 그리기
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rc

    # 한글 폰트 경로 설정
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 사용하고자 하는 한글 폰트 경로로 변경해주세요
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
    
    # 마이너스 부호 깨짐 방지 설정
    plt.rcParams['axes.unicode_minus'] = False
    
    # 박스 플롯 대신 실선 그래프를 사용하여 기온 데이터를 시각화합니다.
    plt.figure(figsize=(10, 6))
    plt.plot(weather_csv_filtered['일시'], temperature_data, color='green', marker='o', linestyle='-')
    plt.xlabel('날짜')
    plt.ylabel('기온 (°C)')
    plt.title('기온 데이터')
    plt.xticks(rotation=45, ha='right')  # x축 라벨 회전
    plt.tight_layout()
    plt.show()


# 함수 호출
weather_file_analyze()


# 함수 호출
bus_user_file_analyze()





# if __name__ == '__main__':
#     # congestion_file_analyze()
#     user_file_analyze()
# user_file_analyze()
