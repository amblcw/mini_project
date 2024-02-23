import numpy as np
import pandas as pd

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
    
    # 일별 강수량 평균을 계산합니다.
    daily_rain_avg = weather_csv_filtered.groupby(weather_csv_filtered['일시'].dt.date)['강수량(mm)'].mean()
    
    return daily_rain_avg

# 함수를 호출하여 일별 강수량 평균을 얻습니다.
daily_avg_rainfall = weather_file_analyze()
print(daily_avg_rainfall)
  