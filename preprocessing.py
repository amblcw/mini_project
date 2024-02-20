import pandas as pd
import numpy as np

def make_passenger_csv():
    ''' 
    from: 2023년 1~8월 이용인원 파일
    return: pd.Dataframe
    
    지하철 이용 승하차 인원 데이터 프레임을 만드는 함수입니다(라벨 인코딩 포함)
    '''
    row_passenger_csv = pd.read_csv("./data/2023년 1~8월 이용인원.csv",index_col=0,encoding="UTF-8")
    subway_line_num = row_passenger_csv['호선']
    subway_line_num = subway_line_num.str.replace("호선","").astype(int)    # 몇호선에서 숫자만 남김
    
    upside_downside = row_passenger_csv['구분']
    upside_downside = upside_downside.replace("승차",0)
    upside_downside = upside_downside.replace("하차",1)     # 승차를 0으로 하차를 1로 라벨링
    
    passenger_csv = row_passenger_csv.drop('역명',axis=1)   # 이미 라벨링 된 역명이 있기에 역명은 제거
    passenger_csv['호선'] = subway_line_num
    passenger_csv['구분'] = upside_downside
    return passenger_csv

def label_encode():
    pass

def scaling():
    pass

if __name__ == "__main__":
    print(make_passenger_csv())