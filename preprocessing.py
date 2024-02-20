import pandas as pd
import numpy as np
import pickle
import os.path

def make_passenger_csv():
    ''' 
    from: 2023년 1~8월 이용인원.csv
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

def make_transfer_csv():
    '''
    from: 서울교통공사_1_8호선 역별 일별 승객유형별 수송인원(환승유입인원 포함) 정보_20221231.csv
    return: pd.Dataframe
    
    각 역별 환승(버스에서 지하철로 추정) 이용객 수 데이터를 만드는 함수입니다
    각종 이용객 유형을 제거하고 단순 이용객 수로 합쳤습니다
    '''
    if os.path.exists('./data/trasfer_list.pkl'):           # 이미 존재하면 있는 파일 읽어서 반환
        return pd.read_pickle('./data/trasfer_list.pkl')
    
    row_transfer_csv = pd.read_csv("./data/서울교통공사_1_8호선 역별 일별 승객유형별 수송인원(환승유입인원 포함) 정보_20221231.csv",index_col=0,encoding="EUC-KR")
    subway_line_num = row_transfer_csv['호선']
    subway_line_num = subway_line_num.str.replace("호선","").astype(int)    # 몇호선에서 숫자만 남김
    
    date_list = np.unique(row_transfer_csv['날짜'])
    subway_line_list = np.unique(row_transfer_csv['호선'])
    station_list = np.unique(row_transfer_csv['역번호'])
    
    transfer_list = pd.DataFrame()
    for date in date_list:
        split_by_date = row_transfer_csv[row_transfer_csv["날짜"] == date].copy()
        for subway_line in subway_line_list:
            split_by_line = split_by_date[split_by_date["호선"] == subway_line].copy()
            for station_num in station_list:
                temp_data = split_by_line[split_by_line["역번호"] == station_num].copy()
                if temp_data.shape[0] == 0:
                    continue
                data = pd.DataFrame({'날짜':[date],'호선':[subway_line],'역번호':[station_num],'환승유입인원':[temp_data['환승유입인원'].sum()]})
                transfer_list = pd.concat([transfer_list,data])

    transfer_list.to_pickle('./data/trasfer_list.pkl')
    return transfer_list

def label_encode():
    pass

def scaling():
    pass

if __name__ == "__main__":
    # print(make_passenger_csv())
    print(make_transfer_csv())