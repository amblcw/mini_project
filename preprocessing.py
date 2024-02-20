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
    
    각 역별 환승(버스에서 지하철로 추정) 이용객 수 데이터를 만드는 함수입니다(라벨인코딩 포함)
    각종 이용객 유형을 제거하고 단순 이용객 수로 합쳤습니다
    '''
    if os.path.exists('./data/trasfer_list.pkl'):           # 이미 존재하면 있는 파일 읽어서 반환
        return pd.read_pickle('./data/trasfer_list.pkl')
    
    row_transfer_csv = pd.read_csv("./data/서울교통공사_1_8호선 역별 일별 승객유형별 수송인원(환승유입인원 포함) 정보_20221231.csv",index_col=0,encoding="EUC-KR")
    
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

    subway_line_num = transfer_list['호선']
    subway_line_num = subway_line_num.str.replace("호선","").astype(int)    # 몇호선에서 숫자만 남김
    transfer_list['호선'] = subway_line_num
    transfer_list = transfer_list.set_index(keys='날짜')
    transfer_list.to_pickle('./data/trasfer_list.pkl')
    return transfer_list

def make_delay_csv():
    '''
    from: 서울교통공사_노선별 지연시간 정보_20230831.csv
    return: pd.Dataframe
    
    지하철 지연 시간 데이터를 만드는 함수입니다(라벨인코딩 포함)
    지연시간은 최악을 상정해야하기에 노선 방향중 제일 긴 지연시간을 선택합니다
    
    형식:
    날짜(index) | 지연시간대 | 1호선지연(분) | 2호선지연(분) | 3호선지연(분) ... 8호선지연(분)
    2023-01-01    09시~18시        0               5              0               0
    2023-01-02    첫차~09시        0               20             0               0
    '''
    # if os.path.exists('./data/delay_list.pkl'):           # 이미 존재하면 있는 파일 읽어서 반환
    #     return pd.read_pickle('./data/delay_list.pkl')
    
    row_delay_csv = pd.read_csv('./data/서울교통공사_노선별 지연시간 정보_20230831.csv',index_col=0,encoding="EUC-KR")
    
    subway_line_num = row_delay_csv['노선'].copy()
    subway_line_num = subway_line_num.str.replace("호선","")
    subway_line_num = subway_line_num.str.split()
    for idx, data in enumerate(subway_line_num):
        subway_line_num[idx+1] = int(data[0])
    row_delay_csv['노선'] = subway_line_num 

    # new_delay_csv = pd.DataFrame(columns=['지연시간대','1호선지연(분)','2호선지연(분)','3호선지연(분)','4호선지연(분)',
    #                                       '5호선지연(분)','6호선지연(분)','7호선지연(분)','8호선지연(분)'])
    
    date_list = np.unique(row_delay_csv['지연일자'])
    subway_line_list = np.unique(row_delay_csv['노선'])
    time_list = np.unique(row_delay_csv['지연시간대'])
    
    new_delay_csv = pd.DataFrame()
    for date in date_list:
        split_by_date = row_delay_csv[row_delay_csv["지연일자"] == date].copy()
        for subway_line in subway_line_list:
            split_by_line = split_by_date[split_by_date["노선"] == subway_line].copy()
            for time_num in time_list:
                temp_data = split_by_line[split_by_line["지연시간대"] == time_num].copy()
                if temp_data.shape[0] == 0:
                    continue
                delay_time = temp_data['최대지연시간'].max()
                delay_time = delay_time.split()[0]
                
                data = pd.DataFrame({'지연일자':[date],'지연시간대':[time_num],})
                for i in range(1,9):
                    if i == subway_line:
                        data[f'{i}호선지연(분)'] = int(delay_time[:-1])
                    else:
                        data[f'{i}호선지연(분)'] = 0
                  
                new_delay_csv = pd.concat([new_delay_csv,data])
    
    new_delay_csv = new_delay_csv.set_index(keys='지연일자')
    new_delay_csv.to_pickle('./data/delay_list.pkl')
    
    return new_delay_csv

def make_weather_csv():
    row_weather_csv = pd.read_csv('./data/SURFACE_ASOS_108_HR_2023_2023_2024.csv',index_col=1,encoding='EUC-KR')
    # print(row_weather_csv.columns)
    '''
    ['지점', '기온(°C)', '강수량(mm)', '풍속(m/s)', '풍향(16방위)', '습도(%)', '증기압(hPa)',
       '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)', '일조(hr)', '일사(MJ/m2)', '적설(cm)',
       '3시간신적설(cm)', '전운량(10분위)', '중하층운량(10분위)', '운형(운형약어)', '최저운고(100m )',
       '시정(10m)', '지면상태(지면상태코드)', '현상번호(국내식)', '지면온도(°C)', '5cm 지중온도(°C)',
       '10cm 지중온도(°C)', '20cm 지중온도(°C)', '30cm 지중온도(°C)']
    '''
    row_weather_csv = row_weather_csv.drop(['지점','풍속(m/s)', '풍향(16방위)', '습도(%)', '증기압(hPa)',
       '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)', '일조(hr)', '일사(MJ/m2)',
       '3시간신적설(cm)', '전운량(10분위)', '중하층운량(10분위)', '운형(운형약어)', '최저운고(100m )',
       '시정(10m)', '지면상태(지면상태코드)', '현상번호(국내식)', '지면온도(°C)', '5cm 지중온도(°C)',
       '10cm 지중온도(°C)', '20cm 지중온도(°C)', '30cm 지중온도(°C)'],axis=1)
    row_weather_csv = row_weather_csv.fillna(0)
    row_weather_csv = row_weather_csv.astype(float)
    return row_weather_csv

def scaling():
    pass

if __name__ == "__main__":
    # print(make_passenger_csv())
    # print(make_transfer_csv())
    # print(make_delay_csv())
    print(make_weather_csv())