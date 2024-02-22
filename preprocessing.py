import pandas as pd
import numpy as np
import pickle
import os.path
import datetime as dt

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
    !! 이미 파일이 존재하면 그냥 읽어서 반환하기에 변경사항 있으면 유의할 것
    '''
    if os.path.exists('./data/trasfer_list.pkl'):           # 이미 존재하면 있는 파일 읽어서 반환
        return pd.read_pickle('./data/trasfer_list.pkl')
    
    row_transfer_csv = pd.read_csv("./data/서울교통공사_1_8호선 역별 일별 승객유형별 수송인원(환승유입인원 포함) 정보_20221231.csv",index_col=0,encoding="EUC-KR")
    
    row_transfer_csv = row_transfer_csv[row_transfer_csv["역번호"] != 2754] # 해당 역들은 passenger_csv에 존재하지 않기에 제거
    row_transfer_csv = row_transfer_csv[row_transfer_csv["역번호"] != 2761]
    
    date_list = np.unique(row_transfer_csv['날짜'])
    subway_line_list = np.unique(row_transfer_csv['호선'])
    station_list = np.unique(row_transfer_csv['역번호'])
    
    transfer_list = pd.DataFrame()
    for date in date_list:
        split_by_date = row_transfer_csv[row_transfer_csv["날짜"] == date].copy()
        for subway_line in subway_line_list:
            split_by_line = split_by_date[split_by_date["호선"] == subway_line].copy()
            for station_num in station_list:
                temp_data = split_by_line[split_by_line["역번호"] == station_num].copy()    # 일련의 for문들은 같은 날짜,같은 호선, 같은 역번호에서 상행 하행등으로 나뉜것을 합치기 위함
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
    
    full_time_list = pd.date_range("2023-01-01 00:00","2023-08-31 23:00",freq='h')
    time_label = ['첫차~09시','09시~18시','18시~막차']
    label_trans = [
        ['05:00:00', '06:00:00', '07:00:00', '08:00:00'],
        ['09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00',
         '14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00'],
        ['19:00:00', '20:00:00', '21:00:00', '22:00:00', '23:00:00'],
        ['00:00:00', '01:00:00', '02:00:00', '03:00:00', '04:00:00']
    ]
    
    line_list = ['1호선지연(분)','2호선지연(분)','3호선지연(분)','4호선지연(분)',
                '5호선지연(분)','6호선지연(분)','7호선지연(분)','8호선지연(분)']
    new_delay_csv = pd.DataFrame(index=full_time_list, columns=line_list).fillna(0)
    
    date_list = np.unique(row_delay_csv['지연일자'])
    subway_line_list = np.unique(row_delay_csv['노선'])
    time_list = np.unique(row_delay_csv['지연시간대'])
    
    for date in date_list:
        split_by_date = row_delay_csv[row_delay_csv["지연일자"] == date].copy()
        for subway_line in subway_line_list:
            split_by_line = split_by_date[split_by_date["노선"] == subway_line].copy()
            # print(split_by_line)
            for idx, time_num in enumerate(time_label):    # 라벨만큼 즉 날짜당 3개를 만들고 거기에 지연 데이터 합산을 저장한다, 항상 덮어쓰기를할까..
                temp_data = split_by_line[split_by_line["지연시간대"] == time_num].copy()

                if len(temp_data) != 0: 
                    delay_time = temp_data['최대지연시간'].max()
                    delay_time = delay_time.split()[0]
                    delay_time = int(delay_time[:-1])
                    for time in label_trans[idx]:
                        row = date+" "+time
                        new_delay_csv.loc[row,line_list[subway_line-1]] = delay_time
                
    new_delay_csv.to_pickle('./data/test_delay_list.pkl')
    # new_delay_csv.to_csv('./data/test_delay_list.csv')
    
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
    row_weather_csv = row_weather_csv.loc[:'2023-08-31 23:00:00']
    return row_weather_csv

def make_bus_csv():
    row_bus_csv = pd.read_csv("./data/2023년1~8월 일별 버스 이용객수2.csv",index_col=0)
    # print(row_bus_csv.head)
    return row_bus_csv

def scaling():
    pass

def extend_delay_csv(dataset) -> pd.DataFrame: # 어느순간부터 입력이 안됨
    
    full_time_list = pd.date_range("2023-01-01 00:00","2023-08-31 23:00",freq='h')
    # day_time_list = pd.date_range("2023-01-01 00:00","2023-08-31 23:00")
    delay_idx_list = dataset.index
    # print(day_time_list)
    # print(dataset.head)
    time_label = ['첫차~09시','09시~18시','18시~막차']
    label_trans = [
        ['05:00:00', '06:00:00', '07:00:00', '08:00:00'],
        ['09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00',
         '14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00'],
        ['19:00:00', '20:00:00', '21:00:00', '22:00:00', '23:00:00'],
        ['00:00:00', '01:00:00', '02:00:00', '03:00:00', '04:00:00']
    ]
    
    new_delay_csv = pd.DataFrame(index=full_time_list, columns=dataset.columns[1:])
    # print(new_delay_csv.head)
    
    # 다 갈아엎기 | 날자 -> old_delay_csv로 순회하면서 old_delay_csv에 해당하는 날자가 없으면 0으로 채우고 있으면 old_delay_csv 날자가 바뀔때까지 진행하면서 데이터에 맞춰서 값을 채워넣기..
    # 아니다 이거 갈어엎지 말고 old_delay_Csv를 고치자
    dataset_idx = 0 
    is_delayed_day = False
    for day_time in full_time_list:
        day, time = str(day_time).split()
        delay_day = delay_idx_list[dataset_idx]
        print('day, time:', day, time)

        if day == delay_day:
            data = dataset.iloc[dataset_idx]
            print("data",data)
            delay_time = data['지연시간대']
            delay_time = label_trans[time_label.index(delay_time)]
            if time in delay_time:
                new_delay_csv.loc[day_time] = data.drop(['지연시간대'])
                if time == delay_time[-1]:
                    dataset_idx += 1
            else:
                # if is_delayed_day:
                #     is_delayed_day = False
                new_delay_csv.loc[day_time].fillna(0, inplace=True)
        else:
            new_delay_csv.loc[day_time].fillna(0, inplace=True)
            # if is_delayed_day and time == '23:00:00':
            #     dataset_idx += 1
            #     print("next day",dataset_idx)
                
        if dataset_idx >= 5:
            break
    
    return new_delay_csv

if __name__ == "__main__":
    passenger_csv = make_passenger_csv()
    # transfer_csv = make_transfer_csv()
    delay_csv = make_delay_csv()
    weather_csv = make_weather_csv()
    bus_csv = make_bus_csv()
    
    print(passenger_csv.shape,delay_csv.shape,weather_csv.shape, bus_csv.shape)
    # (132598, 25) (5832, 8) (5832, 3) (242, 3)

    
    '''
    같은 날자-시간 즉 시간단위로 인덱스를 잡고
    정류장의 경우 옆 컬럼으로서 늘려서 해결한다
    
    최종적으로 모두 weather 행 개수에 맞춰져야(단 00시~04시는 잘라내고)
    passenger   : 일단위에서 시간단위로 바꾸기, 정류장 행이 아니라 컬럼으로 바꾸기
    delay       : 반나절단위에서 시간단위로 바꾸기, 정류장 칼럼으로 바꾸기
    weather     : 00~04시 잘라내기
    bus         : 시간단위로 바꾸기
    '''
            
            
    # make_bus_csv()
    # print(len(weather_csv[weather_csv['강수량(mm)'] != 0.0]))   # 632
    # print(delay_csv[:10])
    # print(passenger_csv.head)
    delay_csv.to_csv('./data/old_delay.csv')
    # new_delay_csv = extend_delay_csv(delay_csv)
    # print(new_delay_csv[9:20])
    # new_delay_csv.to_csv('./data/new_delay.csv')