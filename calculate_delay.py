from passenger_predict_torch_votingEnsemble import passenger_predict
import pandas as pd
import numpy as np
from datetime import datetime
'''
입력: 시작역 - 환승역 ... 환승역 - 도착역
1. 각 역간의 이동 문제로 쪼개기
2. 부분문제의 출발역에서 승차대기시간 구하기
2-1. 출발역 시점에서 지하철이 포화되어있는지 판단하기
2-2. 포화되었다면 얼마만큼 대기해야하는지 계산
3. 각 부분문제에서 노선 자체의 지연시간 가져오기
4. 각 시간들을 합하여 실질지연시간 구하기
'''
def is_max_at_station(departure_station:int,arrival_station:int,date:datetime=datetime.today())->bool:
    '''
    출발역과 도착역을 적으면 탑승시 열차가 가득차있는지 확인해줍니다
    '''
    if departure_station == arrival_station:
        raise Exception(f"departure and arrival is same num {departure_station}")
    line1 = np.arange(150,160)
    line2 = np.arange(201,251)
    line3 = np.arange(309,343)
    line4 = np.arange(409,435)
    line5 = np.arange(2511,2567)
    line6 = np.arange(2611,2649)
    line7 = np.arange(2711,2753)
    line8 = np.arange(2811,2829)
    line_list = [line1,line2,line3,line4,line5,line6,line7,line8]
    line_num = None
    for idx, line in enumerate(line_list):
        if departure_station in line:       # 출발역이 몇호선의 역인지 탐색
            if not(arrival_station in line):# 출발역과 도차역이 같은 호선이 아닌경우
                raise Exception(f"departure station{departure_station} and arrival station{arrival_station} is not same line")
            line_num = idx
            break
    else:   #출발역이 그 어떤 호선에도 존재하지 않는 역인 경우
        raise Exception(f"{departure_station}station is not exist")
    
    ascending = True  # 역번호가 증가하는 방향이면 True, 감소하는 방향이면 False
    target_line = line_list[line_num]
    departure_station_idx = int(np.where(line_list[line_num] == departure_station)[0])
    arrival_station_idx = int(np.where(line_list[line_num] == arrival_station)[0])
    if departure_station_idx > arrival_station_idx: # 하행
        ascending = False
    # print(departure_station_idx,arrival_station_idx)
    target_stations = (target_line[:departure_station_idx] if ascending else target_line[departure_station_idx:])
    # print(target_stations)
    
    date = str(date)[:14] + '00:00'
    print("date",date)
    full_time_list = list(pd.date_range("2023-01-01 00:00","2023-08-31 23:00",freq='h').astype(str))
    date_idx = full_time_list.index(date)
    print(date_idx, full_time_list[date_idx])
    
    passenger = 0
    for station in target_stations:
        passenger, _, __ = passenger_predict(station)
        passenger = int(passenger[date_idx])
        print("",passenger)
    
    
    ''' 
    1이면 낮은쪽부터, -1이면 큰쪽부터 가까운 역까지 승객 변동을 더하기 
    그리고 호선 별 편성당 수송인원과 비교해서 만석인지 아닌지 구하기    
    '''
    isMax = False
    
    return isMax

if __name__ == '__main__':
    # print(type(datetime.today()),datetime.today())
    result = is_max_at_station(156, 153, "2023-01-31 10:00:00")
    print(result)