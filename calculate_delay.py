'''
입력: 시작역 - 환승역 ... 환승역 - 도착역
1. 각 역간의 이동 문제로 쪼개기
2. 부분문제의 출발역에서 승차대기시간 구하기
2-1. 출발역 시점에서 지하철이 포화되어있는지 판단하기
2-2. 포화되었다면 얼마만큼 대기해야하는지 계산
3. 각 부분문제에서 노선 자체의 지연시간 가져오기
4. 각 시간들을 합하여 실질지연시간 구하기
'''