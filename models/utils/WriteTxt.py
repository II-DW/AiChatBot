import pandas as pd
import os
import json

f_txt = open("./utils/말뭉치.txt", "w", encoding='utf8')

train_file = "C:/Users/dowon/data/01.데이터/1.Training/라벨링데이터"
file_list = ["TL_01. KAKAO(1)", "TL_01. KAKAO(2)", "TL_01. KAKAO(3)", "TL_01. KAKAO(4)"] 
df_list = []

for f_n in file_list :
    file_lst = os.listdir(train_file+'/' +f_n)
    # 현재 디렉토리내에 모든 파일 출력
    os.chdir(train_file+'/' +f_n)
    print(f_n + ' 파일 ' + str(len(file_lst)) + '개' + ' 불러오는 중...')
    for file in file_lst :
        try :
            f = open(file, encoding='utf-8')
            raw_data = json.loads(f.read())
            df = pd.DataFrame(raw_data['info'][0]['annotations']['lines'])
            df_list.append(df)
        except Exception as e : 
            print("오류가 발생했습니다. 오류 메세지 :", e, ", 오류 파일명 :", file)
            pass
    print(f_n + ' 파일 ' + str(len(file_lst)) + '개' + ' 완료')


textList, IntendList = [], []
for df in df_list :
    textList.append(df['morpheme'].tolist())
    IntendList.append(df['speechAct'].tolist())

for text in textList :
    for t in text :
        f_txt.write(t + '\n')
f_txt.close()
