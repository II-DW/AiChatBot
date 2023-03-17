import pandas as pd
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate
import os
import json

train_file = "C:/Users/dowon/data/01.데이터/1.Training/라벨링데이터"
file_list = ["TL_01. KAKAO(1)", "TL_01. KAKAO(2)", "TL_01. KAKAO(3)", "TL_01. KAKAO(4)"]
df_list = []

for f_n in file_list :
    file_lst = os.listdir(train_file+'/' +f_n)
    # 현재 디렉토리내에 모든 파일 출력
    print(file_lst)
    os.chdir(train_file+'/' +f_n)
    for file in file_lst :
        try :
            f = open(file, encoding='utf-8')
            raw_data = json.loads(f.read())
            #print(raw_data)
            df = pd.DataFrame(raw_data['info'][0])
            df_list.append(df)
            print(df)
        except : pass

