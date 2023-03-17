import random
import csv
import pickle
import numpy as np
import os

import pandas as pd

import jpype
from konlpy.tag import Komoran

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


train_file = "C:/Users/dowon/data/소상공인 고객 주문 질의-응답 텍스트/Training/라벨링데이터_train"

file_lst = os.listdir(train_file)
# 현재 디렉토리내에 모든 파일 출력
os.chdir(train_file)
dfList=[]
for f_n in file_lst :
    print(f_n + ' 파일 ' + ' 불러오는 중...')
    try :
        dfList.append(pd.read_csv(f_n))
        
        #df = pd.DataFrame(raw_data['info'][0]['annotations']['lines'])
        #df_list.append(df)
    except Exception as e : 
        print("오류가 발생했습니다. 오류 메세지 :", e, ", 오류 파일명 :", f_n)
        pass
    
komoran = Komoran()

def pos(sentence):
    jpype.attachThreadToJVM()
    return komoran.pos(sentence)

words = []
classes = []
documents = []

for df in dfList :
    for L in df :
        print(L)
    #if df['발화자'] == 'c' :
        

