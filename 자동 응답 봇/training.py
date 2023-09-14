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
words = []
classes = []
documents = []

df = dfList[0]

for i in range(len(df['발화문'])) :
    try :
        Text = df['발화문'][i]
        cartegory = df['인텐트'][i]
        word_list = komoran.pos(Text)
        w_d_list = []
        for word in word_list :
            w_d_list.append(word[0])
        words.extend(w_d_list)
        if cartegory not in classes :
            classes.append(cartegory)
        documents.append((w_d_list, cartegory))
    except Exception as e :
        print(e, "라는 이유로 오류가 났습니다.")
        print("해당 구문은 이와 같습니다.")
        print("-------------------------------------------------------------------------------")
        print(df['발화문'][i]) 
        print("-------------------------------------------------------------------------------")


pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)  # [0, 0, 0, 0, 0, 0, 0]

for document in documents:
  bag = []
  word_patterns = document[0]
  for word in words:
    if word in word_patterns:
      bag.append(1)
    else:
      bag.append(0)

  output_row = list(output_empty)
  output_row[classes.index(document[1])] = 1  # 원핫 벡터  [0, 0, 0, 0, 1, 0, 0]
  training.append([bag, output_row])



random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]), ), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', history)
print("Done")


