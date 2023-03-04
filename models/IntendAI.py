import pandas as pd
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate
import os
import json

current_location = os.getcwd() 

train_file = "C:/Users/dowon/data/01.데이터/1.Training/라벨링데이터"
file_list = ["TL_01. KAKAO(1)", "TL_01. KAKAO(2)", "TL_01. KAKAO(3)", "TL_01. KAKAO(4)"] 
df_list = []

print('파일을 불러옵니다.')
print("---------------------------------------------------------------------")
for f_n in file_list :
    file_lst = os.listdir(train_file+'/' +f_n)
    # 현재 디렉토리내에 모든 파일 출력
    os.chdir(train_file+'/' +f_n)
    print(f_n + ' 파일 ' + str(len(file_lst)) + '개' + ' 불러오는 중...')
    for file in file_lst :
        try :
            f = open(file, encoding='utf-8')
            raw_data = json.loads(f.read())
            #print(raw_data)
            df = pd.DataFrame(raw_data['info'][0]['annotations']['lines'])
            df_list.append(df)
        except Exception as e : 
            print("오류가 발생했습니다. 오류 메세지 :", e, ", 오류 파일명 :", file)
            pass
    print(f_n + ' 파일 ' + str(len(file_lst)) + '개' + ' 완료')
print("---------------------------------------------------------------------")



queries, intents = [], []
for df in df_list :
    for query in df['norm_text'].tolist() :
        queries.append(query)

    for intent in df['speechAct'].tolist() :
        intent = intent[1:3]    
        if intent == '단언' :
            intents.append(1)
        elif intent == '지시' :
            intents.append(2)
        elif intent == '언약' :
            intents.append(3)
        elif intent == '표현' :
            intents.append(4)
        elif intent == '채팅' :
            intents.append(5)
        else : # etc
            intents.append(0)
        

os.chdir(current_location)

print('\n\n데이터 처리를 시작합니다.')
print("---------------------------------------------------------------------")
from utils.preprocess import Preprocess
p = Preprocess(word2index_dic='./chatbot_dict.bin', userdic='./models/utils/user_dic.tsv')

# 단어 시퀀스 생성
sequences = []
for sentence in queries :
    pos = p.pos(sentence)
    keywords = p.get_keywords(pos, without_tag=True)
    seq = p.get_wordidx_sequence(keywords)
    sequences.append(seq)
print(sequences)
print("---------------------------------------------------------------------")


print('\n\n 모델 관련 정보 처리를 시작합니다.')
print("---------------------------------------------------------------------")
# 단어 인덱스 시퀀스 벡터 생성
# 단어 시퀀스 벡터 크기 
MAX_SEQ_LEN = 15
padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen = MAX_SEQ_LEN, padding = 'post')

# 학습용, 검증용, 테스트용 데이터셋 생성
# 학습셋:검증셋:테스트셋 = 7:2:1
ds = tf.data.Dataset.from_tensor_slices((padded_seqs, intents))
ds = ds.shuffle(len(queries))

train_size = int(len(padded_seqs) * 0.7)
val_size = int(len(padded_seqs) * 0.2)
test_size = int(len(padded_seqs) * 0.1)

train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).take(val_size).batch(20)
test_ds = ds.skip(train_size + val_size).take(test_size).batch(20)

# 하이퍼파라미터 설정
dropout_prob = 0.5
EMB_SIZE = 128
EPOCH = 5
VOCAB__SIZE = len(p.word_index)

# CNN 모델 정의
input_layer = Input(shape=(MAX_SEQ_LEN,))
embedding_layer = Embedding(VOCAB__SIZE, EMB_SIZE, input_length=MAX_SEQ_LEN)(input_layer)
dropout_emb = Dropout(rate = dropout_prob)(embedding_layer)

conv1 = Conv1D(
    filters = 128,
    kernel_size = 3,
    padding = 'valid',
    activation=tf.nn.relu)(dropout_emb)
pool1 = GlobalMaxPool1D()(conv1)

conv2 = Conv1D(
    filters = 128,
    kernel_size = 4,
    padding = 'valid',
    activation=tf.nn.relu)(dropout_emb)
pool2 = GlobalMaxPool1D()(conv2)

conv3 = Conv1D(
    filters = 128,
    kernel_size = 5,
    padding = 'valid',
    activation=tf.nn.relu)(dropout_emb)
pool3 = GlobalMaxPool1D()(conv3)


# 3, 4, 5-gram 이후 합치기
concat = concatenate([pool1, pool2, pool3])

hidden = Dense(128, activation=tf.nn.relu)(concat)
dropout_hidden = Dropout(rate=dropout_prob)(hidden)
logits = Dense(6, name = 'logits')(dropout_hidden)
predctions = Dense(6, activation = tf.nn.softmax)(logits)
print("---------------------------------------------------------------------")


# 모델 생성
model = Model(inputs=input_layer, outputs=predctions)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print('\n\n모델 학습을 시작합니다.')
print("---------------------------------------------------------------------")
# 모델 학습
model.fit(train_ds, validation_data=val_ds, epochs=EPOCH, verbose=1)

# 모델 평가(테스트 데이터셋 이용)
loss, accuracy = model.evaluate(test_ds, verbose=1)
print('Accuracy:', (accuracy*100))
print('loss:', loss)

# 모델 저장
model.save('intent_model.h5')
print("---------------------------------------------------------------------")
