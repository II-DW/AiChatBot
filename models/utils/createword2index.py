import pickle
from tensorflow.keras import preprocessing


def read_data(filename) :
    with open (filename, 'r', encoding='utf8') as f :
        data = [line.split('\n') for line in f.read().splitlines()]
    return data

d = read_data('./utils/말뭉치.txt')

dict = []
for c in d :
    data = c[0].split('/')
    for Data in data :
        word = ''
        for ch in Data :
            if ch.encode().isalpha() or ch == '+' :
                continue
            word += ch
        if word == '' :
             continue
        dict.append(word)

# 사전에 사용될 word2index 생성
# 사전의 첫 번째 인덱스에는 OOV 사용
tokenizer = preprocessing.text.Tokenizer(oov_token = 'OOV')
tokenizer.fit_on_texts(dict)
word_index = tokenizer.word_index

# 사전 파일 생성
f = open('chatbot_dict.bin', 'wb')
try :
    pickle.dump(word_index, f)
except Exception as e :
    print("오류가 발생했습니다 오류 메세지 : " + str(e))
finally :
    f.close()
print("word2index 사전 생성 완료`")