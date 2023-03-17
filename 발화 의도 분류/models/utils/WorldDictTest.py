import pickle 
from preprocess import Preprocess

f = open("chatbot_dict.bin", "rb")
word_index = pickle.load(f)
f.close()

sent = "안녕 나는 사람이야"

# 전처리 객체 생성
p = Preprocess(userdic="./utils/user_dic.tsv")

# 형태소 분석기 실행 
pos = p.pos(sent)

# 품사 태그와 같이 키워드 출력 
keywords = p.get_keywords(pos, without_tag=True)
for word in keywords :
    try :
        print(word, word_index[word])
    except KeyError :
        print(word, word_index['OOV'])