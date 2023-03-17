from TestIntend import IntentModel
from utils.preprocess import Preprocess


p = Preprocess(word2index_dic='./chatbot_dict.bin', userdic='./models/utils/user_dic.tsv')

intent = IntentModel(model_name='./models/intent_model.h5', preprocess=p)

query = input("발화 의도를 분류할 문장을 입력해주세요 : ")
predict = intent.predict_class(query)
predict_label = intent.labels[predict]

print(query)
print("의도 예측 클래스 :", predict)
print("의도 예측 레이블 :", predict_label)