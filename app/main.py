import nltk
from fastapi import FastAPI

from Train_Test import train_test
from data import load_data
from lstm_network import lstm_net
from lstm_prediction import model_pred
from text_attack import create_attack
from tokenization import tokenize, init_tokenizer


nltk.download('omw-1.4')

RANDOM_STATE = 42

app = FastAPI()

df = load_data()

X_train, X_test, y_train, y_test = train_test(df, RANDOM_STATE)

tokenizer = init_tokenizer(num_words=10000)

X_train_pad, X_test_pad, tokenizer, word_index, maxlen = tokenize(tokenizer, X_train, X_test)

model = lstm_net(X_train_pad, X_test_pad, y_train, y_test, word_index, maxlen)

attack_result = create_attack(model, tokenizer, maxlen, df, RANDOM_STATE)

@app.get('/')
def get_root():
    return {'message': 'Welcome to Text Attack API'}


@app.get('/LSTM')
def lstm_prediction(text):
    '''
    LSTM model prediction
    '''

    return model_pred(text, model, tokenizer, maxlen)


@app.get('/text_attack')
def lstm_text_attack():

    return str(attack_result[0])



# Nltk, textattack download to docker
# attack result to json   !!!!!
# model training cache
#lstm net if model: load model   +++

#create attack cache

