from fastapi import FastAPI

from commands import commands
from lstm_prediction import model_pred
from text_attack import create_attack

app = FastAPI()

runall = commands()

attack_result = create_attack()

@app.get('/')
def get_root():
    return {'message': 'Welcome to Text Attack API'}


@app.get('/LSTM')
def LSTM_prediction(text):
    '''
    LSTM model prediction
    '''
    print(attack_result)
    return model_pred(text)
