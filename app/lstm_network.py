from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.metrics import classification_report

from Train_Test import train_test
from data import load_data
from tokenization import tokenize

df = load_data()

maxlen, tokenizer, X_test_pad, X_train_pad, word_index = tokenize()

X_train, X_test, y_train, y_test = train_test(df)


def lstm_net():
    # initiate LSTM for sequence classification
    model = Sequential()

    # embed each numeric in a 50-dimensional vector
    model.add(Embedding(len(word_index) + 1,
                        50,
                        input_length=maxlen))

    # add bidirectional LSTM layer
    model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

    # add a classifier
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    batch_size = 512
    num_epochs = 1

    # train the model
    model.fit(X_train_pad, y_train,
              epochs=num_epochs,
              batch_size=batch_size)

    """## Evaluation"""

    # evaluate model on the test set
    model.evaluate(X_test_pad, y_test)
    y_test_pred = (model.predict(X_test_pad) >= 0.5).astype("int32")

    print("model save")
    """ Save Model (optional)"""
    # save the entire model
    model.save('data/model/LSTM_Raw_Dataset')

    print("lstm_network, classification_report")
    print(classification_report(y_test, y_test_pred))
    print("lstm Network checkpoint")
    return model
