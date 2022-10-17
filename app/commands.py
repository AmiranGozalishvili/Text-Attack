import nltk
import pandas as pd
import tensorflow as tf
from tqdm.notebook import tqdm_notebook

nltk.download('omw-1.4')

from Train_Test import train_test
from data import load_data
from tokenization import tokenize
from lstm_network import lstm_net

model = lstm_net()

df = load_data()

maxlen, tokenizer, X_test_pad, X_train_pad, word_index = tokenize()

X_train, X_test, y_train, y_test = train_test(df)


def commands():
    # initiate tqdm for pandas.apply() functions
    tqdm_notebook.pandas()

    # expand notebook display options for dataframes
    pd.set_option('display.max_colwidth', 200)
    pd.options.display.max_columns = 999
    pd.options.display.max_rows = 300

    dataset = load_data()

    # check value counts of prediction class
    dataset['Recommended IND'].value_counts()

    # define a random seed for reproducible results
    random_state = 42

    # shuffle the dataset rows
    dataset = dataset.sample(frac=1,
                             random_state=random_state)

    # resize dataset
    dataset = dataset[0:1000]

    # check value counts of prediction class
    dataset['Recommended IND'].value_counts()

    # """## Load Model (optional)"""
    # reload a fresh Keras model from the saved model
    new_model = tf.keras.models.load_model('data/model/LSTM_Raw_Dataset')

    # retrieve the maxlen variable of the model
    model_config = new_model.get_config()
    maxlen = model_config['layers'][0]['config']['batch_input_shape'][1]
    print(maxlen)


    return
