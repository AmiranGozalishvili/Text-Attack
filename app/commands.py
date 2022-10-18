# import nltk
# import pandas as pd
# import tensorflow as tf
# from tqdm.notebook import tqdm_notebook
#
# nltk.download('omw-1.4')
#

# # initiate tqdm for pandas.apply() functions
# tqdm_notebook.pandas()
#
# # expand notebook display options for dataframes
# pd.set_option('display.max_colwidth', 200)
# pd.options.display.max_columns = 999
# pd.options.display.max_rows = 300

#
# def get_model_maxlen(df, random_state):
#
#     # check value counts of prediction class
#     print(df['Recommended IND'].value_counts())
#
#
#
#     # shuffle the dataset rows
#     dataset = df.sample(n=1000, random_state=random_state)
#
#     # check value counts of prediction class
#     print(dataset['Recommended IND'].value_counts())
#
#     # """## Load Model (optional)"""
#     # reload a fresh Keras model from the saved model
#     model = tf.keras.models.load_model('data/model/LSTM_Raw_Dataset')
#
#     # retrieve the maxlen variable of the model
#     model_config = model.get_config()
#     maxlen = model_config['layers'][0]['config']['batch_input_shape'][1]
#     print(maxlen)
#
#     return maxlen