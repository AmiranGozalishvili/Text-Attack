from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


# utilize the most frequently apprearing words in the corpus
# num_words = 10000

def init_tokenizer(num_words = 10000):

    # tokenize the training data
    tokenizer = Tokenizer(num_words=num_words,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n1234567890')
    return tokenizer


def tokenize(tokenizer, X_train, X_test):

    corpus = X_train['Review Text'].tolist() + X_test['Review Text'].tolist()
    tokenizer.fit_on_texts(corpus)

    # define the data word index
    word_index = tokenizer.word_index
    # print(word_index)

    # encode training/test data into sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train['Review Text'].tolist())
    X_test_seq = tokenizer.texts_to_sequences(X_test['Review Text'].tolist())

    # define the max number of words to consider in each review
    maxlen = max([len(x) for x in X_train_seq])
    print(f"Max sequence length: {maxlen}\n")

    # truncate and pad the training/test input sequences
    X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
    X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

    # output the resulting dimensions
    print("Padded shape (training):".ljust(25), X_train_pad.shape)
    print("Padded shape (test):".ljust(25), X_test_pad.shape)
    print("tokenizer checkpoint")

    return X_train_pad, X_test_pad, tokenizer, word_index, maxlen
