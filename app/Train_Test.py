from sklearn.model_selection import train_test_split

from data import load_data

df = load_data()

# define a random seed for reproducible results
random_state = 42



def train_test(df):
    X = df.drop('Recommended IND', axis=1)
    y = df['Recommended IND']

    # split the dataset into an 80% training and 20% test set
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=random_state,
                                                        shuffle=True)
    print("train test checkpoint")
    return X_train, X_test, y_train, y_test

