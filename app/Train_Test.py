from sklearn.model_selection import train_test_split


def train_test(df, random_state):
    X = df.drop('Recommended IND', axis=1)
    y = df['Recommended IND']

    # split the dataset into an 80% training and 20% test set
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=random_state,
                                                        shuffle=True)
    print("train test checkpoint")
    return X_train, X_test, y_train, y_test
