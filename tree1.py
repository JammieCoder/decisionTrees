import pandas as pd
from sklearn.model_selection import train_test_split


def setup() -> pd.DataFrame:
    df = pd.read_csv('wine-quality-white-and-red.csv')
    df.info()
    df = pd.get_dummies(df, columns=['type'])
    print(df.head())
    return df


def regression(data):
    # set the x-axis as all columns except the quality column, so we can predict the quality
    X = data.drop(['quality'], axis=1)
    y = data['quality']  # set the y-axis as the quality column
    return train_test_split(X, y, test_size=0.3, random_state=1234)


def main():
    df = setup()
    print(regression(df))


if __name__ == '__main__':
    main()
