import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def read_from_file(filename, f_type='csv'):
    if f_type == 'csv':
        data = pd.read_csv(filename)
        return data
    else:
        return None


def null_filter(data: pd.DataFrame, column):
    data = data[data[column].notnull()]
    return data


def len_filter(data: pd.DataFrame, column, _len):
    data['_len'] = data[column].apply(lambda x: len(x.split()))
    data = data[data['_len'] < _len]
    data.drop('_len')
    return data


def label_encoder(data: pd.DataFrame, text, label, test_size=0.1):
    data = data[[text, label]]
    data.reset_index(drop=True, inplace=True)
    data.rename(columns={text: 'text', label: 'label'}, inplace=True)

    data['label'] = LabelEncoder().fit_transform(data['label'])
    data['text'] = data['text'].apply(clean_txt)

    train = data.copy()
    train = train.reindex(np.random.permutation(train.index))

    train, test = train_test_split(train, test_size=test_size, random_state=35)

    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    return train, test


def clean_txt(text):
    # TODO: text cleansing operations
    return text
