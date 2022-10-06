import pandas as pd
import os
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

def df_for_analysis(df,analis_column_word,analis_row_word):
    df['date'] = pd.to_datetime(df[analis_column_word])
    lst = list(df.index.values.tolist())
    list_index = set(lst)
    list_index.remove(analis_row_word)
    df.drop(list_index, inplace=True)
    df_for_ds = pd.DataFrame()
    df_for_ds['count car'] = df.groupby(pd.Grouper(key='date', freq='D')) ['date'].count()
    return df_for_ds

def remove_outlier(df_in, col_name):
       q1 = df_in[col_name].quantile(0.25)
       q3 = df_in[col_name].quantile(0.75)
       iqr = q3-q1
       fence_low  = q1-iqr
       fence_high = q3+iqr
       df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
       return df_out

def create_features(df, label=None):
    df['date'] = df.index
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    X = df[['dayofweek','quarter','month','year']]
    if label:
        y = df[label]
        return X, y
    return X

def model_predict (df, analysis_column_word, analysis_row_word):
    df_car = df_for_analysis(df, analysis_column_word, analysis_row_word)
    df_rem = remove_outlier(df_car, 'count car')
    x, Y = create_features(df_rem, 'count car')
    x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.25)
    model = xgb.XGBRegressor(n_estimators=1000)
    model.fit(x_train, Y_train,
            eval_set=[(x_train, Y_train), (x_test, Y_test)],
            early_stopping_rounds=50,
        verbose=False) 
    preds = model.predict(x_test)
    best_preds = np.asarray([np.argmax(line) for line in preds])
    print("Precision = {}".format( precision_score(Y_test, best_preds, average='micro', zero_division=0)))
    print("Recall = {}".format( recall_score(Y_test, best_preds, average='micro')))
    print("Accuracy = {}".format( accuracy_score(Y_test, best_preds)))
    return model


os.chdir('/home/oem/Dropbox/DataScienceTest/Reservation test') # директория содержащая *.csv



name_file_rakuten = 'rakuten_shinchitose.csv'
name_file_jalan = 'jalan_shinchitose.csv'

df_rakuten = pd.read_csv(name_file_rakuten, encoding= 'cp932', index_col=0, dayfirst=True)
df_jalan = pd.read_csv(name_file_jalan, encoding= 'cp932', index_col=1, dayfirst=True)

analis_column_word = '予約受付日時'
analis_row_word = '予約確認済'

jalan_analysis_column_word = '申込日'
jalan_analysis_row_word = '予約成立'

model_predict(df_rakuten, analis_column_word, analis_row_word)
model_predict(df_jalan, jalan_analysis_column_word, jalan_analysis_row_word)