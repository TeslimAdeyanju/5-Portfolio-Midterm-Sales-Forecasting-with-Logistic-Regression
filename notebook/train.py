
# ## Deployment Model

import pickle
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


# 1. parameter defintion
C = 1.0
n_splits = 5
output_file = f'model_C={C}.bin'


# 2. reading the file
df = pd.read_csv('dataset.csv')


# 3. data preparation

# drop the columns that are not needed
df = df.drop(['zip', 'country'], axis=1)


# decompose the 'Order Date' column into year, month, and day
df['order_date'] = pd.to_datetime(df['order_date'])
df['order_year'] = df['order_date'].dt.year
df['order_month'] = df['order_date'].dt.month


# decompose the 'Ship Date' column into year, month, and day
df['ship_date'] = pd.to_datetime(df['ship_date'])
df['ship_year'] = df['ship_date'].dt.year
df['ship_month'] = df['ship_date'].dt.month


# channging the column context which are object to uniform content
object_columns_content = df.select_dtypes(include='object')
object_columns_content.head()

# transforming those columns to uniform content
def clean_column_content(df, column_name):
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].str.lower()
    df[column_name] = df[column_name].str.replace(' ', '_')
    return df

for col in object_columns_content:
    clean_categorical_data = clean_column_content(df, col)


# Create a LabelEncoder object
le = LabelEncoder()

# Fit and transform the product_name column
df['product_name_encoded'] = le.fit_transform(df['product_name'])


# droping the product_name after encoding
df = df.drop('product_name', axis=1)


# Setting the profit taget variable to categorical either profitable or not
df['profit'] = np.where(df['profit'] > 0, 'yes', 'no')


# converting the target column to binary
df.profit = (df.profit == 'yes').astype(int)



df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=8)


numerical_columns = [
            'quantity',
            'sales',
            'profit_margin',
            'order_year',
            'order_month',
            'ship_year',
            'ship_month']



categorical_columns = [
            'order_id',
            'customer',
            'manufactory',
            'product_name_encoded',
            'segment',
            'category',
            'subcategory',
            'region',
            'city',
            'state']


# Training
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical_columns + numerical_columns].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[categorical_columns + numerical_columns].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# 4 Evaluation


print(f'doing validation with C={C}')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

fold = 0

for train_idx, val_idx in kfold.split(df_train_full):
    df_train = df_train_full.iloc[train_idx]
    df_val = df_train_full.iloc[val_idx]

    y_train = df_train.profit.values
    y_val = df_val.profit.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'auc on fold {fold} is {auc}')
    fold = fold + 1

print('validation results:')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# 5 Training the final model

print('training the final model')

dv, model = train(df_train_full, df_train_full.profit.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.profit.values
auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')


# 6. Saving the model to pickle file

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')


import os
print("Current working directory:", os.getcwd())
