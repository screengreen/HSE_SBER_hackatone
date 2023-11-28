import os
import pickle
import pandas as pd
import re
import numpy as np # new

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_DATA = os.path.join(ROOT_DIR, 'data')
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin  # new

class SequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(292, 256)
        self.fc2 = nn.Linear(256, 200)
        self.fc3 = nn.Linear(200, 128)
        self.fc4 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

class Amount_Sizing(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.settings = {}
    
    def fit(self, X, y=None):
        df = X.copy()
        for col in self.columns:
            df[col] = df[col].fillna(0)

            non_zero_values = df.loc[df[col] != 0, col]
            cuts = np.percentile(non_zero_values, q=np.arange(0, 100, 10))
            cuts[0], cuts[-1] = -np.inf, np.inf 
            print(cuts)
            self.settings[col] = cuts
        return self

    def transform(self, X, y=None):
        df = X.copy()
        for col, cuts in self.settings.items():
            df.loc[df[col] == 0, col] = '0_zero'
            df.loc[df[col] != '0_zero', col] = pd.cut(df.loc[df[col] != '0_zero', col], 
                                                        bins=cuts, 
                                                        labels=['1_small','2_medium','3_medium','4_medium', \
                                                        '5_medium', '6_large', '7xlarge', '8_xlarge', '9_xlarge']) #[1]
        return df

def def_pos(value):
    if value > 0: 
        return value
    else:
        return 0 

def def_neg(value):
    if value < 0: 
        return value
    else:
        return 0 

def top3_transactions(row):
  top_3 = row.nlargest(3).index.tolist()
  ans = []
  for top in top_3:
    ans.append(top.replace('mcc_', ''))
  return ans


def get_top3_mcc_features(X):
  cols = []
  for col in X.columns:
    if col.startswith('mcc'):
      cols.append(col)

  top_3 = X[cols].apply(top3_transactions, axis=1, result_type='expand')
  top_3.columns = ['top1', 'top2', 'top3']
  return top_3

def create_bins(df):

    df['money_received'] = df['amount'].apply(def_pos)
    df['money_spent']  = df['amount'].apply(def_neg)

    amountsize_cols = ['money_received', 'money_spent']
    amount_sizing = Amount_Sizing(amountsize_cols)

    amount_sizing.fit(df)
    df[['money_received', 'money_spent']]  = amount_sizing.transform(df)[['money_received', 'money_spent']] 

    df_received = df.groupby(['client_id', 'money_received']).count()[['amount']]


    transactions_unstacked = df_received.unstack('money_received').fillna(0)
    transactions_unstacked.columns = transactions_unstacked.columns.get_level_values(1)
    transactions_unstacked.columns = 'money_received_' + transactions_unstacked.columns.astype(str)
    transactions_unstacked.drop('money_received_0_zero', axis=1, inplace=True)

    df_spent = df.groupby(['client_id', 'money_spent']).count()[['amount']]

    transactions_unstacked_spent = df_spent.unstack('money_spent').fillna(0)
    transactions_unstacked_spent.columns = transactions_unstacked_spent.columns.get_level_values(1)
    transactions_unstacked_spent.columns = 'money_spent_' + transactions_unstacked_spent.columns.astype(str)

    transactions_unstacked_spent.drop('money_spent_0_zero', axis=1, inplace=True)

    transactions_bins = transactions_unstacked_spent.join(transactions_unstacked)
    return transactions_bins



def get_data_frames():
    ttr_mcc_codes = pd.read_csv(os.path.join(PATH_DATA, 'mcc_codes.csv'), sep=';', index_col='mcc_code')
    tr_types = pd.read_csv(os.path.join(PATH_DATA, 'trans_types.csv'), sep=';', index_col='trans_type')

    transactions = pd.read_csv(os.path.join(PATH_DATA, 'transactions.csv'), index_col='client_id')
    gender_train = pd.read_csv(os.path.join(PATH_DATA, 'train.csv'), index_col='client_id')
    gender_test = pd.read_csv(os.path.join(PATH_DATA, 'test.csv'), index_col='client_id')
    transactions_train = transactions.join(gender_train, how='inner')
    transactions_test = transactions.join(gender_test, how='inner')
    return transactions_test, ttr_mcc_codes, tr_types
    

def read_pkl(pkl_name='model.pkl'):
    with open(os.path.join(ROOT_DIR, pkl_name), 'rb') as f:
        obj = pickle.load(f)
    return obj
    

def features_creation(x):
    """
    This function takes a DataFrame 'x' as input and generates various features based on different columns.

    Parameters:
    - x (pd.DataFrame): Input DataFrame containing transaction data.

    Returns:
    pd.Series: Concatenated series of features including MCC (Merchant Category Code), transaction type,
               day of the week, hour of the day, and a binary indicator for night transactions.
    """

    # Extracting features based on 'mcc_code', 'trans_type', 'day', 'hour', and 'night'
    features = [
        pd.Series(x['mcc_code'].value_counts().add_prefix('mcc_')),
        pd.Series(x['trans_type'].value_counts().add_prefix('tr_')),
        pd.Series(x['day'].value_counts().add_prefix('day_')),
        pd.Series(x['hour'].value_counts().add_prefix('h_')),
        pd.Series(x['night'].value_counts().add_prefix('night_'))
    ]

    # Concatenating the features into a single series
    return pd.concat(features)

def collect_mcc_features(mcc_code_info, X, top3):
    X = X.join(top3)
    X['top1'] = X['top1'].astype('int')
    X['top2'] = X['top2'].astype('int')
    X['top3'] = X['top3'].astype('int')
    X = X.reset_index()
    X = X.merge(mcc_code_info, right_on='mcc_code', left_on='top1')
    X = X.merge(mcc_code_info, right_on='mcc_code', left_on='top2')
    X = X.merge(mcc_code_info, right_on='mcc_code', left_on='top3')
    X.drop(columns=['top1', 'top2', 'top3'], axis=1, inplace=True)
    X.drop(columns=['mcc_code', 'mcc_code_x', 'mcc_code_y'], axis=1, inplace=True)
    X = X.set_index('client_id')
    return X

def make_all_preprocessing(X, mcc_code_indo):
    X['day'] = X['trans_time'].str.split().apply(lambda x: int(x[0]) % 7)
    X['hour'] = X['trans_time'].apply(lambda x: re.search(' \d*', x).group(0)).astype(int)
    X['night'] = ~X['hour'].between(6, 22).astype(int)

    X = X.reset_index('client_id')
    data = X.groupby(X['client_id']).apply(features_creation).unstack(-1).fillna(0)

    df_bins = create_bins(X)
    data = data.join(df_bins)
    top3 = get_top3_mcc_features(data)
    data = collect_mcc_features(mcc_code_info, data, top3)
    return data

def make_scoring(model, X, cols, mode='ML'):
  for col in set(cols).difference(X.columns):
        X[col] = 0
  if mode == 'ML':
    X['probability'] = model.predict_proba(X[model.feature_names_])[:, 1]
  else:
    test_tensor = torch.from_numpy(X[cols].values).float()
    y_pred_tensor = model(test_tensor)
    y_pred = y_pred_tensor.detach().numpy()
    X['probability'] = y_pred
  return X['probability'].reset_index()

def write_df(df):
    df.to_csv('result.csv')

def add_proba(scoring_df, proba_df):
    scoring_df['pred_nn'] = proba_df['probability']
    return scoring_df

test_df, mcc_codes_df, tr_types_df = get_data_frames()
mcc_code_info = pd.read_csv(os.path.join(ROOT_DIR, 'mcc_codes_info.csv'))
scoring_df = make_all_preprocessing(test_df, mcc_code_info)

#reading NN model 
model_base = read_pkl('final_model.pkl')

# reading data frame with final columns
final_cols = read_pkl('final_cols.pkl')

init_cols = read_pkl('columns.pkl')

model_emb = SequentialModel()
model_emb.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'model_seqNN.pkl')))
model_emb.eval()

scored_df_to_next = make_scoring(model_emb, scoring_df, init_cols, 'DL')
prepare_df = add_proba(scoring_df, scored_df_to_next)
scored_df = make_scoring(model_base, prepare_df, final_cols)
write_df(scored_df)
