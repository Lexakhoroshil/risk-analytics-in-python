import numpy as np
import pandas as pd

from typing import Dict, Tuple
from optbinning import BinningProcess
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

def get_index(df:pd.DataFrame, col_to_index:str, index_name:str)->pd.DataFrame:
    """
    rebuild indexes
    """
    df_indexed = df.rename(columns={col_to_index:index_name}).set_index(index_name)
    return df_indexed



def _get_killer_feature(df:pd.DataFrame)->pd.DataFrame:
    """
    
    """
    df_ = df.copy(deep=True)
    for x in ['flag_bad_utilization', 'flag_bad_dlq', 'flag_bad_noopencreds']:
        df_[x+'_modified'] = x +'='+ df_[x].astype(str) + '&'
    df_['feat_killer'] = df_[[x for x in df_ if '_modified' in x]].sum(axis=1)
    return df_['feat_killer']


def get_extra_features(df:pd.DataFrame)->pd.DataFrame:
    """
    feature engineering
        Debt: DebtRatio * MonthlyIncome
        NumberOfTime30+DaysPastDueNotWorse: sum('NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTimes90DaysLate')
        avg_debt_per_credit: Debt / NumberOfOpenCreditLinesAndLoans where NumberOfOpenCreditLinesAndLoans > 0 else NULL
        feat_killer: 3 flags of bad clients (flag_bad_utilization, flag_bad_dlq, flag_bad_noopencreds)

    """
    df['Debt'] =df['DebtRatio']*df['MonthlyIncome'].fillna(1)
    df['NumberOfTime30+DaysPastDueNotWorse'] = df[['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTimes90DaysLate']].sum(axis=1)
    df['avg_debt_per_credit'] =  np.where( df['NumberOfOpenCreditLinesAndLoans']>0, df['Debt']/df['NumberOfOpenCreditLinesAndLoans'], np.nan)

    df['flag_bad_utilization'] = np.where(df['RevolvingUtilizationOfUnsecuredLines']>1, 1, 0)
    df['flag_bad_dlq'] = np.where(df['NumberOfTime30+DaysPastDueNotWorse']>1, 1, 0)
    df['flag_bad_noopencreds'] = np.where(df['NumberOfOpenCreditLinesAndLoans']== 0, 1, 0)
    df['feat_killer'] = _get_killer_feature(df)
    return df


def get_train_test(df: pd.DataFrame, target, test_size:float, random_state:int)->pd.DataFrame:
    """
    divide developmant sample on train/test
    """
    X_train, X_test, y_train, y_test = train_test_split(
    df[[x for x in df.columns if x!=target ]], df[target], test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def get_prebinning(df_X_train:pd.DataFrame, 
            series_y_train:pd.Series, 
            variable_names:list, 
            categorical_variables:list,
            manual_cols_to_bin_splits:dict,
            ):
    """
    binning with manual splits corrections
    """

    binning_fit_params={}
    if len(manual_cols_to_bin_splits)>0:
        for feat in manual_cols_to_bin_splits.keys():
            binning_fit_params[feat] = {
                'user_splits': np.array(manual_cols_to_bin_splits[feat][0], dtype=object),
                'user_splits_fixed': manual_cols_to_bin_splits[feat][1]
            }
    else:
        pass
    
    bp = BinningProcess(variable_names=variable_names, 
                        categorical_variables=categorical_variables, 
                        binning_fit_params=binning_fit_params)
                        
    bp.fit(df_X_train[variable_names], series_y_train)

    return bp

def get_woe_binned_features(df:pd.DataFrame, bp, metric:str, metric_missing:str ):
    """
    transformation to binned
    """
    df_woe = bp.transform(df[bp.variable_names], metric=metric, metric_missing=metric_missing )
    return df_woe
 
