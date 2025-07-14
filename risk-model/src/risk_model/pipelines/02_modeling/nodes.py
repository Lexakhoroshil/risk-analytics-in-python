import numpy as np
import pandas as pd

from optbinning import BinningProcess
from optbinning import Scorecard
from typing import Dict, Tuple
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.evaluate import PredefinedHoldoutSplit
from sklearn.linear_model import LogisticRegression


import warnings
warnings.filterwarnings('ignore')


def feature_selection_logreg_predefined_test( df_dev_woe:pd.DataFrame, 
                        df_dev:pd.DataFrame, 
                        target:str,
                        X_test:pd.DataFrame, 
                        features_in: list,
                        seed:int,
                        selection_forward:bool,
                        selection_feature_metric:str,
                        n_jobs=-1
                        ):
    """
    feature selection with predefined test sample
    """
    test_index = PredefinedHoldoutSplit(X_test.index.to_list())
    y_dev = df_dev[target]
    np.bool = np.bool_ # old numpy
    lr = LogisticRegression(random_state=seed)
    sfs = SFS( estimator=lr,
                k_features=(1, df_dev_woe[features_in].shape[1]),
                forward=selection_forward, 
                floating=False, 
                scoring=selection_feature_metric,
                cv=test_index)
    sfs.fit(df_dev_woe[features_in], y_dev)
    return sfs


def get_binning_short_list(df_X_train:pd.DataFrame, 
            series_y_train:pd.Series, 
            sfs, 
            categorical_variables:list,
            manual_cols_to_bin_splits:dict,
            ):
    """
    binning with manual splits corrections
    """

    variable_names = list(sfs.k_feature_names_)
    categorical_variables_in = [x for x in categorical_variables if x in variable_names]
    binning_fit_params={}
    if len(manual_cols_to_bin_splits)>0:
        for feat in manual_cols_to_bin_splits.keys():
            binning_fit_params[feat] = {
                'user_splits': np.array(manual_cols_to_bin_splits[feat][0], dtype=object),
                'user_splits_fixed': manual_cols_to_bin_splits[feat][1]
            }
    else:
        pass
    
    bp_short = BinningProcess(variable_names=variable_names, 
                        categorical_variables=categorical_variables_in, 
                        binning_fit_params=binning_fit_params)
                        
    bp_short.fit(df_X_train[bp_short.variable_names], series_y_train)

    return bp_short    



def get_woe_binned_features_short_list(df:pd.DataFrame, bp_short, metric:str, metric_missing:str ):
    """
    transformation to binned
    """
    df_woe = bp_short.transform(df[bp_short.variable_names], metric=metric, metric_missing=metric_missing )
    return df_woe
 


def model_scorecard_logreg_sfs(X_train:pd.DataFrame,
                     y_train:pd.Series,
                     bp_short,
                     seed:int):
    """
    modeling
    """                     
    lr = LogisticRegression(random_state=seed)

    scorecard = Scorecard(binning_process=bp_short,
                      estimator=lr, 
                      scaling_method=None,
                      )

    scorecard.fit(X_train[list(bp_short.variable_names)], y_train)
    
    return lr, scorecard