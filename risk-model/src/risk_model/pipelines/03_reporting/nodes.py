import numpy as np
import pandas as pd
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from optbinning import Scorecard
from optbinning.scorecard import ScorecardMonitoring
from optbinning.scorecard import plot_auc_roc, plot_ks
from optbinning import BinningProcess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.ticker as mtick
import plotly.graph_objects as go 



def get_scored_sample(df:pd.DataFrame, X_train, X_test, scorecard, metric:str, metric_missing:str):
    """
    scoring function (binning included)
    """
    df['type_sample'] = np.NaN
    df.loc[df.index.intersection(X_train.index), 'type_sample'] = 'train'
    df.loc[df.index.intersection(X_test.index), 'type_sample'] = 'test'

    df[[x + '_woe' for x in scorecard.binning_process_.variable_names]] =  scorecard.binning_process_.transform(df,  metric = metric,
                                                                                                                    metric_missing = metric_missing 
                                                                                                                    )
    df['prediction'] = scorecard.predict_proba(df)[:,1]

    return df

def get_result_to_kaggle(df:pd.DataFrame, scorecard):
    """
    result: upload to Kaggle
    """
    df['Probability'] = scorecard.predict_proba(df)[:,1]

    return df.reset_index()[['Id', 'Probability']]



def get_table_binning(bp):
    """get binning long_list
    """
    t=[]
    for feature in list(bp.variable_names):
        t.append(bp.get_binned_variable(feature).binning_table.build(show_digits=4))
    table=pd.concat(t , keys=list(bp.variable_names))
    
    return table


def get_table_features_quality(bp):
    """get features quality (IV, Gini)
    """
    table_info_from_binning = bp.summary()
    return table_info_from_binning


def get_table_features_quality_short_list(bp_short, df_dev_scored, target):
    """get features quality (IV, Gini)
    """
    table_info_from_binning = bp_short.summary()

    table_gini_train_test = pd.DataFrame([])
    for type_sampe in ['train', 'test']:
        table_info_gini = {}

        for feature_woe in [ x for x in df_dev_scored.columns if 'woe' in x]:
            table_info_gini[feature_woe] = (2*roc_auc_score(df_dev_scored[df_dev_scored.type_sample == type_sampe][target], 
                                                            df_dev_scored[df_dev_scored.type_sample == type_sampe][feature_woe]) - 1) * (-1)

        table_gini_train_test = pd.concat([table_gini_train_test, pd.DataFrame.from_dict(table_info_gini, orient='index', columns=['gini_{}'.format(type_sampe)])], axis=1)
    table_gini_train_test = table_gini_train_test.reset_index().rename(columns = {'index':'name'})
    table_gini_train_test['name'] = table_gini_train_test['name'].str.replace('_woe','', regex=True)

    table = table_info_from_binning.merge(table_gini_train_test, how='left', on = 'name')
    return table

def get_correlation_table_short_list(X_train_woe):
    """get corr table
    """
    table_corr = X_train_woe.corr().round(2)
    fig = plt.figure(figsize=(15, 10)) 
    fig = sns.heatmap(table_corr, annot=True).get_figure()
    
    return table_corr, fig


def get_table_model_scorecard(model):
    """
    get model scorecard info
    """
    table = model.table(style='detailed')
    return table

def get_table_model_scorecard_feature_coefs(model):
    """
    get model scorecard info
    """
    weights = {}
    weights['const'] = np.round(model.estimator_.intercept_[0],4)
    weights['features'] = dict(zip(model.estimator_.feature_names_in_, np.round(model.estimator_.coef_[0],4)))
    return weights    


def get_plot_feature_selection(sfs):
    """
    plot quality depending on the features
    """
    plot = plot_sfs(sfs.get_metric_dict(),figsize=(10, 8))[0]
    return plot


class ScorecardMonitoringMod(ScorecardMonitoring):
    def psi_plot(self, savefig=None):
        """Plot Population Stability Index (PSI).

        Parameters
        ----------
        return plot
        """
        self._check_is_fitted()

        fig, ax1 = plt.subplots()

        n_bins = len(self._n_records_a)
        indices = np.arange(n_bins)
        width = np.min(np.diff(indices))/3

        p_records_a = self._n_records_a / self._n_records_a.sum() * 100.0
        p_records_e = self._n_records_e / self._n_records_e.sum() * 100.0

        p1 = ax1.bar(indices-width, p_records_a, width, color='tab:red',
                     label="Records Actual", alpha=0.75)
        p2 = ax1.bar(indices, p_records_e, width, color='tab:blue',
                     label="Records Expected", alpha=0.75)

        handles = [p1[0], p2[0]]
        labels = ['Actual', 'Expected']

        ax1.set_xlabel("Bin ID", fontsize=12)
        ax1.set_ylabel("Population distribution", fontsize=13)
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter())

        ax2 = ax1.twinx()

        if self._target_dtype == "binary":
            metric_label = "Event rate"
        elif self._target_dtype == "continuous":
            metric_label = "Mean"

        ax2.plot(indices, self._metric_a, linestyle="solid", marker="o",
                 color='tab:red')
        ax2.plot(indices, self._metric_e,  linestyle="solid", marker="o",
                 color='tab:blue')

        ax2.set_ylabel(metric_label, fontsize=13)
        ax2.xaxis.set_major_locator(mtick.MultipleLocator(1))

        ax2.set_xlim(-width * 2, n_bins - width * 2)

        plt.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=12)

        plt.tight_layout()

        if savefig is None:
            #plt.show()
            pass
        else:
            if not isinstance(savefig, str):
                raise TypeError("savefig must be a string path; got {}."
                                .format(savefig))
            plt.savefig(savefig)
            plt.close()
        return fig

def get_statistics_monitoring(scorecard, psi_method:str, psi_n_bins:int, target,
                                X_actual, y_actual, X_expected, y_expected, inplace_y_actual):
    """
    file with scorecard metrics and statics (Gini, KS, PSI, etc.) 
    inplace_y_actual : value
    """
    scorecard_monitoring = ScorecardMonitoringMod(scorecard=scorecard, psi_method=psi_method,
                                 psi_n_bins = psi_n_bins, verbose=False)
                          
    if  (type(y_actual) == str) and (target != None): 
        y_actual = X_actual[target].fillna(inplace_y_actual)
    else:
        pass
    if  (type(y_expected) == str) and (target != None): 
        y_expected = X_expected[target]
    else:
        pass    
    scorecard_monitoring.fit(X_actual, y_actual, X_expected, y_expected)                                 
    return scorecard_monitoring

def get_tables_psi_features_detailed(scorecard_monitoring):
    """
    get table detailed
    """
    return scorecard_monitoring.psi_variable_table(style="detailed")

def get_tables_psi_features_summary(scorecard_monitoring):
    """
    get table summary
    """
    return scorecard_monitoring.psi_variable_table()

def get_table_psi_scorecard(scorecard_monitoring):
    """
    get table psi scorecard
    """
    return scorecard_monitoring.psi_table()

def get_plot_psi_scorecard(scorecard_monitoring):
    """
    get plot psi_scorecard 
    """
    return scorecard_monitoring.psi_plot()

def get_table_statistical_tests(scorecard_monitoring):
    """ Null hypothesis: actual == expected
        Chi-square test - binary target
    """
    return scorecard_monitoring.tests_table()

def _calc_plot_auc_roc_mod(y, y_pred, title=None, xlabel=None, ylabel=None,
                    fname=None, **kwargs):
        """Plot Area Under the Receiver Operating Characteristic Curve (AUC ROC).

        """

        fpr, tpr, _ = roc_curve(y, y_pred)
        auc_roc = roc_auc_score(y, y_pred)

        # Define the plot settings
        if title is None:
            title = "ROC curve"
        if xlabel is None:
            xlabel = "False Positive Rate"
        if ylabel is None:
            ylabel = "True Positive Rate"

        plt.plot(fpr, fpr, linestyle="--", color="k", label="Random Model")
        plt.plot(fpr, tpr,  label="Model (AUC: {:.5f})".format(auc_roc))
        plt.title(title, fontdict={"fontsize": 14})
        plt.xlabel(xlabel, fontdict={"fontsize": 12})
        plt.ylabel(ylabel, fontdict={"fontsize": 12})
        plt.legend(loc='lower right')


        return plt.gcf()

def get_plot_roc_auc_train_test(scorecard, X_train, y_train, X_test, y_test):
    """ plot train test roc_auc plots
    """
    train_roc_auc = round(roc_auc_score(y_train, pd.Series(scorecard.predict_proba(X_train)[:, 1])),2)
    test_roc_auc = round(roc_auc_score(y_test, pd.Series(scorecard.predict_proba(X_test)[:, 1])),2)
    g1 = _calc_plot_auc_roc_mod(y_train, pd.Series(scorecard.predict_proba(X_train)[:, 1]))
    g1 = _calc_plot_auc_roc_mod(y_test, scorecard.predict_proba(X_test)[:, 1], title='ROC Train {} Test {}'.format(train_roc_auc, test_roc_auc))
    return g1


def get_plot_ks_train_test(scorecard, X_train, y_train, X_test, y_test ):
    plt.figure(figsize=(6.4, 4.8))
    plt.subplot(121)
    plot_ks(y_train, scorecard.predict_proba(X_train)[:, 1], title='KS Train')
    plt.subplot(122)
    plot_ks(y_test, scorecard.predict_proba(X_test)[:, 1], title='KS Test')
    #plt.suptitle('KS')
    g1 = plt.gcf()
    return g1


def get_plot_coefs_scorecard_model (scorecard_monitoring):
    """
    get plot of coefs scorecard
    """
    #plt.figure(figsize=(3.2, 2.4))
    reg_coef = pd.DataFrame((zip(scorecard_monitoring.scorecard.estimator_.feature_names_in_, scorecard_monitoring.scorecard.estimator_.coef_[0])), columns=['name', 'reg_coef'])
    g1 = sns.barplot(round(reg_coef,2), x="reg_coef", y="name", errorbar=None)
    g1.bar_label(g1.containers[0], fontsize=8)


    return g1.get_figure()

def get_general_statistics_report(scorecard_monitoring):
    scorecard_monitoring.system_stability_report()