from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func = get_scored_sample,
                inputs = dict( df = "04_feature_development",
                               X_train = "05_model_input_X_train",
                               X_test = "05_model_input_X_test",
                               scorecard = "06_models_scorecard",
                               metric="params:binning_parameters.metric", 
                               metric_missing="params:binning_parameters.metric_missing",
                ),
                outputs = "07_model_output_development_scored",
                name = "get_scored_sample_development"
            ),

            node(
                func = get_scored_sample,
                inputs = dict( df = "04_feature_validation",
                               X_train = "05_model_input_X_train",
                               X_test = "05_model_input_X_test",
                               scorecard = "06_models_scorecard",
                               metric="params:binning_parameters.metric", 
                               metric_missing="params:binning_parameters.metric_missing",
                ),
                outputs = "07_model_output_validation_scored",
                name = "get_scored_sample_validation"
            ),         

            node(
                func = get_result_to_kaggle,
                inputs = dict( df = "04_feature_validation",
                               scorecard = "06_models_scorecard"

                ),
                outputs = "07_model_output_sample_to_kaggle",
                name = "get_result_to_kaggle"
            ),  

            node(
                func = get_table_binning,
                inputs = dict( bp = "06_models_binning"

                ),
                outputs = "08_reporting_variables_binning_table",
                name = "get_table_binning_long_list",
                tags="info_table"
            ),  

            node(
                func = get_table_binning,
                inputs = dict( bp = "06_models_binning_short_list"

                ),
                outputs = "08_reporting_variables_binning_table_short_list",
                name = "get_table_binning_short_list",
                tags="info_table"
            ),  

            node(
                func = get_table_features_quality,
                inputs = dict( bp = "06_models_binning"

                ),
                outputs = "08_reporting_variables_summary_table",
                name = "get_table_features_quality_long_list",
                tags="info_table"
            ),  

            node(
                func = get_table_features_quality_short_list,
                inputs = dict( bp_short = "06_models_binning_short_list", 
                               df_dev_scored = "07_model_output_development_scored", 
                               target = "params:cols.col_target"

                ),
                outputs = "08_reporting_variables_summary_table_short_list",
                name = "get_table_features_quality_short_list",
                tags="info_table"
            ),  

            node(
                func = get_correlation_table_short_list,
                inputs = dict( X_train_woe = "05_model_input_X_train_woe_short_list", 

                ),
                outputs = ["08_reporting_model_feature_correlation", "08_reporting_model_feature_correlation_plot" ],
                name = "get_correlation_table_short_list",
                tags="info_plot",
            ),  

            node(
                func = get_table_model_scorecard,
                inputs = dict( model = "06_models_scorecard"

                ),
                outputs = "08_reporting_model_scorecard_table",
                name = "get_table_model_scorecard",
                tags="info_table"
            ),  
            node(
                func = get_table_model_scorecard_feature_coefs,
                inputs = dict( model = "06_models_scorecard"

                ),
                outputs = "08_reporting_model_scorecard_feature_coefs",
                name = "get_table_model_scorecard_feature_coefs",
                tags="info_table"
            ),  

            node(
                func = get_plot_feature_selection,
                inputs = dict( sfs = "06_models_feature_selection"

                ),
                outputs = "08_reporting_model_scorecard_feature_selection_plot",
                name = "get_plot_feature_selection",
                tags="info_plot"
            ),  

            node(
                func = get_statistics_monitoring,
                inputs = dict(  scorecard = "06_models_scorecard", 
                                psi_method = "params:monitoring_parameters.psi_method", 
                                psi_n_bins = "params:monitoring_parameters.psi_n_bins",
                                target = "params:cols.col_target",
                                X_actual = "05_model_input_X_test", 
                                y_actual = "05_model_input_y_test", 
                                X_expected = "05_model_input_X_train", 
                                y_expected = "05_model_input_y_train",
                                inplace_y_actual = "params:monitoring_parameters.inplace_y_actual"

                ),
                outputs = "08_reporting_statistics_monitoring_train_test",
                name = "get_statistics_monitoring_train_test"
            ),  

            node(
                func = get_statistics_monitoring,
                inputs = dict(  scorecard = "06_models_scorecard", 
                                psi_method = "params:monitoring_parameters.psi_method", 
                                psi_n_bins = "params:monitoring_parameters.psi_n_bins",
                                target = "params:cols.col_target",
                                X_actual = "04_feature_validation", 
                                y_actual = "params:monitoring_parameters.none_type", 
                                X_expected = "04_feature_development", 
                                y_expected = "params:monitoring_parameters.none_type",
                                inplace_y_actual = "params:monitoring_parameters.inplace_y_actual"

                ),
                outputs = "08_reporting_statistics_monitoring_dev_valid",
                name = "get_statistics_monitoring_dev_valid"
            ),  

            node(
                func = get_tables_psi_features_detailed,
                inputs = dict( scorecard_monitoring = "08_reporting_statistics_monitoring_train_test"
                ),
                outputs = "08_reporting_model_scorecard_features_psi_detailed_train_test",
                name = "get_tables_psi_features_detailed_train_test",
                tags="info_table"
            ),  

            node(
                func = get_tables_psi_features_detailed,
                inputs = dict( scorecard_monitoring = "08_reporting_statistics_monitoring_dev_valid"
                ),
                outputs = "08_reporting_model_scorecard_features_psi_detailed_dev_valid",
                name = "get_tables_psi_features_detailed_dev_valid",
                tags="info_table"
            ),  

            node(
                func = get_tables_psi_features_summary,
                inputs = dict( scorecard_monitoring = "08_reporting_statistics_monitoring_train_test"
                ),
                outputs = "08_reporting_model_scorecard_features_psi_summary_train_test",
                name = "get_tables_psi_features_summary_train_test",
                tags="info_table"
            ),  

            node(
                func = get_tables_psi_features_summary,
                inputs = dict( scorecard_monitoring = "08_reporting_statistics_monitoring_dev_valid"
                ),
                outputs = "08_reporting_model_scorecard_features_psi_summary_dev_valid",
                name = "get_tables_psi_features_summary_dev_valid",
                tags="info_table"
            ),  

            node(
                func = get_table_psi_scorecard,
                inputs = dict( scorecard_monitoring = "08_reporting_statistics_monitoring_train_test"
                ),
                outputs = "08_reporting_model_scorecard_psi_summary_train_test",
                name = "get_table_psi_scorecard_train_test",
                tags="info_table"
            ),  

            node(
                func = get_table_psi_scorecard,
                inputs = dict( scorecard_monitoring = "08_reporting_statistics_monitoring_dev_valid"
                ),
                outputs = "08_reporting_model_scorecard_psi_summary_dev_valid",
                name = "get_table_psi_scorecard_dev_valid",
                tags="info_table"
            ),  
#

            node(
                func = get_plot_psi_scorecard,
                inputs = dict( scorecard_monitoring = "08_reporting_statistics_monitoring_train_test"
                ),
                outputs = "08_reporting_model_scorecard_psi_summary_train_test_plot",
                name = "get_plot_psi_scorecard_train_test",
                tags="info_plot"
            ),  

            node(
                func = get_plot_psi_scorecard,
                inputs = dict( scorecard_monitoring = "08_reporting_statistics_monitoring_dev_valid"
                ),
                outputs = "08_reporting_model_scorecard_psi_summary_dev_valid_plot",
                name = "get_plot_psi_scorecard_dev_valid",
                tags="info_plot"
            ),  

            node(
                func = get_table_statistical_tests,
                inputs = dict( scorecard_monitoring = "08_reporting_statistics_monitoring_train_test"
                ),
                outputs = "08_reporting_model_scorecard_statistical_tests",
                name = "get_table_statistical_tests",
                tags="info_table"
            ),  

            node(
                func = get_plot_roc_auc_train_test,
                inputs = dict(  scorecard = "06_models_scorecard",
                                X_train = "05_model_input_X_train", 
                                y_train = "05_model_input_y_train", 
                                X_test = "05_model_input_X_test", 
                                y_test = "05_model_input_y_test"
                ),
                outputs = "08_reporting_model_scorecard_roc_auc_train_test_plot",
                name = "get_plot_roc_auc_train_test",
                tags="info_plot"
            ),


            node(
                func = get_plot_ks_train_test,
                inputs = dict(  scorecard = "06_models_scorecard",
                                X_train = "05_model_input_X_train", 
                                y_train = "05_model_input_y_train", 
                                X_test = "05_model_input_X_test", 
                                y_test = "05_model_input_y_test"
                ),
                outputs = "08_reporting_model_scorecard_ks_plot",
                name = "get_plot_ks_train_test",
                tags="info_plot"
            ),

            node(
                func = get_plot_coefs_scorecard_model,
                inputs = dict(  scorecard_monitoring = "08_reporting_statistics_monitoring_train_test",
 
                ),
                outputs = "08_reporting_model_scorecard_coefs_plot" ,
                name = "get_plot_coefs_scorecard_model",
                tags="info_plot"
            ),

            node(
                func = get_general_statistics_report,
                inputs = dict(  scorecard_monitoring = "08_reporting_statistics_monitoring_train_test",
 
                ),
                outputs = None ,
                name = "get_general_statistics_report"
            ),

        ]
    )

