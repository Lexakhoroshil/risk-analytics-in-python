from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func = feature_selection_logreg_predefined_test,
                inputs = dict(  df_dev_woe = "05_model_input_development_woe",
                                df_dev = "04_feature_development",
                                target = "params:cols.col_target",  
                                X_test ="05_model_input_X_test_woe", 
                                features_in = "params:cols.cols_variables" ,
                                seed="params:global_parameters.seed",
                                selection_forward="params:feature_selection_parameters.selection_forward",
                                selection_feature_metric ="params:feature_selection_parameters.selection_feature_metric"),
                outputs = "06_models_feature_selection",
                name = "feature_selection_logreg_predefined_test"
            ),

            node(
                func = get_binning_short_list,
                inputs = dict(df_X_train = "05_model_input_X_train", 
                              series_y_train = "05_model_input_y_train", 
                              sfs = "06_models_feature_selection", 
                              categorical_variables = "params:cols.cols_categorial",
                              manual_cols_to_bin_splits = "params:manual_binning.splits"),
                outputs = "06_models_binning_short_list",
                name = "get_binning_short_list"
            ),

            node(
                func = get_woe_binned_features_short_list,
                inputs = dict(df = "05_model_input_X_train", 
                              bp_short="06_models_binning_short_list", 
                              metric="params:binning_parameters.metric", 
                              metric_missing="params:binning_parameters.metric_missing",
                             ),
                outputs="05_model_input_X_train_woe_short_list",
                name="get_woe_binned_features_X_train_short_list",
            ),

            node(
                func = get_woe_binned_features_short_list,
                inputs = dict(df = "05_model_input_X_test", 
                              bp_short="06_models_binning_short_list", 
                              metric="params:binning_parameters.metric", 
                              metric_missing="params:binning_parameters.metric_missing",
                             ),
                outputs="05_model_input_X_test_woe_short_list",
                name="get_woe_binned_features_X_test_short_list",
            ),

            node(
                func = get_woe_binned_features_short_list,
                inputs = dict(df = "04_feature_development", 
                              bp_short="06_models_binning_short_list", 
                              metric="params:binning_parameters.metric", 
                              metric_missing="params:binning_parameters.metric_missing",
                             ),
                outputs="05_model_input_development_woe_short_list",
                name="get_woe_binned_features_development_short_list",
            ),

            node(
                func = get_woe_binned_features_short_list,
                inputs = dict(df = "04_feature_validation", 
                              bp_short="06_models_binning_short_list", 
                              metric="params:binning_parameters.metric", 
                              metric_missing="params:binning_parameters.metric_missing",
                             ),
                outputs="05_model_input_validation_woe_short_list",
                name="get_woe_binned_features_validation_short_list",
            ),


            node(
                func = model_scorecard_logreg_sfs,
                inputs = dict(  X_train = "05_model_input_X_train",
                                y_train = "05_model_input_y_train",
                                bp_short = "06_models_binning_short_list",
                                seed = "params:global_parameters.seed"),
                outputs = ["06_models_lr", "06_models_scorecard"],
                name = "model_scorecard_logreg_sfs"
            ),              
        ]
    )


