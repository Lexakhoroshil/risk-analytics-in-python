from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func = get_index,
                inputs = dict(df = "01_raw_development",  col_to_index ="params:cols.col_raw_id", index_name = "params:cols.col_final_id" ),
                outputs="02_intermediate_development",
                name="get_index_development",

            ),
            node(
                func = get_index,
                inputs = dict(df = "01_raw_validation",  col_to_index ="params:cols.col_raw_id", index_name = "params:cols.col_final_id" ),
                outputs="02_intermediate_validation",
                name="get_index_validation",

            ),
            node(
                func = get_extra_features,
                inputs = "02_intermediate_development",
                outputs="04_feature_development",
                name="get_extra_features_development",

            ),

            node(
                func = get_extra_features,
                inputs = "02_intermediate_validation",
                outputs="04_feature_validation",
                name="get_extra_features_validation",

            ),

            node(
                func = get_train_test,
                inputs = dict(df = "04_feature_development", target="params:cols.col_target", test_size="params:global_parameters.test_ratio", random_state="params:global_parameters.seed"),
                outputs=["05_model_input_X_train", "05_model_input_X_test", "05_model_input_y_train", "05_model_input_y_test"],
                name="get_train_test",

            ),

            node(
                func = get_prebinning,
                inputs = dict(df_X_train = "05_model_input_X_train", 
                              series_y_train="05_model_input_y_train", 
                              variable_names="params:cols.cols_variables", 
                              categorical_variables="params:cols.cols_categorial",
                              manual_cols_to_bin_splits="params:manual_binning.splits",
                             ),
                outputs="06_models_binning",
                name="get_prebinning",

            ),

            node(
                func = get_woe_binned_features,
                inputs = dict(df = "05_model_input_X_train", 
                              bp="06_models_binning", 
                              metric="params:binning_parameters.metric", 
                              metric_missing="params:binning_parameters.metric_missing",
                             ),
                outputs="05_model_input_X_train_woe",
                name="get_woe_binned_features_X_train",

            ),
            node(
                func = get_woe_binned_features,
                inputs = dict(df = "05_model_input_X_test", 
                              bp="06_models_binning", 
                              metric="params:binning_parameters.metric", 
                              metric_missing="params:binning_parameters.metric_missing",
                             ),
                outputs="05_model_input_X_test_woe",
                name="get_woe_binned_features_X_test",

            ),
            node(
                func = get_woe_binned_features,
                inputs = dict(df = "04_feature_development", 
                              bp="06_models_binning", 
                              metric="params:binning_parameters.metric", 
                              metric_missing="params:binning_parameters.metric_missing",
                             ),
                outputs="05_model_input_development_woe",
                name="get_woe_binned_features_development",

            ),

            node(
                func = get_woe_binned_features,
                inputs = dict(df = "04_feature_validation", 
                              bp="06_models_binning", 
                              metric="params:binning_parameters.metric", 
                              metric_missing="params:binning_parameters.metric_missing",
                             ),
                outputs="05_model_input_validation_woe",
                name="get_woe_binned_features_validation",

            ),            
        ]
    )

    
