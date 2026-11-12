
from ai_library import train,infer, show_config, update_config

update_config(None, {
    "gmlp.model_params.epochs": 500,          # Sets prediction_target to None properly
    "itransformer.model_params.epochs": 500,
    "xgb.model_params.n_estimators": 700,
    "xgb.model_params.max_depth": 10
})
update_config(None, {
    "pipeline_type": "itransformer"
})
show_config()
train()

predictions_it, first_predicted_it, last_predicted_it, data_frequency_it, conformal_q_it = infer()

update_config(None, {
    "pipeline_type": "gmlp"
})
train()
predictions_gmlp, first_predicted_gmlp, last_predicted_gmlp, data_frequency_gmlp, conformal_q_gmlp = infer()

update_config(None, {
    "pipeline_type": "xgb"
})
train()
predictions_xgb, first_predicted_xgb, last_predicted_xgb, data_frequency_xgb, conformal_q_xgb = infer()
print("confq")
print(conformal_q_xgb)

import pandas as pd

def build_combined_dataframe(preds_dict, conformal_q, first_pred, last_pred, model_suffix):
    """Converts a model's prediction dictionary into a DataFrame with timesteps and suffixes."""
    df = pd.DataFrame(preds_dict)
    
    # Generate timesteps as the row index (matching the number of horizon steps)
    df["timestamp"] = pd.date_range(start=first_pred, end=last_pred, periods=len(df))
    df.set_index("timestamp", inplace=True)
    
    # Add model suffix to metric prediction columns
    df = df.add_suffix(f"_{model_suffix}") 
    for col, q_val in zip(df.columns, conformal_q):
        df[f"conf_q_{col}"] = q_val
    return df

# 1. Build individual DataFrames for each model
df_it_combined = build_combined_dataframe(predictions_it, conformal_q_it, first_predicted_it, last_predicted_it, "itransformer")
df_gmlp_combined = build_combined_dataframe(predictions_gmlp, conformal_q_gmlp, first_predicted_gmlp, last_predicted_gmlp, "gmlp")
df_xgb_combined = build_combined_dataframe(predictions_xgb, conformal_q_xgb, first_predicted_xgb, last_predicted_xgb, "xgb")

# 2. Merge all models side-by-side on the timestamp index
final_df = pd.concat([df_it_combined, df_gmlp_combined, df_xgb_combined], axis=1)

# Reset index if you want timestamps back as a regular column
final_df.reset_index(inplace=True)

# 3. Export to Parquet
final_df.to_parquet("all_models_predictions_combined.parquet", index=False)
print(final_df.head())