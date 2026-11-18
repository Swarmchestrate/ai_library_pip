
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

preds_dict_itransformer = infer()

update_config(None, {
    "pipeline_type": "gmlp"
})
train()
preds_dict_gmlp = infer()

update_config(None, {
    "pipeline_type": "xgb"
})
train()
preds_dict_xgb = infer()

print(preds_dict_itransformer)

from ai_library import add_to_cron,remove_from_cron

add_to_cron() #add train to cron with retrain_frequency frequency
remove_from_cron() #remove train from crontab
print("done")