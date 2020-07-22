def make_config_file(file_path, df_len, weight_path=None):
  run = wandb.init(project="covid_forecast", entity="covid")
  wandb_config = wandb.config
  train_number = df_len*.6
  validation_number = df_len *.9
  config_default={                 
    "model_name": "CustomTransformerDecoder",
    "model_type": "PyTorch",
    "model_params": {
        "seq_length":wandb_config["forecast_history"],
        "n_time_series":9,
        "output_seq_length":wandb_config["out_seq_length"],
        "n_layers_encoder": wandb_config["number_encoder_layers"],
        "use_mask": wandb_config["use_mask"]
    },
    "dataset_params":
    {  "class": "default",
       "training_path": file_path,
       "validation_path": file_path,
       "test_path": file_path,
       "forecast_test_len":14,
       "batch_size":wandb_config["batch_size"],
       "forecast_history":wandb_config["forecast_history"],
       "forecast_length":wandb_config["out_seq_length"],
       "train_end": int(train_number),
       "valid_start":int(train_number+1),
       "valid_end": int(validation_number),
       "test_start":int(train_number+1),
       "target_col": ["diff_rolling"],
       "relevant_cols": ["diff_rolling", "month", "weekday", "mobility_retail_recreation",	"mobility_grocery_pharmacy",	"mobility_parks",	"mobility_transit_stations",	"mobility_workplaces",	"mobility_residential"],
       "scaler": "StandardScaler", 
       "interpolate": False
    },
    "training_params":
    {
       "criterion":"MSE",
       "optimizer": "Adam",
       "optim_params":
       {

       },
       "lr": wandb_config["lr"],
       "epochs": 10,
       "batch_size":wandb_config["batch_size"]
    
    },
    "GCS": True,
    "early_stopping":
    {
        "patience":3
    },
    "sweep":True,
    "wandb":False,
    "forward_params":{},
   "metrics":["MSE"],
   "inference_params":
   {     
         "datetime_start":"2020-06-19",
          "hours_to_forecast":16, 
          "num_prediction_samples": 100,
          "test_csv_path":file_path,
          "decoder_params":{
              "decoder_function": "simple_decode", 
            "unsqueeze_dim": 1
          },
          "dataset_params":{
             "file_path": file_path,
             "forecast_history":wandb_config["forecast_history"],
             "forecast_length":wandb_config["out_seq_length"],
             "relevant_cols": ["diff_rolling", "month", "weekday", "mobility_retail_recreation",	"mobility_grocery_pharmacy",	"mobility_parks",	"mobility_transit_stations",	"mobility_workplaces",	"mobility_residential"],
             "target_col": ["diff_rolling"],
             "scaling": "StandardScaler",
             "interpolate_param": False
          }
      }, 
      "weight_path_add":{
      "excluded_layers":["out_length_lay.weight", "out_length_lay.bias", "dense_shape.weight", "dense_shape.bias"]
      }
  }
  if weight_path: 
    config_default["weight_path"] = weight_path
  wandb.config.update(config_default)
  return config_default

wandb_sweep_config_full = {
  "name": "Default sweep",
  "method": "grid",
  "parameters": {
        "batch_size": {
            "values": [10, 25, 30]
        },
        "lr":{
            "values":[0.001, 0.0001, .01, .1]
        },
        "forecast_history":{
            "values":[5, 10, 11, 15]
        },
        "out_seq_length":{
            "values":[1, 2, 5]
        },
        "number_encoder_layers":
        {
            "values":[1, 2, 4, 5, 6]
        },
        "use_mask":{
            "values":[True, False]
        }
    }
}
