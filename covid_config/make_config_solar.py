
def make_config_file(file_path, df_len):
  run = wandb.init(project="pretrained-solar-updated")
  wandb_config = wandb.config
  train_number = df_len * .7
  validation_number = df_len *.9
  config_default={                 
    "model_name": "MultiAttnHeadSimple",
    "model_type": "PyTorch",
    "model_params": {
      "number_time_series":3,
      "seq_len":wandb_config["forecast_history"], 
      "output_seq_len":wandb_config["out_seq_length"],
      "forecast_length":wandb_config["out_seq_length"]
     },
    "dataset_params":
    {  "class": "default",
       "training_path": file_path,
       "validation_path": file_path,
       "test_path": file_path,
       "batch_size":wandb_config["batch_size"],
       "forecast_history":wandb_config["forecast_history"],
       "forecast_length":wandb_config["out_seq_length"],
       "train_end": int(train_number),
       "valid_start":int(train_number+1),
       "valid_end": int(validation_number),
       "target_col": ["Power(MW)"],
       "relevant_cols": ["Power(MW)", "month", "weekday"],
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
    "GCS": False,
    
    "sweep":True,
    "wandb":False,
    "forward_params":{},
   "metrics":["MSE"],
   "inference_params":
   {     
         "datetime_start":"2006-08-22",
          "hours_to_forecast":150, 
          "test_csv_path":file_path,
          "decoder_params":{
              "decoder_function": "simple_decode", 
            "unsqueeze_dim": 1
          },
          "dataset_params":{
             "file_path": file_path,
             "forecast_history":wandb_config["forecast_history"],
             "forecast_length":wandb_config["out_seq_length"],
             "relevant_cols": ["Power(MW)", "month", "weekday"],
             "target_col": ["Power(MW)"],
             "scaling": "StandardScaler",
             "interpolate_param": False
          }
      }
  }
  wandb.config.update(config_default)
  return config_default

sweep_config = {
  "name": "Default sweep",
  "method": "grid",
  "parameters": {
        "batch_size": {
            "values": [2, 3, 4]
        },
        "lr":{
            "values":[0.001, 0.01]
        },
        "forecast_history":{
            "values":[1, 2, 3, 5]
        },
        "out_seq_length":{
            "values":[1, 2]
        }
    }
}
