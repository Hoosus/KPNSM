
def check_config(runname, config):
  args = runname.split("_")

  assert args[0] == "run"
  
  assert "msm" in args or "sm" in args

  # check filter
  filter_config = config["filter"]
  if filter_config["filter_size"] == 1:
    assert filter_config["dilation"] == 0
    assert "filter1" in args
  elif filter_config["filter_size"] == 5:
    assert "filter5" in args
    assert filter_config["dilation"] in [2, 3, 4]
    assert f"dilation{filter_config["dilation"]}" in args
  else: # big kernel
    assert filter_config["filter_size"] in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 29, 31]
    assert filter_config["dilation"] == 0
    assert f"filter{filter_config["filter_size"]}" in args
  
  # check msm
  msm = config["dataset"]["configs"]["use_msm"]
  if msm == True:
    assert "msm" in args and "sm" not in args
  else:
    assert "sm" in args and "msm" not in args

  # check sm and d
  if config["include_sm"] == True:
    assert "nosm" not in args
  else:
    assert "nosm" in args
    assert "msm" not in args
  if config["include_de_ds"] == True:
    assert "nod" not in args
  else:
    assert "nod" in args
  
  # check loss
  loss_config = config["loss"]
  assert loss_config["type"] == "combine"
  assert  loss_config["configs"]["l1_weight"] == 1.0
  assert "l2_weight" not in loss_config["configs"]
  assert "gd_weight" not in loss_config["configs"]
  if "vgg_weight" in loss_config["configs"]:
    assert loss_config["configs"]["vgg_weight"] == 0.1 and "vgg0.1" in args
  else:
    assert "vgg0.1" not in args
  
  # check temporal loss
  temporal_loss_weight_dict = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
  if config["temporal_weight"] == 0.0:
    for w in temporal_loss_weight_dict:
      assert f"tl1vgg{w}" not in args and f"tl1{w}" not in args
  else:
    assert f"t{config["temporal_loss"]["type"]}{config["temporal_weight"] }" in args
    if config["temporal_loss"]["proj"] == True:
      assert "proj" in args
    else:
      assert False, "noproj no longer used"
      assert "noproj" in args

  # check shuffle
  if config["pixel_shuffle"] == False:
    assert "noshuffle" in args
  
  # check model
  model_config = config["model"]
  assert model_config["layer_num"] in [3, 4, 5]
  assert f"layer{model_config["layer_num"]}" in args
  assert model_config["up"] == "transconv" and model_config["down"] == "maxpool"
  assert model_config["less_channel"] == False and model_config["light_conv"] == False
  assert model_config["skip"] == "add"

  # check dataset
  assert config["dataset"]["configs"]["penumbra_clamp"] == 50
  assert config["dataset"]["configs"]["temporal_group_num"] == 2
  assert config["dataset"]["configs"]["cv_div_d"] == True
  if config["dataset"]["configs"]["ce_add_re"] == True:
    assert "ce+R" in args
  penumbra = config["dataset"]["configs"]["penumbra_width_choice"] 
  if penumbra == "w_sqrt":
    assert "wsqrt" in args
  elif penumbra == "w_sqrt_fixed":
    assert "wsqrtfixed" in args 
  elif penumbra == "R":
    assert "R" in args
  elif penumbra == "zero":
    assert config["dataset"]["configs"]["ce_add_re"] == True
  elif penumbra == "w":
    assert "w" in args
  elif penumbra == "ce+R":
    assert "wce+R" in args
  else:
    assert False
    
  
  # check training
  assert config["training"]["max_step"] == 100000000000
  if config["training"]["learning_rate"] != 0.001:
    assert "smalllr" in args
  assert config["training"]["train_batch_accum"] == 4
  if config["training"]["train_batch_size"] != 2:
    assert f"batch{config["training"]["train_batch_size"]}" in args
  
  # check val
  assert config["val"]["val_batch_size"] in [2, 3]
  assert config["val"]["val_frequency"] == 5000

  if "gradient_clip" in config["training"]:
    if config["training"]["gradient_clip"] > 0:
      assert f"clip{config["training"]["gradient_clip"]}" in args