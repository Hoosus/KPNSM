from unet.unet import UNet, PixelShuffleUNet

def create_model(config):
  filter_config = config["filter"]
  model_config = config["model"]
  model = PixelShuffleUNet if config.get("pixel_shuffle", False) == True else UNet
  in_channels = 8
  if not config["include_sm"]:
      in_channels -= 1
  if not config["include_de_ds"]:
      in_channels -= 2
  if config["dataset"]["configs"]["penumbra_width_choice"] == "zero":
      in_channels -= 1
  print("input channel numbers:", in_channels)
  return model(in_channels=in_channels, filter_size=filter_config["filter_size"], dilation=filter_config["dilation"], **model_config)