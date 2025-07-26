# KPNSM
Official Code for "Kernel Predicting Neural Shadow Maps"

SIGGRAPH 2025 Conference Paper

## Training
```
python train.py [--config config/default.yaml]
```
By default it use config/default.yaml.

You would need to create your own dataset for training. Note that in current dataset format, 2 consecutive data samples formed a perturbed pair (1+2, 3+4,...).

## Testing
test_scene.py by default uses config/evaluate.yaml.
First copy the config used to training to config/evaluate.yaml, then add an additional option `ckpt` to it, pointing to the checkpoint file (.pth).
```
python test_scene.py [--config config/evaluate.yaml]
```

## Model Weights
We will upload checkpoints in a few days.

## Acknowledgement
The pytorch_msssim module is from https://github.com/VainF/pytorch-msssim, and the VGG loss implementation is from https://github.com/crowsonkb/vgg_loss, both under MIT licenses.
