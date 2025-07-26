import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms

class L1Loss:
    def __init__(self):
        pass

    def apply(self, x, gt, mask=None):
        if mask is None:
            return torch.mean(torch.abs(x - gt))
        else:
            return torch.mean(torch.abs(x - gt) * mask)

class L2Loss:
    def __init__(self):
        pass

    def apply(self, x, gt, mask=None):
        if mask is None:
            return torch.mean((x - gt) ** 2)
        else:
            return torch.mean(((x - gt) ** 2) * mask)

class GDLoss:
    def __init__(self):
        pass

    def apply(self, x, gt, mask=None):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if gt.dim() == 3:
            gt = gt.unsqueeze(0)

        # dx_pred = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        # dy_pred = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        dx_pred = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy_pred = x[:, :, 1:, :] - x[:, :, :-1, :]

        # dx_gt = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
        # dy_gt = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])
        dx_gt = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        dy_gt = gt[:, :, 1:, :] - gt[:, :, :-1, :]

        dx_loss = torch.abs(dx_pred - dx_gt)
        dy_loss = torch.abs(dy_pred - dy_gt)

        loss = torch.mean(dx_loss) + torch.mean(dy_loss)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(0) 
            loss = torch.mean(dx_loss * mask[:, :, :, :-1]) + torch.mean(dy_loss * mask[:, :, :-1, :])
        return loss
   
# borrowed from https://github.com/crowsonkb/vgg_loss/blob/master/vgg_loss.py
class VGGLoss(nn.Module):
    """Computes the VGG perceptual loss between two batches of images.

    The input and target must be 4D tensors with three channels
    ``(B, 3, H, W)`` and must have equivalent shapes. Pixel values should be
    normalized to the range 0–1.

    The VGG perceptual loss is the mean squared difference between the features
    computed for the input and target at layer :attr:`layer` (default 8, or
    ``relu2_2``) of the pretrained model specified by :attr:`model` (either
    ``'vgg16'`` (default) or ``'vgg19'``).

    If :attr:`shift` is nonzero, a random shift of at most :attr:`shift`
    pixels in both height and width will be applied to all images in the input
    and target. The shift will only be applied when the loss function is in
    training mode, and will not be applied if a precomputed feature map is
    supplied as the target.

    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``'mean'``.

    :meth:`get_features()` may be used to precompute the features for the
    target, to speed up the case where inputs are compared against the same
    target over and over. To use the precomputed features, pass them in as
    :attr:`target` and set :attr:`target_is_features` to :code:`True`.

    Instances of :class:`VGGLoss` must be manually converted to the same
    device and dtype as their inputs.
    """

    models = {'vgg16': models.vgg16, 'vgg19': models.vgg19}

    def __init__(self, model='vgg19', layer=8, shift=0, reduction='mean'):
        super().__init__()
        self.shift = shift
        self.reduction = reduction
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.model = self.models[model](pretrained=True).features[:layer+1]
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(torch.device("cuda:0"))

    def get_features(self, input):
        input = input.repeat((1, 3, 1, 1)) # single channel to 3 channels
        return self.model(self.normalize(input))

    def train(self, mode=True):
        self.training = mode

    def forward(self, input, target, target_is_features=False):
        input = input.repeat((1, 3, 1, 1)) # single channel to 3 channels
        if target_is_features:
            input_feats = self.get_features(input)
            target_feats = target
        else:
            target = target.repeat((1, 3, 1, 1)) # single channel to 3 channels
            
            with torch.no_grad():
                target_feats = self.get_features(target)
            input_feats = self.get_features(input)
        return F.mse_loss(input_feats, target_feats, reduction=self.reduction)

class CombinedLoss:
    def __init__(self, gd_weight=0.0, l2_weight=0.0, l1_weight=0.0, vgg_weight=0.0):
        self.gd_weight = gd_weight
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.vgg_weight = vgg_weight
        self.gd_loss = GDLoss()
        self.l1_loss = L1Loss()
        self.l2_loss = L2Loss()
        if self.vgg_weight > 0:
            self.vgg_loss = VGGLoss()
        print(f"loss weight: l1 {l1_weight} l2 {l2_weight} gd {gd_weight} vgg {vgg_weight}")

    def apply(self, x, gt, mask=None, return_vgg=False):
        final_loss = 0
        if self.l1_weight > 0:
            final_loss += self.l1_weight * self.l1_loss.apply(x, gt, mask)
        if self.l2_weight > 0:
            final_loss += self.l2_weight * self.l2_loss.apply(x, gt, mask)
        if self.gd_weight > 0:
            final_loss += self.gd_weight * self.gd_loss.apply(x, gt, mask)
        if self.vgg_weight > 0:
            feat_x = self.vgg_loss.get_features(x)
            feat_gt = self.vgg_loss.get_features(gt)
            final_loss += self.vgg_weight * F.mse_loss(feat_x, feat_gt, reduction='mean')
        if return_vgg:
            assert self.vgg_weight > 0
            return final_loss, feat_x
        else:
            return final_loss

def create_loss(loss_config):
    if loss_config["type"] == "l2":
        return L2Loss()
    elif loss_config["type"] == "l1":
        return L1Loss()
    elif loss_config["type"] == "gd":
        return GDLoss()
    elif loss_config["type"] == "combine":
        return CombinedLoss(**loss_config["configs"])
    else:
        assert False

class L1LossTemporal:
    def __init__(self):
        pass

    def use_vgg(self):
        return False

    def apply(self, x, prev_x, mask=None):
        if mask is None:
            return torch.mean(torch.abs(x - prev_x))
        else:
            return torch.mean(torch.abs(x - prev_x) * mask)

class L2LossTemporal:
    def __init__(self):
        pass

    def use_vgg(self):
        return False

    def apply(self, x, prev_x, mask=None):
        if mask is None:
            return torch.mean((x - prev_x) ** 2)
        else:
            return torch.mean(((x - prev_x) ** 2) * mask)

# class GDLossTemporal:
#     def __init__(self):
#         assert False

#     def apply(self, x, prev_x, gt, prev_gt, mask=None):
#         if mask is None:
#             return torch.mean(torch.abs(x - prev_x - (gt - prev_gt)))
#         else:
#             return torch.mean(torch.abs((x - prev_x - (gt - prev_gt))) * mask)

class L1VGGLossTemporal:
    def __init__(self):
        pass

    def use_vgg(self):
        return True

    def apply(self, x, prev_x, vgg, prev_vgg, mask=None):
        if mask is None:
            return torch.mean(torch.abs(x - prev_x)) + 0.1 * F.mse_loss(vgg, prev_vgg, reduction="mean")
        else:
            return torch.mean(torch.abs(x - prev_x) * mask) + 0.1 * F.mse_loss(vgg, prev_vgg, reduction="mean")

def create_temporal_loss(tloss_config):
    print("creating temporal loss type =", tloss_config["type"])
    if tloss_config["type"] == "l2":
        return L2LossTemporal()
    elif tloss_config["type"] == "l1":
        return L1LossTemporal()
    # elif tloss_config["type"] == "gd":
    #     return GDLossTemporal()
    elif tloss_config["type"] == "l1vgg":
        return L1VGGLossTemporal()
    else:
        assert False
