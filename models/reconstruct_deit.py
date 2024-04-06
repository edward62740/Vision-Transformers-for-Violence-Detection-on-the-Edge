import timm

onnx_model_path = "deit_onnx_intermediate_model.onnx"

import deit
from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


def gelu():
    return nn.GELU(approximate='tanh')


class GELUapprx(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x * 1.702)


def deit_tiny_distilled_patch16_224(pretrained=True, **kwargs):
    model = deit.DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0, act_layer=gelu, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


sample_input = torch.rand((1, 3, 224, 224))


class CustomDeiTWithPreprocessing(nn.Module):
    def __init__(self, **kwargs):
        super(CustomDeiTWithPreprocessing, self).__init__()

        # Preprocessing layers
        self.normalization = nn.BatchNorm2d(3)
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255])
        self.variance = torch.tensor([(0.229 * 255) ** 2, (0.224 * 255) ** 2, (0.225 * 255) ** 2])

        # DeiT model
        self.deit_model = deit_tiny_distilled_patch16_224()

    def forward(self, x):
        # Apply normalization
        x = self.normalization(x)

        # Apply mean and variance scaling
        x = (x - self.mean.view(1, 3, 1, 1)) / torch.sqrt(self.variance.view(1, 3, 1, 1))

        # Forward pass through DeiT model
        x = self.deit_model(x)
        return x


# Create an instance of the custom model
custom_model = CustomDeiTWithPreprocessing()

torch.onnx.export(
    custom_model,  # PyTorch Model
    sample_input,  # Input tensor
    onnx_model_path,  # Output file (eg. 'output_model.onnx')
    opset_version=14,  # Operator support version
    input_names=['input'],
    output_names=['output']
)
