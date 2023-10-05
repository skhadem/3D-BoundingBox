import torch
from torchvision.models import vgg
from torch_lib.Model import Model


checkpoint = "/home/thoro-ml/work/ml/3D-BoundingBox/weights/epoch_2.pkl"
backbone = vgg.vgg19_bn(weights=vgg.VGG19_BN_Weights.IMAGENET1K_V1)
model = Model(features=backbone.features)
ckpt = torch.load(checkpoint)
model.load_state_dict(ckpt['model_state_dict'])
fake_input = torch.randn((1, 3, 224, 224))

torch.onnx.export(
    model,
    fake_input,
    "Deep3DBox.onnx",
    export_params=True,
    opset_version=9,
    do_constant_folding=True,
    input_names=['modelInput'],
    output_names=['Orientation', 'Confidence']
)