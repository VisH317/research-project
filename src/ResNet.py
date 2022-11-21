import torch
from torchvision.models.resnet import resnet50, ResNet50_Weights

weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
    
device = torch.device(dev)

model = resnet50(weights=weights).to(device)
feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))