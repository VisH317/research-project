# embedder
import torch
import torch.nn as nn
from img2vec_pytorch import Img2Vec
from Pillow import Image

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
import Generator
from torchvision.models.resnet import resnet50, ResNet50_Weights


feature_extractor = feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device)


class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        super(FeatureExtractor, self).__init__(observation_space, features_dim)
        
        self.image = observation_space.spaces["image"]
        self.text = observation_space.spaces["text"]
        
        # networks
        self.generator = Generator()
        self.generator.load_state_dict(torch.load('./Models/generator.pt'))
        self.feature_extractor = Img2Vec(cuda=True, model='resnet50')
        
        # load torch parameters for generator
        
        self.concat_processor = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, features_dim),
            nn.ReLU()
        )
        
    def forward(self):
        with torch.no_grad():
            I = Image.fromarray(self.image)
            image_embed = self.feature_extractor.get_vec(I)
            text_embed = torch.squeeze(self.generator(self.text))        
        concat_embed = torch.cat((image_embed, text_embed), dim=0)
        out = self.concat_processor(concat_embed)
        return out