import torch
from ai2thorGym import AI2thor
from stable_baselines3 import PPO
from FeatureExtractor import FeatureExtractor

if __name__ == '__main__':
    
    env = AI2thor('train')
    policy_kwargs = dict(features_extractor_class = FeatureExtractor)
    device = torch.device("cuda:0")

    model = PPO(
        env=env,
        learning_rate = 0.0004,
        batch_size = 16,
        verbose=1,
        device=device,
        policy_kwargs=policy_kwargs
    )

# to do: download model parameters set up the model loading and try to compile the program