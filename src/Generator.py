import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# CGAN Generator Model
class FFTranslator(nn.Module):
    def __init__(self) -> None:
        super(FFTranslator, self).__init__()
        
        self.linear1 = nn.Linear(512, 1024)
        self.linear2 = nn.Linear(1024, 1536)
        self.linear3 = nn.Linear(1536, 2048)
    
    def forward(self, input):
        out = F.leaky_relu(self.linear1(input))
        out = F.leaky_relu(self.linear2(out))
        out = self.linear3(out)
        return out

class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        self.translator = FFTranslator()

        # hyperparams and train params

    def forward(self, input):
        noise = self.generate_noise().to(device=torch.device("cuda:0"))
        out = torch.cat((torch.squeeze(input), torch.squeeze(noise)), dim=0)
        out = self.translator(out)
        return out

    def generate_noise(self):
        return torch.normal(mean=torch.zeros(128), std=torch.zeros(128))
    
class GeneratorPreprocess(nn.Module):
    def __init__(self):
        super(GeneratorPreprocess, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
    @torch.no_grad()
    def forward(self, text):
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device=torch.device("cuda:0"))
        out = self.model(**encoded_input)
        out = mean_pooling(out, encoded_input["attention_mask"])
        out = F.normalize(out)
        return out