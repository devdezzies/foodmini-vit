"""
contains pytorch model code to instantiate a TinyVGG model.
"""
import torch 
from torch import nn 
import torchvision
from utils import set_seeds

def create_model_baseline_vit(out_feats: int, device: torch.device = None) -> torch.nn.Module: 
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT 
    pretrained_vit = torchvision.models.vit_b_16(weights=weights).to(device)

    for param in pretrained_vit.parameters():
        param.requires_grad = False

    set_seeds()
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=out_feats).to(device)
    return pretrained_vit