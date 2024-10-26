import torch
import torch.nn as nn
from torchvision import models

class RegressionResNet(nn.Module):
    def __init__(self, pretrained=True, bias=False, num_outputs=2):
        super(RegressionResNet, self).__init__()
        resnet_model = models.resnet18(pretrained=pretrained)
        
        # Modify backbone to exclude the last layer containing ReLU
        self.backbone = nn.Sequential(*list(resnet_model.children())[:-1])  # Excludes the final layer (which has ReLU)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.feat = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(resnet_model.fc.in_features, resnet_model.fc.in_features)
        )
        
        self.fc = nn.Linear(resnet_model.fc.in_features, num_outputs, bias=bias)
        self.embeddings = None

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_avg_pool(x)
        x = self.feat(x)
        return self.fc(x)

    def get_last_layer_embeddings(self, x):
        """Extract embeddings from the last layer using a hook and global average pooling."""
        def hook_fn(module, input, output):
            self.embeddings = output.detach() 
            
        # Register the hook on the appropriate layer
        hook = self.feat[-1].register_forward_hook(hook_fn)
        self.forward(x)
        hook.remove()
        return self.embeddings
