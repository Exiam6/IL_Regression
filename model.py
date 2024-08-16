import torch
import torch.nn as nn
from torchvision import models

class RegressionResNet(nn.Module):
    def __init__(self, pretrained=True, bias=False, num_outputs=2):
        super(RegressionResNet, self).__init__()
        resnet_model = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet_model.children())[:-2])
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
            pooled_output = self.global_avg_pool(output)
            self.embeddings = pooled_output.view(pooled_output.size(0), -1).detach()

        # Register the hook on the appropriate layer
        hook = self.backbone[-1].register_forward_hook(hook_fn)
        self.forward(x)
        hook.remove()
        return self.embeddings
