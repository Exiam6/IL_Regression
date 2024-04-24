import torch
import torch.nn as nn
from torchvision import models

class RegressionResNet(nn.Module):
    def __init__(self, pretrained=True, num_outputs=2):
        super(RegressionResNet, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        #self.model.conv1 = torch.nn.Conv2d(200, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_outputs)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        return self.model(x)

    def get_last_layer_embeddings(self, x):
        """Extract embeddings from the last layer using a hook and global average pooling."""
        def hook_fn(module, input, output):
            pooled_output = self.global_avg_pool(output)
            self.embeddings = pooled_output.view(pooled_output.size(0), -1).detach()

        hook = self.model.layer4.register_forward_hook(hook_fn)
        self.forward(x)
        hook.remove()
        return self.embeddings
