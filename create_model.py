import torch
import torch.nn as nn
import torchvision.models as models

class TomatoLeafNet(nn.Module):
    def __init__(self, num_classes=4):
        super(TomatoLeafNet, self).__init__()
        # Use a pre-trained ResNet18 model
        self.resnet = models.resnet18(pretrained=True)
        # Replace the final layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def main():
    # Create model
    model = TomatoLeafNet(num_classes=4)
    
    # Set to evaluation mode
    model.eval()
    
    # Save the model state
    torch.save(model.state_dict(), 'tomato_model_state.pth')
    print("Model state saved successfully!")

if __name__ == '__main__':
    main() 