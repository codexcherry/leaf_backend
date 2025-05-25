import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np

class TomatoLeafNet(nn.Module):
    def __init__(self, num_classes=4):
        super(TomatoLeafNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def convert_weights(tf_model, pytorch_model):
    # Get weights from TensorFlow model
    tf_weights = tf_model.get_weights()
    
    # Convert convolutional layers
    pytorch_model.features[0].weight.data = torch.from_numpy(np.transpose(tf_weights[0], (3, 2, 0, 1)))
    pytorch_model.features[0].bias.data = torch.from_numpy(tf_weights[1])
    
    pytorch_model.features[3].weight.data = torch.from_numpy(np.transpose(tf_weights[2], (3, 2, 0, 1)))
    pytorch_model.features[3].bias.data = torch.from_numpy(tf_weights[3])
    
    pytorch_model.features[6].weight.data = torch.from_numpy(np.transpose(tf_weights[4], (3, 2, 0, 1)))
    pytorch_model.features[6].bias.data = torch.from_numpy(tf_weights[5])
    
    # Convert fully connected layers
    pytorch_model.classifier[1].weight.data = torch.from_numpy(tf_weights[6].T)
    pytorch_model.classifier[1].bias.data = torch.from_numpy(tf_weights[7])
    
    pytorch_model.classifier[4].weight.data = torch.from_numpy(tf_weights[8].T)
    pytorch_model.classifier[4].bias.data = torch.from_numpy(tf_weights[9])

def main():
    # Load TensorFlow model
    tf_model = tf.keras.models.load_model('tomato_model.h5')
    
    # Create PyTorch model
    pytorch_model = TomatoLeafNet(num_classes=4)
    
    # Convert weights
    convert_weights(tf_model, pytorch_model)
    
    # Save PyTorch model
    torch.save(pytorch_model, 'tomato_model.pth')
    print("Model converted successfully!")

if __name__ == '__main__':
    main() 