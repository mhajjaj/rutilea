import torch.nn as nn

# Define a simple convolutional layer
conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

# Forward pass through the convolutional layer
output = conv_layer(input_tensor)
