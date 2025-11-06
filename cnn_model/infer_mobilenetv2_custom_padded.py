import torch
import torch.nn as nn
from torchvision import models

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the customized model
model = models.mobilenet_v2(pretrained=False)  # Structure only

# Reapply the same first conv customization (since loading state_dict doesn't include structure changes)
first_conv = model.features[0][0]
new_first_conv = nn.Conv2d(1, first_conv.out_channels, kernel_size=first_conv.kernel_size,
                           stride=first_conv.stride, padding=first_conv.padding, bias=first_conv.bias)
# Note: Weights will be loaded from state_dict, so no need to average again here

model.features[0][0] = new_first_conv

# Modify classifier
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 363)

model.load_state_dict(torch.load('mobilenetv2_custom_padded.pth'))
model = model.to(device)
model.eval()

# Function for inference on a [24,5] tensor
def infer_input(input_tensor):
    inp = input_tensor.float().unsqueeze(0)  # [1, 24, 5]
    # Pad to [1, 32, 32]
    pad_left = 13
    pad_right = 14
    pad_top = 4
    pad_bottom = 4
    inp = nn.functional.pad(inp, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    # Normalize (same as training)
    inp = (inp - 0.485) / 0.229
    
    inp = inp.unsqueeze(0).to(device)  # [1, 1, 32, 32]
    
    with torch.no_grad():
        outputs = model(inp)
        preds = (torch.sigmoid(outputs) > 0.5).float().squeeze()  # [363]
    
    return preds

# Example: Use your provided input
input_tensor = torch.tensor([[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [1, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 1],
                             [0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [1, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0]])

predicted_output = infer_input(input_tensor)
print(f'Predicted binary output (first 20 elements shown): {predicted_output[:20]}')
print(f'Full predicted output shape: {predicted_output.shape}')