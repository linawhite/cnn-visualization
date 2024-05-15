from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA
import numpy as np
import base64
from io import BytesIO
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt 

def load_image(image, img_width=150, img_height=150):
    transform = transforms.Compose([
        transforms.Resize((img_width, img_height)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform(image)

def visualize_activations(activations, layers):
    fig, axes = plt.subplots(len(layers), 5, figsize=(20, 10))

    for idx, layer_idx in enumerate(layers):
        layer_activation = activations[f'layer{layer_idx+1}']
        for neuron_idx in range(5):
            ax = axes[idx, neuron_idx]
            ax.imshow(layer_activation[0][neuron_idx].cpu().numpy(), cmap='gray')
            ax.axis('off')
            ax.set_title(f'Layer {layer_idx+1} Feature {neuron_idx+1}')

    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig) 
    buf.seek(0) 
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8') 

    return image_base64
