from flask import Flask, request, jsonify
import torch
from model import ConvNet
from utilities import load_image, visualize_activations
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:8000"])

model = ConvNet(num_classes=2)
model.load_state_dict(torch.load('model_best_2.pth', map_location=torch.device('cpu')))
model.eval()

activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

hook_handles = []
layers = [0, 3, 6]
for layer_idx in layers:
    handle = model.features[layer_idx].register_forward_hook(get_activation(f'layer{layer_idx+1}'))
    hook_handles.append(handle)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    input_tensor = load_image(image)

    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))

    predicted_class = torch.argmax(output, dim=1).item()
    visualizations = visualize_activations(activations, layers)

    activations.clear()

    return jsonify({'prediction': predicted_class, 'visualizations': visualizations})

if __name__ == '__main__':
    app.run(debug=True)