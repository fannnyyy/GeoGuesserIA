import streamlit as st

import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import datetime
import itertools
import io
from PIL import Image
import torch
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms

PATH = "/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/model/cnn/geoguesser_model.pth"


@st.cache_resource(show_spinner=False)
def load_cnn1():
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'), weights_only=False)
    num_classes = checkpoint['num_classes']
    model_ = tv_models.resnet18(weights=None)
    model_.fc = torch.nn.Linear(512, num_classes)
    model_.load_state_dict(checkpoint['model_state_dict'])
    model_.eval()
    idx_to_label = checkpoint.get('idx_to_label', {})
    return model_, idx_to_label


def preprocessed_image(file):
    image = file.convert("RGB").resize((224, 224), Image.LANCZOS)
    # Conversion sans passer par le bridge numpy (incompatible numpy 2.x / torch)
    raw = bytearray(image.tobytes())           # bytes bruts HWC uint8
    tensor = torch.frombuffer(raw, dtype=torch.uint8).clone()
    tensor = tensor.reshape(224, 224, 3).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    return tensor


def get_activations(model, input_tensor):
    """Capture outputs of each direct child layer via forward hooks."""
    activations = []
    hooks = []

    def make_hook():
        def hook_fn(module, input, output):
            activations.append(output.detach().cpu())
        return hook_fn

    for layer in model.children():
        hooks.append(layer.register_forward_hook(make_hook()))

    with torch.no_grad():
        model(input_tensor)

    for hook in hooks:
        hook.remove()

    return activations


def predict(model, input_tensor, idx_to_label):
    with torch.no_grad():
        output = model(input_tensor)
    idx = torch.argmax(output, dim=-1).item()
    return idx_to_label.get(idx, str(idx))


RESNET_LAYER_NAMES = [
    "Conv1", "BatchNorm1", "ReLU", "MaxPool",
    "Layer1 (2x BasicBlock 64ch)", "Layer2 (2x BasicBlock 128ch)",
    "Layer3 (2x BasicBlock 256ch)", "Layer4 (2x BasicBlock 512ch)",
    "AdaptiveAvgPool", "FC"
]

def display_activation(activations, act_index):
    activation = activations[act_index]  # (batch, C, H, W) tensor
    if activation.dim() != 4:
        st.write(f"Output shape: {list(activation.shape)} (pas de carte spatiale à afficher)")
        return
    n_show = min(activation.shape[1], 32)
    col_size = 8
    row_size = max(1, (n_show + col_size - 1) // col_size)
    fig, ax = plt.subplots(row_size, col_size, figsize=(col_size * 2, row_size * 2))
    if row_size == 1:
        ax = [ax]
    for row in range(row_size):
        for col in range(col_size):
            ch = row * col_size + col
            if ch < n_show:
                ax[row][col].imshow(activation[0, ch, :, :].tolist(), cmap='gray')
            ax[row][col].axis('off')
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.title('GeoGuessrIA')
    st.sidebar.title('Web Apps using Streamlit')
    st.sidebar.text(""" Project to visualize the CNN layers on GeoGuessrIA image""")

    menu = {1:"Home",2:"Visualization of Dataset",3:"Perform Prediction"}
    def format_func(option):
        return menu[option]
    choice= st.sidebar.selectbox("Menu",options=list(menu.keys()), format_func=format_func)
    if choice == 1 :
        st.subheader("CentraleSupelec final project")
        st.markdown("#### Goal")
        """

        //////
        """

        st.markdown("#### Dataset OSV5M")
        """
        //////
        """

        st.markdown("#### Differents models")
        """
        //////
        """
    elif choice== 2 :
        st.subheader("Sample Data")
        sample1 = Image.open('/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/visualisation/tokyo_street.jpg')
        st.image(sample1,caption='Parasitized Cells', width='stretch')
        sample2 = Image.open('/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/visualisation/tokyo_tower.jpg')
        st.image(sample2,caption='Uninfected Cells', width='stretch')
        st.markdown("#### Training and Testing Sets")
        """
        Description dataset
        """
    elif choice == 3 :
        st.subheader("CNN Models")
        st.markdown("#### CNN from scratch, ResNet18 fine-tune, ViT")
        """
        Description of the différents models
        """

        models = st.sidebar.radio(" Select model to perform prediction", ("CNN from scratch", "ResNet18 fine-tune", "ViT"))
        if models=="ResNet18 fine-tune":
            model_1, idx_to_label_1 = load_cnn1()
            """
            \n ** ResNet18 fine-tune architecture preview**

            Description of the architecture => ajouté confiance
            """
            st.subheader('Test on an Image')
            images = st.file_uploader('Upload Image',type=['jpg','png','jpeg'])
            if images is not None:
                images = Image.open(images)
                st.text("Image Uploaded!")
                st.image(images,width=300)
                used_images = preprocessed_image(images)
                prediction = predict(model_1, used_images, idx_to_label_1)
                st.info(f"Predicted country: **{prediction}**")

                st.sidebar.subheader('Visualization in ResNet18 fine-tune')
                activations = get_activations(model_1, used_images)
                n_layers = len(activations)
                layer_idx = st.sidebar.slider('Which layer do you want to see ?', 0, n_layers, 0, format="no %d ")
                st.subheader('Visualize Layer')
                if layer_idx > 0:
                    act_i = layer_idx - 1
                    name = RESNET_LAYER_NAMES[act_i] if act_i < len(RESNET_LAYER_NAMES) else f"Layer {act_i}"
                    st.write(f"Layer {layer_idx}: {name}")
                    display_activation(activations, act_i)

        elif models=="Simple CNN":
            """
            /////
            """


if __name__ == "__main__":
    main()
