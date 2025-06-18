import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input  # type: ignore
from streamlit_cropper import st_cropper

# Load model once
@st.cache_resource
def load_trained_model():
    model = load_model("best_model.keras")
    return model

model = load_trained_model()

class_names = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']

# Custom ReLU for Guided Grad-CAM
@tf.custom_gradient
def guided_relu(x):
    def grad(dy):
        return tf.cast(dy > 0, tf.float32) * tf.cast(x > 0, tf.float32) * dy
    return tf.nn.relu(x), grad

def modify_relu(model):
    for layer in model.layers:
        if hasattr(layer, "activation") and layer.activation == tf.keras.activations.relu:
            layer.activation = guided_relu

# Guided Grad-CAM function
def guided_gradcam(img_array, model, last_conv_layer_name="conv5_block3_out"):
    modify_relu(model)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    # Guided backpropagation
    with tf.GradientTape() as tape2:
        tape2.watch(img_array)
        _, predictions2 = grad_model(img_array)
        loss2 = predictions2[:, class_idx]
    guided_grads = tape2.gradient(loss2, img_array)[0]

    return heatmap.numpy(), guided_grads.numpy(), class_idx.numpy(), predictions.numpy()

# Overlay Grad-CAM heatmap
def overlay_heatmap(original_img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed_img

# Streamlit UI
st.title("🍎 Fruit Freshness Classifier with Guided Grad-CAM")
st.markdown("Upload a fruit image or capture using camera to classify freshness and visualize attention.")

upload_option = st.radio("Choose input method:", ("Upload Image", "Use Camera"))

cropped_img = None
if upload_option == "Upload Image":
    input_img = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])
    if input_img:
        image = Image.open(input_img).convert("RGB")
        cropped_img = image  # no cropping
elif upload_option == "Use Camera":
    input_img = st.camera_input("Take a photo of a fruit")
    if input_img:
        image = Image.open(input_img).convert("RGB")
        st.markdown("### ✂️ Crop the fruit manually below:")
        cropped_img = st_cropper(image, aspect_ratio=None, box_color='#FF4B4B')

# Run prediction if an image is available
if cropped_img:
    st.image(cropped_img, caption="Input Image", use_container_width=True)

    resized_img = cropped_img.resize((224, 224))
    array = np.array(resized_img) / 255.0
    batch = tf.convert_to_tensor(np.expand_dims(array, axis=0), dtype=tf.float32)


    # Run Guided Grad-CAM
    heatmap, guided_output, class_idx, probs = guided_gradcam(batch, model)
    class_label = class_names[class_idx]
    confidence = probs[0][class_idx] * 100

    st.markdown(f"### 🔍 Prediction: **{class_label}** ({confidence:.2f}%)")

    # Show Grad-CAM overlay
    original_bgr = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR)
    overlay_img = overlay_heatmap(original_bgr, heatmap)
    overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

    st.image(overlay_rgb, caption="📊 Guided Grad-CAM Visualization", use_container_width=True)
