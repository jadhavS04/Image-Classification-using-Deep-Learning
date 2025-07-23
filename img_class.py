
import streamlit as st
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
model = EfficientNetB0(weights='imagenet')

st.title("📸 Image Classification with EfficientNetB0")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    # Predict
    preds = model.predict(x)
    results = decode_predictions(preds, top=3)[0]

    st.subheader("Predictions:")
    for i, (imagenetID, label, prob) in enumerate(results):
        st.write(f"{i+1}. **{label}** — {prob*100:.2f}%")
