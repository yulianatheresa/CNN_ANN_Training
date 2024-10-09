import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load model
model = keras.models.load_model('model_cnn.h5')

# Title
st.title('Animal Faces Prediction')

# Upload
image = st.file_uploader(label='Animal', type=['png', 'jpg'])

# Display Image
if image is not None:
    img = Image.open(image)
    img = np.array(img)
    st.image(img, caption='Uploaded Image')
    st.write('')
    st.write('Classifying')

    def prediction(file):
        img = Image.open(file)
        img = img.resize((150, 150))
        x = np.array(img) / 255.0 
        x = np.expand_dims(x, axis=0)
        classes = model.predict(x, batch_size=1)
        idx = np.argmax(classes)
        clas = ['cat', 'dog', 'wild']
        return clas[idx]

    result = prediction(image)
    st.write(f'Prediction: {result}')
