import streamlit as st
import tensorflow as tf
import os

st.set_page_config(page_title="H5 to TFLite Converter")

st.title("🛠️ H5 to TFLite Converter")
st.write("Since you can't use Colab/Local, use this tool to convert your model.")

uploaded_file = st.file_uploader("Upload your keras_model.h5 file", type=["h5"])

if uploaded_file is not None:
    st.write("Processing...")
    
    # 1. Save the uploaded h5 file temporarily
    with open("temp_model.h5", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # 2. Load the Keras model
        st.info("Loading Keras model... (This might take a minute)")
        model = tf.keras.models.load_model("temp_model.h5", compile=False)
        
        # 3. Convert to TFLite
        st.info("Converting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        # 4. Create the download button
        st.success("Conversion Successful! Download your file below:")
        
        st.download_button(
            label="⬇️ Download model.tflite",
            data=tflite_model,
            file_name="model.tflite",
            mime="application/octet-stream"
        )
        
    except Exception as e:
        st.error(f"Error during conversion: {e}")

    # Cleanup temp file
    if os.path.exists("temp_model.h5"):
        os.remove("temp_model.h5")
