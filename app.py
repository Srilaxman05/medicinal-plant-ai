import streamlit as st
import os

# --- 1. SETUP PAGE ---
st.set_page_config(page_title="H5 to TFLite Converter")
st.title("🛠️ H5 to TFLite Converter")
st.caption("A tool to convert Keras models to TFLite for faster loading.")

# --- 2. UPLOAD FILE ---
uploaded_file = st.file_uploader("Upload your keras_model.h5", type=["h5"])

# --- 3. CONVERSION LOGIC ---
if uploaded_file is not None:
    if st.button("Start Conversion"):
        
        # A. Save uploaded file
        with open("temp_model.h5", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        status = st.status("Initializing conversion...", expanded=True)
        
        try:
            # B. Lazy Import TensorFlow (Only happens when button is clicked)
            status.write("Loading TensorFlow Library (This takes 10-20 seconds)...")
            import tensorflow as tf
            
            # C. Load Model
            status.write("Reading Keras Model...")
            model = tf.keras.models.load_model("temp_model.h5", compile=False)
            
            # D. Convert
            status.write("Converting to TFLite...")
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            
            status.update(label="Conversion Complete!", state="complete", expanded=False)
            
            # E. Download Button
            st.success("Success! Download your file below:")
            st.download_button(
                label="⬇️ Download model.tflite",
                data=tflite_model,
                file_name="model.tflite",
                mime="application/octet-stream"
            )
            
        except Exception as e:
            st.error(f"Error: {e}")
            status.update(label="Failed", state="error")
        
        # Cleanup
        if os.path.exists("temp_model.h5"):
            os.remove("temp_model.h5")

# --- 4. INSTRUCTIONS ---
st.markdown("---")
st.info("""
**Instructions:**
1. Upload your `.h5` file.
2. Click **Start Conversion**.
3. Wait for the 'Loading TensorFlow' step (it happens only once).
4. Download the `.tflite` file.
5. Afterwards, replace this code with your Leaf Scanner code!
""")
