import streamlit as st
import os
import tensorflow as tf

# --- 1. SETUP PAGE ---
st.set_page_config(page_title="H5 to TFLite Converter (Fixed)")
st.title("🛠️ H5 to TFLite Converter (Fixed)")
st.caption("Auto-fixes the 'groups=1' error for Teachable Machine models.")

# --- 2. THE FIX (Monkey Patch) ---
# This class intercepts the layer and removes the bad parameter
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        # The error happens because 'groups' is passed but not expected.
        # We simply pop it out before passing kwargs to the parent class.
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

# --- 3. UI & LOGIC ---
uploaded_file = st.file_uploader("Upload your keras_model.h5", type=["h5"])

if uploaded_file is not None:
    if st.button("Start Conversion"):
        
        # Save temp file
        with open("temp_model.h5", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        status = st.status("Processing...", expanded=True)
        
        try:
            status.write("Applying 'DepthwiseConv2D' Patch...")
            
            # LOAD WITH THE FIX
            # We pass the custom class to custom_objects
            model = tf.keras.models.load_model(
                "temp_model.h5", 
                compile=False,
                custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D} 
            )
            
            status.write("Model Loaded! Converting to TFLite...")
            
            # CONVERT
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            
            status.update(label="Conversion Successful!", state="complete", expanded=False)
            
            # DOWNLOAD
            st.success("✅ Fixed & Converted! Download below:")
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
