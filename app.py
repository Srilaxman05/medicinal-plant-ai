import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import os
import tensorflow as tf  # Changed from tflite_runtime to tensorflow

# --- 1. CONFIGURATION (Must be the first line) ---
st.set_page_config(
    page_title="AyurVision: Medicinal Plant Scanner",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. UI HEADER ---
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>Medicinal Leaf Identification</h1>", unsafe_allow_html=True)
st.caption("Upload a clear leaf image to identify its species and medicinal uses.")
st.markdown("---")

# --- 3. DATA: PLANT NAMES & INFO ---
LEAF_NAMES = [
    "Alpinia Galanga (Rasna)", "Amaranthus Viridis (Arive-Dantu)", "Artocarpus Heterophyllus (Jackfruit)",
    "Azadirachta Indica (Neem)", "Basella Alba (Basale)", "Brassica Juncea (Indian Mustard)",
    "Carissa Carandas (Karanda)", "Citrus Limon (Lemon)", "Ficus Auriculata (Roxburgh fig)",
    "Ficus Religiosa (Peepal Tree)", "Hibiscus Rosa-sinensis", "Jasminum (Jasmine)",
    "Mangifera Indica (Mango)", "Mentha (Mint)", "Moringa Oleifera (Drumstick)",
    "Muntingia Calabura (Jamaica Cherry-Gasagase)", "Murraya Koenigii (Curry)",
    "Nerium Oleander (Oleander)", "Nyctanthes Arbor-tristis (Parijata)", "Ocimum Tenuiflorum (Tulsi)",
    "Piper Betle (Betel)", "Plectranthus Amboinicus (Mexican Mint)", "Pongamia Pinnata (Indian Beech)",
    "Psidium Guajava (Guava)", "Punica Granatum (Pomegranate)", "Santalum Album (Sandalwood)",
    "Syzygium Cumini (Jamun)", "Syzygium Jambos (Rose Apple)",
    "Tabernaemontana Divaricata (Crape Jasmine)", "Trigonella Foenum-graecum (Fenugreek)"
]

PLANT_INFO = {
    "Alpinia Galanga (Rasna)": "Used for rheumatism, respiratory ailments, and digestion.",
    "Amaranthus Viridis (Arive-Dantu)": "Rich in vitamins; used for general health and digestion.",
    "Artocarpus Heterophyllus (Jackfruit)": "Roots used for asthma; leaves for skin diseases.",
    "Azadirachta Indica (Neem)": "Powerful antiseptic, antifungal, and blood purifier.",
    "Basella Alba (Basale)": "Cooling effect, good for mouth ulcers and digestion.",
    "Brassica Juncea (Indian Mustard)": "Oil used for joint pain; seeds for digestion.",
    "Carissa Carandas (Karanda)": "Rich in Iron/Vitamin C; treats anemia and digestion.",
    "Citrus Limon (Lemon)": "Immunity booster, digestive aid, rich in Vitamin C.",
    "Ficus Auriculata (Roxburgh fig)": "Used for wounds, cuts, and digestive issues.",
    "Ficus Religiosa (Peepal Tree)": "Treats asthma, skin disorders, and kidney issues.",
    "Hibiscus Rosa-sinensis": "Good for hair growth, blood pressure, and heart health.",
    "Jasminum (Jasmine)": "Stress relief, skin health, and wound healing.",
    "Mangifera Indica (Mango)": "Leaves help regulate insulin levels and treat burns.",
    "Mentha (Mint)": "Relieves indigestion, nausea, and headache.",
    "Moringa Oleifera (Drumstick)": "Superfood; treats joint pain, anemia, and diabetes.",
    "Muntingia Calabura (Jamaica Cherry-Gasagase)": "Pain relief, antibacterial properties.",
    "Murraya Koenigii (Curry)": "Good for hair, digestion, and controlling blood sugar.",
    "Nerium Oleander (Oleander)": "⚠️ CAUTION: Toxic if ingested. Used externally for skin conditions.",
    "Nyctanthes Arbor-tristis (Parijata)": "Treats sciatica, arthritis, and fever.",
    "Ocimum Tenuiflorum (Tulsi)": "Holy Basil. Treats colds, coughs, and boosts immunity.",
    "Piper Betle (Betel)": "Digestion, oral health, and wound healing.",
    "Plectranthus Amboinicus (Mexican Mint)": "Treats cough, cold, and asthma (Karpooravalli).",
    "Pongamia Pinnata (Indian Beech)": "Oil used for skin diseases and rheumatism.",
    "Psidium Guajava (Guava)": "Treats diarrhea, toothache, and gum infections.",
    "Punica Granatum (Pomegranate)": "Digestion, heart health, and anemia.",
    "Santalum Album (Sandalwood)": "Skin care, cooling, and mental clarity.",
    "Syzygium Cumini (Jamun)": "Excellent for diabetes management and digestion.",
    "Syzygium Jambos (Rose Apple)": "Treats smallpox, joints, and eye inflammation.",
    "Tabernaemontana Divaricata (Crape Jasmine)": "Used for wounds, eye diseases, and toothache.",
    "Trigonella Foenum-graecum (Fenugreek)": "Controls blood sugar, digestion, and hair health."
}

# --- 4. MODEL ENGINE (Using standard TensorFlow) ---
@st.cache_resource
def load_model():
    """Load the TFLite model using TensorFlow."""
    try:
        model_path = "model.tflite"
        
        if not os.path.exists(model_path):
            st.error(f"❌ File not found: {model_path}. Please upload it to your repo.")
            return None
            
        # Initialize the TFLite interpreter using TensorFlow
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading AI Engine: {e}")
        return None

def predict_image(interpreter, image):
    """Run inference on the image."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 1. Resize to 224x224 (Standard for Teachable Machine)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # 2. Convert to Array & Normalize
    img_array = np.asarray(image)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    
    # 3. Reshape for Model (1, 224, 224, 3)
    data = normalized_image_array[np.newaxis, ...]

    # 4. Run Inference
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    return prediction

# --- 5. MAIN APPLICATION ---
file = st.file_uploader("📂 Upload Image", type=["jpg", "png", "jpeg"])

if file:
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.subheader("📸 Uploaded Leaf")
        image = Image.open(file).convert("RGB")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🧬 Identification")
        
        # Use st.status to show progress without blocking
        with st.status("Initializing AI Brain...", expanded=True) as status:
            interpreter = load_model()
            
            if interpreter:
                status.write("Scanning leaf patterns...")
                predictions = predict_image(interpreter, image)
                
                # Get Result
                index = np.argmax(predictions)
                confidence = predictions[0][index]
                
                status.update(label="Analysis Complete", state="complete", expanded=False)

                # Display Logic
                if index < len(LEAF_NAMES):
                    name = LEAF_NAMES[index]
                    details = PLANT_INFO.get(name, "No specific info available.")
                    
                    if confidence > 0.65:
                        st.success(f"**Identified:** {name}")
                        st.progress(int(confidence * 100))
                        st.caption(f"Confidence: **{confidence*100:.2f}%**")
                        st.info(f"**💊 Medicinal Uses:**\n\n{details}")
                    else:
                        st.warning(f"**Possible Match:** {name}")
                        st.write(f"Confidence: {confidence*100:.2f}%")
                        st.error("⚠️ Low confidence. Please upload a clearer image.")
                else:
                    st.error("❌ Unknown Plant Species")
            else:
                status.update(label="Error", state="error")

# --- 6. SIDEBAR INFO ---
with st.sidebar:
    st.title("🌿 AyurVision")
    st.info("This app uses a TFLite model to identify 30 different medicinal plants common in India.")
    st.write("---")
    st.write("**Supported Plants:**")
    st.caption(", ".join(LEAF_NAMES[:5]) + " and more...")
