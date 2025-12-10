import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# --- 1. SET PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AyurVision: Medicinal Plant Scanner",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. DEFINE THE PLANT LIST & MEDICINAL INFO ---
# The order here MUST match your Teachable Machine Class order (1, 2, 3...)
LEAF_NAMES = [
    "Alpinia Galanga (Rasna)",
    "Amaranthus Viridis (Arive-Dantu)",
    "Artocarpus Heterophyllus (Jackfruit)",
    "Azadirachta Indica (Neem)",
    "Basella Alba (Basale)",
    "Brassica Juncea (Indian Mustard)",
    "Carissa Carandas (Karanda)",
    "Citrus Limon (Lemon)",
    "Ficus Auriculata (Roxburgh fig)",
    "Ficus Religiosa (Peepal Tree)",
    "Hibiscus Rosa-sinensis",
    "Jasminum (Jasmine)",
    "Mangifera Indica (Mango)",
    "Mentha (Mint)",
    "Moringa Oleifera (Drumstick)",
    "Muntingia Calabura (Jamaica Cherry-Gasagase)",
    "Murraya Koenigii (Curry)",
    "Nerium Oleander (Oleander)",
    "Nyctanthes Arbor-tristis (Parijata)",
    "Ocimum Tenuiflorum (Tulsi)",
    "Piper Betle (Betel)",
    "Plectranthus Amboinicus (Mexican Mint)",
    "Pongamia Pinnata (Indian Beech)",
    "Psidium Guajava (Guava)",
    "Punica Granatum (Pomegranate)",
    "Santalum Album (Sandalwood)",
    "Syzygium Cumini (Jamun)",
    "Syzygium Jambos (Rose Apple)",
    "Tabernaemontana Divaricata (Crape Jasmine)",
    "Trigonella Foenum-graecum (Fenugreek)"
]

# Dictionary to display usage info based on the detected plant
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

# --- 3. LOAD MODEL ---
@st.cache_resource
def load_classifier():
    try:
        model = load_model("keras_model.h5", compile=False)
        return model
    except Exception as e:
        return None

model = load_classifier()

if model is None:
    st.error("⚠️ Error: 'keras_model.h5' not found. Please place your model file in the same folder as this script.")
    st.stop()

# --- 4. PREDICTION ENGINE ---
def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    data = (img_reshape.astype(np.float32) / 127.5) - 1
    prediction = model.predict(data)
    return prediction

# --- 5. UI LAYOUT ---

# Sidebar
with st.sidebar:
    st.title("🌿 AyurVision")
    st.subheader("Medicinal Plant Identifier")
    st.write("This AI-powered tool identifies 30 common Indian medicinal plants from leaf images.")
    
    with st.expander("See Supported Plants"):
        st.write("\n".join([f"- {name}" for name in LEAF_NAMES]))
        
    st.markdown("---")
    st.caption("Built with Python & Streamlit")

# Main Interface
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>Medicinal Leaf Identification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a clear image of a leaf to identify its species and medicinal uses.</p>", unsafe_allow_html=True)
st.markdown("---")

file = st.file_uploader("📂 Upload Leaf Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if file is None:
    st.info("Please upload an image to begin analysis.")
else:
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.subheader("📸 Uploaded Image")
        image = Image.open(file).convert("RGB")
        st.image(image, use_container_width=True, style={"border-radius": "10px"})

    with col2:
        st.subheader("🧬 Analysis Results")
        
        with st.spinner('Scanning leaf patterns...'):
            prediction = import_and_predict(image, model)
            index = np.argmax(prediction)
            confidence_score = prediction[0][index]
            
            # Logic to handle index errors
            if index < len(LEAF_NAMES):
                class_name = LEAF_NAMES[index]
                plant_details = PLANT_INFO.get(class_name, "No specific info available.")
            else:
                class_name = "Unknown Species"
                plant_details = "Please check your model classes."

        # Display Logic
        if confidence_score > 0.60:  # Threshold for accuracy
            st.success(f"**Identified:** {class_name}")
            
            # Confidence Bar
            st.write(f"Confidence: **{confidence_score * 100:.2f}%**")
            st.progress(int(confidence_score * 100))
            
            # Medicinal Info Card
            st.info(f"💊 **Medicinal Uses:**\n\n{plant_details}")
            
        else:
            st.warning(f"**Identified:** {class_name} (Low Confidence)")
            st.write(f"Confidence: {confidence_score * 100:.2f}%")
            st.error("⚠️ The image is unclear or the leaf is not in our database. Please try again.")