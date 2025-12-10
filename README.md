# 🌿 AyurVision: AI-Powered Medicinal Plant Identifier

**AyurVision** is a Deep Learning application designed to identify medicinal plants from leaf images and provide their therapeutic uses. Utilizing **Transfer Learning** and **Computer Vision**, this project aims to bridge the gap between traditional Ayurvedic knowledge and modern technology.

---

## 📖 Table of Contents
1. [Project Overview](#project-overview)
2. [How It Works (The Process)](#how-it-works-the-process)
3. [Tech Stack](#tech-stack)
4. [Dataset Details](#dataset-details)
5. [Installation & Usage](#installation--usage)
6. [Results & Accuracy](#results--accuracy)

---

## 🔍 Project Overview
India is home to a vast variety of medicinal flora, but identifying them requires expert botanical knowledge. This app allows users to simply upload a photo of a leaf, and the system identifies the species among **30 supported classes** (e.g., Neem, Tulsi, Aloe Vera) and displays its medicinal benefits instantly.

---

## ⚙️ How It Works (The Process)

The core functionality relies on a Convolutional Neural Network (CNN) trained via Google Teachable Machine. The process flow is as follows:

### 1. Image Acquisition
*   The user uploads an image via the **Streamlit** web interface.
*   The system accepts `JPG`, `JPEG`, or `PNG` formats.

### 2. Pre-Processing
Before the AI model can analyze the image, it undergoes specific transformations:
*   **Resizing:** The image is resized to **224x224 pixels** using the Lanczos resampling method (via the Pillow library) to match the input layer of the neural network.
*   **Normalization:** The pixel color values (0-255) are converted to a normalized float range of **-1 to 1**.
    *   *Formula used:* `(Image_Pixel / 127.5) - 1`
    *   This step is crucial for the model to converge faster and make accurate predictions.

### 3. Feature Extraction (Transfer Learning)
*   The processed image is passed into the **MobileNet** architecture (a lightweight CNN).
*   MobileNet extracts high-level visual features such as edge detection, texture analysis, and vein patterns.
*   We utilize **Transfer Learning**, meaning the model was pre-trained on millions of generic images (ImageNet) and fine-tuned for our specific leaf dataset.

### 4. Classification & Inference
*   The extracted features are passed to a **Dense (Fully Connected) Layer** specifically trained on our 30 plant classes.
*   The model outputs a probability distribution (Confidence Score) for all 30 classes using the **Softmax** activation function.

### 5. Output Generation
*   The system identifies the class with the highest probability (e.g., *Ocimum Tenuiflorum* - 98%).
*   If the confidence score exceeds the threshold (60%), the app fetches the corresponding medicinal data from the internal dictionary and displays it to the user.

---

## 🛠 Tech Stack

*   **Language:** Python 3.9+
*   **Framework:** Streamlit (For Web UI)
*   **Machine Learning:** TensorFlow / Keras
*   **Image Processing:** PIL (Python Imaging Library) & NumPy
*   **Training Platform:** Google Teachable Machine
*   **Model Architecture:** MobileNet (Transfer Learning)

---

## 📂 Dataset Details

The model was trained on the **Indian Medicinal Leaf Image Dataset**.
*   **Total Classes:** 30
*   **Images per Class:** ~40-50 high-quality images.
*   **Augmentation:** To improve accuracy, the training process involved random flipping, minor rotations, and noise addition.

**Supported Plants:**
*Alpinia Galanga, Amaranthus Viridis, Artocarpus Heterophyllus, Azadirachta Indica (Neem), Basella Alba, Brassica Juncea, Carissa Carandas, Citrus Limon, Ficus Auriculata, Ficus Religiosa, Hibiscus Rosa-sinensis, Jasminum, Mangifera Indica (Mango), Mentha (Mint), Moringa Oleifera, Muntingia Calabura, Murraya Koenigii (Curry), Nerium Oleander, Nyctanthes Arbor-tristis, Ocimum Tenuiflorum (Tulsi), Piper Betle, Plectranthus Amboinicus, Pongamia Pinnata, Psidium Guajava (Guava), Punica Granatum, Santalum Album, Syzygium Cumini, Syzygium Jambos, Tabernaemontana Divaricata, Trigonella Foenum-graecum.*

---

## 🚀 Installation & Usage

### Prerequisites
Ensure you have Python installed.

### 1. Clone the Repository
```bash
git clone https://github.com/Srilaxman-EU/medicinal-plant-ai.git
cd medicinal-plant-ai
