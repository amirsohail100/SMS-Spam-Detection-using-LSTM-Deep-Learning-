import streamlit as st
import numpy as np
import time
import pickle
import random

# Page configuration - Modern Messaging Theme
st.set_page_config(
    page_title="SecureScan SMS Classifier",
    page_icon="💬",
    layout="centered"
)

# --- TENSORFLOW AUTOMATIC DETECTION ---
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# --- PREMIUM CUSTOM CSS (FIXED TEXT COLOR & VISIBILITY) ---
st.markdown("""
    <style>
    /* Main container background */
    .main {
        background-color: #fafafa;
    }
    /* Textarea styling and text visibility fix */
    .stTextArea textarea {
        border-radius: 14px;
        border: 2px solid #dcdcdc;
        font-size: 16px;
        background-color: #fcfcfc !important;
        color: #1e1e1e !important; /* Isse typed text ekdam saaf dikhega */
    }
    /* Textarea focus glow */
    .stTextArea textarea:focus {
        border-color: #0b132b;
        box-shadow: 0 0 10px rgba(11, 19, 43, 0.1);
    }
    .main-title {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #0b132b;
        font-weight: 800;
        text-align: center;
    }
    .subtitle {
        color: #5c677d;
        text-align: center;
        margin-bottom: 35px;
        font-size: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title & Description
st.markdown("<h1 class='main-title'>💬 SecureScan: Deep Learning SMS Guard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>LSTM-Powered Neural Network for Real-time Message Classification</p>", unsafe_allow_html=True)

# --- 1. SAFE MODEL LOADING ---
@st.cache_resource
def load_sms_assets():
    if TF_AVAILABLE:
        try:
            from tensorflow.keras.models import load_model
            model = load_model('model.h5')
            with open('tokenizer.pkl', 'rb') as handle:
                tokenizer = pickle.load(handle)
            return model, tokenizer
        except Exception as e:
            return None, None
    return None, None

model, tokenizer = load_sms_assets()

# Smart Environment Status Badge
if TF_AVAILABLE and model is not None:
    st.success("⚡ **Production Engine Active:** TensorFlow Neural Network successfully engaged.")
else:
    st.info("✨ **Visual Demo Active:** Running on interface simulation mode (Zero TensorFlow Dependency).")

st.markdown("---")

# --- 2. SINGLE USER INPUT: THE SMS CONTENT ---
st.markdown("### 📥 Paste the SMS Message")
user_sms = st.text_area(
    "", 
    height=150, 
    placeholder="Enter the suspicious SMS text or regular message here to classify..."
)

# Live Statistics Counters
c1, c2 = st.columns(2)
with c1:
    st.caption(f"📊 Words: **{len(user_sms.split()) if user_sms.strip() else 0}**")
with c2:
    st.caption(f"🔤 Characters: **{len(user_sms)}**")

st.markdown("<br>", unsafe_allow_html=True)

# --- 3. CLASSIFICATION & INFERENCE PIPELINE ---
if st.button("🛡️ Audit Message", type="primary", use_container_width=True):
    if user_sms.strip() == "":
        st.warning("⚠️ Kripya analyze karne ke liye pehle koi SMS text type ya paste karein!")
    else:
        with st.spinner("🤖 Extracting token sequences and evaluating temporal dependencies..."):
            time.sleep(1.0) # Elegant processing delay
            
            # --- CASE A: REAL PRODUCTION INFERENCE ---
            if TF_AVAILABLE and model is not None and tokenizer is not None:
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                sequences = tokenizer.texts_to_sequences([user_sms])
                padded_seq = pad_sequences(sequences, maxlen=80, padding='post', truncating='post')
                probability = float(model.predict(padded_seq, verbose=0)[0][0])
            
            # --- CASE B: SIMULATED SIMULATION (Demo Mode) ---
            else:
                spam_keywords = ["win", "free", "money", "click", "prize", "lottery", "cash", "urgent", "subscribe", "offer"]
                if any(word in user_sms.lower() for word in spam_keywords):
                    probability = random.uniform(0.78, 0.99)
                else:
                    probability = random.uniform(0.01, 0.35)
            
            # --- 4. PREMIUM RESULTS PRESENTATION ---
            st.markdown("### 🎯 Threat Intelligence Report")
            metrics_col1, metrics_col2 = st.columns(2)
            
            if probability > 0.5:
                spam_score = probability * 100
                with metrics_col1:
                    st.metric(label="Message Status", value="🚨 SPAM DETECTED")
                with metrics_col2:
                    st.metric(label="Spam Risk Probability", value=f"{spam_score:.2f}%")
                
                st.error("🚫 **Security Alert:** Is message mein suspicious token patterns hain. Iske links par click na karein!")
            else:
                ham_score = (1 - probability) * 100
                with metrics_col1:
                    st.metric(label="Message Status", value="🟢 HAM (SAFE)")
                with metrics_col2:
                    st.metric(label="Safety Confidence", value=f"{ham_score:.2f}%")
                
                st.success("🟢 **Verified Safe:** Yeh message bilkul normal aur safe lag raha hai.")
            
            st.progress(probability)

# Footer
st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a2a8d3; font-size: 13px;'>SecureScan LSTM System | Developed as an Engineering Learning Portfolio Project</p>", unsafe_allow_html=True)