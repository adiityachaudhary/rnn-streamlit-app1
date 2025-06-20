import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load Model and Encoders ---
model = load_model('rnn_model.keras')
MAX_LEN = 30  # same as training

with open('item_encoder.pkl', 'rb') as f:
    item_encoder = pickle.load(f)

with open('cat_encoder.pkl', 'rb') as f:
    cat_encoder = pickle.load(f)

# --- Build Reverse Maps for Dropdowns ---
item_map = {i: label for i, label in enumerate(item_encoder.classes_)}
cat_map = {i: label for i, label in enumerate(cat_encoder.classes_)}

item_name_to_id = {v: k for k, v in item_map.items()}
cat_name_to_id = {v: k for k, v in cat_map.items()}

# --- Streamlit UI ---
st.title("🛍️ Predict User's Next Action")
st.write("Build a session below by selecting item, category and hour:")

if "sequence" not in st.session_state:
    st.session_state.sequence = []

# Input selection
item_name = st.selectbox("Select Item", sorted(item_name_to_id.keys()))
cat_name = st.selectbox("Select Category", sorted(cat_name_to_id.keys()))
hour = st.slider("Select Hour (0-23)", 0, 23)

# Add step
if st.button("➕ Add to Sequence"):
    st.session_state.sequence.append({
        "item": item_name_to_id[item_name],
        "cat": cat_name_to_id[cat_name],
        "hour": hour
    })

# Show session
if st.session_state.sequence:
    st.write("🧾 Current Sequence:")
    st.table(st.session_state.sequence)

    if st.button("🚀 Predict Next Action"):
        item_seq = [x['item'] for x in st.session_state.sequence]
        cat_seq = [x['cat'] for x in st.session_state.sequence]
        hour_seq = [x['hour'] for x in st.session_state.sequence]

        X_item = pad_sequences([item_seq], maxlen=MAX_LEN)
        X_cat = pad_sequences([cat_seq], maxlen=MAX_LEN)
        X_hour = pad_sequences([hour_seq], maxlen=MAX_LEN)

        pred_prob = model.predict([X_item, X_cat, X_hour])[0][0]
        pred_class = int(pred_prob > 0.5)

        st.success(f"📈 Predicted Action: **{'Cart/Purchase (1)' if pred_class == 1 else 'View (0)'}**")
        st.write(f"Prediction Confidence: `{pred_prob:.2f}`")

# Reset
if st.button("🔁 Reset Sequence"):
    st.session_state.sequence = []
