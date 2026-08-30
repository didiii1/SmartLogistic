import streamlit as st
import pickle
import numpy as np

# Konfigurasi Halaman (Harus di baris paling atas)
st.set_page_config(
    page_title="Smart Logistic AI - Dashboard", 
    page_icon="🧊",
    layout="centered"
)

# --- CUSTOM CSS (Clean Glassmorphism) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

/* Main Background (Modern Dark Gradient) */
.stApp {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    background-attachment: fixed;
}

/* Glassmorphism Container */
.block-container {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 24px !important;
    padding: 3rem !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    margin-top: 3rem !important;
}

/* Customizing Button */
.stButton > button {
    background: linear-gradient(135deg, #00ffcc 0%, #00b386 100%) !important;
    color: #0f2027 !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(0, 255, 204, 0.3) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 255, 204, 0.6) !important;
}

/* Alerts / Boxes Glassmorphism */
.stAlert {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 15px !important;
}

/* Horizontal line */
hr {
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
}
</style>
""", unsafe_allow_html=True)

# Load Model AI
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Header Dashboard
st.markdown("<h1 style='text-align: center; color: #00ffcc; font-weight: 800; letter-spacing: 2px;'>SMART LOGISTIC AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e; font-weight: 300;'>Sistem Prediksi & Mitigasi Risiko Keterlambatan Pengiriman Berbasis Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")

# Layout Form Menggunakan Kolom Ala Dashboard Korporat
st.markdown("<h3 style='color: #e0e0e0; font-weight: 600;'>Panel Parameter Pengiriman</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    warehouse_label = st.selectbox("Blok Gudang (Warehouse)", ["Blok A", "Blok B", "Blok C", "Blok D", "Blok F"])
    warehouse_map = {"Blok A": 0, "Blok B": 1, "Blok C": 2, "Blok D": 3, "Blok F": 4}
    warehouse = warehouse_map[warehouse_label]

    shipment_label = st.selectbox("Metode Pengiriman", ["Pesawat (Flight)", "Kapal Laut (Ship)", "Truk (Road)"])
    shipment_map = {"Pesawat (Flight)": 0, "Kapal Laut (Ship)": 1, "Truk (Road)": 2}
    shipment = shipment_map[shipment_label]

    calls = st.slider("Jumlah Kontak Komplain (Customer Care Calls)", 2, 7, 4)
    rating = st.slider("Rating Kepuasan Pelanggan", 1, 5, 3)

with col2:
    # Rentang disesuaikan dengan data training asli (model tidak bisa ekstrapolasi
    # dengan andal di luar rentang ini -- lihat catatan di bawah form).
    cost = st.number_input("Harga Produk ($)", min_value=96, max_value=310, value=214)
    prior_purchases = st.number_input("Riwayat Pembelian Sebelumnya", min_value=2, max_value=10, value=3)
    discount = st.number_input("Besaran Diskon Promo (%)", min_value=1, max_value=65, value=7)
    weight = st.number_input("Berat Paket (Gram)", min_value=1001, max_value=7846, value=4149)

st.caption(
    "Rentang input di atas dibatasi sesuai data yang dipelajari model saat training. "
    "Nilai di luar rentang ini tidak pernah dilihat model, sehingga hasil prediksinya tidak bisa diandalkan."
)

st.markdown("---")

# Tombol Prediksi
if st.button("ANALISIS RISIKO PENGIRIMAN", use_container_width=True):
    # Menyusun data sesuai urutan fitur saat training:
    # ['Warehouse_block', 'Mode_of_Shipment', 'Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product', 'Prior_purchases', 'Discount_offered', 'Weight_in_gms']
    fitur_input = np.array([[warehouse, shipment, calls, rating, cost, prior_purchases, discount, weight]])
    
    # Menggunakan Probabilitas agar lebih seimbang (Threshold 50%)
    probabilitas = model.predict_proba(fitur_input)
    prob_terlambat = probabilitas[0][1] # Persentase kemungkinan terlambat
    
    # Menampilkan Hasil Interaktif
    st.markdown("<h3 style='color: #00ffcc; font-weight: 600;'>Hasil Analisis AI</h3>", unsafe_allow_html=True)
    
    # Jika probabilitas terlambat di atas 50% baru nyatakan terlambat
    if prob_terlambat > 0.50:
        st.error(f"**STATUS: BERISIKO TERLAMBAT!**\n\nTingkat Keyakinan Model: **{prob_terlambat*100:.1f}%**")
        st.info("**Rekomendasi Tindakan:** Alihkan pengiriman prioritas atau berikan notifikasi antisipasi dini kepada pelanggan.")
    else:
        st.success(f"**STATUS: AMAN (TEPAT WAKTU)**\n\nTingkat Keyakinan Model Tepat Waktu: **{(1-prob_terlambat)*100:.1f}%**")

    # Peringatan kalau model sedang "ragu-ragu" (probabilitas dekat 50%)
    if 0.40 <= prob_terlambat <= 0.60:
        st.warning(
            "Probabilitas mendekati 50:50 — model tidak terlalu yakin pada kombinasi input ini. "
            "Anggap hasil ini sebagai indikasi awal, bukan keputusan final."
        )

    # Peringatan khusus: berdasarkan analisis data, Discount_offered <= 10%
    # secara statistik adalah zona yang nyaris tidak informatif (~50:50 di data asli),
    # sehingga keyakinan tinggi pada rentang ini patut dicurigai overconfident.
    if discount <= 10:
        st.info(
            "**Catatan Analisis:** Pada rentang diskon ≤10%, data historis menunjukkan hasil "
            "hampir 50:50 (tidak terlalu informatif untuk membedakan Terlambat/Tepat Waktu). "
            "Jika model menampilkan keyakinan tinggi di rentang ini, anggap hasil dengan hati-hati "
            "dan pertimbangkan faktor lain di luar model."
        )