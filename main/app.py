import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Sistem Seleksi Beasiswa",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #3B82F6;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .prediction-accepted {
        background-color: #D1FAE5;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #10B981;
        text-align: center;
    }
    .prediction-rejected {
        background-color: #FEE2E2;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #EF4444;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ========== FUNGSI VISUALISASI ==========
def plot_logistic_regression_explanation():
    """Plot visual explanation of Logistic Regression as a classifier"""
    
    fig = plt.figure(figsize=(15, 10))
    
    # Create 3 subplots
    gs = fig.add_gridspec(2, 2)
    
    # Plot 1: Sigmoid Function
    ax1 = fig.add_subplot(gs[0, 0])
    z = np.linspace(-10, 10, 100)
    sigmoid = 1 / (1 + np.exp(-z))
    
    ax1.plot(z, sigmoid, 'b-', linewidth=3, label='Sigmoid Function')
    ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
    ax1.axvline(x=0, color='g', linestyle='--', linewidth=2, label='Decision Boundary (z=0)')
    
    # Color areas
    ax1.fill_between(z, 0, 0.5, where=(sigmoid < 0.5), alpha=0.2, color='red', label='Kelas 0: Tidak Diterima')
    ax1.fill_between(z, 0.5, 1, where=(sigmoid >= 0.5), alpha=0.2, color='green', label='Kelas 1: Diterima')
    
    ax1.set_xlabel('z = β₀ + β₁X₁ + β₂X₂ + ...', fontsize=12)
    ax1.set_ylabel('Probability P(Y=1)', fontsize=12)
    ax1.set_title('① Sigmoid Function: Transformasi Linear ke Probability', fontsize=14, fontweight='bold')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Decision Boundary Example
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
    # Class 0 (Red): IPK rendah, pendapatan tinggi
    class0_ipk = np.random.normal(2.5, 0.5, n_samples)
    class0_income = np.random.normal(25, 5, n_samples)
    
    # Class 1 (Green): IPK tinggi, pendapatan rendah
    class1_ipk = np.random.normal(3.5, 0.5, n_samples)
    class1_income = np.random.normal(10, 5, n_samples)
    
    ax2.scatter(class0_ipk, class0_income, color='red', alpha=0.6, label='Tidak Diterima (0)', s=50)
    ax2.scatter(class1_ipk, class1_income, color='green', alpha=0.6, label='Diterima (1)', s=50)
    
    # Add decision boundary line (example)
    x_boundary = np.linspace(1.5, 4.5, 100)
    y_boundary = 40 - 10 * x_boundary  # Example boundary
    ax2.plot(x_boundary, y_boundary, 'k--', linewidth=3, label='Decision Boundary')
    
    ax2.set_xlabel('IPK', fontsize=12)
    ax2.set_ylabel('Pendapatan Orang Tua (juta)', fontsize=12)
    ax2.set_title('② Decision Boundary: Memisahkan 2 Kelas', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Probability Distribution
    ax3 = fig.add_subplot(gs[1, :])
    
    # Generate probability distributions
    probabilities_0 = np.random.beta(2, 5, 500)  # Most probabilities low
    probabilities_1 = np.random.beta(5, 2, 500)  # Most probabilities high
    
    bins = np.linspace(0, 1, 30)
    ax3.hist(probabilities_0, bins=bins, alpha=0.6, color='red', label='Aktual: Tidak Diterima', density=True)
    ax3.hist(probabilities_1, bins=bins, alpha=0.6, color='green', label='Aktual: Diterima', density=True)
    
    ax3.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold = 0.5')
    
    ax3.set_xlabel('Predicted Probability', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('③ Probability Output: Prediksi 0-1', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_linear_vs_logistic_comparison():
    """Comparison between Linear and Logistic Regression"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Generate data
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    
    # Linear Regression
    y_linear = 2 * X + 3 + np.random.normal(0, 2, 100)
    
    # Logistic Regression (binary)
    y_logistic_prob = 1 / (1 + np.exp(-(0.5 * X - 2.5)))
    y_logistic_class = (y_logistic_prob > 0.5).astype(int)
    
    # Plot Linear Regression
    ax1.scatter(X, y_linear, alpha=0.6, color='blue')
    ax1.plot(X, 2 * X + 3, 'r-', linewidth=3, label='Regression Line')
    ax1.set_xlabel('Feature X')
    ax1.set_ylabel('Continuous Target Y')
    ax1.set_title('LINEAR REGRESSION\n(Prediksi Nilai Kontinu)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Logistic Regression
    colors = ['red' if c == 0 else 'green' for c in y_logistic_class]
    ax2.scatter(X, y_logistic_prob, c=colors, alpha=0.6, s=50)
    ax2.plot(X, y_logistic_prob, 'b-', linewidth=3, label='Sigmoid Curve')
    ax2.axhline(y=0.5, color='k', linestyle='--', label='Decision Threshold')
    
    # Annotate classes
    ax2.text(8, 0.2, 'Kelas 0\n(Tidak Diterima)', 
             fontsize=10, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.3))
    ax2.text(8, 0.8, 'Kelas 1\n(Diterima)', 
             fontsize=10, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.3))
    
    ax2.set_xlabel('Feature X (contoh: IPK)')
    ax2.set_ylabel('Probability P(Y=1)')
    ax2.set_title('LOGISTIC REGRESSION\n(Prediksi Probability & Klasifikasi)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_comprehensive_sigmoid():
    """Plot comprehensive sigmoid function with annotations"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Sigmoid Function
    z = np.linspace(-10, 10, 100)
    sigmoid = 1 / (1 + np.exp(-z))
    
    ax1.plot(z, sigmoid, 'b-', linewidth=3)
    ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=2)
    ax1.axvline(x=0, color='g', linestyle='--', linewidth=2)
    
    # Color areas
    ax1.fill_between(z, 0, 0.5, where=(sigmoid < 0.5), alpha=0.2, color='red')
    ax1.fill_between(z, 0.5, 1, where=(sigmoid >= 0.5), alpha=0.2, color='green')
    
    # Annotations
    ax1.annotate('Tidak Diterima', xy=(-5, 0.2), xytext=(-8, 0.3),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=12, color='red')
    
    ax1.annotate('Diterima', xy=(5, 0.8), xytext=(3, 0.9),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=12, color='green')
    
    ax1.set_xlabel('Linear Score (z)', fontsize=12)
    ax1.set_ylabel('Probability P(Y=1)', fontsize=12)
    ax1.set_title('Sigmoid Function: Transformasi Linear ke Probability', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Example for beasiswa
    ax2.clear()
    
    # Example scores for different applicants
    scores = np.array([-3, -1.5, -0.5, 0, 0.5, 1.5, 3])
    labels = ['Sangat Rendah', 'Rendah', 'Agak Rendah', 'Netral', 'Agak Tinggi', 'Tinggi', 'Sangat Tinggi']
    probabilities = 1 / (1 + np.exp(-scores))
    
    colors = ['darkred', 'red', 'orange', 'yellow', 'lightgreen', 'green', 'darkgreen']
    
    bars = ax2.barh(labels, probabilities, color=colors)
    ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
    
    # Add probability values
    for i, (prob, bar) in enumerate(zip(probabilities, bars)):
        ax2.text(prob + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1%}', va='center', fontweight='bold')
    
    ax2.set_xlabel('Probability Diterima', fontsize=12)
    ax2.set_title('Contoh: Score vs Probability untuk Beasiswa', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig

# ========== FUNGSI UTAMA ==========
def load_data():
    """Load and preprocess the data"""
    try:
        # Load the data
        df = pd.read_csv('beasiswa_800.csv')
        
        # Data preprocessing
        # Handle categorical variables
        categorical_cols = ['Asal_Sekolah', 'Lokasi_Domisili', 'Gender', 'Status_Disabilitas']
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        # Feature and target separation
        X = df.drop('Diterima_Beasiswa', axis=1)
        y = df['Diterima_Beasiswa']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_cols = ['IPK', 'Pendapatan_Orang_Tua', 'Keikutsertaan_Organisasi', 
                         'Pengalaman_Sosial', 'Prestasi_Akademik', 'Prestasi_Non_Akademik']
        
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        return df, X_train, X_test, y_train, y_test, scaler, label_encoders
        
    except FileNotFoundError:
        st.error("File 'beasiswa_800.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        return None, None, None, None, None, None, None

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train and evaluate Logistic Regression CLASSIFICATION model"""
    
    st.info("""
    🎯 **LOGISTIC REGRESSION ADALAH MODEL KLASIFIKASI**
    
    Meski namanya mengandung "regression", ini adalah algoritma **klasifikasi biner** 
    yang memprediksi probability (0-1) suatu instance masuk ke kelas tertentu.
    """)
    
    # Show explanation
    with st.expander("📚 Klik untuk lihat penjelasan visual Logistic Regression"):
        fig = plot_logistic_regression_explanation()
        st.pyplot(fig)
    
    # Train the CLASSIFICATION model
    model = LogisticRegression(
        random_state=42, 
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced'  # Handle imbalanced data
    )
    
    # Training phase
    with st.spinner("🔧 Melatih model klasifikasi Logistic Regression..."):
        progress_bar = st.progress(0)
        model.fit(X_train, y_train)
        progress_bar.progress(100)
    
    st.success("✅ Model Logistic Regression berhasil dilatih!")
    
    # Make CLASSIFICATIONS
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return model, y_pred, y_pred_proba, accuracy, precision, recall, f1

# ========== MAIN FUNCTION ==========
def main():
    # Header
    st.markdown('<h1 class="main-header">Sistem Seleksi Beasiswa dengan Machine Learning</h1>', unsafe_allow_html=True)
    st.markdown("**Universitas X - Beasiswa Generasi Unggul Nusantara**")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/student-center.png", width=100)
        st.title("Navigasi")
        menu = st.radio(
            "Pilih Menu:",
            ["📊 Eksplorasi Data", 
             "📚 Logistic Regression",
             "🤖 Model Machine Learning", 
             "📈 Evaluasi Model", 
             "🔮 Prediksi Baru"]
        )
        
        st.markdown("---")
        st.info(
            "Tugas Pembelajaran Mesin\n\n"
            "M. Riansyah Lubis - 072925002\n"
            "Sumiati - 072925003"
        )
    
    # Load data
    df, X_train, X_test, y_train, y_test, scaler, label_encoders = load_data()
    
    if df is None:
        return
    
    # Menu 1: Data Exploration
    if menu == "📊 Eksplorasi Data":
        st.markdown('<h2 class="sub-header">📊 Eksplorasi Data Pendaftar Beasiswa</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Preview Data")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.subheader("Statistik Data")
            st.metric("Total Pendaftar", len(df))
            st.metric("Diterima", df['Diterima_Beasiswa'].sum())
            st.metric("Tidak Diterima", len(df) - df['Diterima_Beasiswa'].sum())
            st.metric("Persentase Diterima", f"{(df['Diterima_Beasiswa'].sum()/len(df)*100):.1f}%")
        
        # Data Description
        st.subheader("Deskripsi Data")
        with st.expander("Lihat Deskripsi Statistik"):
            st.dataframe(df.describe(), use_container_width=True)
        
        # Visualizations
        st.subheader("Visualisasi Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            df['Diterima_Beasiswa'].value_counts().plot(kind='bar', ax=ax, color=['#EF4444', '#10B981'])
            ax.set_title('Distribusi Status Penerimaan Beasiswa')
            ax.set_xlabel('Status Penerimaan')
            ax.set_ylabel('Jumlah')
            ax.set_xticklabels(['Tidak Diterima', 'Diterima'], rotation=0)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            df['IPK'].hist(bins=20, ax=ax, color='#3B82F6', alpha=0.7)
            ax.set_title('Distribusi IPK Pendaftar')
            ax.set_xlabel('IPK')
            ax.set_ylabel('Frekuensi')
            st.pyplot(fig)
        
        # Correlation matrix
        st.subheader("Korelasi antar Fitur")
        fig, ax = plt.subplots(figsize=(12, 8))
        numeric_df = df.select_dtypes(include=[np.number])
        correlation = numeric_df.corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Matriks Korelasi')
        st.pyplot(fig)
    
    # Menu 2: Logistic Regression 101
    elif menu == "📚 Logistic Regression":
        st.markdown('<h1 class="main-header">📚 Logistic Regression</h1>', unsafe_allow_html=True)
        st.markdown("### **Memahami Logistic Regression sebagai Model Klasifikasi**")
        
        # Tab untuk penjelasan
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Konsep Dasar", 
            "📈 Visualisasi", 
            "⚖️ Decision Boundary", 
            "📊 Aplikasi di Beasiswa"
        ])
        
        with tab1:
            st.markdown("### **🎯 Konsep Dasar Logistic Regression**")
            
            # Two columns for better layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                #### **Apa itu Logistic Regression?**
                
                Meski namanya "regression", ini adalah algoritma **klasifikasi biner**:
                
                **Tahapan:**
                1. Hitung **score linear**: z = β₀ + β₁X₁ + β₂X₂ + ...
                2. Transform ke **probability**: P = 1 / (1 + e^(-z))
                3. Bandingkan dengan **threshold** (biasanya 0.5)
                4. Hasil akhir: **Kelas 0 atau 1**
                
                #### **Contoh di Seleksi Beasiswa:**
                
                **Fitur Input:**
                - IPK: 3.5 → (+) meningkatkan peluang
                - Pendapatan: 15 juta → (-) mengurangi peluang
                - Prestasi: 5 → (+) meningkatkan peluang
                
                **Proses:**
                ```
                z = (1.2 × 3.5) + (-0.08 × 15) + (0.5 × 5) - 2.5
                z = 1.7
                
                P = 1 / (1 + e^(-1.7))
                P = 0.85 (85% probability diterima)
                
                Karena P > 0.5 → KEPUTUSAN: DITERIMA
                ```
                """)
            
            with col2:
                # Tampilkan visualisasi comprehensive
                st.markdown("#### **📈 Visualisasi Sigmoid Function**")
                
                # Interactive controls untuk score
                score_input = st.slider(
                    "Coba Score Linear (z):",
                    min_value=-5.0,
                    max_value=5.0,
                    value=0.0,
                    step=0.1,
                    help="Score dari kombinasi linear fitur"
                )
                
                # Calculate probability
                probability = 1 / (1 + np.exp(-score_input))
                
                # Create plot sederhana untuk score yang dipilih
                fig_simple, ax_simple = plt.subplots(figsize=(8, 5))
                
                # Sigmoid curve
                z_curve = np.linspace(-5, 5, 100)
                p_curve = 1 / (1 + np.exp(-z_curve))
                ax_simple.plot(z_curve, p_curve, 'b-', linewidth=3, alpha=0.7, label='Sigmoid Function')
                
                # Current point
                ax_simple.plot(score_input, probability, 'ro', markersize=10, 
                              label=f'Score={score_input:.1f}, P={probability:.2f}')
                ax_simple.vlines(score_input, 0, probability, colors='r', linestyles='dashed', alpha=0.5)
                ax_simple.hlines(probability, -5, score_input, colors='r', linestyles='dashed', alpha=0.5)
                
                # Threshold
                ax_simple.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='Threshold (0.5)')
                ax_simple.axvline(x=0, color='orange', linestyle='--', linewidth=2, label='Decision Boundary')
                
                # Color regions
                ax_simple.fill_between(z_curve, 0, 0.5, where=(p_curve < 0.5), 
                                      alpha=0.2, color='red', label='Tidak Diterima')
                ax_simple.fill_between(z_curve, 0.5, 1, where=(p_curve >= 0.5), 
                                      alpha=0.2, color='green', label='Diterima')
                
                # Labels
                ax_simple.set_xlabel('Score Linear (z)', fontsize=11)
                ax_simple.set_ylabel('Probability P(Y=1)', fontsize=11)
                ax_simple.set_title('Sigmoid Function dengan Score Interaktif', fontsize=13, fontweight='bold')
                ax_simple.legend(loc='upper left', fontsize=9)
                ax_simple.grid(True, alpha=0.3)
                
                st.pyplot(fig_simple)
                
                # Result display
                if probability >= 0.5:
                    st.success(f"**Probability: {probability:.1%}** → **DITERIMA** 🎉")
                else:
                    st.error(f"**Probability: {probability:.1%}** → **TIDAK DITERIMA** ❌")
            
            # Tambahkan visualisasi comprehensive di bawah
            st.markdown("---")
            st.markdown("#### **📊 Visualisasi Lengkap Sigmoid & Contoh Beasiswa**")
            
            # Tampilkan comprehensive plot
            fig_comprehensive = plot_comprehensive_sigmoid()
            st.pyplot(fig_comprehensive)
            
            # Penjelasan
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                st.markdown("""
                **Grafik Kiri (Sigmoid Function):**
                - **Garis Biru**: Kurva sigmoid
                - **Garis Merah**: Threshold (0.5)
                - **Garis Hijau**: Decision boundary (z=0)
                - **Area Merah**: Probability < 0.5 → Tidak Diterima
                - **Area Hijau**: Probability ≥ 0.5 → Diterima
                """)
            
            with col_exp2:
                st.markdown("""
                **Grafik Kanan (Contoh Beasiswa):**
                - **Bar Horizontal**: Berbagai level score
                - **Warna**: Merah → Hijau (Rendah → Tinggi)
                - **Angka**: Probability untuk setiap score
                - **Garis Hitam**: Threshold 0.5
                
                **Interpretasi:**
                - Score "Sangat Rendah" (-3) → 4.7% → Tidak Diterima
                - Score "Tinggi" (1.5) → 81.8% → Diterima
                """)
        
        with tab2:
            st.markdown("### **📈 Visualisasi: Linear vs Logistic Regression**")
            
            # Comparison plot
            fig = plot_linear_vs_logistic_comparison()
            st.pyplot(fig)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### **Linear Regression:**
                - **Output**: Nilai kontinu
                - **Contoh**: Prediksi harga rumah, temperatur
                - **Grafik**: Garis lurus
                - **Equation**: y = β₀ + β₁X
                """)
            
            with col2:
                st.markdown("""
                ### **Logistic Regression:**
                - **Output**: Probability (0-1)
                - **Contoh**: Diterima/Tidak, Spam/Not Spam
                - **Grafik**: Kurva S (sigmoid)
                - **Equation**: P = 1/(1 + e^(-z))
                """)
        
        with tab3:
            st.markdown("### **⚖️ Decision Boundary & Threshold**")
            
            # Interactive threshold slider
            threshold = st.slider(
                "Atur Decision Threshold:",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05,
                help="Threshold untuk menentukan kapan prediksi menjadi 'Diterima'"
            )
            
            # Plot with interactive threshold
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Generate sigmoid curve
            z = np.linspace(-10, 10, 100)
            sigmoid = 1 / (1 + np.exp(-z))
            
            ax.plot(z, sigmoid, 'b-', linewidth=3, label='Sigmoid Function')
            ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, 
                      label=f'Decision Threshold ({threshold})')
            
            # Find corresponding z value for threshold
            z_threshold = -np.log((1/threshold) - 1)
            ax.axvline(x=z_threshold, color='g', linestyle='--', linewidth=2,
                      label=f'Decision Boundary (z={z_threshold:.2f})')
            
            # Color areas
            ax.fill_between(z, 0, threshold, where=(sigmoid < threshold), 
                           alpha=0.2, color='red', label='Kelas 0: Tidak Diterima')
            ax.fill_between(z, threshold, 1, where=(sigmoid >= threshold), 
                           alpha=0.2, color='green', label='Kelas 1: Diterima')
            
            ax.set_xlabel('z = β₀ + β₁X₁ + β₂X₂ + ... (Linear Score)', fontsize=12)
            ax.set_ylabel('Probability P(Y=1)', fontsize=12)
            ax.set_title(f'Decision Boundary dengan Threshold = {threshold}', fontsize=14, fontweight='bold')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            st.markdown(f"""
            ### **Interpretasi:**
            
            Dengan threshold = **{threshold}**:
            
            - **Score (z) > {z_threshold:.2f}** → Probability > {threshold} → **DITERIMA**
            - **Score (z) < {z_threshold:.2f}** → Probability < {threshold} → **TIDAK DITERIMA**
            
            **Note:** 
            - Threshold lebih tinggi (0.7) = lebih strict, lebih sedikit yang diterima
            - Threshold lebih rendah (0.3) = lebih lenient, lebih banyak yang diterima
            """)
        
        with tab4:
            st.markdown("### **📊 Aplikasi Logistic Regression di Seleksi Beasiswa**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### **Input Features:**
                
                **Data Pelamar:**
                1. IPK (3.5) → **Pengaruh positif**
                2. Pendapatan Orang Tua (15 juta) → **Pengaruh negatif**
                3. Prestasi Akademik (5) → **Pengaruh positif**
                4. Pengalaman Sosial (150 jam) → **Pengaruh positif**
                5. Keikutsertaan Organisasi (2) → **Pengaruh positif**
                
                ### **Proses Klasifikasi:**
                
                1. **Hitung Score:**
                ```
                z = \beta_{\text{IPK}} \times 3.5 + \beta_{\text{Pendapatan}} \times 15 + \dots + \beta_0
                ```
                
                2. **Transform ke Probability:**
                ```
                P = 1 / (1 + e^(-z))
                ```
                
                3. **Bandingkan dengan Threshold:**
                ```
                Jika P ≥ 0.5 → "DITERIMA"
                Jika P < 0.5 → "TIDAK DITERIMA"
                ```
                """)
            
            with col2:
                # Create example calculation
                st.markdown("### **Contoh Perhitungan:**")
                
                # Input for example
                st.markdown("**Contoh Data Pelamar:**")
                ipk_ex = st.number_input("IPK", 0.0, 4.0, 3.5, 0.1, key="ex_ipk")
                income_ex = st.number_input("Pendapatan (juta)", 0, 50, 15, key="ex_income")
                
                # Example coefficients
                coeff_ipk = 1.2
                coeff_income = -0.08
                intercept = -2.5
                
                # Calculate
                score = (coeff_ipk * ipk_ex) + (coeff_income * income_ex) + intercept
                probability = 1 / (1 + np.exp(-score))
                
                # Display result
                st.markdown(f"""
                **Perhitungan:**
                ```
                z = (1.2 × {ipk_ex}) + (-0.08 × {income_ex}) - 2.5
                z = {score:.2f}
                
                P = 1 / (1 + e^(-{score:.2f}))
                P = {probability:.2f}
                ```
                
                **Hasil:**
                - Probability: **{probability:.1%}**
                - Threshold: 50%
                - Keputusan: **{"DITERIMA 🎉" if probability >= 0.5 else "TIDAK DITERIMA ❌"}**
                """)
                
                # Visual gauge
                st.markdown("**Visualisasi Probability:**")
                gauge_value = probability
                st.progress(float(gauge_value))
                
                if gauge_value >= 0.5:
                    st.success(f"✅ {gauge_value:.1%} → DITERIMA")
                else:
                    st.error(f"❌ {gauge_value:.1%} → TIDAK DITERIMA")
    
    # Menu 3: Machine Learning Model
    elif menu == "🤖 Model Machine Learning":
        st.markdown('<h2 class="sub-header">🤖 Model Logistic Regression</h2>', unsafe_allow_html=True)
        
        st.info("**Logistic Regression** adalah model **klasifikasi biner** yang digunakan untuk masalah klasifikasi. Model ini memprediksi probabilitas bahwa suatu instance masuk ke kelas tertentu.")
        
        # Train model
        model, y_pred, y_pred_proba, accuracy, precision, recall, f1 = train_logistic_regression(
            X_train, X_test, y_train, y_test
        )
        
        # Display model coefficients
        st.subheader("Koefisien Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature importance
            feature_names = X_train.columns
            coefficients = model.coef_[0]
            
            importance_df = pd.DataFrame({
                'Fitur': feature_names,
                'Koefisien': coefficients,
                'Pengaruh': ['Positif' if x > 0 else 'Negatif' for x in coefficients]
            })
            importance_df = importance_df.sort_values('Koefisien', ascending=False)
            
            st.dataframe(importance_df, use_container_width=True)
        
        with col2:
            st.subheader("Interpretasi Koefisien")
            st.markdown("""
            - **Koefisien Positif**: Meningkatkan kemungkinan diterima
            - **Koefisien Negatif**: Menurunkan kemungkinan diterima
            
            Contoh:
            - IPK yang lebih tinggi → koefisien positif → meningkatkan peluang diterima
            - Pendapatan orang tua tinggi → koefisien negatif → menurunkan peluang diterima
            """)
        
        # Model details
        with st.expander("Detail Teknis Model"):
            st.code(f"""
            Parameter Model:
            - Random State: 42
            - Max Iterations: 1000
            - Solver: lbfgs
            - C (Regularization): 1.0
            
            Ukuran Data:
            - Training: {len(X_train)} sampel
            - Testing: {len(X_test)} sampel
            """)
    
    # Menu 4: Model Evaluation
    elif menu == "📈 Evaluasi Model":
        st.markdown('<h2 class="sub-header">📈 Evaluasi Performa Model</h2>', unsafe_allow_html=True)
        
        # Train model
        model, y_pred, y_pred_proba, accuracy, precision, recall, f1 = train_logistic_regression(
            X_train, X_test, y_train, y_test
        )
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Accuracy", f"{accuracy:.2%}")
            st.caption("Proporsi prediksi benar dari total prediksi")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Precision", f"{precision:.2%}")
            st.caption("Proporsi prediksi diterima yang benar-benar diterima")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Recall", f"{recall:.2%}")
            st.caption("Proporsi yang diterima berhasil diprediksi")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("F1-Score", f"{f1:.2%}")
            st.caption("Harmonic mean dari precision dan recall")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        col1, col2 = st.columns([3, 2])
        
        with col1:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Prediksi Tidak', 'Prediksi Ya'],
                       yticklabels=['Aktual Tidak', 'Aktual Ya'])
            ax.set_title('Confusion Matrix')
            ax.set_ylabel('Aktual')
            ax.set_xlabel('Prediksi')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Interpretasi")
            st.markdown(f"""
            **True Negative (TN):** {cm[0,0]}
            - Diprediksi tidak diterima dan benar
            
            **False Positive (FP):** {cm[0,1]}
            - Diprediksi diterima tapi sebenarnya tidak
            
            **False Negative (FN):** {cm[1,0]}
            - Diprediksi tidak diterima tapi sebenarnya diterima
            
            **True Positive (TP):** {cm[1,1]}
            - Diprediksi diterima dan benar
            """)
        
        # Classification Report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        # ROC Curve
        st.subheader("ROC Curve dan Probabilitas")
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # ROC Curve
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend(loc="lower right")
        ax1.grid(True)
        
        # Probability Distribution
        ax2.hist(y_pred_proba[y_test == 0], bins=20, alpha=0.5, label='Tidak Diterima', color='red')
        ax2.hist(y_pred_proba[y_test == 1], bins=20, alpha=0.5, label='Diterima', color='green')
        ax2.set_xlabel('Probability of Acceptance')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Predicted Probabilities')
        ax2.legend()
        ax2.grid(True)
        
        st.pyplot(fig)
    
    # Menu 5: New Prediction
    elif menu == "🔮 Prediksi Baru":
        st.markdown('<h2 class="sub-header">🔮 Prediksi Penerimaan Beasiswa Baru</h2>', unsafe_allow_html=True)
        
        # Input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Data Akademik")
                ipk = st.slider("IPK (0-4)", 0.0, 4.0, 3.0, 0.01)
                prestasi_akademik = st.number_input("Jumlah Prestasi Akademik", 0, 20, 5)
                prestasi_non_akademik = st.number_input("Jumlah Prestasi Non-Akademik", 0, 20, 3)
                keikutsertaan_organisasi = st.number_input("Jumlah Organisasi Aktif", 0, 10, 2)
            
            with col2:
                st.subheader("Data Pribadi")
                pendapatan_orang_tua = st.number_input("Pendapatan Orang Tua (juta)", 0, 50, 15)
                pengalaman_sosial = st.number_input("Pengalaman Sosial (jam)", 0, 500, 150)
                gender = st.selectbox("Gender", ["L", "P"])
                status_disabilitas = st.selectbox("Status Disabilitas", ["Tidak", "Ya"])
                tahun_pendaftaran = st.selectbox("Tahun Pendaftaran", [2024, 2023, 2022, 2021, 2020])
                asal_sekolah = st.selectbox("Asal Sekolah", 
                    ["Negeri_Kota", "Negeri_Desa", "Swasta_Kota", "Swasta_Desa"])
                lokasi_domisili = st.text_input("Lokasi Domisili", "Kota Bandung")
            
            submitted = st.form_submit_button("🚀 Prediksi Penerimaan")
        
        if submitted:
            # Train model
            model, _, _, _, _, _, _ = train_logistic_regression(
                X_train, X_test, y_train, y_test
            )
            
            # Prepare input data
            input_data = {
                'Tahun_Pendaftaran': tahun_pendaftaran,
                'IPK': ipk,
                'Pendapatan_Orang_Tua': pendapatan_orang_tua,
                'Asal_Sekolah': asal_sekolah,
                'Lokasi_Domisili': lokasi_domisili,
                'Keikutsertaan_Organisasi': keikutsertaan_organisasi,
                'Pengalaman_Sosial': pengalaman_sosial,
                'Gender': gender,
                'Status_Disabilitas': status_disabilitas,
                'Prestasi_Akademik': prestasi_akademik,
                'Prestasi_Non_Akademik': prestasi_non_akademik
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            for col in ['Asal_Sekolah', 'Lokasi_Domisili', 'Gender', 'Status_Disabilitas']:
                if input_df[col].iloc[0] in label_encoders[col].classes_:
                    input_df[col] = label_encoders[col].transform([input_df[col].iloc[0]])
                else:
                    # If new category, use most frequent
                    input_df[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])
            
            # Scale numerical features
            numerical_cols = ['IPK', 'Pendapatan_Orang_Tua', 'Keikutsertaan_Organisasi', 
                             'Pengalaman_Sosial', 'Prestasi_Akademik', 'Prestasi_Non_Akademik']
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            # Display result
            st.markdown("---")
            
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-accepted">
                    <h2>🎉 SELAMAT! DIPREDIKSI DITERIMA</h2>
                    <h3>Probabilitas Penerimaan: {probability:.1%}</h3>
                    <p>Berdasarkan analisis model, calon ini memiliki karakteristik yang sesuai dengan penerima beasiswa sebelumnya.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show influencing factors
                st.subheader("Faktor Pendukung:")
                coefficients = model.coef_[0]
                feature_names = X_train.columns
                
                for feature, coef in zip(feature_names, coefficients):
                    if coef > 0 and feature in input_data:
                        st.write(f"✓ **{feature}**: {input_data[feature]} (Meningkatkan peluang)")
                
            else:
                st.markdown(f"""
                <div class="prediction-rejected">
                    <h2>⚠️ DIPREDIKSI TIDAK DITERIMA</h2>
                    <h3>Probabilitas Penerimaan: {probability:.1%}</h3>
                    <p>Berdasarkan analisis model, calon ini memerlukan perbaikan pada beberapa aspek untuk meningkatkan peluang penerimaan.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show improvement suggestions
                st.subheader("Saran Perbaikan:")
                
                suggestions = {
                    'IPK': 'Tingkatkan IPK di atas 3.0',
                    'Prestasi_Akademik': 'Tambah prestasi akademik',
                    'Prestasi_Non_Akademik': 'Ikuti lebih banyak kegiatan non-akademik',
                    'Keikutsertaan_Organisasi': 'Aktif dalam organisasi',
                    'Pengalaman_Sosial': 'Tambah pengalaman sosial/kemasyarakatan'
                }
                
                for feature, suggestion in suggestions.items():
                    if feature in input_data and input_data[feature] < df[feature].median():
                        st.write(f"• **{suggestion}**")
            
            # Show probability breakdown
            with st.expander("📊 Detail Probabilitas"):
                st.write(f"**Probabilitas Diterima:** {probability:.2%}")
                st.write(f"**Probabilitas Tidak Diterima:** {1-probability:.2%}")
                st.progress(float(probability))
                
                if probability > 0.7:
                    st.success("Peluang sangat tinggi (≥70%)")
                elif probability > 0.5:
                    st.warning("Peluang sedang (50-70%)")
                else:
                    st.error("Peluang rendah (<50%)")

if __name__ == "__main__":
    main()