import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import warnings
import io
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Sistem Seleksi Beasiswa",
    layout="wide"
)

# ========== FUNGSI VISUALISASI ==========
def plot_logistic_regression_explanation():
    """Plot visual explanation of Logistic Regression"""
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Plot 1: Sigmoid Function
    ax1 = fig.add_subplot(gs[0, 0])
    z = np.linspace(-10, 10, 100)
    sigmoid = 1 / (1 + np.exp(-z))
    
    ax1.plot(z, sigmoid, 'b-', linewidth=3)
    ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=2)
    ax1.axvline(x=0, color='g', linestyle='--', linewidth=2)
    ax1.fill_between(z, 0, 0.5, where=(sigmoid < 0.5), alpha=0.2, color='red')
    ax1.fill_between(z, 0.5, 1, where=(sigmoid >= 0.5), alpha=0.2, color='green')
    ax1.set_xlabel('z = β₀ + β₁X₁ + β₂X₂ + ...')
    ax1.set_ylabel('Probability P(Y=1)')
    ax1.set_title('Sigmoid Function: Transformasi Linear ke Probability')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Decision Boundary
    ax2 = fig.add_subplot(gs[0, 1])
    np.random.seed(42)
    n_samples = 100
    
    class0_ipk = np.random.normal(2.5, 0.5, n_samples)
    class0_income = np.random.normal(25, 5, n_samples)
    class1_ipk = np.random.normal(3.5, 0.5, n_samples)
    class1_income = np.random.normal(10, 5, n_samples)
    
    ax2.scatter(class0_ipk, class0_income, color='red', alpha=0.6, s=50)
    ax2.scatter(class1_ipk, class1_income, color='green', alpha=0.6, s=50)
    x_boundary = np.linspace(1.5, 4.5, 100)
    y_boundary = 40 - 10 * x_boundary
    ax2.plot(x_boundary, y_boundary, 'k--', linewidth=3)
    ax2.set_xlabel('IPK')
    ax2.set_ylabel('Pendapatan Orang Tua (juta)')
    ax2.set_title('Decision Boundary: Memisahkan 2 Kelas')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Probability Distribution
    ax3 = fig.add_subplot(gs[1, :])
    probabilities_0 = np.random.beta(2, 5, 500)
    probabilities_1 = np.random.beta(5, 2, 500)
    bins = np.linspace(0, 1, 30)
    ax3.hist(probabilities_0, bins=bins, alpha=0.6, color='red', density=True)
    ax3.hist(probabilities_1, bins=bins, alpha=0.6, color='green', density=True)
    ax3.axvline(x=0.5, color='black', linestyle='--', linewidth=2)
    ax3.set_xlabel('Predicted Probability')
    ax3.set_ylabel('Density')
    ax3.set_title('Probability Output: Prediksi 0-1')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ========== FUNGSI UTAMA ==========
def load_data(uploaded_file=None):
    """Load data from uploaded file or default"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state['uploaded_data'] = df
            st.session_state['data_source'] = 'uploaded'
            return df, True
        else:
            df = pd.read_csv('beasiswa_800.csv')
            st.session_state['data_source'] = 'default'
            return df, False
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, False

def preprocess_data(df):
    """Preprocess the data"""
    # Identify categorical columns
    categorical_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'Diterima_Beasiswa':
            categorical_cols.append(col)
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
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
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    return df, X_train, X_test, y_train, y_test, scaler, label_encoders

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train and evaluate Logistic Regression model"""
    model = LogisticRegression(
        random_state=42, 
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return model, y_pred, y_pred_proba, accuracy, precision, recall, f1

# ========== MAIN FUNCTION ==========
def main():
    st.title("Sistem Seleksi Beasiswa dengan Machine Learning")
    
    # Sidebar for file upload
    with st.sidebar:
        st.title("Pengaturan Data")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Unggah file CSV baru",
            type=['csv'],
            help="Unggah file CSV dengan format yang sesuai"
        )
        
        # Load default or uploaded data
        if uploaded_file is not None:
            df, is_uploaded = load_data(uploaded_file)
            if df is not None:
                st.success(f"File berhasil diunggah: {uploaded_file.name}")
                st.write(f"Jumlah data: {len(df)} baris, {len(df.columns)} kolom")
        else:
            if 'uploaded_data' in st.session_state:
                df = st.session_state['uploaded_data']
                is_uploaded = True
            else:
                df, is_uploaded = load_data()
        
        # Data preview in sidebar
        if df is not None:
            with st.expander("Preview Data"):
                st.dataframe(df.head(5))
                
            # Show data info
            st.write("**Kolom dalam data:**")
            for col in df.columns:
                st.write(f"- {col}: {df[col].dtype}")
        
        # Navigation
        st.markdown("---")
        menu = st.radio(
            "Menu Utama:",
            ["Eksplorasi Data", 
             "Logistic Regression",
             "Model Machine Learning", 
             "Evaluasi Model", 
             "Prediksi Baru"]
        )
    
    if df is None:
        st.warning("Silakan unggah file CSV atau pastikan file default tersedia")
        return
    
    # Preprocess data
    df_processed, X_train, X_test, y_train, y_test, scaler, label_encoders = preprocess_data(df.copy())
    
    # Menu 1: Data Exploration
    if menu == "Eksplorasi Data":
        st.header("Eksplorasi Data")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Preview Data")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.subheader("Statistik Data")
            st.metric("Total Pendaftar", len(df))
            if 'Diterima_Beasiswa' in df.columns:
                diterima = df['Diterima_Beasiswa'].sum()
                st.metric("Diterima", diterima)
                st.metric("Tidak Diterima", len(df) - diterima)
                st.metric("Persentase Diterima", f"{(diterima/len(df)*100):.1f}%")
        
        # Data Description
        with st.expander("Deskripsi Statistik"):
            st.dataframe(df.describe(), use_container_width=True)
        
        # Visualizations
        st.subheader("Visualisasi Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Diterima_Beasiswa' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                df['Diterima_Beasiswa'].value_counts().plot(kind='bar', ax=ax, color=['red', 'green'])
                ax.set_title('Distribusi Status Penerimaan')
                ax.set_xlabel('Status Penerimaan')
                ax.set_ylabel('Jumlah')
                ax.set_xticklabels(['Tidak Diterima', 'Diterima'], rotation=0)
                st.pyplot(fig)
        
        with col2:
            if 'IPK' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                df['IPK'].hist(bins=20, ax=ax, color='blue', alpha=0.7)
                ax.set_title('Distribusi IPK Pendaftar')
                ax.set_xlabel('IPK')
                ax.set_ylabel('Frekuensi')
                st.pyplot(fig)
        
        # Correlation matrix
        if len(df.select_dtypes(include=[np.number]).columns) > 0:
            st.subheader("Korelasi antar Fitur")
            fig, ax = plt.subplots(figsize=(10, 8))
            numeric_df = df.select_dtypes(include=[np.number])
            correlation = numeric_df.corr()
            sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Matriks Korelasi')
            st.pyplot(fig)
    
    # Menu 2: Logistic Regression
    elif menu == "Logistic Regression":
        st.header("Logistic Regression")
        
        tab1, tab2 = st.tabs(["Konsep Dasar", "Visualisasi"])
        
        with tab1:
            st.markdown("### Konsep Dasar Logistic Regression")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Apa itu Logistic Regression?**
                
                Algoritma klasifikasi biner yang memprediksi probability:
                
                1. Hitung score linear: z = β₀ + β₁X₁ + β₂X₂ + ...
                2. Transform ke probability: P = 1 / (1 + e^(-z))
                3. Bandingkan dengan threshold (biasanya 0.5)
                4. Hasil akhir: Kelas 0 atau 1
                
                **Contoh di Seleksi Beasiswa:**
                
                Fitur Input:
                - IPK: 3.5 → meningkatkan peluang
                - Pendapatan: 15 juta → mengurangi peluang
                - Prestasi: 5 → meningkatkan peluang
                """)
            
            with col2:
                score_input = st.slider(
                    "Score Linear (z):",
                    min_value=-5.0,
                    max_value=5.0,
                    value=0.0,
                    step=0.1
                )
                
                probability = 1 / (1 + np.exp(-score_input))
                
                fig, ax = plt.subplots(figsize=(7, 4))
                z_curve = np.linspace(-5, 5, 100)
                p_curve = 1 / (1 + np.exp(-z_curve))
                ax.plot(z_curve, p_curve, 'b-', linewidth=3, alpha=0.7)
                ax.plot(score_input, probability, 'ro', markersize=10)
                ax.axhline(y=0.5, color='green', linestyle='--', linewidth=2)
                ax.axvline(x=0, color='orange', linestyle='--', linewidth=2)
                ax.fill_between(z_curve, 0, 0.5, where=(p_curve < 0.5), alpha=0.2, color='red')
                ax.fill_between(z_curve, 0.5, 1, where=(p_curve >= 0.5), alpha=0.2, color='green')
                ax.set_xlabel('Score Linear (z)')
                ax.set_ylabel('Probability P(Y=1)')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                if probability >= 0.5:
                    st.success(f"Probability: {probability:.1%} → DITERIMA")
                else:
                    st.error(f"Probability: {probability:.1%} → TIDAK DITERIMA")
        
        with tab2:
            fig = plot_logistic_regression_explanation()
            st.pyplot(fig)
    
    # Menu 3: Machine Learning Model
    elif menu == "Model Machine Learning":
        st.header("Model Logistic Regression")
        
        if 'Diterima_Beasiswa' not in df.columns:
            st.error("Kolom target 'Diterima_Beasiswa' tidak ditemukan dalam data")
            return
        
        model, y_pred, y_pred_proba, accuracy, precision, recall, f1 = train_logistic_regression(
            X_train, X_test, y_train, y_test
        )
        
        st.success("Model berhasil dilatih")
        
        # Display model coefficients
        st.subheader("Koefisien Model")
        
        feature_names = X_train.columns
        coefficients = model.coef_[0]
        
        importance_df = pd.DataFrame({
            'Fitur': feature_names,
            'Koefisien': coefficients,
            'Pengaruh': ['Positif' if x > 0 else 'Negatif' for x in coefficients]
        })
        importance_df = importance_df.sort_values('Koefisien', ascending=False)
        
        st.dataframe(importance_df, use_container_width=True)
        
        with st.expander("Detail Model"):
            st.write(f"**Data Source:** {st.session_state.get('data_source', 'default')}")
            st.write(f"**Ukuran Training:** {len(X_train)} sampel")
            st.write(f"**Ukuran Testing:** {len(X_test)} sampel")
    
    # Menu 4: Model Evaluation
    elif menu == "Evaluasi Model":
        st.header("Evaluasi Performa Model")
        
        if 'Diterima_Beasiswa' not in df.columns:
            st.error("Kolom target 'Diterima_Beasiswa' tidak ditemukan")
            return
        
        model, y_pred, y_pred_proba, accuracy, precision, recall, f1 = train_logistic_regression(
            X_train, X_test, y_train, y_test
        )
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.2%}")
        
        with col2:
            st.metric("Precision", f"{precision:.2%}")
        
        with col3:
            st.metric("Recall", f"{recall:.2%}")
        
        with col4:
            st.metric("F1-Score", f"{f1:.2%}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Prediksi Tidak', 'Prediksi Ya'],
                       yticklabels=['Aktual Tidak', 'Aktual Ya'])
            ax.set_title('Confusion Matrix')
            ax.set_ylabel('Aktual')
            ax.set_xlabel('Prediksi')
            st.pyplot(fig)
        
        with col2:
            st.write(f"**True Negative (TN):** {cm[0,0]}")
            st.write(f"**False Positive (FP):** {cm[0,1]}")
            st.write(f"**False Negative (FN):** {cm[1,0]}")
            st.write(f"**True Positive (TP):** {cm[1,1]}")
        
        # ROC Curve
        st.subheader("ROC Curve")
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True)
        st.pyplot(fig)
    
    # Menu 5: New Prediction
    elif menu == "Prediksi Baru":
        st.header("Prediksi Penerimaan Beasiswa")
        
        if 'Diterima_Beasiswa' not in df.columns:
            st.error("Data tidak memiliki kolom target yang sesuai")
            return
        
        model, _, _, _, _, _, _ = train_logistic_regression(
            X_train, X_test, y_train, y_test
        )
        
        # Get feature names from training data
        feature_names = X_train.columns.tolist()
        
        # Create input form
        st.subheader("Input Data Calon Penerima")
        
        input_data = {}
        cols_per_row = 3
        features_count = len(feature_names)
        
        for i in range(0, features_count, cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i + j
                if idx < features_count:
                    feature = feature_names[idx]
                    with cols[j]:
                        # Get appropriate input based on data type
                        if df[feature].dtype in ['int64', 'float64']:
                            min_val = float(df[feature].min())
                            max_val = float(df[feature].max())
                            default_val = float(df[feature].median())
                            
                            if feature in ['IPK']:
                                input_data[feature] = st.number_input(
                                    feature,
                                    min_value=0.0,
                                    max_value=4.0,
                                    value=float(default_val),
                                    step=0.1
                                )
                            else:
                                input_data[feature] = st.number_input(
                                    feature,
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=float(default_val),
                                    step=1.0 if df[feature].dtype == 'int64' else 0.1
                                )
                        else:
                            # For categorical, show unique values
                            unique_vals = df[feature].unique()
                            input_data[feature] = st.selectbox(
                                feature,
                                options=unique_vals
                            )
        
        if st.button("Prediksi"):
            # Prepare input for prediction
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            for col in input_df.columns:
                if col in label_encoders:
                    try:
                        input_df[col] = label_encoders[col].transform([str(input_df[col].iloc[0])])
                    except:
                        input_df[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])
            
            # Ensure all columns are numeric
            for col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            
            # Scale numerical features
            numerical_cols = input_df.select_dtypes(include=[np.number]).columns
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
            
            # Make prediction
            try:
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
                
                st.markdown("---")
                st.subheader("Hasil Prediksi")
                
                if prediction == 1:
                    st.success(f"**DIPREDIKSI DITERIMA**")
                else:
                    st.error(f"**DIPREDIKSI TIDAK DITERIMA**")
                
                st.write(f"**Probabilitas Diterima:** {probability:.2%}")
                st.write(f"**Probabilitas Tidak Diterima:** {1-probability:.2%}")
                
                # Show progress bar
                st.progress(float(probability))
                
                # Show influencing factors
                st.subheader("Faktor yang Mempengaruhi:")
                coefficients = model.coef_[0]
                
                positive_factors = []
                negative_factors = []
                
                for feature, coef in zip(feature_names, coefficients):
                    if coef > 0:
                        positive_factors.append(feature)
                    else:
                        negative_factors.append(feature)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Meningkatkan peluang:**")
                    for factor in positive_factors[:5]:  # Show top 5
                        st.write(f"✓ {factor}")
                
                with col2:
                    st.write("**Menurunkan peluang:**")
                    for factor in negative_factors[:5]:  # Show top 5
                        st.write(f"✗ {factor}")
                        
            except Exception as e:
                st.error(f"Error dalam prediksi: {str(e)}")

if __name__ == "__main__":
    main()