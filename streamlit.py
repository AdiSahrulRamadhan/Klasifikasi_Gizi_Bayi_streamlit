import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Perceptron
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from streamlit_option_menu import option_menu
from collections import Counter

data = pd.read_csv('data_gizi.csv', encoding='latin1')
df = pd.read_csv('output_data.csv', encoding='latin1')

# Preprocessing steps
X = df.drop(columns=['BB/TB'])
y = df['BB/TB']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

smote_enn = SMOTEENN(random_state=42, smote=SMOTE(k_neighbors=1))
Xtrain_resampled, ytrain_resampled = smote_enn.fit_resample(X_train, y_train)

# Build and train the model
model = Perceptron(max_iter=1000, tol=1e-5, random_state=42)
model.fit(Xtrain_resampled, ytrain_resampled)

# Evaluate the model using cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=skf)
mean_accuracy = cv_scores.mean()
y_pred = model.predict(X_test)

# Menampilkan aplikasi Streamlit
def main():
    st.set_page_config(page_title="Klasifikasi Status Gizi Balita", layout="wide")

    st.markdown("""
        <style>
            .main { background-color: #f0f0f5; }
            .stButton>button { background-color: #4CAF50; color: white; }
            .stTextInput>div>div>input { font-size: 1.2rem; }
            h3 { color: green; }
            .orange { color: orange; }
            .red { color: red; }
            .split-container { display: flex; }
            .split-container > div { flex: 1; padding: 10px; }
            .split-container > div:first-child { margin-right: 10px; }
            .scrollable { max-height: 400px; overflow-y: auto; }
        </style>
    """, unsafe_allow_html=True)

    st.title("Klasifikasi Status Gizi Balita")
    page = option_menu(
        None, 
        ["Data Understanding","Preprocessing", 'Modeling', "Implementasi"], 
        icons=['table', 'gear', 'diagram-3', 'play-circle'], 
        menu_icon="cast", 
        default_index=0, 
        orientation="horizontal"
    )

    if page == "Data Understanding":
        st.header("Tentang Data")
        st.subheader("Memahami Data")

        st.markdown("""
            Dataset ini digunakan untuk mengklasifikasikan status gizi balita berdasarkan beberapa fitur yang telah dikumpulkan. 
            Berikut adalah penjelasan dari fitur-fitur yang ada:
            - **Berat:** Berat badan balita dalam kilogram.
            - **Tinggi:** Tinggi badan balita dalam centimeter.
            - **Usia:** Usia balita dalam bulan.
            - **BB/TB:** Kategori status gizi berdasarkan berat badan dan tinggi badan, yang menjadi target klasifikasi.
            
            Dataset yang digunakan untuk modeling adalah hasil preprocessing dari data asli.

            Metode yang digunakan dalam penelitian ini adalah jaringan syaraf tiruan dengan menggunakan perceptron. Metode perceptron merupakan metode pembelajaran terawasi dalam sistem jaringan syaraf tiruan. Saat merancang jaringan saraf, seseorang harus mempertimbangkan jumlah spesifikasi yang akan ditentukan. Jaringan syaraf tiruan terdiri dari beberapa neuron dan beberapa input.
            Metode Perceptron adalah metode yang dilatih dengan menggunakan sekumpulan sampel yang diberikan secara iteratif selama proses pelatihan. Setiap sampel yang diberikan adalah sepasang input dan sampel yang diinginkan.
        """)

        st.subheader("Data Awal")
        st.write(data)

    elif page == "Preprocessing":
        st.header("Preprocessing")

        st.subheader("Identifikasi dan Menghapus Fitur yang Tidak Digunakan")
        st.markdown("""
            Tidak semua fitur yang ada dalam dataset akan berguna atau relevan untuk model prediksi yang kita bangun. 
            Mengidentifikasi fitur yang tidak diperlukan adalah langkah penting untuk mengurangi kompleksitas model dan meningkatkan kinerjanya.
        """)
        
        # Menampilkan data awal
        st.subheader("Data Awal")
        st.write(data)
        
        # Pengecekan nilai yang hilang (missing values)
        missing_values = df.isnull().sum()

        st.markdown("""
            <div class="split-container">
                <div style="flex: 1; margin-right: 10px;">
                    <h3>Hasil Setelah Menghapus Fitur-Fitur Tidak Perlu</h3>
                    <p>Setelah Menghapus Fitur tidak perlu maka kita bisa melakukan proses selanjutnya yakni pengecekan missing value, karena missing value sangat bepengaruh untuk akurasi data yang dihasilkan.</p>
                    <div class="scrollable">""" + df.to_html(index=False) + """</div>
                </div>
                <div style="flex: 1;">
                    <h3>Pengecekan Missing Values</h3>
                    <p>Seperti Hasil Pengecekan Missing Value dibawah ini kita dapat mengetahui bahwasannya tidak terdapat missing value yang kami gunakan maka kita bisa melakukan proses selanjutnya yakni proses Split Data dipecah menjadi data training dan testing dengan perbandingan 70:30. Selanjutnya, data dinormalisasi menggunakan MinMaxScaler untuk memastikan bahwa semua fitur berada dalam skala yang sama, yaitu antara 0 dan 1. Ini penting untuk meningkatkan kinerja model dalam tahap pelatihan.</p>
                    <div class="scrollable">""" + missing_values.to_frame().to_html(header=False) + """</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="split-container">
                <div>
                    <h3>Data Yang Dipakai</h3>
                    <p>Pada tahap awal, dataset yang digunakan telah diperiksa dan dipastikan tidak mengandung nilai yang hilang (missing values). Ini sangat penting untuk memastikan kualitas data dan integritas analisis yang dilakukan. Seluruh fitur dalam dataset telah terisi dengan data yang lengkap dan valid.</p>
                    <div class="scrollable">""" + df.to_html(index=False) + """</div>
                </div>
                <div>
                    <h3>Split Data Dan Normalisasi Data</h3>
                    <p>Data dipecah menjadi data training dan testing dengan perbandingan 70:30. Selanjutnya, data dinormalisasi menggunakan MinMaxScaler untuk memastikan bahwa semua fitur berada dalam skala yang sama, yaitu antara 0 dan 1. Ini penting untuk meningkatkan kinerja model dalam tahap pelatihan.</p>
                    <div style="display: flex;">
                        <div style="flex: 1; margin-right: 10px;">
                            <h4>Data Training</h4>
                            <div class="scrollable">""" + pd.DataFrame(X_train).to_html(index=False) + """</div>
                            <p>Shape: """ + str(X_train.shape) + """</p>
                        </div>
                        <div style="flex: 1;">
                            <h4>Data Testing</h4>
                            <div class="scrollable">""" + pd.DataFrame(X_test).to_html(index=False) + """</div>
                            <p>Shape: """ + str(X_test.shape) + """</p>
                        </div>
                    </div>
                    <h3>Balancing Data</h3>
                    <p>Setelah data dipecah dan dinormalisasi, langkah selanjutnya adalah menyeimbangkan data training menggunakan metode SMOTEENN. Ini bertujuan untuk mengatasi masalah ketidakseimbangan kelas dalam data training, sehingga model yang dihasilkan lebih robust dan tidak bias terhadap kelas mayoritas.</p>
                    <div style="display: flex;">
                        <div style="flex: 1; margin-right: 10px;">
                            <h4>Sebelum di Balancing</h4>
                            <p>""" + str(Counter(y_train)) + """</p>
                        </div>
                        <div style="flex: 1;">
                            <h4>Sesudah di Balancing</h4>
                            <p>""" + str(Counter(ytrain_resampled)) + """</p>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)


    elif page == "Modeling":
        st.header("Model Yang Digunakan adalah ANN Perceptron")
        st.write(f"Cross-validated Accuracy: {mean_accuracy:.2f}")
        st.subheader("Laporan Akurasi")

        report_dict = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()
        st.table(df_report[['precision', 'recall', 'f1-score', 'support']])
        
    elif page == "Implementasi":
        st.header("Implementasi")

        st.subheader("Input Data")
        berat = st.number_input("Berat", min_value=0.0, value=0.0)
        tinggi = st.number_input("Tinggi", min_value=0.0, value=0.0)
        usia_input = st.number_input("Usia (Bulan)", min_value=0.0, value=0.0)

        if st.button("Predict"):
            if usia_input == 0 or berat == 0 or tinggi == 0:
                st.write("Masukkan Data dengan benar!")
            else:
                user_data = scaler.transform([[usia_input, berat, tinggi]])
                prediction = model.predict(user_data)
                prediction_text = prediction[0]
                
                color = "green" if prediction_text == "Gizi Baik" else "orange" if prediction_text == "Risiko Gizi Lebih" else "red"
                st.markdown(f"Prediction: <h3 class='{color}'>{prediction_text}</h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
