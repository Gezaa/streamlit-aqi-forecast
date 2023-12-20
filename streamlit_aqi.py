import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

# Define CSS styles
st.markdown(
    """
    <style>
    .sidebar {
        background-color: #c0d6df;
    }
    .dataframe {
        background-color: #dbe9ee;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    df = pd.read_csv("aqi_jakarta_data.csv")
    return df

@st.cache_data
def predict_aqi_single(model_params, df, date):
    
    len_df = len(df)
    model = SimpleExpSmoothing(
        endog=df['AQI_Value'],
        initialization_method=model_params.get('initialization_method', 'estimated'),
    )
    result = model.fit(smoothing_level=model_params.get('smoothing_level', None))
    pred = result.predict(start=len_df, end=len_df + date - 1)
    pred = pd.DataFrame(pred, columns=['AQI_Value'])
    return pred

def predict_aqi_double(model_params, df, date):
    
    len_df = len(df)
    model = ExponentialSmoothing(
        endog=df['AQI_Value'],
        trend=model_params.get('trend', 'add'),
        initialization_method=model_params.get('initialization_method', 'estimated'),
    )
    result = model.fit(smoothing_level=model_params.get('smoothing_level'))
    pred = result.predict(start=len_df, end=len_df + date - 1)
    pred = pd.DataFrame(pred, columns=['AQI_Value'])
    return pred

def main():
    st.title('AQI Value Forecast with Exponential Smoothing')
    st.divider()
    df = load_data()
    # Membuat sidebar untuk input
    with st.sidebar:
        st.image("logo_if.png", use_column_width=True)
        forecast_models = st.selectbox("Pilih Model", ["Single Exponential Smoothing", "Double Exponential Smoothing"])
        day_forecast = st.slider("Tentukan Hari", 1, 30, step=1)
        generate_button = st.button('Gunakan Model')

    if generate_button:  # Tombol "Generate" ditekan
        if forecast_models == "Single Exponential Smoothing":
            st.header("Pilihan Model = Single Exponential Smoothing")
            model_params = pickle.load(open('SES_Model_Prediksi.sav', 'rb'))
            pred = predict_aqi_single(model_params, df, day_forecast)
            col1, col2 = st.columns([2, 3])
            with col1:
                st.dataframe(pred)
            with col2:
                # Plot data frame
                fig, ax = plt.subplots()
                df['AQI_Value'].plot(label='Known', ax=ax)

                # Plot prediksi
                pred['AQI_Value'].plot(label='Hasil Peramalan', color='red', linestyle='dashed', ax=ax)

                # Menambahkan legenda dan label
                ax.legend()
                ax.set_xlabel('Date')
                ax.set_ylabel('AQI Value')

                # Menentukan batas zoom pada sumbu x
                zoom_start = 380  # Ganti dengan indeks awal untuk zoom
                zoom_end = 480    # Ganti dengan indeks akhir untuk zoom
                ax.set_xlim(zoom_start, zoom_end)

                # Menampilkan grafik
                st.pyplot(fig)

        elif forecast_models == "Double Exponential Smoothing":
            st.header("Pilihan Model = Double Exponential Smoothing")
            model_params = pickle.load(open('DES_Model_Prediksi.sav', 'rb'))
            pred = predict_aqi_double(model_params['parameters'], df, day_forecast)
            col1, col2 = st.columns([2, 3])
            with col1:
                st.dataframe(pred)
            with col2:
                fig, ax = plt.subplots()
                # Plot data frame
                df['AQI_Value'].plot(label='Known', ax=ax)

                # Plot prediksi
                pred['AQI_Value'].plot(label='Hasil Peramalan', color='red', linestyle='dashed', ax=ax)

                # Menambahkan legenda dan label
                ax.legend()
                ax.set_xlabel('Date')
                ax.set_ylabel('AQI Value')

                # Menentukan batas zoom pada sumbu x
                zoom_start = 380  # Ganti dengan indeks awal untuk zoom
                zoom_end = 480    # Ganti dengan indeks akhir untuk zoom
                ax.set_xlim(zoom_start, zoom_end)

                # Menampilkan grafik
                st.pyplot(fig)
        else:
            st.info("Tekan tombol 'Gunakan Model' untuk menggunakan model.")

if __name__ == "__main__":
    main()