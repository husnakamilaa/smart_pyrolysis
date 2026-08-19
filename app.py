import joblib
from model_pipeline import PyrolysisReactorPipeline
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Virtual Pyrolysis Reactor Predictor",
    layout="wide",
    page_icon="⚗️",
)

st.title("⚗️ Virtual Pyrolysis Reactor Predictor")
st.caption(
    "Simulasi Termokimia Pirolisis Biomassa Berbasis Multi-Model XGBoost &"
    " Random Forest"
)


#LOAD DATA
@st.cache_resource
def load_pipeline():
  return joblib.load("pyrolysis_pipeline.joblib")


pipeline = load_pipeline()
features = pipeline.feature_names_in_

#INPUTAN
with st.sidebar:
  st.header("⚙️ Parameter Masukan")

  with st.expander("1. Analisis Proksimat (% wt)", expanded=True):
    val_M = st.number_input(
        "M (Moisture)",
        min_value=0.0,
        max_value=100.0,
        value=8.50,
        step=0.1,
        format="%.2f",
    )
    val_Ash = st.number_input(
        "Ash (Kadar Abu)",
        min_value=0.0,
        max_value=100.0,
        value=2.10,
        step=0.1,
        format="%.2f",
    )
    val_VM = st.number_input(
        "VM (Volatile Matter)",
        min_value=0.0,
        max_value=100.0,
        value=72.40,
        step=0.5,
        format="%.2f",
    )
    val_FC = st.number_input(
        "FC (Fixed Carbon)",
        min_value=0.0,
        max_value=100.0,
        value=17.00,
        step=0.5,
        format="%.2f",
    )

  with st.expander("2. Analisis Ultimat (% wt)", expanded=True):
    val_C = st.number_input(
        "C (Karbon)",
        min_value=0.0,
        max_value=100.0,
        value=48.20,
        step=0.5,
        format="%.2f",
    )
    val_H = st.number_input(
        "H (Hidrogen)",
        min_value=0.0,
        max_value=100.0,
        value=5.80,
        step=0.1,
        format="%.2f",
    )
    val_O = st.number_input(
        "O (Oksigen)",
        min_value=0.0,
        max_value=100.0,
        value=44.50,
        step=0.5,
        format="%.2f",
    )
    val_N = st.number_input(
        "N (Nitrogen)",
        min_value=0.0,
        max_value=100.0,
        value=0.40,
        step=0.05,
        format="%.2f",
    )

  with st.expander("3. Kondisi Operasi Reaktor", expanded=True):
    val_FT = st.slider(
        "FT - Final Temp (°C)", 300.0, 900.0, 500.0, step=10.0, format="%.1f"
    )
    val_HR = st.slider(
        "HR - Heating Rate (°C/min)", 5.0, 150.0, 20.0, step=5.0, format="%.1f"
    )
    val_FR = st.slider(
        "FR - Flow Rate (mL/min)", 10.0, 500.0, 100.0, step=10.0, format="%.1f"
    )
    val_PS = st.number_input(
        "PS - Particle Size (mm)",
        min_value=0.01,
        max_value=20.0,
        value=0.50,
        step=0.05,
        format="%.2f",
    )

  btn_predict = st.button(
      "🚀 Jalankan Simulasi Reaktor",
      type="primary",
      use_container_width=True,
  )

#VARIABLE => NYESUAIIN YG ADA DI MODEL
input_dict = {
    "M": val_M,
    "Ash ": val_Ash,
    "VM": val_VM,
    "FC": val_FC,
    "C": val_C,
    "H": val_H,
    "O": val_O,
    "N": val_N,
    "FT": val_FT,
    "HR": val_HR,
    "FR": val_FR,
    "PS": val_PS,
}
input_df = pd.DataFrame([input_dict])[features]

#Button Predict dijalankan
if btn_predict:
  hasil = pipeline.predict(input_df)
  s_val = float(hasil["solid"][0])
  l_val = float(hasil["liquid"][0])
  g_val = float(hasil["gas"][0])

  st.divider()

  #hasil predict
  col1, col2, col3 = st.columns(3)
  col1.metric("🪨 Solid (Bio-Char)", f"{s_val:.2f} %")
  col2.metric("🧪 Liquid (Bio-Oil)", f"{l_val:.2f} %")
  col3.metric("💨 Gas (Syngas By-Diff)", f"{g_val:.2f} %")

  st.subheader("📊 Distribusi Yield & Karakteristik Sampel")
  g_col1, g_col2 = st.columns(2)

  with g_col1:
    df_pie = pd.DataFrame({
        "Fraksi": ["Solid (Char)", "Liquid (Bio-oil)", "Gas (Syngas)"],
        "Yield (%)": [s_val, l_val, g_val],
    })
    fig_pie = px.pie(
        df_pie,
        values="Yield (%)",
        names="Fraksi",
        hole=0.45,
        title="Neraca Massa Yield Pirolisis (Total 100%)",
        color="Fraksi",
        color_discrete_map={
            "Solid (Char)": "#4A5568",
            "Liquid (Bio-oil)": "#E53E3E",
            "Gas (Syngas)": "#3182CE",
        },
    )
    st.plotly_chart(fig_pie, use_container_width=True)

  with g_col2:
    df_biomass = pd.DataFrame({
        "Komponen": ["M", "Ash ", "VM", "FC", "C", "H", "O", "N"],
        "Nilai (%)": [
            val_M,
            val_Ash,
            val_VM,
            val_FC,
            val_C,
            val_H,
            val_O,
            val_N,
        ],
        "Tipe": [
            "Proksimat",
            "Proksimat",
            "Proksimat",
            "Proksimat",
            "Ultimat",
            "Ultimat",
            "Ultimat",
            "Ultimat",
        ],
    })
    fig_bar = px.bar(
        df_biomass,
        x="Komponen",
        y="Nilai (%)",
        color="Tipe",
        title="Profil Sifat Fisis-Kimia Input",
        text_auto=".2f",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

  st.subheader("📈 Analisis Sensitivitas Temperatur (FT) terhadap Yield")
  st.caption(
      "Simulasi perubahan fraksi yield jika suhu reaktor dinaikkan bertahap"
      " dari 350°C hingga 800°C:"
  )

  temps = np.linspace(350, 800, 20)
  temp_sim_df = pd.concat([input_df] * len(temps), ignore_index=True)
  temp_sim_df["FT"] = temps

  sim_res = pipeline.predict(temp_sim_df)
  df_curve = pd.DataFrame({
      "Suhu (°C)": temps,
      "Solid (%)": sim_res["solid"],
      "Liquid (%)": sim_res["liquid"],
      "Gas (%)": sim_res["gas"],
  })

  fig_line = px.line(
      df_curve,
      x="Suhu (°C)",
      y=["Solid (%)", "Liquid (%)", "Gas (%)"],
      title="Kurva Dekomposisi Termal",
      markers=True,
      color_discrete_map={
          "Solid (%)": "#4A5568",
          "Liquid (%)": "#E53E3E",
          "Gas (%)": "#3182CE",
      },
  )
  st.plotly_chart(fig_line, use_container_width=True)

  st.subheader("🔍 Interpretasi Model: Fitur Paling Berpengaruh")
  st.caption(
      "Bobot pengaruh variabel fisis terhadap estimasi yield pada model XGBoost"
      " dan Random Forest:"
  )

  imp_solid = pipeline.solid_model.feature_importances_
  imp_liquid = pipeline.liquid_model.feature_importances_

  df_imp = pd.DataFrame({
      "Fitur": features,
      "Solid (XGBoost)": imp_solid,
      "Liquid (RF)": imp_liquid,
  })

  fi_col1, fi_col2 = st.columns(2)

  with fi_col1:
    fig_xgb = px.bar(
        df_imp.sort_values(by="Solid (XGBoost)", ascending=True),
        x="Solid (XGBoost)",
        y="Fitur",
        orientation="h",
        title="Fitur Penentu Solid Yield (XGBoost)",
        color_discrete_sequence=["#4A5568"],
    )
    st.plotly_chart(fig_xgb, use_container_width=True)

  with fi_col2:
    fig_rf = px.bar(
        df_imp.sort_values(by="Liquid (RF)", ascending=True),
        x="Liquid (RF)",
        y="Fitur",
        orientation="h",
        title="Fitur Penentu Liquid Yield (Random Forest)",
        color_discrete_sequence=["#E53E3E"],
    )
    st.plotly_chart(fig_rf, use_container_width=True)
else:
  st.info(
      "👈 Masukkan nilai parameter pada sidebar di sebelah kiri, lalu klik"
      " tombol **'Jalankan Simulasi Reaktor'** untuk melihat hasil analisis."
  )