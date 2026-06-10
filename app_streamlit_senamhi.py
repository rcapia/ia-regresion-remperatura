import streamlit as st
import pandas as pd
from joblib import load
import datetime

# 1. Configuración de la interfaz web
st.set_page_config(page_title="Pronóstico Global SENAMHI", page_icon="🌍", layout="centered")

st.title("🌍 Sistema de Pronóstico del Clima Global (SENAMHI IA)")
st.markdown(
    "**Modelo de Regresión Masivo a escala mundial.** "
    "Seleccione un país, ingrese el año y el mes deseado para pronosticar de forma inteligente "
    "la temperatura media estimada de la superficie terrestre."
)
st.markdown("Developed by Reynaldo Capia Capia")
st.markdown("---")

# 2. Cargar el modelo global y el codificador de países en caché para alta velocidad
@st.cache_resource

# 2. Cargar el modelo global y el codificador de países en caché para alta velocidad
@st.cache_resource
def cargar_recursos_ia():
    # Eliminamos la ruta de tu computadora C:/... y dejamos solo los nombres de los archivos
    modelo = load('modelo_senamhi_regresion.joblib')
    encoder = load('encoder_paises.joblib')
    return modelo, encoder

try:
    clf_regresor, encoder_paises = cargar_recursos_ia()
    
    # 3. Formulario interactivo
    with st.form("senamhi_global_form"):
        st.subheader("📊 Parámetros de Selección Geográfica y Temporal")
        
        # Lista de países desplegable ordenada alfabéticamente directo del dataset histórico
        lista_paises = sorted(list(encoder_paises.classes_))
        pais_seleccionado = st.selectbox("**Seleccione el País**", options=lista_paises)
        
        col1, col2 = st.columns(2)
        with col1:
            anio_actual = datetime.date.today().year
            anio_input = st.number_input("**Año a pronosticar**", min_value=1750, max_value=2100, value=anio_actual, step=1)
        with col2:
            mes_opciones = {
                1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
                7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
            }
            mes_seleccionado = st.selectbox(
                "**Mes a pronosticar**", 
                options=list(mes_opciones.keys()), 
                format_func=lambda x: mes_opciones[x]
            )
            
        st.markdown("---")
        predict_button = st.form_submit_button("🔮 Ejecutar Pronóstico de Temperatura Global")

    # 4. Lógica de Predicción Inteligente
    if predict_button:
        # Transformar el país seleccionado por el usuario a su código numérico correspondiente
        pais_codigo = encoder_paises.transform([pais_seleccionado])[0]
        
        # Estructurar la entrada idéntica al entrenamiento: Anio, Mes, Country_Code
        data_input = pd.DataFrame({
            'Anio': [anio_input],
            'Mes': [mes_seleccionado],
            'Country_Code': [pais_codigo]
        })
        
        # Ejecutar regresión
        temperatura_predicha = clf_regresor.predict(data_input)[0]
        
        # 5. Presentación de Resultados Estéticos
        #st.markdown("### 🎯 Resultado del Análisis Climático:")
        #st.success(f"### 🌡️ Temperatura Media Estimada para {pais_seleccionado}: {temperatura_predicha:.2f} °C")
        
        #st.metric(
        #    label=f"Predicción Central para {mes_opciones[mes_seleccionado]} del {anio_input}", 
        #    value=f"{temperatura_predicha:.2f} °C"
        #)
        #st.markdown(
        #    f"*Nota: Estimación de la temperatura media del aire en superficie calculada a través del "
        #    f"análisis de tendencias de más de 540,000 registros históricos globales.*"
        #)
        # ==========================================
        # BLOQUE 3: LÓGICA DE COLOR DINÁMICO (STREAMLIT)
        # ==========================================
        st.markdown("### 🎯 Resultado del Análisis Climático:")
        
        # Evaluamos el signo de la temperatura predicha para armar el juego de colores
        if temperatura_predicha >= 0:
            # Estilo dinámico Azul para temperaturas sobre cero
            estilo_alerta = (
                "background-color: #e8f4fd; "
                "color: #0b5790; "
                "border: 1px solid #b8dbf4; "
                "padding: 18px; "
                "border-radius: 8px; "
                "font-weight: bold; "
                "font-size: 20px;"
            )
            mensaje_color = f"☀️ Temperatura Media Estimada (Sobre Cero): {temperatura_predicha:.2f} °C"
            
            # Mostramos la caja personalizada y luego la alerta nativa azul (info)
            st.markdown(f"<div style='{estilo_alerta}'>{mensaje_color}</div>", unsafe_allow_html=True)
            st.info(f"El modelo pronostica un clima templado/cálido para {pais_seleccionado}.")
        else:
            # Estilo dinámico Rojo para temperaturas bajo cero (Negativas)
            estilo_alerta = (
                "background-color: #fce8e6; "
                "color: #c5221f; "
                "border: 1px solid #fad2cf; "
                "padding: 18px; "
                "border-radius: 8px; "
                "font-weight: bold; "
                "font-size: 20px;"
            )
            mensaje_color = f"❄️ Temperatura Media Estimada (Bajo Cero): {temperatura_predicha:.2f} °C"
            
            # Mostramos la caja personalizada y luego la alerta nativa roja (error)
            st.markdown(f"<div style='{estilo_alerta}'>{mensaje_color}</div>", unsafe_allow_html=True)
            st.error(f"¡Atención! Alerta de helada o clima extremadamente frío en {pais_seleccionado}.")

        # Marcador ejecutivo tradicional abajo
        st.markdown("---")
        st.metric(
            label=f"Predicción Central para {mes_opciones[mes_seleccionado]} del {anio_input}", 
            value=f"{temperatura_predicha:.2f} °C"
        )

except FileNotFoundError:
    st.error("❌ Archivos esenciales del modelo (.joblib) no encontrados. Ejecute el cuaderno de entrenamiento primero.")
except Exception as e:
    st.error(f"❌ Ocurrió un error en el procesamiento: {e}")
