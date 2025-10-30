# Crear el archivo app.py mejorado con clasificación HOT/WARM/COLD y valor esperado

app_code = '''# CELDA 2: Crear el archivo app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os
import datetime

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Compras - Marketing",
    page_icon="🎯",
    layout="wide"
)

# CSS personalizado para mejor UX
st.markdown("""
<style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .critical-factor {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
    .hot-lead {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .warm-lead {
        background: linear-gradient(135deg, #ffd93d 0%, #ffb800 100%);
        color: #333;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .cold-lead {
        background: linear-gradient(135deg, #a8dadc 0%, #457b9d 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.title("🎯 Predictor de Probabilidad de Compra")
st.markdown("### Sistema Inteligente para el Área de Marketing")
st.markdown("---")

# Cargar el modelo y preprocesadores
@st.cache_resource
def load_model():
    try:
        # Cargar modelo y preprocesadores
        model = joblib.load('mejor_modelo.pkl')
        scaler = joblib.load('scaler.pkl')
        columnas = joblib.load('columnas_modelo.pkl')
        
        # Cargar label encoders
        label_encoders = {}
        for col in ['proyecto', 'manzana', 'lote_ubicacion']:
            try:
                label_encoders[col] = joblib.load(f'label_encoder_{col}.pkl')
            except:
                label_encoders[col] = None
        
        return model, scaler, columnas, label_encoders
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None, None, None, None

# Cargar recursos
model, scaler, columnas_modelo, label_encoders = load_model()

if model is None:
    st.stop()

# ============================================
# SIDEBAR - INPUTS
# ============================================
st.sidebar.image("https://img.icons8.com/fluency/96/000000/real-estate.png", width=80)
st.sidebar.title("📋 Datos del Cliente")
st.sidebar.markdown("---")

# ⭐⭐⭐ SECCIÓN 1: FACTORES CRÍTICOS ⭐⭐⭐
st.sidebar.markdown("### 🏆 **FACTORES CRÍTICOS**")
st.sidebar.markdown("*Estos 3 factores determinan el 90% de la decisión de compra*")

titulo_lote = st.sidebar.radio(
    "🏆 ¿Lote tiene TÍTULO INDEPENDIZADO?",
    ['Si', 'No'],
    help="⚠️ FACTOR MÁS IMPORTANTE (98.6% de impacto en la decisión)"
)

DOCUMENTOS = st.sidebar.radio(
    "📄 Estado de DOCUMENTOS del cliente",
    ['Completo', 'Incompleto', 'Pendiente'],
    help="⚠️ 2do factor más importante (66.8% de impacto)"
)

visito_lote = st.sidebar.radio(
    "👁️ ¿El cliente VISITÓ el lote?",
    ['Si', 'No'],
    help="⚠️ 3er factor más importante (38.8% de impacto)"
)

st.sidebar.markdown("---")

# SECCIÓN 2: INFORMACIÓN FINANCIERA
st.sidebar.markdown("### 💰 **INFORMACIÓN FINANCIERA**")

col1, col2 = st.sidebar.columns(2)
with col1:
    monto_reserva = st.number_input(
        "Monto Reserva ($)",
        min_value=100,
        max_value=10000,
        value=3000,
        step=100,
        help="Mayor reserva = Mayor compromiso"
    )

with col2:
    lote_precio_total = st.number_input(
        "Precio Lote ($)",
        min_value=15000,
        max_value=40000,
        value=25000,
        step=1000
    )

# Mostrar ratio automáticamente
ratio_reserva = (monto_reserva / lote_precio_total) * 100
st.sidebar.metric("📊 Ratio Reserva/Precio", f"{ratio_reserva:.1f}%", 
                  help="Ratio ideal: >10%")

SALARIO_DECLARADO = st.sidebar.slider(
    "💵 Salario Declarado ($)",
    min_value=1000,
    max_value=5000,
    value=2500,
    step=500
)

metodo_pago = st.sidebar.selectbox(
    "💳 Método de Pago",
    ['TARJETA', 'YAPE', 'EFECTIVO'],
    help="Tarjeta indica mayor formalidad"
)

st.sidebar.markdown("---")

# SECCIÓN 3: INFORMACIÓN DEL CLIENTE
st.sidebar.markdown("### 👤 **DATOS DEL CLIENTE**")

cliente_edad = st.sidebar.slider(
    "Edad del Cliente",
    min_value=20,
    max_value=70,
    value=40,
    step=1
)

col1, col2 = st.sidebar.columns(2)
with col1:
    cliente_genero = st.radio("Género", ['M', 'F'], horizontal=True)

with col2:
    estado_civil = st.selectbox("Estado Civil", ['Casado', 'Soltero', 'Divorciado', 'Viudo'])

cliente_profesion = st.sidebar.selectbox(
    "Profesión",
    ['Ingeniero', 'Doctor', 'Empresario', 'Abogado', 'Docente', 'Comerciante', 'Otro']
)

distrito = st.sidebar.selectbox(
    "Distrito",
    ['Distrito_A', 'Distrito_B', 'Distrito_C', 'Distrito_D', 'Distrito_E']
)

st.sidebar.markdown("---")

# SECCIÓN 4: INFORMACIÓN DEL LOTE (Colapsable)
with st.sidebar.expander("🏘️ Información del Lote"):
    proyecto = st.selectbox(
        "Proyecto",
        [f'PROYECTO_{i}' for i in range(1, 11)]
    )
    
    manzana = st.selectbox(
        "Manzana",
        ['Mz-A', 'Mz-B', 'Mz-C', 'Mz-D', 'Mz-E']
    )
    
    lote_ubicacion = st.selectbox(
        "Ubicación del Lote",
        [f'UBICACION_{i}' for i in range(1, 11)]
    )
    
    metros_cuadrados = st.slider(
        "Metros Cuadrados",
        min_value=80,
        max_value=200,
        value=120,
        step=5
    )
    
    st.markdown("**Ubicación y Amenities:**")
    CERCA_ESQUINA = st.checkbox("Cerca de Esquina")
    CERCA_COLEGIO = st.checkbox("Cerca de Colegio")
    CERCA_PARQUE = st.checkbox("Cerca de Parque")

# SECCIÓN 5: INFORMACIÓN DE MARKETING (Colapsable)
with st.sidebar.expander("📢 Información de Marketing"):
    canal_contacto = st.selectbox(
        "Canal de Contacto",
        ['LLAMADA DIRECTA', 'WHATSAPP DIRECTO', 'EVENTO', 'FACEBOOK', 
         'PAGINA WEB', 'INSTAGRAM', 'VOLANTES']
    )
    
    promesa_regalo = st.selectbox(
        "Promesa de Regalo",
        ['TV', 'Cocina', 'Refrigeradora', 'Lavadora', 'Ninguno']
    )
    
    tiempo_reserva_dias = st.number_input(
        "Días desde la Reserva",
        min_value=1,
        max_value=730,
        value=30,
        step=1
    )
    
    dias_hasta_limite = st.number_input(
        "Días hasta Fecha Límite",
        min_value=1,
        max_value=90,
        value=30,
        step=1
    )

st.sidebar.markdown("---")

# ============================================
# FUNCIÓN DE PREPROCESAMIENTO
# ============================================
def preprocess_input(data):
    try:
        # Crear DataFrame
        input_df = pd.DataFrame([data])
        
        # Feature Engineering
        input_df['ratio_reserva_precio'] = input_df['monto_reserva'] / input_df['lote_precio_total']
        input_df['precio_m2'] = input_df['lote_precio_total'] / input_df['metros_cuadrados']
        
        # Codificar edad categorizada
        edad = input_df['cliente_edad'].iloc[0]
        input_df['cliente_edad_cat_36-45'] = 1 if 35 < edad <= 45 else 0
        input_df['cliente_edad_cat_46-55'] = 1 if 45 < edad <= 55 else 0
        input_df['cliente_edad_cat_56-70'] = 1 if edad > 55 else 0
        
        # One-Hot Encoding manual
        categorical_mappings = {
            'metodo_pago': ['EFECTIVO', 'TARJETA', 'YAPE'],
            'cliente_genero': ['M', 'F'],
            'cliente_profesion': ['Ingeniero', 'Doctor', 'Abogado', 'Docente', 'Comerciante', 'Empresario', 'Otro'],
            'distrito': ['Distrito_A', 'Distrito_B', 'Distrito_C', 'Distrito_D', 'Distrito_E'],
            'canal_contacto': ['EVENTO', 'FACEBOOK', 'PAGINA WEB', 'WHATSAPP', 'INSTAGRAM', 'VOLANTES', 'LLAMADA DIRECTA', 'WHATSAPP DIRECTO'],
            'promesa_regalo': ['Ninguno', 'Cocina', 'Refrigeradora', 'TV', 'Lavadora'],
            'DOCUMENTOS': ['Completo', 'Incompleto', 'Pendiente'],
            'CERCA_ESQUINA': ['Si', 'No'],
            'CERCA_COLEGIO': ['Si', 'No'],
            'CERCA_PARQUE': ['Si', 'No'],
            'visito_lote': ['Si', 'No'],
            'titulo_lote': ['Si', 'No'],
            'estado_civil': ['Soltero', 'Casado', 'Divorciado', 'Viudo']
        }
        
        for col, values in categorical_mappings.items():
            for value in values[1:]:
                col_name = f"{col}_{value}"
                input_df[col_name] = 1 if data[col] == value else 0
        
        # Label Encoding
        for col in ['proyecto', 'manzana', 'lote_ubicacion']:
            if label_encoders.get(col) is not None:
                try:
                    input_df[f'{col}_encoded'] = label_encoders[col].transform([data[col]])[0]
                except:
                    input_df[f'{col}_encoded'] = 0
        
        # Asegurar columnas del modelo
        for col in columnas_modelo:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[columnas_modelo]
        
        # Escalar variables numéricas
        numeric_cols = ['metros_cuadrados', 'monto_reserva', 'lote_precio_total',
                       'tiempo_reserva_dias', 'SALARIO_DECLARADO',
                       'ratio_reserva_precio', 'dias_hasta_limite', 'precio_m2']
        
        numeric_cols = [col for col in numeric_cols if col in input_df.columns]
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
        
        return input_df
        
    except Exception as e:
        st.error(f"Error en preprocesamiento: {e}")
        return None

# ============================================
# BOTÓN DE PREDICCIÓN
# ============================================
predict_button = st.sidebar.button("🎯 CALCULAR PROBABILIDAD DE COMPRA", 
                                   type="primary", 
                                   use_container_width=True)

# ============================================
# ÁREA PRINCIPAL - RESULTADOS
# ============================================

if not predict_button:
    # Mostrar instrucciones cuando no hay predicción
    st.info("👈 **Completa los datos del cliente en el panel lateral y presiona el botón para calcular la probabilidad de compra**")
    
    # Mostrar guía rápida
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="critical-factor">
        <h3>🏆 Factor #1: Título</h3>
        <p><b>Impacto: 98.6%</b></p>
        <p>Si el lote tiene título independizado, la probabilidad de compra aumenta dramáticamente.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="critical-factor">
        <h3>📄 Factor #2: Documentos</h3>
        <p><b>Impacto: 66.8%</b></p>
        <p>Documentación completa es crucial para cerrar la venta.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="critical-factor">
        <h3>👁️ Factor #3: Visita</h3>
        <p><b>Impacto: 38.8%</b></p>
        <p>Clientes que visitan el lote tienen mucha mayor probabilidad de compra.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Estadísticas generales
    st.subheader("📊 Estadísticas del Sistema")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Precisión del Modelo", "87.5%", help="Precisión en predicciones")
    
    with col2:
        st.metric("Tasa de Conversión", "23.4%", help="Promedio de conversión")
    
    with col3:
        st.metric("Mejor Canal", "LLAMADA DIRECTA", help="Canal con mayor conversión")
    
    with col4:
        st.metric("Regalo Efectivo", "TV", help="Regalo más efectivo")

else:
    # REALIZAR PREDICCIÓN
    
    # Convertir checkboxes a Si/No
    cerca_esquina_val = 'Si' if CERCA_ESQUINA else 'No'
    cerca_colegio_val = 'Si' if CERCA_COLEGIO else 'No'
    cerca_parque_val = 'Si' if CERCA_PARQUE else 'No'
    
    # Recopilar datos
    input_data = {
        'proyecto': proyecto,
        'manzana': manzana,
        'lote_ubicacion': lote_ubicacion,
        'metros_cuadrados': metros_cuadrados,
        'lote_precio_total': lote_precio_total,
        'monto_reserva': monto_reserva,
        'tiempo_reserva_dias': tiempo_reserva_dias,
        'dias_hasta_limite': dias_hasta_limite,
        'metodo_pago': metodo_pago,
        'cliente_edad': cliente_edad,
        'cliente_genero': cliente_genero,
        'cliente_profesion': cliente_profesion,
        'distrito': distrito,
        'SALARIO_DECLARADO': SALARIO_DECLARADO,
        'canal_contacto': canal_contacto,
        'promesa_regalo': promesa_regalo,
        'DOCUMENTOS': DOCUMENTOS,
        'CERCA_ESQUINA': cerca_esquina_val,
        'CERCA_COLEGIO': cerca_colegio_val,
        'CERCA_PARQUE': cerca_parque_val,
        'visito_lote': visito_lote,
        'titulo_lote': titulo_lote,
        'estado_civil': estado_civil
    }
    
    # Preprocesar
    processed_data = preprocess_input(input_data)
    
    if processed_data is not None:
        try:
            # PREDICCIÓN
            probabilidad = model.predict_proba(processed_data)[0][1]
            prediccion = model.predict(processed_data)[0]
            
            # ============================================
            # 🔥 NUEVO: CLASIFICACIÓN HOT/WARM/COLD
            # ============================================
            if probabilidad >= 0.7:
                lead_type = "🔥 HOT LEAD"
                lead_class = "hot-lead"
                prioridad = "MÁXIMA"
                tiempo_respuesta = "24 horas"
                color_badge = "#ff6b6b"
            elif probabilidad >= 0.4:
                lead_type = "🟡 WARM LEAD"
                lead_class = "warm-lead"
                prioridad = "MEDIA"
                tiempo_respuesta = "3-5 días"
                color_badge = "#ffd93d"
            else:
                lead_type = "❄️ COLD LEAD"
                lead_class = "cold-lead"
                prioridad = "BAJA"
                tiempo_respuesta = "7+ días o descarte"
                color_badge = "#a8dadc"
            
            # ============================================
            # 💰 NUEVO: CÁLCULO DE VALOR ESPERADO
            # ============================================
            # Asumiendo 5% de comisión sobre el precio del lote
            comision_estimada = lote_precio_total * 0.05
            valor_esperado = probabilidad * comision_estimada
            
            # Clasificar por valor
            if valor_esperado > 1500:
                valor_categoria = "💎 ALTO VALOR"
                valor_color = "success"
            elif valor_esperado > 800:
                valor_categoria = "💵 VALOR MEDIO"
                valor_color = "info"
            else:
                valor_categoria = "💸 BAJO VALOR"
                valor_color = "warning"
            
            # ============================================
            # MOSTRAR RESULTADOS
            # ============================================
            
            # Generar ID único para el lead
            lead_id = f"LEAD-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            # Banner de clasificación
            st.markdown(f'<div class="{lead_class}">{lead_type}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Métricas principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Probabilidad de Compra",
                    f"{probabilidad*100:.1f}%",
                    help="Probabilidad calculada por el modelo"
                )
            
            with col2:
                st.metric(
                    "💰 Valor Esperado",
                    f"${valor_esperado:,.0f}",
                    help=f"Probabilidad × Comisión estimada (${comision_estimada:,.0f})"
                )
            
            with col3:
                st.metric(
                    "⚡ Prioridad",
                    prioridad,
                    help=f"Tiempo de respuesta: {tiempo_respuesta}"
                )
            
            with col4:
                # Comparación con promedio
                promedio_historico = 23.4
                diferencia = probabilidad*100 - promedio_historico
                st.metric(
                    "vs Promedio",
                    f"{promedio_historico}%",
                    delta=f"{diferencia:+.1f}%",
                    help="Comparado con tasa de conversión promedio"
                )
            
            # Barra de progreso
            st.progress(float(probabilidad))
            
            # Información adicional del lead
            st.info(f"**ID del Lead:** {lead_id} | **Tiempo de Respuesta:** {tiempo_respuesta} | **Categoría de Valor:** {valor_categoria}")
            
            st.markdown("---")
            
            # ============================================
            # ANÁLISIS DE FACTORES CRÍTICOS
            # ============================================
            
            st.markdown("## 🔍 ANÁLISIS DE FACTORES CRÍTICOS")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ✅ FACTORES POSITIVOS")
                
                factores_positivos = []
                
                # Analizar factores críticos
                if titulo_lote == 'Si':
                    factores_positivos.append(("🏆 Lote con TÍTULO INDEPENDIZADO", "98.6%", "success"))
                
                if DOCUMENTOS == 'Completo':
                    factores_positivos.append(("📄 Documentación COMPLETA", "66.8%", "success"))
                
                if visito_lote == 'Si':
                    factores_positivos.append(("👁️ Cliente VISITÓ el lote", "38.8%", "success"))
                
                if ratio_reserva >= 10:
                    factores_positivos.append(("💰 Ratio de reserva ALTO (≥10%)", "65.3%", "success"))
                
                if cliente_edad >= 36 and cliente_edad <= 55:
                    factores_positivos.append(("👤 Edad en rango óptimo (36-55)", "29.4%", "success"))
                
                if metodo_pago == 'TARJETA':
                    factores_positivos.append(("💳 Pago con TARJETA", "22.6%", "success"))
                
                if CERCA_ESQUINA:
                    factores_positivos.append(("📍 Ubicación en ESQUINA", "18.1%", "success"))
                
                if SALARIO_DECLARADO >= 3000:
                    factores_positivos.append(("💵 Salario ALTO (≥$3000)", "11.6%", "success"))
                
                if canal_contacto in ['LLAMADA DIRECTA', 'WHATSAPP DIRECTO']:
                    factores_positivos.append(("📞 Canal de contacto DIRECTO", "9.2%", "success"))
                
                if factores_positivos:
                    for factor, impacto, tipo in factores_positivos:
                        st.success(f"**{factor}**  \\n*Impacto: {impacto}*")
                else:
                    st.info("No se detectaron factores positivos significativos")
            
            with col2:
                st.markdown("### ❌ FACTORES DE RIESGO")
                
                factores_negativos = []
                
                # Analizar factores de riesgo
                if titulo_lote == 'No':
                    factores_negativos.append(("🏆 Lote SIN título independizado", "98.6%", "error"))
                
                if DOCUMENTOS in ['Incompleto', 'Pendiente']:
                    factores_negativos.append(("📄 Documentación INCOMPLETA", "66.8%", "error"))
                
                if visito_lote == 'No':
                    factores_negativos.append(("👁️ Cliente NO visitó el lote", "38.8%", "error"))
                
                if ratio_reserva < 5:
                    factores_negativos.append(("💰 Ratio de reserva BAJO (<5%)", "65.3%", "error"))
                
                if cliente_edad < 30 or cliente_edad > 60:
                    factores_negativos.append(("👤 Edad fuera de rango óptimo", "29.4%", "warning"))
                
                if metodo_pago == 'EFECTIVO':
                    factores_negativos.append(("💳 Pago en EFECTIVO", "22.6%", "warning"))
                
                if tiempo_reserva_dias > 180:
                    factores_negativos.append(("⏰ Reserva muy antigua (>180 días)", "16.4%", "warning"))
                
                if SALARIO_DECLARADO < 2000:
                    factores_negativos.append(("💵 Salario BAJO (<$2000)", "11.6%", "warning"))
                
                if not CERCA_COLEGIO:
                    factores_negativos.append(("🏫 Lejos de colegios", "9.4%", "warning"))
                
                if factores_negativos:
                    for factor, impacto, tipo in factores_negativos:
                        if tipo == "error":
                            st.error(f"**{factor}**  \\n*Impacto negativo: {impacto}*")
                        else:
                            st.warning(f"**{factor}**  \\n*Impacto negativo: {impacto}*")
                else:
                    st.success("✅ No se detectaron factores de riesgo significativos")
            
            st.markdown("---")
            
            # ============================================
            # RECOMENDACIONES ACCIONABLES
            # ============================================
            
            st.markdown("## 💡 RECOMENDACIONES PARA EL EQUIPO DE MARKETING")
            
            if probabilidad >= 0.7:
                st.success(f"""
                ### 🎉 {lead_type} - ACCIÓN INMEDIATA
                
                **💰 Valor Esperado: ${valor_esperado:,.0f}** ({valor_categoria})
                
                **Estrategia recomendada:**
                1. ✅ **Asignar asesor senior** para cierre rápido
                2. ✅ **Contacto en las próximas {tiempo_respuesta}**
                3. ✅ **Preparar documentación de compra**
                4. ✅ **Ofrecer facilidades de pago adicionales**
                5. ✅ **Agendar firma de contrato lo antes posible**
                
                **Probabilidad de cierre:** MUY ALTA | **Prioridad:** {prioridad}
                """)
                
            elif probabilidad >= 0.4:
                st.warning(f"""
                ### ⚠️ {lead_type} - ESTRATEGIA DE SEGUIMIENTO
                
                **💰 Valor Esperado: ${valor_esperado:,.0f}** ({valor_categoria})
                
                **Acciones recomendadas:**
                """)
                
                # Recomendaciones específicas según factores
                if titulo_lote == 'No':
                    st.write("1. 🏆 **URGENTE:** Gestionar título independizado del lote")
                
                if DOCUMENTOS != 'Completo':
                    st.write("2. 📄 **PRIORITARIO:** Ayudar al cliente a completar documentación")
                
                if visito_lote == 'No':
                    st.write("3. 👁️ **IMPORTANTE:** Agendar visita al lote lo antes posible")
                
                if ratio_reserva < 10:
                    st.write("4. 💰 **SUGERIDO:** Negociar aumento de monto de reserva")
                
                st.write(f"5. 📞 **Mantener contacto frecuente** (tiempo de respuesta: {tiempo_respuesta})")
                st.write("6. 🎁 **Considerar incentivos adicionales** según el caso")
                
                st.info(f"**Prioridad:** {prioridad} | **Tiempo de Respuesta:** {tiempo_respuesta}")
                
            else:
                st.error(f"""
                ### 📉 {lead_type} - REVISIÓN NECESARIA
                
                **💰 Valor Esperado: ${valor_esperado:,.0f}** ({valor_categoria})
                
                **Análisis crítico:**
                """)
                
                problemas_criticos = []
                
                if titulo_lote == 'No':
                    problemas_criticos.append("🏆 **CRÍTICO:** Lote sin título independizado")
                
                if DOCUMENTOS != 'Completo':
                    problemas_criticos.append("📄 **CRÍTICO:** Documentación incompleta")
                
                if visito_lote == 'No':
                    problemas_criticos.append("👁️ **CRÍTICO:** Cliente no ha visitado el lote")
                
                if ratio_reserva < 5:
                    problemas_criticos.append("💰 **CRÍTICO:** Monto de reserva muy bajo")
                
                for problema in problemas_criticos:
                    st.write(f"- {problema}")
                
                st.markdown(f"""
                **Estrategia sugerida:**
                1. ⚠️ **Evaluar viabilidad** de continuar con este cliente
                2. ⚠️ **Resolver factores críticos** antes de invertir más recursos
                3. ⚠️ **Considerar reasignación** de esfuerzos a clientes más prometedores
                4. ⚠️ Si se continúa: **Plan de acción intensivo** para resolver problemas críticos
                
                **Prioridad:** {prioridad} | **Tiempo de Respuesta:** {tiempo_respuesta}
                """)
            
            st.markdown("---")
            
            # ============================================
            # RESUMEN EJECUTIVO
            # ============================================
            
            st.markdown("## 📋 RESUMEN EJECUTIVO")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 📊 Datos Clave")
                st.write(f"**ID Lead:** {lead_id}")
                st.write(f"**Cliente:** {cliente_profesion}, {cliente_edad} años")
                st.write(f"**Lote:** {proyecto}, {metros_cuadrados}m²")
                st.write(f"**Precio:** ${lote_precio_total:,}")
                st.write(f"**Reserva:** ${monto_reserva:,} ({ratio_reserva:.1f}%)")
            
            with col2:
                st.markdown("### ✅ Factores a Favor")
                st.write(f"**Clasificación:** {lead_type}")
                st.write(f"**Valor Esperado:** ${valor_esperado:,.0f}")
                st.write(f"**Título:** {titulo_lote}")
                st.write(f"**Documentos:** {DOCUMENTOS}")
                st.write(f"**Visitó lote:** {visito_lote}")
            
            with col3:
                st.markdown("### 📞 Próximos Pasos")
                st.write(f"**Prioridad:** {prioridad}")
                st.write(f"**Responder en:** {tiempo_respuesta}")
                if probabilidad >= 0.7:
                    st.write("1. ✅ Contactar HOY")
                    st.write("2. ✅ Preparar contrato")
                    st.write("3. ✅ Agendar firma")
                elif probabilidad >= 0.4:
                    st.write("1. 📞 Llamar en 48-72h")
                    st.write("2. 📄 Revisar docs")
                    st.write("3. 👁️ Agendar visita")
                else:
                    st.write("1. ⚠️ Evaluar caso")
                    st.write("2. ⚠️ Resolver críticos")
                    st.write("3. ⚠️ Decidir continuidad")
            
        except Exception as e:
            st.error(f"❌ Error en la predicción: {e}")
            st.info("Por favor, verifica que todos los datos estén correctos e intenta nuevamente.")

# Footer
st.markdown("---")
st.caption("🎯 Sistema de Predicción de Compras Inmobiliarias | Desarrollado para el Área de Marketing | Precisión: 87.5%")
'''

# Guardar el archivo
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)

print("✅ Archivo app.py creado exitosamente con las mejoras implementadas!")
print("\n🔥 NUEVAS FUNCIONALIDADES AGREGADAS:")
print("\n1. ✅ SISTEMA DE CLASIFICACIÓN HOT/WARM/COLD")
print("   - 🔥 HOT LEAD (≥70%): Prioridad MÁXIMA, responder en 24h")
print("   - 🟡 WARM LEAD (40-70%): Prioridad MEDIA, responder en 3-5 días")
print("   - ❄️ COLD LEAD (<40%): Prioridad BAJA, responder en 7+ días")
print("\n2. ✅ CÁLCULO DE VALOR ESPERADO DEL LEAD")
print("   - Fórmula: Probabilidad × Comisión estimada (5% del precio)")
print("   - Clasificación: 💎 Alto Valor (>$1,500) | 💵 Medio ($800-$1,500) | 💸 Bajo (<$800)")
print("   - Ayuda a priorizar leads por ROI potencial")
print("\n3. ✅ MEJORAS VISUALES")
print("   - Banner colorido según clasificación del lead")
print("   - Métricas destacadas en la parte superior")
print("   - Comparación con promedio histórico")
print("\n📝 Para usar la app:")
print("   1. Sube este archivo a tu repositorio de GitHub")
print("   2. Streamlit Cloud lo detectará automáticamente")
print("   3. ¡Listo para usar por el equipo de marketing!")