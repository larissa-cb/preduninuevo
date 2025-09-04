# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Deserción Estudiantil",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título de la aplicación
st.title("🎓 Predictor de Deserción y Éxito Académico Estudiantil")
st.markdown("""
Esta aplicación utiliza machine learning para predecir la probabilidad de que un estudiante 
abandone sus estudios, permanezca enrolado o se gradúe exitosamente.
""")

# Cargar modelo y preprocesador
@st.cache_resource
def load_model():
    """Cargar modelo entrenado"""
    try:
        model = joblib.load('best_model.pkl')
        return model
    except:
        st.error("No se pudo cargar el modelo. Asegúrate de que 'best_model.pkl' existe.")
        return None

@st.cache_resource
def load_scaler():
    """Cargar scaler"""
    scaler = RobustScaler()
    return scaler

@st.cache_data
def load_feature_names():
    """Cargar nombres de características"""
    # Estos deberían coincidir con tu dataset de entrenamiento
    feature_names = [
        'Marital status', 'Application mode', 'Application order', 'Course', 
        'Daytime/evening attendance', 'Previous qualification', 
        'Previous qualification (grade)', 'Nacionality', 'Mother\'s qualification',
        'Father\'s qualification', 'Father\'s occupation', 'Admission grade',
        'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date',
        'Gender', 'Scholarship holder', 'Age at enrollment', 'International',
        'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
        'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
        'Unemployment rate', 'Inflation rate', 'GDP', 'performance_ratio_1st_sem',
        'improvement_ratio', 'academic_load', 'success_momentum', 'academic_efficiency'
    ]
    return feature_names

def preprocess_input(input_data, feature_names, scaler):
    """Preprocesar datos de entrada"""
    # Convertir a DataFrame
    df = pd.DataFrame([input_data], columns=feature_names)
    
    # Aplicar transformaciones (debes usar el mismo scaler que en entrenamiento)
    # En una app real, deberías guardar y cargar el scaler entrenado
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def main():
    # Cargar recursos
    model = load_model()
    scaler = load_scaler()
    feature_names = load_feature_names()
    
    if model is None:
        return
    
    # Sidebar para navegación
    st.sidebar.title("Navegación")
    app_mode = st.sidebar.selectbox(
        "Selecciona el modo",
        ["Predicción Individual", "Análisis de Datos", "Acerca de"]
    )
    
    if app_mode == "Predicción Individual":
        st.header("📊 Predicción Individual")
        st.markdown("Ingresa los datos del estudiante para predecir su probabilidad de éxito.")
        
        # Formulario de entrada de datos
        with st.form("student_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Datos Demográficos")
                age = st.slider("Edad al ingreso", 17, 70, 20)
                gender = st.selectbox("Género", ["Masculino", "Femenino"])
                international = st.selectbox("Estudiante internacional", ["No", "Sí"])
                marital_status = st.selectbox("Estado civil", [
                    "Soltero", "Casado", "Viudo", "Divorciado", 
                    "Unión de hecho", "Separado legalmente"
                ])
                
            with col2:
                st.subheader("Datos Académicos")
                admission_grade = st.slider("Nota de admisión", 0, 200, 120)
                previous_qualification = st.selectbox("Calificación previa", [
                    "Educación secundaria", "Bachillerato", "Grado universitario",
                    "Maestría", "Doctorado", "Otro"
                ])
                curricular_units_1st = st.slider("Materias inscritas 1er sem", 0, 10, 5)
                curricular_approved_1st = st.slider("Materias aprobadas 1er sem", 0, 10, 3)
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Situación Económica")
                debtor = st.selectbox("Es deudor", ["No", "Sí"])
                scholarship = st.selectbox("Tiene beca", ["No", "Sí"])
                tuition_fees = st.selectbox("Matrícula al día", ["Sí", "No"])
                
            with col4:
                st.subheader("Rendimiento Académico")
                grade_1st_sem = st.slider("Promedio 1er semestre", 0, 20, 12)
                grade_2nd_sem = st.slider("Promedio 2do semestre", 0, 20, 13)
                improvement = grade_2nd_sem - grade_1st_sem
                st.metric("Mejora entre semestres", f"{improvement:.1f}")
            
            submitted = st.form_submit_button("Predecir")
        
        if submitted:
            # Preparar datos para predicción
            input_data = {
                'Age at enrollment': age,
                'Gender': 1 if gender == "Masculino" else 0,
                'International': 1 if international == "Sí" else 0,
                'Marital status': ["Soltero", "Casado", "Viudo", "Divorciado", "Unión de hecho", "Separado legalmente"].index(marital_status) + 1,
                'Admission grade': admission_grade,
                'Previous qualification (grade)': 150,  # Valor por defecto
                'Debtor': 1 if debtor == "Sí" else 0,
                'Scholarship holder': 1 if scholarship == "Sí" else 0,
                'Tuition fees up to date': 1 if tuition_fees == "Sí" else 0,
                'Curricular units 1st sem (enrolled)': curricular_units_1st,
                'Curricular units 1st sem (approved)': curricular_approved_1st,
                'Curricular units 1st sem (grade)': grade_1st_sem,
                'Curricular units 2nd sem (grade)': grade_2nd_sem,
                'performance_ratio_1st_sem': curricular_approved_1st / curricular_units_1st if curricular_units_1st > 0 else 0,
                'improvement_ratio': (grade_2nd_sem - grade_1st_sem) / grade_1st_sem if grade_1st_sem > 0 else 0,
                # ... completar con más características según sea necesario
            }
            
            # Llenar valores faltantes con ceros
            for feature in feature_names:
                if feature not in input_data:
                    input_data[feature] = 0
            
            # Ordenar según feature_names
            input_data_ordered = [input_data[feature] for feature in feature_names]
            
            # Preprocesar y predecir
            processed_data = preprocess_input(input_data_ordered, feature_names, scaler)
            prediction = model.predict(processed_data)
            probabilities = model.predict_proba(processed_data)[0]
            
            # Mapear predicciones
            class_names = ["Abandono", "Enrolado", "Graduado"]
            predicted_class = class_names[prediction[0]]
            
            # Mostrar resultados
            st.success(f"**Predicción:** {predicted_class}")
            
            # Gráfico de probabilidades
            fig = go.Figure(data=[
                go.Bar(x=class_names, y=probabilities, 
                      marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ])
            fig.update_layout(
                title="Probabilidades de Predicción",
                xaxis_title="Categoría",
                yaxis_title="Probabilidad",
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig)
            
            # Interpretación de resultados
            st.subheader("📋 Interpretación de Resultados")
            if predicted_class == "Abandono":
                st.warning("""
                **Recomendaciones:**
                - Implementar programa de mentoría
                - Ofrecer asesoramiento académico
                - Revisar situación económica del estudiante
                - Monitorear rendimiento continuo
                """)
            elif predicted_class == "Enrolado":
                st.info("""
                **Recomendaciones:**
                - Mantener programas de apoyo actuales
                - Monitorear progreso académico
                - Ofrecer oportunidades de desarrollo
                """)
            else:
                st.success("""
                **Recomendaciones:**
                - Continuar con el apoyo actual
                - Ofrecer oportunidades de investigación/pasantías
                - Preparar para transición al mercado laboral
                """)
    
    elif app_mode == "Análisis de Datos":
        st.header("📈 Análisis de Datos")
        
        # Aquí puedes agregar visualizaciones de análisis exploratorio
        st.subheader("Distribución de la Variable Objetivo")
        
        # Datos de ejemplo (deberías cargar datos reales)
        target_data = pd.DataFrame({
            'Categoría': ['Abandono', 'Enrolado', 'Graduado'],
            'Cantidad': [46.3, 26.6, 27.1]
        })
        
        fig = px.pie(target_data, values='Cantidad', names='Categoría',
                    title='Distribución de Resultados Académicos')
        st.plotly_chart(fig)
        
        # Más visualizaciones pueden ir aquí...
        
    else:
        st.header("ℹ️ Acerca de")
        st.markdown("""
        ## Predictor de Deserción Estudiantil
        
        **Tecnologías utilizadas:**
        - Machine Learning: Random Forest, XGBoost, LightGBM
        - Framework: Scikit-learn, Streamlit
        - Visualización: Matplotlib, Seaborn, Plotly
        
        **Características:**
        - Predicción de riesgo de deserción estudiantil
        - Análisis de factores influyentes
        - Interfaz amigable para usuarios no técnicos
        - Recomendaciones accionables basadas en predicciones
        
        **Desarrollado por:** [Tu Nombre]
        """)

if __name__ == "__main__":
    main()