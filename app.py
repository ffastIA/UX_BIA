import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import io
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Importações dos módulos personalizados com tratamento de erro
try:
    from src.data_processor import DataProcessor
    from src.statistical_analyzer import StatisticalAnalyzer
    from src.visualizations import ChartGenerator
    from config.settings import (
        SHEETS_URLS, PROFESSOR_ICON_PATH, PROFESSOR_ASSISTANT_URL,
        APP_TITLE, APP_ICON, APP_LAYOUT,
        DEFAULT_CONFIDENCE_LEVEL, DEFAULT_CORRECTION_FACTOR
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    # Valores padrão caso os módulos não estejam disponíveis
    SHEETS_URLS = {
        'tilapias': 'https://docs.google.com/spreadsheets/d/1zoO2Eq-h2mx4i6p6i6bUhGCEXtVWXEZGSRYjnDa13dA/export?format=csv',
        'racao': 'https://docs.google.com/spreadsheets/d/1i-QwgMjC9ZgWymtS_0h0amlAsu9Vu8JvEGpSzTUs_WE/export?format=csv'
    }
    APP_TITLE = "Aquicultura Analytics Pro"
    APP_ICON = "🐟"
    APP_LAYOUT = "wide"
    DEFAULT_CONFIDENCE_LEVEL = 0.95
    DEFAULT_CORRECTION_FACTOR = 1.0
    PROFESSOR_ASSISTANT_URL = "https://ffastia-bia-rag-bia-chain-mem-vgkrw6.streamlit.app/"

# Configuração da página
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded"
)

# CSS Avançado Completo
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .alert-card {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        margin: 0.8rem 0;
        box-shadow: 0 8px 20px rgba(239,68,68,0.3);
    }
    .success-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        margin: 0.8rem 0;
        box-shadow: 0 8px 20px rgba(16,185,129,0.3);
    }
    .filter-section {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #cbd5e1;
    }
    .stat-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #3b82f6;
    }
    .analysis-section {
        background: #fafafa;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
    }
    .professor-container-aligned {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin: 2rem auto;
        width: 100%;
        max-width: 300px;
    }
    .professor-image-link {
        text-decoration: none;
        display: block;
        transition: all 0.3s ease;
    }
    .professor-image-aligned {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
        border: 4px solid white;
        animation: pulse-aligned 3s ease-in-out infinite;
        position: relative;
        overflow: hidden;
        padding: 5px;
        margin: 0 auto 1.5rem auto;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .professor-image-aligned img {
        width: 100%;
        height: 100%;
        border-radius: 50%;
        object-fit: cover;
        transition: all 0.3s ease;
    }
    .professor-image-aligned .emoji-fallback-aligned {
        font-size: 2.5rem;
        color: white;
    }
    .professor-image-link:hover .professor-image-aligned {
        transform: scale(1.1);
        box-shadow: 0 15px 40px rgba(59, 130, 246, 0.7);
    }
    .professor-image-link:hover .professor-image-aligned img {
        transform: scale(1.05);
        filter: brightness(1.15);
    }
    .professor-image-link:active .professor-image-aligned {
        transform: scale(0.95);
    }
    .professor-text-center {
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
        width: 100%;
        margin: 1rem auto;
    }
    .professor-text-center p {
        color: #1e40af;
        font-size: 1rem;
        font-weight: 600;
        margin: 0;
        text-align: center;
        line-height: 1.4;
        max-width: 280px;
        word-wrap: break-word;
    }
    @keyframes pulse-aligned {
        0%, 100% { 
            box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
        }
        50% { 
            box-shadow: 0 12px 30px rgba(59, 130, 246, 0.6);
        }
    }
    @media (max-width: 768px) {
        .professor-container-aligned {
            max-width: 250px;
        }
        .professor-image-aligned {
            width: 70px;
            height: 70px;
        }
        .professor-image-aligned .emoji-fallback-aligned {
            font-size: 2rem;
        }
        .professor-text-center p {
            font-size: 0.9rem;
            max-width: 220px;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300, show_spinner=False)
def load_sheets():
    """Carrega planilhas do Google Sheets"""
    data = {}
    errors = {}

    for name, url in SHEETS_URLS.items():
        try:
            df = pd.read_csv(url)
            df = df.dropna(how='all').dropna(axis=1, how='all')

            if len(df) == 0:
                errors[name] = "Planilha vazia ou sem dados válidos"
            else:
                data[name] = df

        except Exception as e:
            errors[name] = f"Erro ao carregar dados: {str(e)}"

    return data, errors

def add_consultor_icon():
    """Adiciona ícone do consultor virtual com CSS integrado"""
    import base64
    import os

    def get_image_base64(image_path):
        try:
            if os.path.exists(image_path):
                with open(image_path, "rb") as img_file:
                    return base64.b64encode(img_file.read()).decode()
            return None
        except Exception:
            return None

    # Tentar carregar a imagem
    image_path = "images/Tilap-IA.png" if 'PROFESSOR_ICON_PATH' not in globals() else PROFESSOR_ICON_PATH
    img_base64 = get_image_base64(image_path)

    # Centralizar todo o container
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        # Container principal
        st.markdown('<div class="professor-container-aligned">', unsafe_allow_html=True)

        # Imagem clicável com link
        if img_base64:
            html_image = f'<a href="{PROFESSOR_ASSISTANT_URL}" target="_blank" rel="noopener noreferrer" class="professor-image-link" title="Clique para acessar o Prof. Tilap-IA"><div class="professor-image-aligned"><img src="data:image/png;base64,{img_base64}" alt="Prof. Tilap-IA" /></div></a>'
            st.markdown(html_image, unsafe_allow_html=True)
        else:
            html_fallback = f'<a href="{PROFESSOR_ASSISTANT_URL}" target="_blank" rel="noopener noreferrer" class="professor-image-link" title="Clique para acessar o Prof. Tilap-IA"><div class="professor-image-aligned"><span class="emoji-fallback-aligned">🐟🤓</span></div></a>'
            st.markdown(html_fallback, unsafe_allow_html=True)

        # Texto centralizado
        st.markdown('<div class="professor-text-center"><p>👆 Clique na imagem para acessar nosso consultor virtual Dr. Tilap-IA!</p></div>',
                    unsafe_allow_html=True)

        # Fechar container
        st.markdown('</div>', unsafe_allow_html=True)


def calculate_feed_conversion_rate(data):
    """Calcula a Taxa de Conversão Alimentar (TCA)"""
    if data is None or data.empty:
        return data

    data = data.sort_values(['tanque', 'data'])
    tca_results = []

    for tanque in data['tanque'].unique():
        tank_data = data[data['tanque'] == tanque].copy()

        if len(tank_data) < 2:
            tank_data['tca_periodo'] = np.nan
            tank_data['peso_inicial'] = np.nan
            tank_data['peso_final'] = np.nan
            tank_data['ganho_peso'] = np.nan
            tca_results.append(tank_data)
            continue

        peso_inicial = tank_data['peso_medio'].iloc[0]
        peso_final = tank_data['peso_medio'].iloc[-1]
        total_racao_periodo = tank_data['total_racao'].sum()
        ganho_peso = peso_final - peso_inicial

        if ganho_peso > 0:
            tca = total_racao_periodo / ganho_peso
        else:
            tca = np.nan if ganho_peso < 0 else 0

        tank_data['tca_periodo'] = tca
        tank_data['peso_inicial'] = peso_inicial
        tank_data['peso_final'] = peso_final
        tank_data['ganho_peso'] = ganho_peso

        tca_results.append(tank_data)

    if tca_results:
        return pd.concat(tca_results, ignore_index=True)
    else:
        data['tca_periodo'] = np.nan
        data['peso_inicial'] = np.nan
        data['peso_final'] = np.nan
        data['ganho_peso'] = np.nan
        return data


def calculate_daily_growth(data):
    """Calcula crescimento diário"""
    if data is None or data.empty:
        return data

    data = data.sort_values(['tanque', 'data'])
    growth_results = []

    for tanque in data['tanque'].unique():
        tank_data = data[data['tanque'] == tanque].copy()
        tank_data['crescimento_diario'] = tank_data['peso_medio'].diff()
        tank_data['taxa_crescimento_pct'] = tank_data['peso_medio'].pct_change() * 100

        if not tank_data.empty and 'peso_medio' in tank_data.columns and not tank_data['peso_medio'].empty:
            initial_weight = tank_data['peso_medio'].iloc[0]
            tank_data['crescimento_acumulado'] = tank_data['peso_medio'] - initial_weight
        else:
            tank_data['crescimento_acumulado'] = np.nan

        growth_results.append(tank_data)

    if growth_results:
        return pd.concat(growth_results, ignore_index=True)
    else:
        data['crescimento_diario'] = np.nan
        data['taxa_crescimento_pct'] = np.nan
        data['crescimento_acumulado'] = np.nan
        return data


def calculate_advanced_weight(fish_data, correction_factor):
    """Cálculo avançado de peso com fallback"""
    if MODULES_AVAILABLE:
        try:
            from src.weight_estimator import WeightEstimator
            weight_estimator = WeightEstimator(correction_factor=correction_factor)

            peso_estimado = fish_data.apply(
                lambda row: weight_estimator.estimate_weight_advanced(
                    row['largura'],
                    row['altura'],
                    row.get('temperatura'),
                    row.get('ph'),
                    row.get('o2')
                ), axis=1
            )

            return peso_estimado.clip(0.01, 50.0)
        except Exception as e:
            st.warning(f"⚠️ Erro ao usar WeightEstimator: {e}. Usando método padrão.")

    # Método padrão como fallback
    densidade_peixe = 1.05  # kg/L
    fator_forma = 0.7
    fator_profundidade = 0.6

    profundidade_estimada = fish_data['largura'] * fator_profundidade
    volume_cm3 = fish_data['largura'] * fish_data['altura'] * profundidade_estimada * fator_forma
    volume_litros = volume_cm3 / 1000
    peso_base = volume_litros * densidade_peixe * correction_factor

    fator_ambiental = pd.Series(1.0, index=fish_data.index)

    # Correção por temperatura
    if 'temperatura' in fish_data.columns:
        temp = fish_data['temperatura']
        temp_factor = temp.apply(lambda t:
                                 1.0 if pd.isna(t) else
                                 1.0 if 26 <= t <= 30 else
                                 0.98 if 24 <= t <= 32 else
                                 0.95 if 22 <= t <= 34 else
                                 0.90 if 20 <= t <= 36 else 0.85
                                 )
        fator_ambiental *= temp_factor

    peso_final = peso_base * fator_ambiental
    return peso_final.clip(0.01, 50.0)


def process_advanced_data(sheets_data, start_date, end_date, selected_tanks, confidence_level, correction_factor):
    """Processamento avançado dos dados com fallback"""

    if MODULES_AVAILABLE:
        try:
            data_processor = DataProcessor(correction_factor=correction_factor)

            processed_data = data_processor.process_integrated_data(
                sheets_data.get('tilapias'),
                sheets_data.get('racao'),
                start_date, end_date, selected_tanks
            )

            if processed_data is None or processed_data.empty:
                raise ValueError("DataProcessor retornou dados vazios")

            return processed_data

        except Exception as e:
            st.warning(f"⚠️ Erro ao usar DataProcessor: {e}. Usando método padrão.")

    # Método padrão como fallback
    fish_df = sheets_data.get('tilapias')
    feed_df = sheets_data.get('racao')

    if fish_df is None or feed_df is None:
        raise ValueError("Dados não disponíveis")

    # Limpar e preparar dados dos peixes
    fish_clean = fish_df.copy()
    fish_clean['data'] = pd.to_datetime(fish_clean['data'], errors='coerce')
    fish_clean['largura'] = pd.to_numeric(fish_clean['largura'], errors='coerce')
    fish_clean['altura'] = pd.to_numeric(fish_clean['altura'], errors='coerce')

    # Filtrar por período
    mask = (fish_clean['data'].dt.date >= start_date) & (fish_clean['data'].dt.date <= end_date)
    fish_filtered = fish_clean[mask]

    # Remover dados inválidos
    fish_filtered = fish_filtered.dropna(subset=['data', 'largura', 'altura'])
    fish_filtered = fish_filtered[(fish_filtered['largura'] > 0) & (fish_filtered['altura'] > 0)]

    # Calcular peso estimado
    fish_filtered['peso_estimado'] = calculate_advanced_weight(fish_filtered, correction_factor)

    # Agrupar por data e tanque
    agg_dict = {
        'peso_estimado': ['mean', 'std', 'count', 'sum', 'min', 'max'],
        'largura': ['mean', 'std'],
        'altura': ['mean', 'std']
    }

    fish_grouped = fish_filtered.groupby(['data', 'tanque']).agg(agg_dict).reset_index()
    new_columns = ['data', 'tanque', 'peso_medio', 'peso_std', 'n_peixes', 'peso_total', 'peso_min', 'peso_max',
                   'largura_media', 'largura_std', 'altura_media', 'altura_std']
    fish_grouped.columns = new_columns

    # Processar dados de ração
    feed_clean = feed_df.copy()
    feed_clean['data'] = pd.to_datetime(feed_clean['data'], errors='coerce')
    feed_clean['peso'] = pd.to_numeric(feed_clean['peso'], errors='coerce')

    mask_feed = (feed_clean['data'].dt.date >= start_date) & (feed_clean['data'].dt.date <= end_date)
    feed_filtered = feed_clean[mask_feed]

    if selected_tanks:
        feed_filtered = feed_filtered[feed_filtered['tanque'].isin(selected_tanks)]

    feed_grouped = feed_filtered.groupby(['data', 'tanque'])['peso'].agg(['sum', 'count', 'mean']).reset_index()
    feed_grouped.columns = ['data', 'tanque', 'total_racao', 'n_alimentacoes', 'racao_media']

    # Integrar dados
    integrated = pd.merge(fish_grouped, feed_grouped, on=['data', 'tanque'], how='left')
    integrated = integrated.fillna(0)

    # Calcular métricas
    integrated['eficiencia_alimentar'] = integrated.apply(
        lambda row: row['peso_total'] / row['total_racao'] if row['total_racao'] > 0 else 0,
        axis=1
    )

    integrated['consumo_per_capita'] = integrated.apply(
        lambda row: row['total_racao'] / row['n_peixes'] if row['n_peixes'] > 0 else 0,
        axis=1
    )

    # TCA e crescimento
    integrated = calculate_feed_conversion_rate(integrated)
    integrated = calculate_daily_growth(integrated)

    return integrated


def perform_advanced_analysis_with_analyzer(data: pd.DataFrame, analyzer) -> dict:
    """Realiza análises estatísticas avançadas"""
    results = {}

    if data is None or data.empty:
        return results

    # Calcular TCA e crescimento
    if 'tca_periodo' not in data.columns:
        data = calculate_feed_conversion_rate(data)
    if 'crescimento_diario' not in data.columns:
        data = calculate_daily_growth(data)

    # Análise de correlação
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        try:
            correlation_matrix = analyzer.correlation_matrix(data[numeric_cols])
            results['correlations'] = correlation_matrix
        except Exception as e:
            st.warning(f"⚠️ Erro na análise de correlação: {e}")

    # Regressão
    if 'total_racao' in data.columns and 'peso_medio' in data.columns:
        df_reg = data[['total_racao', 'peso_medio']].dropna()
        if not df_reg.empty:
            try:
                regression_results = analyzer.regression_analysis(df_reg['total_racao'], df_reg['peso_medio'])
                results['regression'] = regression_results
            except Exception as e:
                st.warning(f"⚠️ Erro na análise de regressão: {e}")

    return results


def perform_advanced_analysis_fallback(data, confidence_level):
    """Análises estatísticas usando métodos padrão"""
    results = {}

    if data is None or data.empty:
        return results

    # Análise de correlação
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        correlation_matrix = data[numeric_cols].corr()
        results['correlations'] = correlation_matrix

    # Regressão linear
    if 'total_racao' in data.columns and 'peso_medio' in data.columns:
        X = data[['total_racao']].values
        y = data['peso_medio'].values

        mask = ~(np.isnan(X.flatten()) | np.isnan(y))
        X_clean = X[mask].reshape(-1, 1)
        y_clean = y[mask]

        if len(X_clean) > 1:
            model = LinearRegression()
            model.fit(X_clean, y_clean)
            y_pred = model.predict(X_clean)

            results['regression'] = {
                'slope': model.coef_[0],
                'intercept': model.intercept_,
                'r2': r2_score(y_clean, y_pred),
                'predictions': y_pred,
                'X': X_clean.flatten(),
                'y': y_clean
            }

    # Estatísticas descritivas
    if 'peso_medio' in data.columns and not data['peso_medio'].empty:
        clean_data = data['peso_medio'].dropna()
        if len(clean_data) > 0:
            mean = clean_data.mean()
            std = clean_data.std()

            results['peso_statistics'] = {
                'mean': mean,
                'std': std,
                'median': clean_data.median(),
                'min': clean_data.min(),
                'max': clean_data.max(),
                'count': len(clean_data)
            }

    # Teste de normalidade
    if 'peso_medio' in data.columns and len(data['peso_medio'].dropna()) > 3:
        try:
            shapiro_stat, shapiro_p = stats.shapiro(data['peso_medio'].dropna())
            results['normality_test'] = {
                'shapiro': {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                }
            }
        except Exception:
            pass

    # Análise por tanque
    if 'tanque' in data.columns:
        tank_analysis = {}
        for tanque in data['tanque'].unique():
            tank_data = data[data['tanque'] == tanque]
            if not tank_data.empty:
                tank_analysis[tanque] = {
                    'peso_medio': tank_data['peso_medio'].mean() if 'peso_medio' in tank_data.columns else np.nan,
                    'peso_std': tank_data['peso_medio'].std() if 'peso_medio' in tank_data.columns else np.nan,
                    'eficiencia_media': tank_data[
                        'eficiencia_alimentar'].mean() if 'eficiencia_alimentar' in tank_data.columns else np.nan,
                    'tca_media': tank_data['tca_periodo'].mean() if 'tca_periodo' in tank_data.columns else np.nan,
                    'n_registros': len(tank_data)
                }
        results['tank_analysis'] = tank_analysis

    return results


def main():
    st.markdown('<h1 class="main-header">🐟 Aquicultura Analytics Pro</h1>',
                unsafe_allow_html=True)

    # Session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'sheets_data' not in st.session_state:
        st.session_state.sheets_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    # Adicionar ícone do consultor
    add_consultor_icon()

    # Sidebar
    with st.sidebar:
        st.header("🔗 Conexão e Filtros")

        # Carregar dados
        if st.button("🔄 Carregar/Atualizar Dados", type="primary", use_container_width=True):
            with st.spinner("Carregando dados do Google Sheets..."):
                data, errors = load_sheets()

                if data:
                    st.session_state.sheets_data = data
                    for name, df in data.items():
                        st.success(f"✅ {name}: {len(df)} registros")

                if errors:
                    for name, error in errors.items():
                        st.error(f"❌ {name}: {error}")

        # Filtros
        if st.session_state.sheets_data:
            st.markdown('<div class="filter-section">', unsafe_allow_html=True)

            # Período
            st.subheader("📅 Período de Análise")

            fish_df = st.session_state.sheets_data.get('tilapias')
            if fish_df is not None:
                fish_df['data'] = pd.to_datetime(fish_df['data'], errors='coerce')
                valid_dates = fish_df['data'].dropna()

                if not valid_dates.empty:
                    min_date = valid_dates.min().date()
                    max_date = valid_dates.max().date()
                else:
                    min_date = date.today() - timedelta(days=30)
                    max_date = date.today()
            else:
                min_date = date.today() - timedelta(days=30)
                max_date = date.today()

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("📅 Data Inicial", value=min_date, format="DD/MM/YYYY")
            with col2:
                end_date = st.date_input("📅 Data Final", value=max_date, format="DD/MM/YYYY")

            # Tanques
            st.subheader("🏊 Seleção de Tanques")

            feed_df = st.session_state.sheets_data.get('racao')
            if feed_df is not None:
                available_tanks = sorted([t for t in feed_df['tanque'].unique() if pd.notna(t)])
                all_tanks = st.checkbox("✅ Selecionar todos os tanques", value=True)

                if all_tanks:
                    selected_tanks = available_tanks
                else:
                    selected_tanks = st.multiselect(
                        "Escolha os tanques:",
                        available_tanks,
                        default=available_tanks[:3] if len(available_tanks) > 3 else available_tanks
                    )
            else:
                selected_tanks = []

            # Configurações
            st.subheader("📊 Configurações Estatísticas")
            confidence_level = st.selectbox("🎯 Nível de Confiança:", [90, 95, 99], index=1)
            correction_factor = st.slider("🔧 Fator de Correção do Peso:", 0.5, 2.0, DEFAULT_CORRECTION_FACTOR, 0.1)

            st.markdown('</div>', unsafe_allow_html=True)

            # Processar - CORRIGIDO SEM LOOP
            if st.button("🚀 Processar Análise Completa", use_container_width=True):
                with st.spinner("Processando análises avançadas..."):
                    try:
                        if not MODULES_AVAILABLE:
                            st.warning("⚠️ Módulos avançados não disponíveis. Usando métodos padrão.")

                        # Processar dados
                        processed_data = process_advanced_data(
                            st.session_state.sheets_data,
                            start_date, end_date, selected_tanks,
                            confidence_level, correction_factor
                        )

                        if processed_data is None or processed_data.empty:
                            st.warning("⚠️ O processamento de dados não retornou dados válidos.")
                            return

                        # Análises estatísticas
                        if MODULES_AVAILABLE:
                            try:
                                statistical_analyzer = StatisticalAnalyzer(confidence_level=confidence_level / 100)
                                analysis_results = perform_advanced_analysis_with_analyzer(
                                    processed_data, statistical_analyzer
                                )
                            except Exception as e:
                                st.warning(f"⚠️ Erro ao usar StatisticalAnalyzer: {e}. Usando método padrão.")
                                analysis_results = perform_advanced_analysis_fallback(processed_data, confidence_level)
                        else:
                            analysis_results = perform_advanced_analysis_fallback(processed_data, confidence_level)

                        # Salvar no session state
                        st.session_state.processed_data = processed_data
                        st.session_state.analysis_results = analysis_results

                        st.success("✅ Análise completa concluída!")
                        # REMOVIDO st.rerun() para evitar loop infinito

                    except Exception as e:
                        st.error(f"❌ Erro no processamento: {e}")
                        import traceback
                        st.code(traceback.format_exc())

    # Área principal - CORRIGIDA
    if st.session_state.processed_data is not None:
        display_advanced_dashboard()
    elif st.session_state.sheets_data:
        display_data_preview()
    else:
        display_welcome_advanced()


def display_advanced_dashboard():
    """Dashboard avançado com todas as análises"""
    data = st.session_state.processed_data
    analysis = st.session_state.analysis_results

    if data is None or data.empty:
        st.error("❌ Dados não disponíveis para exibição")
        return

    # Instanciar ChartGenerator se disponível
    chart_generator = None
    if MODULES_AVAILABLE:
        try:
            chart_generator = ChartGenerator()
        except Exception as e:
            st.warning(f"⚠️ Erro ao instanciar ChartGenerator: {e}")

    # KPIs Avançados
    display_advanced_kpis(data)

    # Abas principais
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Análise Temporal",
        "📈 Taxa de Conversão",
        "📉 Curva de Gauss",
        "🔬 Estatísticas Avançadas",
        "🤖 Assistente IA",
        "📋 Relatórios",
        "💾 Exportação"
    ])

    with tab1:
        display_temporal_analysis(data, chart_generator)

    with tab2:
        display_feed_conversion_analysis(data, chart_generator)

    with tab3:
        display_gaussian_analysis(data, analysis, chart_generator)

    with tab4:
        display_advanced_statistics(data, analysis, chart_generator)

    with tab5:
        display_ai_assistant_placeholder()

    with tab6:
        display_automated_reports(data, analysis)

    with tab7:
        display_advanced_export(data, analysis)


def display_advanced_kpis(data):
    """KPIs avançados"""
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.subheader("📊 Indicadores de Performance Avançados")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        peso_medio = data['peso_medio'].mean()
        peso_trend = ((data['peso_medio'].iloc[-1] - data['peso_medio'].iloc[0]) / data['peso_medio'].iloc[
            0] * 100) if len(data) > 1 else 0
        st.metric("🐟 Peso Médio", f"{peso_medio:.3f} kg", f"{peso_trend:+.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        eff_media = data['eficiencia_alimentar'].mean()
        st.metric("⚡ Eficiência Alimentar", f"{eff_media:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'tca_periodo' in data.columns:
            tca_media = data['tca_periodo'].mean()
            st.metric("🔄 TCA Média", f"{tca_media:.2f}")
        else:
            st.metric("🔄 TCA Média", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'crescimento_diario' in data.columns:
            crescimento = data['crescimento_diario'].mean()
            st.metric("📈 Crescimento Diário", f"{crescimento:.4f} kg")
        else:
            st.metric("📈 Crescimento Diário", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)

    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_racao = data['total_racao'].sum()
        st.metric("🍽️ Ração Total", f"{total_racao:.1f} kg")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def display_temporal_analysis(data, chart_generator=None):
    """Análise temporal da relação peso x ração"""
    st.subheader("📊 Análise Temporal: Peso vs Ração")

    if chart_generator is not None:
        try:
            fig_evolution = chart_generator.create_evolution_chart(data)
            st.plotly_chart(fig_evolution, use_container_width=True)

            st.subheader("🔗 Correlação Peso x Ração no Tempo")
            fig_correlation = chart_generator.create_main_correlation_chart(data)
            st.plotly_chart(fig_correlation, use_container_width=True)
            return
        except Exception as e:
            st.warning(f"⚠️ Erro ao usar ChartGenerator: {e}. Usando gráficos padrão.")

    # Gráfico padrão (fallback)
    display_temporal_analysis_fallback(data)


def display_temporal_analysis_fallback(data):
    """Análise temporal usando plotly padrão"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Evolução do Peso Médio por Tanque",
            "Consumo de Ração por Tanque",
            "Eficiência Alimentar no Tempo",
            "Crescimento Acumulado"
        ),
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    colors = px.colors.qualitative.Set3

    # Gráfico 1: Peso médio
    for i, tanque in enumerate(data['tanque'].unique()):
        tank_data = data[data['tanque'] == tanque]
        fig.add_trace(
            go.Scatter(
                x=tank_data['data'],
                y=tank_data['peso_medio'],
                mode='lines+markers',
                name=f'Tanque {tanque}',
                line=dict(color=colors[i % len(colors)], width=2),
                legendgroup=f'tanque_{tanque}'
            ),
            row=1, col=1
        )

    # Gráfico 2: Ração
    for i, tanque in enumerate(data['tanque'].unique()):
        tank_data = data[data['tanque'] == tanque]
        fig.add_trace(
            go.Bar(
                x=tank_data['data'],
                y=tank_data['total_racao'],
                name=f'Ração T{tanque}',
                marker_color=colors[i % len(colors)],
                opacity=0.7,
                legendgroup=f'tanque_{tanque}',
                showlegend=False
            ),
            row=1, col=2
        )

    # Gráfico 3: Eficiência
    for i, tanque in enumerate(data['tanque'].unique()):
        tank_data = data[data['tanque'] == tanque]
        fig.add_trace(
            go.Scatter(
                x=tank_data['data'],
                y=tank_data['eficiencia_alimentar'],
                mode='lines+markers',
                name=f'Eff T{tanque}',
                line=dict(color=colors[i % len(colors)], dash='dash'),
                legendgroup=f'tanque_{tanque}',
                showlegend=False
            ),
            row=2, col=1
        )

    # Gráfico 4: Crescimento acumulado
    if 'crescimento_acumulado' in data.columns:
        for i, tanque in enumerate(data['tanque'].unique()):
            tank_data = data[data['tanque'] == tanque]
            fig.add_trace(
                go.Scatter(
                    x=tank_data['data'],
                    y=tank_data['crescimento_acumulado'],
                    mode='lines+markers',
                    name=f'Cresc T{tanque}',
                    line=dict(color=colors[i % len(colors)], width=3),
                    legendgroup=f'tanque_{tanque}',
                    showlegend=False
                ),
                row=2, col=2
            )

    fig.update_layout(
        title="Análise Temporal Completa",
        height=700,
        template="plotly_white",
        showlegend=True
    )

    fig.update_yaxes(title_text="Peso (kg)", row=1, col=1)
    fig.update_yaxes(title_text="Ração (kg)", row=1, col=2)
    fig.update_yaxes(title_text="Eficiência", row=2, col=1)
    fig.update_yaxes(title_text="Crescimento Acum. (kg)", row=2, col=2)

    st.plotly_chart(fig, use_container_width=True)


def display_feed_conversion_analysis(data, chart_generator=None):
    """Análise da Taxa de Conversão Alimentar"""
    st.subheader("🔄 Taxa de Conversão Alimentar (TCA)")

    if 'tca_periodo' not in data.columns:
        st.warning("⚠️ Dados de TCA não disponíveis")
        return

    # Informações sobre TCA
    st.info("""
    **Taxa de Conversão Alimentar (TCA)** = Quantidade de ração fornecida (kg) ÷ (Peso final - Peso inicial) (kg)

    - **TCA < 1.5**: Excelente eficiência
    - **TCA 1.5-2.0**: Boa eficiência  
    - **TCA 2.0-2.5**: Eficiência regular
    - **TCA > 2.5**: Baixa eficiência
    """)

    # Métricas de TCA
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        tca_media = data['tca_periodo'].mean()
        st.metric("📊 TCA Média Geral", f"{tca_media:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        tca_melhor = data['tca_periodo'].min()
        st.metric("🏆 Melhor TCA", f"{tca_melhor:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        tca_pior = data['tca_periodo'].max()
        st.metric("⚠️ Pior TCA", f"{tca_pior:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        ganho_total = data['ganho_peso'].sum()
        st.metric("📈 Ganho Total", f"{ganho_total:.2f} kg")
        st.markdown('</div>', unsafe_allow_html=True)

        # Gráfico de TCA por tanque
        if chart_generator is not None:
            try:
                fig_tca = chart_generator.create_efficiency_by_tank(
                    data.rename(columns={'tca_periodo': 'eficiencia_alimentar'}))
                fig_tca.update_layout(title="Taxa de Conversão Alimentar por Tanque")
                fig_tca.update_yaxes(title_text="TCA (kg ração / kg ganho)")
                st.plotly_chart(fig_tca, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Erro ao usar ChartGenerator para TCA: {e}. Usando gráfico padrão.")
                display_tca_chart_fallback(data)
        else:
            display_tca_chart_fallback(data)

        # Tabela detalhada de TCA
        st.subheader("📋 Detalhamento da TCA por Tanque")

        tca_summary = []
        for tanque in data['tanque'].unique():
            tank_data = data[data['tanque'] == tanque].iloc[0]  # TCA é constante por tanque

            status = "🟢 Excelente" if tank_data['tca_periodo'] < 1.5 else \
                "🔵 Bom" if tank_data['tca_periodo'] < 2.0 else \
                    "🟡 Regular" if tank_data['tca_periodo'] < 2.5 else \
                        "🔴 Baixo"

            tca_summary.append({
                'Tanque': f'Tanque {tanque}',
                'Peso Inicial (kg)': f"{tank_data['peso_inicial']:.3f}",
                'Peso Final (kg)': f"{tank_data['peso_final']:.3f}",
                'Ganho de Peso (kg)': f"{tank_data['ganho_peso']:.3f}",
                'Ração Total (kg)': f"{data[data['tanque'] == tanque]['total_racao'].sum():.2f}",
                'TCA': f"{tank_data['tca_periodo']:.2f}",
                'Status': status
            })

        tca_df = pd.DataFrame(tca_summary)
        st.dataframe(tca_df, use_container_width=True, hide_index=True)

def display_tca_chart_fallback(data):
    """Gráfico de TCA padrão (fallback)"""
    fig_tca = go.Figure()

    tank_tca = data.groupby('tanque')['tca_periodo'].first()  # TCA é constante por tanque

    # Definir cores baseadas na eficiência
    colors_tca = []
    for tca in tank_tca.values:
        if tca < 1.5:
            colors_tca.append('#10b981')  # Verde - Excelente
        elif tca < 2.0:
            colors_tca.append('#3b82f6')  # Azul - Bom
        elif tca < 2.5:
            colors_tca.append('#f59e0b')  # Amarelo - Regular
        else:
            colors_tca.append('#ef4444')  # Vermelho - Baixo

    fig_tca.add_trace(
        go.Bar(
            x=[f'Tanque {t}' for t in tank_tca.index],
            y=tank_tca.values,
            marker_color=colors_tca,
            text=[f'{tca:.2f}' for tca in tank_tca.values],
            textposition='auto',
            name='TCA por Tanque'
        )
    )

    # Linhas de referência
    fig_tca.add_hline(y=1.5, line_dash="dash", line_color="green",
                      annotation_text="Excelente (< 1.5)")
    fig_tca.add_hline(y=2.0, line_dash="dash", line_color="blue",
                      annotation_text="Bom (< 2.0)")
    fig_tca.add_hline(y=2.5, line_dash="dash", line_color="orange",
                      annotation_text="Regular (< 2.5)")

    fig_tca.update_layout(
        title="Taxa de Conversão Alimentar por Tanque",
        xaxis_title="Tanque",
        yaxis_title="TCA (kg ração / kg ganho)",
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig_tca, use_container_width=True)

def display_gaussian_analysis(data, analysis, chart_generator=None):
    """Análise da Curva de Gauss (Distribuição Normal)"""
    st.subheader("📉 Curva de Gauss - Distribuição do Peso dos Peixes")

    peso_data = data['peso_medio'].dropna()

    if len(peso_data) == 0:
        st.warning("⚠️ Dados de peso não disponíveis")
        return

    # Estatísticas da distribuição
    mean = peso_data.mean()
    std = peso_data.std()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.metric("📊 Média (μ)", f"{mean:.3f} kg")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.metric("📏 Desvio Padrão (σ)", f"{std:.3f} kg")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        cv = (std / mean) * 100 if mean != 0 else 0
        st.metric("📈 Coef. Variação", f"{cv:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.metric("🔢 Amostras", f"{len(peso_data)}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Gráfico da distribuição
    if chart_generator is not None:
        try:
            fig_gaussian = chart_generator.create_histogram_with_normal(peso_data)

            fig_gaussian.add_vline(x=mean, line_dash="dash", line_color="green",
                                   annotation_text=f"μ = {mean:.3f}",
                                   annotation_position="top right")
            fig_gaussian.add_vline(x=mean - std, line_dash="dot", line_color="orange",
                                   annotation_text=f"μ - σ",
                                   annotation_position="bottom left")
            fig_gaussian.add_vline(x=mean + std, line_dash="dot", line_color="orange",
                                   annotation_text=f"μ + σ",
                                   annotation_position="bottom right")

            fig_gaussian.update_layout(
                title=f"Distribuição Normal do Peso (μ={mean:.3f}, σ={std:.3f})",
                xaxis_title="Peso (kg)",
                yaxis_title="Densidade de Probabilidade",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig_gaussian, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Erro ao usar ChartGenerator para Gaussiana: {e}. Usando gráfico padrão.")
            display_gaussian_chart_fallback(peso_data, mean, std)
    else:
        display_gaussian_chart_fallback(peso_data, mean, std)

    # Teste de normalidade
    if analysis and 'normality_test' in analysis:
        st.subheader("🔬 Teste de Normalidade (Shapiro-Wilk)")

        normality = analysis['normality_test']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            if 'shapiro' in normality:
                st.metric("📊 Estatística W", f"{normality['shapiro']['statistic']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            if 'shapiro' in normality:
                st.metric("📈 p-valor", f"{normality['shapiro']['p_value']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)

        is_normal = normality.get('shapiro', {}).get('is_normal', False)

        if is_normal:
            st.success("✅ **Distribuição Normal**: Os dados seguem uma distribuição normal (p > 0.05)")
        else:
            st.warning("⚠️ **Distribuição Não-Normal**: Os dados não seguem uma distribuição normal (p ≤ 0.05)")

def display_gaussian_chart_fallback(peso_data, mean, std):
    """Gráfico de distribuição Gaussiana padrão (fallback)"""
    fig = go.Figure()

    # Histograma dos dados reais
    fig.add_trace(
        go.Histogram(
            x=peso_data,
            nbinsx=25,
            name='Distribuição Observada',
            opacity=0.7,
            marker_color='lightblue',
            histnorm='probability density'
        )
    )

    # Curva de Gauss teórica
    x_range = np.linspace(peso_data.min() - 2 * std, peso_data.max() + 2 * std, 100)
    y_gauss = stats.norm.pdf(x_range, mean, std)

    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_gauss,
            mode='lines',
            name='Curva de Gauss Teórica',
            line=dict(color='red', width=3)
        )
    )

    # Linhas de referência
    fig.add_vline(x=mean, line_dash="dash", line_color="green",
                  annotation_text=f"μ = {mean:.3f}")
    fig.add_vline(x=mean - std, line_dash="dot", line_color="orange",
                  annotation_text=f"μ - σ")
    fig.add_vline(x=mean + std, line_dash="dot", line_color="orange",
                  annotation_text=f"μ + σ")

    fig.update_layout(
        title=f"Distribuição Normal do Peso (μ={mean:.3f}, σ={std:.3f})",
        xaxis_title="Peso (kg)",
        yaxis_title="Densidade de Probabilidade",
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

def display_advanced_statistics(data, analysis, chart_generator=None):
    """Exibe estatísticas avançadas e correlações"""
    st.subheader("🔬 Análises Estatísticas Avançadas")

    # Matriz de correlação
    if analysis and 'correlations' in analysis:
        st.subheader("📊 Matriz de Correlação")

        corr_matrix = analysis['correlations']

        # Filtrar apenas correlações relevantes
        relevant_cols = ['peso_medio', 'total_racao', 'eficiencia_alimentar', 'n_peixes']
        if 'tca_periodo' in corr_matrix.columns:
            relevant_cols.append('tca_periodo')

        available_cols = [col for col in relevant_cols if col in corr_matrix.columns]

        if len(available_cols) > 1:
            corr_subset = corr_matrix.loc[available_cols, available_cols]

            # Usar ChartGenerator se disponível
            if chart_generator is not None:
                try:
                    fig_corr = chart_generator.create_correlation_matrix(corr_subset)
                    st.plotly_chart(fig_corr, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Erro ao usar ChartGenerator para correlação: {e}. Usando heatmap padrão.")
                    display_correlation_heatmap_fallback(corr_subset)
            else:
                display_correlation_heatmap_fallback(corr_subset)

            # Interpretação das correlações
            st.subheader("🔍 Interpretação das Correlações")

            strong_correlations = []
            for i in range(len(available_cols)):
                for j in range(i + 1, len(available_cols)):
                    corr_value = corr_subset.iloc[i, j]
                    var1 = available_cols[i]
                    var2 = available_cols[j]

                    if abs(corr_value) > 0.7:
                        strength = "muito forte"
                        color = "🔴" if corr_value > 0 else "🔵"
                    elif abs(corr_value) > 0.5:
                        strength = "forte"
                        color = "🟠" if corr_value > 0 else "🟦"
                    elif abs(corr_value) > 0.3:
                        strength = "moderada"
                        color = "🟡" if corr_value > 0 else "🟪"
                    else:
                        continue

                    direction = "positiva" if corr_value > 0 else "negativa"
                    strong_correlations.append(
                        f"{color} **{var1}** vs **{var2}**: Correlação {strength} {direction} (r = {corr_value:.3f})"
                    )

            if strong_correlations:
                for corr_text in strong_correlations:
                    st.write(corr_text)
            else:
                st.info("ℹ️ Não foram encontradas correlações significativas (|r| > 0.3)")

    # Regressão linear
    if analysis and 'regression' in analysis:
        st.subheader("📈 Análise de Regressão: Ração vs Peso")

        reg = analysis['regression']

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.metric("📊 Coeficiente Angular", f"{reg['slope']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.metric("📈 Intercepto", f"{reg['intercept']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.metric("🎯 R² (Ajuste)", f"{reg['r2']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Gráfico de regressão
        if chart_generator is not None:
            try:
                fig_reg = chart_generator.create_regression_plot(reg['X'], reg['y'], reg)
                st.plotly_chart(fig_reg, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Erro ao usar ChartGenerator para regressão: {e}. Usando gráfico padrão.")
                display_regression_chart_fallback(reg)
        else:
            display_regression_chart_fallback(reg)

        # Interpretação da regressão
        st.info(f"""
        **Interpretação da Regressão:**
        - Para cada 1 kg de ração adicional, o peso aumenta em média {reg['slope']:.4f} kg
        - O modelo explica {reg['r2'] * 100:.1f}% da variação no peso dos peixes
        - {"Modelo com bom ajuste" if reg['r2'] > 0.7 else "Modelo com ajuste moderado" if reg['r2'] > 0.5 else "Modelo com ajuste fraco"}
        """)

    # Estatísticas detalhadas
    if analysis and 'peso_statistics' in analysis:
        st.subheader("📋 Estatísticas Detalhadas do Peso")

        peso_stats = analysis['peso_statistics']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.write(f"**Média:** {peso_stats['mean']:.3f} kg")
            st.write(f"**Mediana:** {peso_stats.get('median', 0):.3f} kg")
            st.write(f"**Desvio Padrão:** {peso_stats['std']:.3f} kg")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.write(f"**Mínimo:** {peso_stats['min']:.3f} kg")
            st.write(f"**Máximo:** {peso_stats['max']:.3f} kg")
            st.write(f"**Amplitude:** {peso_stats['max'] - peso_stats['min']:.3f} kg")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.write(f"**Amostras:** {peso_stats['count']}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.write("**Distribuição:**")
            st.write("Normal" if analysis.get('normality_test', {}).get('shapiro', {}).get('is_normal',
                                                                                           False) else "Não-Normal")
            st.markdown('</div>', unsafe_allow_html=True)

def display_correlation_heatmap_fallback(corr_subset):
    """Heatmap de correlação padrão (fallback)"""
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_subset.values,
        x=corr_subset.columns,
        y=corr_subset.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_subset.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))

    fig_corr.update_layout(
        title="Matriz de Correlação entre Variáveis",
        height=500,
        template="plotly_white"
    )

    st.plotly_chart(fig_corr, use_container_width=True)

def display_regression_chart_fallback(reg):
    """Gráfico de regressão padrão (fallback)"""
    fig_reg = go.Figure()

    # Pontos observados
    fig_reg.add_trace(go.Scatter(
        x=reg['X'],
        y=reg['y'],
        mode='markers',
        name='Dados Observados',
        marker=dict(size=8, color='blue', opacity=0.6)
    ))

    # Linha de regressão
    fig_reg.add_trace(go.Scatter(
        x=reg['X'],
        y=reg['predictions'],
        mode='lines',
        name=f'Regressão (R² = {reg["r2"]:.3f})',
        line=dict(color='red', width=3)
    ))

    fig_reg.update_layout(
        title=f"Regressão Linear: Peso = {reg['slope']:.4f} × Ração + {reg['intercept']:.4f}",
        xaxis_title="Ração Consumida (kg)",
        yaxis_title="Peso Médio (kg)",
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig_reg, use_container_width=True)

def display_ai_assistant_placeholder():
    """Placeholder para o assistente de IA"""
    st.subheader("🤖 Assistente de IA - Interpretação Inteligente")

    st.info(f"""
    **🚀 Funcionalidade em Desenvolvimento**

    O Assistente de IA Prof. Tilap-IA está disponível através do ícone no topo da página.

    **Funcionalidades disponíveis:**
    - 📊 Análises rápidas e personalizadas
    - 💬 Chat interativo sobre seus dados
    - 📋 Relatórios automatizados com IA
    - 💡 Recomendações específicas para aquicultura

    **Como usar:**
    1. Clique no ícone do Prof. Tilap-IA (🧑‍🏫)
    2. Acesse o assistente especializado
    3. Faça perguntas sobre seus dados
    4. Receba insights e recomendações
    """)

    # Botão para acessar o assistente
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Acessar Prof. Tilap-IA", use_container_width=True):
            st.markdown(f"""
            <script>
            window.open('{PROFESSOR_ASSISTANT_URL}', '_blank');
            </script>
            """, unsafe_allow_html=True)
            st.success("✅ Abrindo Prof. Tilap-IA em nova aba...")

def display_automated_reports(data, analysis):
    """Gera relatórios automatizados"""
    st.subheader("📋 Relatórios Automatizados")

    # Relatório Executivo
    st.subheader("📊 Relatório Executivo")

    # Resumo geral
    total_peixes = data['n_peixes'].sum()
    peso_medio_geral = data['peso_medio'].mean()
    total_racao = data['total_racao'].sum()
    eficiencia_geral = data['eficiencia_alimentar'].mean()

    st.markdown(f"""
    ### 📈 **Resumo do Período**

    **🐟 População Total:** {total_peixes:,.0f} peixes  
    **⚖️ Peso Médio Geral:** {peso_medio_geral:.3f} kg  
    **🍽️ Ração Total Consumida:** {total_racao:.1f} kg  
    **⚡ Eficiência Alimentar Média:** {eficiencia_geral:.2f}  

    ---
    """)

    # Análise de performance
    if 'tca_periodo' in data.columns:
        tca_media = data['tca_periodo'].mean()

        if tca_media < 1.5:
            performance = "🟢 **EXCELENTE**"
            recomendacao = "Manter as práticas atuais de manejo."
        elif tca_media < 2.0:
            performance = "🔵 **BOA**"
            recomendacao = "Otimizar horários de alimentação e qualidade da ração."
        elif tca_media < 2.5:
            performance = "🟡 **REGULAR**"
            recomendacao = "Revisar estratégia alimentar e monitorar qualidade da água."
        else:
            performance = "🔴 **BAIXA**"
            recomendacao = "Intervenção necessária: revisar ração, densidade e ambiente."

        st.markdown(f"""
        ### 🎯 **Avaliação de Performance**

        **Taxa de Conversão Alimentar:** {tca_media:.2f}  
        **Status:** {performance}  
        **Recomendação:** {recomendacao}

        ---
        """)

    # Alertas e recomendações
    st.subheader("🚨 Alertas e Recomendações")

    alertas = []

    # Verificar eficiência baixa
    tanques_baixa_eff = data[data['eficiencia_alimentar'] < 1.0]['tanque'].unique()
    if len(tanques_baixa_eff) > 0:
        alertas.append(f"⚠️ **Eficiência baixa** nos tanques: {', '.join(map(str, tanques_baixa_eff))}")

    # Verificar variabilidade alta
    cv_peso = (data['peso_medio'].std() / data['peso_medio'].mean()) * 100
    if cv_peso > 20:
        alertas.append(f"📊 **Alta variabilidade** no peso (CV = {cv_peso:.1f}%)")

    # Verificar crescimento
    if 'crescimento_diario' in data.columns:
        crescimento_negativo = data[data['crescimento_diario'] < 0]
        if len(crescimento_negativo) > 0:
            alertas.append("📉 **Períodos de crescimento negativo** detectados")

    if alertas:
        for alerta in alertas:
            st.warning(alerta)
    else:
        st.success("✅ **Nenhum alerta crítico identificado**")

    # Tendências identificadas
    st.subheader("📈 Tendências Identificadas")

    if len(data) > 1:
        # Tendência de peso
        peso_inicial = data['peso_medio'].iloc[0]
        peso_final = data['peso_medio'].iloc[-1]
        variacao_peso = ((peso_final - peso_inicial) / peso_inicial) * 100

        if variacao_peso > 5:
            trend_peso = f"📈 **Crescimento positivo** de {variacao_peso:.1f}%"
        elif variacao_peso < -5:
            trend_peso = f"📉 **Declínio** de {abs(variacao_peso):.1f}%"
        else:
            trend_peso = f"➡️ **Estabilidade** (variação de {variacao_peso:.1f}%)"

        st.write(trend_peso)

        # Tendência de eficiência
        if 'eficiencia_alimentar' in data.columns:
            eff_inicial = data['eficiencia_alimentar'].iloc[0]
            eff_final = data['eficiencia_alimentar'].iloc[-1]

            if eff_final > eff_inicial * 1.1:
                trend_eff = "📈 **Melhoria na eficiência alimentar**"
            elif eff_final < eff_inicial * 0.9:
                trend_eff = "📉 **Declínio na eficiência alimentar**"
            else:
                trend_eff = "➡️ **Eficiência estável**"

            st.write(trend_eff)

    # Recomendações específicas
    st.subheader("💡 Recomendações Específicas")

    recomendacoes = [
        "🔄 **Monitoramento contínuo** da TCA para otimização",
        "📊 **Análise semanal** dos indicadores de performance",
        "🌡️ **Controle rigoroso** dos parâmetros ambientais",
        "📈 **Ajuste da estratégia alimentar** baseado nos dados"
    ]

    for rec in recomendacoes:
        st.write(rec)

def display_advanced_export(data, analysis):
    """Exportação avançada com múltiplos formatos"""
    st.subheader("💾 Exportação Avançada")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Opções de Exportação")

        export_format = st.selectbox(
            "📁 Formato:",
            ["CSV", "Excel Completo", "JSON Dados"],
        )

        include_charts = st.checkbox("📊 Incluir gráficos", True)
        include_statistics = st.checkbox("📈 Incluir estatísticas", True)
        include_analysis = st.checkbox("🔬 Incluir análises", True)

    with col2:
        st.subheader("⚙️ Configurações")

        date_format = st.selectbox(
            "📅 Formato de data:",
            ["DD/MM/YYYY", "YYYY-MM-DD", "MM/DD/YYYY"],
        )

        decimal_places = st.slider("🔢 Casas decimais:", 1, 4, 3)

        include_metadata = st.checkbox("📝 Incluir metadados", True)

    # Botões de exportação
    st.subheader("📥 Download")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Exportar CSV", use_container_width=True):
            csv_data = prepare_csv_export(data, decimal_places, date_format)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"aquicultura_completo_{timestamp}.csv"

            st.download_button(
                "📥 Download CSV",
                csv_data,
                filename,
                "text/csv",
            )

    with col2:
        if st.button("📊 Exportar Excel", use_container_width=True):
            excel_data = prepare_excel_export(
                data, analysis, include_charts,
                include_statistics, decimal_places
            )

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"relatorio_aquicultura_{timestamp}.xlsx"

            st.download_button(
                "📥 Download Excel",
                excel_data,
                filename,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with col3:
        if st.button("🔗 Exportar JSON", use_container_width=True):
            json_data = prepare_json_export(data, analysis, include_metadata)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"dados_aquicultura_{timestamp}.json"

            st.download_button(
                "📥 Download JSON",
                json_data,
                filename,
                "application/json",
            )

    # Preview dos dados
    st.subheader("👀 Preview dos Dados")

    preview_data = data.copy()
    numeric_cols = preview_data.select_dtypes(include=[np.number]).columns
    preview_data[numeric_cols] = preview_data[numeric_cols].round(decimal_places)

    if date_format == "DD/MM/YYYY":
        preview_data['data'] = preview_data['data'].dt.strftime('%d/%m/%Y')
    elif date_format == "MM/DD/YYYY":
        preview_data['data'] = preview_data['data'].dt.strftime('%m/%d/%Y')
    else:
        preview_data['data'] = preview_data['data'].dt.strftime('%Y-%m-%d')

    st.dataframe(preview_data.head(10), use_container_width=True)

    # Informações do dataset
    st.subheader("ℹ️ Informações do Dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📊 Total de Registros", len(data))

    with col2:
        st.metric("📅 Período", f"{len(data['data'].unique())} dias")

    with col3:
        st.metric("🏊 Tanques", len(data['tanque'].unique()))

def prepare_csv_export(data, decimal_places, date_format):
    """Prepara dados para exportação CSV"""
    export_data = data.copy()

    # Formatar data
    if date_format == "DD/MM/YYYY":
        export_data['data'] = export_data['data'].dt.strftime('%d/%m/%Y')
    elif date_format == "MM/DD/YYYY":
        export_data['data'] = export_data['data'].dt.strftime('%m/%d/%Y')
    else:
        export_data['data'] = export_data['data'].dt.strftime('%Y-%m-%d')

    # Arredondar valores numéricos
    numeric_cols = export_data.select_dtypes(include=[np.number]).columns
    export_data[numeric_cols] = export_data[numeric_cols].round(decimal_places)

    return export_data.to_csv(index=False)

def prepare_excel_export(data, analysis, include_charts, include_statistics, decimal_places):
    """Prepara dados para exportação Excel"""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Dados principais
        main_data = data.copy()
        numeric_cols = main_data.select_dtypes(include=[np.number]).columns
        main_data[numeric_cols] = main_data[numeric_cols].round(decimal_places)
        main_data.to_excel(writer, sheet_name='Dados Principais', index=False)

        # Estatísticas
        if include_statistics and analysis:
            if 'peso_statistics' in analysis:
                peso_stats = pd.DataFrame([analysis['peso_statistics']])
                peso_stats.to_excel(writer, sheet_name='Estatísticas Peso', index=False)

            if 'tank_analysis' in analysis:
                tank_stats = pd.DataFrame(analysis['tank_analysis']).T
                tank_stats.to_excel(writer, sheet_name='Análise por Tanque')

        # Correlações
        if analysis and 'correlations' in analysis:
            analysis['correlations'].to_excel(writer, sheet_name='Correlações')

        # Resumo por tanque
        if 'tanque' in data.columns:
            resumo_tanque = data.groupby('tanque').agg({
                'peso_medio': ['mean', 'std', 'min', 'max'],
                'total_racao': ['sum', 'mean'],
                'eficiencia_alimentar': ['mean', 'std'],
                'n_peixes': 'mean'
            }).round(decimal_places)

            resumo_tanque.to_excel(writer, sheet_name='Resumo por Tanque')

    return output.getvalue()

def prepare_json_export(data, analysis, include_metadata):
    """Prepara dados para exportação JSON"""
    import json

    # Converter dados para formato serializável
    data_dict = data.copy()

    # Converter datetime para string
    if 'data' in data_dict.columns:
        data_dict['data'] = data_dict['data'].dt.strftime('%Y-%m-%d')

    # Converter numpy types para tipos Python nativos
    for col in data_dict.select_dtypes(include=[np.number]).columns:
        data_dict[col] = data_dict[col].astype(float)

    export_dict = {
        "dados": data_dict.to_dict('records'),
        "estatisticas": analysis if analysis else {},
        "resumo": {
            "total_registros": int(len(data)),
            "periodo_dias": int(len(data['data'].unique())),
            "tanques": int(len(data['tanque'].unique())),
            "peso_medio_geral": float(data['peso_medio'].mean()),
            "total_racao": float(data['total_racao'].sum()),
            "eficiencia_media": float(data['eficiencia_alimentar'].mean())
        }
    }

    if include_metadata:
        export_dict["metadata"] = {
            "data_exportacao": datetime.now().isoformat(),
            "versao": "1.0",
            "fonte": "Aquicultura Analytics Pro",
            "colunas": list(data.columns),
            "tipos_dados": {col: str(dtype) for col, dtype in data.dtypes.items()}
        }

    return json.dumps(export_dict, indent=2, ensure_ascii=False)

def display_data_preview():
    """Mostra preview dos dados carregados do Google Sheets"""
    st.subheader("📊 Preview dos Dados do Google Sheets")

    sheets_data = st.session_state.sheets_data

    tab1, tab2 = st.tabs(["🐟 Dados das Tilápias", "🍽️ Dados de Ração"])

    with tab1:
        if 'tilapias' in sheets_data:
            tilapias_df = sheets_data['tilapias']
            st.write(f"**Total de registros:** {len(tilapias_df)}")
            st.write(f"**Colunas:** {list(tilapias_df.columns)}")

            # Estatísticas básicas
            if 'data' in tilapias_df.columns:
                try:
                    tilapias_df['data'] = pd.to_datetime(tilapias_df['data'], errors='coerce')
                    valid_dates = tilapias_df['data'].dropna()
                    if not valid_dates.empty:
                        date_range = f"{valid_dates.min().strftime('%d/%m/%Y')} a {valid_dates.max().strftime('%d/%m/%Y')}"
                        st.write(f"**Período:** {date_range}")
                except:
                    st.write("**Período:** Não foi possível determinar")

            # Mostrar amostra
            st.dataframe(tilapias_df.head(10), use_container_width=True)

            # Estatísticas das colunas numéricas
            numeric_cols = tilapias_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.write("**Estatísticas das colunas numéricas:**")
                st.dataframe(tilapias_df[numeric_cols].describe(), use_container_width=True)
        else:
            st.error("❌ Dados das tilápias não carregados")

    with tab2:
        if 'racao' in sheets_data:
            racao_df = sheets_data['racao']
            st.write(f"**Total de registros:** {len(racao_df)}")
            st.write(f"**Colunas:** {list(racao_df.columns)}")

            # Estatísticas básicas
            if 'data' in racao_df.columns:
                try:
                    racao_df['data'] = pd.to_datetime(racao_df['data'], errors='coerce')
                    valid_dates = racao_df['data'].dropna()
                    if not valid_dates.empty:
                        date_range = f"{valid_dates.min().strftime('%d/%m/%Y')} a {valid_dates.max().strftime('%d/%m/%Y')}"
                        st.write(f"**Período:** {date_range}")
                except:
                    st.write("**Período:** Não foi possível determinar")

            # Mostrar amostra
            st.dataframe(racao_df.head(10), use_container_width=True)

            # Estatísticas das colunas numéricas
            numeric_cols = racao_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.write("**Estatísticas das colunas numéricas:**")
                st.dataframe(racao_df[numeric_cols].describe(), use_container_width=True)
        else:
            st.error("❌ Dados de ração não carregados")

def display_welcome_advanced():
    """Tela de boas-vindas avançada"""
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2 style="color: #1e3a8a;">🔗 Sistema Avançado de Aquicultura</h2>
        <p style="font-size: 1.1rem; color: #64748b;">
            Conectado ao Google Sheets com análises estatísticas avançadas
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        ### 🚀 Funcionalidades Avançadas

        #### 🔍 **Filtros Inteligentes**
        - **Período personalizável**: Formato dd/mm/aa
        - **Seleção de tanques**: Um, vários ou todos
        - **Nível de confiança**: 90%, 95% ou 99%

        #### 📊 **Análises Estatísticas**
        - **Taxa de Conversão Alimentar (TCA)**
        - **Curva de Gauss** do peso dos peixes
        - **Regressões e correlações** avançadas
        - **Análise temporal** peso vs ração

        #### 📈 **Gráficos Sofisticados**
        - **Distribuições** e tendências
        - **Correlações** multivariadas
        - **Evolução temporal** por tanque
        - **Dashboards** interativos

        #### 📋 **Relatórios Automatizados**
        - **Exportação CSV/Excel/JSON** com gráficos
        - **Dashboards** executivos
        - **Análises** detalhadas por tanque

        #### 🤖 **Assistente de IA**
        - **Prof. Tilap-IA** disponível no topo da página
        - **Interpretação inteligente** dos dados
        - **Recomendações personalizadas**
        - **Chat especializado** em aquicultura
        """)

        st.info("""
        💡 **Como usar:**
        1. Clique em "Carregar/Atualizar Dados"
        2. Configure os filtros desejados
        3. Clique em "Processar Análise Completa"
        4. Explore as abas de análises
        5. Use o Prof. Tilap-IA para insights adicionais
        """)

def display_welcome_advanced():
    """Tela de boas-vindas avançada"""
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2 style="color: #1e3a8a;">🔗 Sistema Avançado de Aquicultura</h2>
        <p style="font-size: 1.1rem; color: #64748b;">
            Conectado ao Google Sheets com análises estatísticas avançadas
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        ### 🚀 Funcionalidades Avançadas

        #### 🔍 **Filtros Inteligentes**
        - **Período personalizável**: Formato dd/mm/aa
        - **Seleção de tanques**: Um, vários ou todos
        - **Nível de confiança**: 90%, 95% ou 99%

        #### 📊 **Análises Estatísticas**
        - **Taxa de Conversão Alimentar (TCA)**
        - **Curva de Gauss** do peso dos peixes
        - **Regressões e correlações** avançadas
        - **Análise temporal** peso vs ração

        #### 📈 **Gráficos Sofisticados**
        - **Distribuições** e tendências
        - **Correlações** multivariadas
        - **Evolução temporal** por tanque
        - **Dashboards** interativos

        #### 📋 **Relatórios Automatizados**
        - **Exportação CSV/Excel/JSON** com gráficos
        - **Dashboards** executivos
        - **Análises** detalhadas por tanque

        #### 🤖 **Assistente de IA**
        - **Prof. Tilap-IA** disponível no topo da página
        - **Interpretação inteligente** dos dados
        - **Recomendações personalizadas**
        - **Chat especializado** em aquicultura
        """)

        st.info("""
        💡 **Como usar:**
        1. Clique em "Carregar/Atualizar Dados"
        2. Configure os filtros desejados
        3. Clique em "Processar Análise Completa"
        4. Explore as abas de análises
        5. Use o Prof. Tilap-IA para insights adicionais
        """)

def display_data_preview():
    """Mostra preview dos dados carregados do Google Sheets"""
    st.subheader("📊 Preview dos Dados do Google Sheets")

    if 'sheets_data' not in st.session_state or st.session_state.sheets_data is None:
        st.warning("⚠️ Nenhum dado carregado. Clique em 'Carregar/Atualizar Dados' na sidebar.")
        return

    sheets_data = st.session_state.sheets_data

    tab1, tab2 = st.tabs(["🐟 Dados das Tilápias", "🍽️ Dados de Ração"])

    with tab1:
        if 'tilapias' in sheets_data:
            tilapias_df = sheets_data['tilapias']
            st.write(f"**Total de registros:** {len(tilapias_df)}")
            st.write(f"**Colunas:** {list(tilapias_df.columns)}")

            # Mostrar amostra
            st.dataframe(tilapias_df.head(10), use_container_width=True)

            # Estatísticas das colunas numéricas
            numeric_cols = tilapias_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.write("**Estatísticas das colunas numéricas:**")
                st.dataframe(tilapias_df[numeric_cols].describe(), use_container_width=True)
        else:
            st.error("❌ Dados das tilápias não carregados")

    with tab2:
        if 'racao' in sheets_data:
            racao_df = sheets_data['racao']
            st.write(f"**Total de registros:** {len(racao_df)}")
            st.write(f"**Colunas:** {list(racao_df.columns)}")

            # Mostrar amostra
            st.dataframe(racao_df.head(10), use_container_width=True)

            # Estatísticas das colunas numéricas
            numeric_cols = racao_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.write("**Estatísticas das colunas numéricas:**")
                st.dataframe(racao_df[numeric_cols].describe(), use_container_width=True)
        else:
            st.error("❌ Dados de ração não carregados")

def display_ai_assistant_placeholder():
    """Placeholder para o assistente de IA"""
    st.subheader("🤖 Assistente de IA - Interpretação Inteligente")

    st.info(f"""
    **🚀 Funcionalidade em Desenvolvimento**

    O Assistente de IA Prof. Tilap-IA está disponível através do ícone no topo da página.

    **Funcionalidades disponíveis:**
    - 📊 Análises rápidas e personalizadas
    - 💬 Chat interativo sobre seus dados
    - 📋 Relatórios automatizados com IA
    - 💡 Recomendações específicas para aquicultura

    **Como usar:**
    1. Clique no ícone do Prof. Tilap-IA (🧑‍🏫)
    2. Acesse o assistente especializado
    3. Faça perguntas sobre seus dados
    4. Receba insights e recomendações
    """)

    # Botão para acessar o assistente
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Acessar Prof. Tilap-IA", use_container_width=True):
            st.markdown(f"""
            <script>
            window.open('{PROFESSOR_ASSISTANT_URL}', '_blank');
            </script>
            """, unsafe_allow_html=True)
            st.success("✅ Abrindo Prof. Tilap-IA em nova aba...")

def display_data_preview():
    """Mostra preview dos dados carregados do Google Sheets"""
    st.subheader("📊 Preview dos Dados do Google Sheets")

    if 'sheets_data' not in st.session_state or st.session_state.sheets_data is None:
        st.warning("⚠️ Nenhum dado carregado. Clique em 'Carregar/Atualizar Dados' na sidebar.")
        return

    sheets_data = st.session_state.sheets_data

    tab1, tab2 = st.tabs(["🐟 Dados das Tilápias", "🍽️ Dados de Ração"])

    with tab1:
        if 'tilapias' in sheets_data:
            tilapias_df = sheets_data['tilapias']
            st.write(f"**Total de registros:** {len(tilapias_df)}")
            st.write(f"**Colunas:** {list(tilapias_df.columns)}")

            # Estatísticas básicas
            if 'data' in tilapias_df.columns:
                try:
                    tilapias_df['data'] = pd.to_datetime(tilapias_df['data'], errors='coerce')
                    valid_dates = tilapias_df['data'].dropna()
                    if not valid_dates.empty:
                        date_range = f"{valid_dates.min().strftime('%d/%m/%Y')} a {valid_dates.max().strftime('%d/%m/%Y')}"
                        st.write(f"**Período:** {date_range}")
                except:
                    st.write("**Período:** Não foi possível determinar")

            # Mostrar amostra
            st.dataframe(tilapias_df.head(10), use_container_width=True)

            # Estatísticas das colunas numéricas
            numeric_cols = tilapias_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.write("**Estatísticas das colunas numéricas:**")
                st.dataframe(tilapias_df[numeric_cols].describe(), use_container_width=True)
        else:
            st.error("❌ Dados das tilápias não carregados")

    with tab2:
        if 'racao' in sheets_data:
            racao_df = sheets_data['racao']
            st.write(f"**Total de registros:** {len(racao_df)}")
            st.write(f"**Colunas:** {list(racao_df.columns)}")

            # Estatísticas básicas
            if 'data' in racao_df.columns:
                try:
                    racao_df['data'] = pd.to_datetime(racao_df['data'], errors='coerce')
                    valid_dates = racao_df['data'].dropna()
                    if not valid_dates.empty:
                        date_range = f"{valid_dates.min().strftime('%d/%m/%Y')} a {valid_dates.max().strftime('%d/%m/%Y')}"
                        st.write(f"**Período:** {date_range}")
                except:
                    st.write("**Período:** Não foi possível determinar")

            # Mostrar amostra
            st.dataframe(racao_df.head(10), use_container_width=True)

            # Estatísticas das colunas numéricas
            numeric_cols = racao_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.write("**Estatísticas das colunas numéricas:**")
                st.dataframe(racao_df[numeric_cols].describe(), use_container_width=True)
        else:
            st.error("❌ Dados de ração não carregados")

def display_welcome_advanced():
    """Tela de boas-vindas avançada"""
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2 style="color: #1e3a8a;">🔗 Sistema Avançado de Aquicultura</h2>
        <p style="font-size: 1.1rem; color: #64748b;">
            Conectado ao Google Sheets com análises estatísticas avançadas
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        ### 🚀 Funcionalidades Avançadas

        #### 🔍 **Filtros Inteligentes**
        - **Período personalizável**: Formato dd/mm/aa
        - **Seleção de tanques**: Um, vários ou todos
        - **Nível de confiança**: 90%, 95% ou 99%

        #### 📊 **Análises Estatísticas**
        - **Taxa de Conversão Alimentar (TCA)**
        - **Curva de Gauss** do peso dos peixes
        - **Regressões e correlações** avançadas
        - **Análise temporal** peso vs ração

        #### 📈 **Gráficos Sofisticados**
        - **Distribuições** e tendências
        - **Correlações** multivariadas
        - **Evolução temporal** por tanque
        - **Dashboards** interativos

        #### 📋 **Relatórios Automatizados**
        - **Exportação CSV/Excel/JSON** com gráficos
        - **Dashboards** executivos
        - **Análises** detalhadas por tanque

        #### 🤖 **Assistente de IA**
        - **Prof. Tilap-IA** disponível no topo da página
        - **Interpretação inteligente** dos dados
        - **Recomendações personalizadas**
        - **Chat especializado** em aquicultura
        """)

        st.info("""
        💡 **Como usar:**
        1. Clique em "Carregar/Atualizar Dados"
        2. Configure os filtros desejados
        3. Clique em "Processar Análise Completa"
        4. Explore as abas de análises
        5. Use o Prof. Tilap-IA para insights adicionais
        """)






    # Executar aplicação principal
    if __name__ == "__main__":
        main()