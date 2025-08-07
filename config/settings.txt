"""
Configurações do Sistema Aquicultura Analytics
"""

# Configurações de dados
DATA_CONFIG = {
    'max_file_size_mb': 100,
    'supported_formats': ['.xlsx', '.csv'],
    'required_fish_columns': ['largura', 'altura'],
    'required_feed_columns': ['peso', 'data'],
    'date_formats': ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y'],
    'encoding': 'utf-8'
}

# Configurações estatísticas
STATS_CONFIG = {
    'confidence_levels': [0.90, 0.95, 0.99],
    'default_confidence': 0.95,
    'outlier_methods': ['iqr', 'zscore', 'modified_zscore'],
    'normality_tests': ['shapiro', 'ks', 'anderson'],
    'min_sample_size': 3
}

# Configurações de visualização
VIZ_CONFIG = {
    'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'template': 'plotly_white',
    'default_height': 500,
    'default_width': 800,
    'font_family': 'Arial, sans-serif',
    'font_size': 12
}

# Configurações de peso (CALIBRAÇÃO NECESSÁRIA)
WEIGHT_CONFIG = {
    'correction_factor_range': (0.1, 10.0),
    'default_correction_factor': 1.0,
    'density_fish': 1.05,  # kg/L - AJUSTAR
    'form_factor': 0.7,  # AJUSTAR
    'depth_factor': 0.6,  # AJUSTAR
    'allometric_coefficient': 3.2,  # AJUSTAR
    'intercept': 0.001,  # AJUSTAR
    'min_weight': 0.01,  # kg
    'max_weight': 100.0,  # kg
    'timestamp_format': '%Y%m%d_%H%M%S',
    'water_quality_ranges': {
        'temperature': (20, 30),  # °C
        'ph': (6.5, 8.5),
        'oxygen': (5, 15)  # mg/L
    }
}

# Configurações de tanques
TANK_CONFIG = {
    'max_tanks': 100,
    'default_tank_capacity': 1000,  # L
    'optimal_density': 20,  # kg/m³
}

# Configurações de interface
UI_CONFIG = {
    'page_title': 'Aquicultura Analytics Pro',
    'page_icon': '🐟',
    'layout': 'wide',
    'sidebar_state': 'expanded',
    'theme': 'light',
    'language': 'pt-BR'
}

# Configurações de exportação
EXPORT_CONFIG = {
    'formats': ['csv', 'xlsx', 'json'],
    'default_format': 'csv',
    'include_metadata': True,
    'timestamp_format': '%Y%m%d_%H%M%S'
}

# Mensagens do sistema
MESSAGES = {
    'pt-BR': {
        'upload_success': '✅ Dados carregados com sucesso: {} registros',
        'upload_error_fish': '❌ Erro ao carregar dados dos peixes: {}',
        'upload_error_feed': '❌ Erro ao carregar dados de ração: {}',
        'connect_success_fish': '✅ Dados dos peixes conectados: {} registros',
        'connect_success_feed': '✅ Dados de ração conectados: {} registros',
        'connect_error_fish': '❌ Erro ao conectar planilha de peixes: {}',
        'connect_error_feed': '❌ Erro ao conectar planilha de ração: {}',
        'invalid_google_sheet_url': '⚠️ URL do Google Sheets inválida ou sem permissão de acesso público.',
        'processing_error': '⚠️ Erro no processamento de dados: {}',
        'processing_failed_or_empty': '⚠️ O processamento dos dados falhou ou resultou em um conjunto de dados vazio.',
        'insufficient_data': 'ℹ️ Dados insuficientes para análise.',
        'calibration_needed': '🔧 Calibração necessária para maior precisão.',
        'outliers_detected': '⚠️ {} outliers detectados ({:.1f}%).',
        'normal_distribution_message': '✅ Distribuição normal (p-value: {:.4f}).',
        'non_normal_distribution_message': '⚠️ Distribuição não-normal (p-value: {:.4f}).',

        # Mensagens específicas para tratamento de erros
        'no_fish_data_for_dates': "Dados de peixes não carregados ou coluna 'data' ausente. Usando data padrão.",
        'no_feed_data_for_tanks': "Dados de ração ou coluna 'tanque' ausentes para seleção de tanques.",
        'no_tanks_found': "Nenhum tanque encontrado nos dados de ração.",
        'no_data_after_date_filter': "Nenhum dado encontrado após o filtro de período selecionado.",
        'no_data_after_tank_filter': "Nenhum dado encontrado após o filtro de tanques selecionado.",
        'missing_date_col_fish': "Coluna de data (e.g., 'data', 'date') não encontrada nos dados dos peixes.",
        'missing_dim_cols_fish': "Colunas de dimensão (e.g., 'largura', 'altura') não encontradas nos dados dos peixes.",
        'datetime_combine_warning': "Não foi possível combinar 'data' e 'hora' em 'datetime'. Usando apenas 'data'.",
        'clean_fish_data_error': "Erro na limpeza dos dados dos peixes: {}",
        'missing_date_col_feed': "Coluna de data (e.g., 'data', 'date') não encontrada nos dados de ração.",
        'missing_weight_col_feed': "Coluna de quantidade de ração (e.g., 'peso', 'racao') não encontrada nos dados de ração.",
        'no_tank_col_feed': "Coluna 'tanque' não encontrada nos dados de ração. Assumindo 'Tanque Único'.",
        'clean_feed_data_error': "Erro na limpeza dos dados de ração: {}",
        'data_integration_error': "Erro na integração dos dados: {}",
        'no_data_for_selection': "Nenhum dado disponível para a seleção atual. Ajuste os filtros.",
        'no_efficiency_data': "Dados de eficiência alimentar não disponíveis.",
        'no_tank_data_for_analysis': "Dados de tanque não disponíveis para análise detalhada.",
        'no_weight_data_for_outliers': "Dados de peso estimados não disponíveis para detecção de outliers.",
        'no_data_to_download': "Nenhum dado para download.",

        # Headers e Labels
        'header_main_indicators': "📊 Indicadores Principais",
        'metric_avg_weight': "Peso Médio dos Peixes",
        'metric_daily_feed': "Ração Média Diária",
        'metric_feed_efficiency': "Eficiência Alimentar",
        'metric_feed_efficiency_unit': "kg peixe/kg ração",
        'metric_analyzed_period': "Período Analisado",
        'days': "dias",
        'day': "dia",
        'records': "registros",
        'ci_fish_weight': "**Intervalo de Confiança (95%) para Peso Médio:** {:.2f} - {:.2f} kg",
        'header_distribution_analysis': "📊 Análise de Distribuição",
        'chart_fish_weight_distribution': "Distribuição Normal - Peso dos Peixes (kg)",
        'chart_feed_consumption_distribution': "Distribuição Normal - Ração Utilizada (kg)",
        'header_growth_analysis': "📈 Análise de Crescimento Temporal",
        'metric_growth_rate': "Taxa de Crescimento",
        'metric_avg_weight_short': "Peso médio",
        'metric_r2_trend': "R² da Tendência",
        'metric_quality_adjustment': "Qualidade do ajuste",
        'metric_feed_increase': "Aumento de Ração",
        'metric_trend': "Tendência",
        'header_correlation_analysis': "🔄 Análise de Correlações",
        'chart_feed_weight_relation': "Relação Ração vs Peso dos Peixes",
        'chart_feed_efficiency_over_time': "Eficiência Alimentar ao Longo do Tempo",
        'header_tank_analysis': "🎯 Análise por Tanque",
        'col_avg_weight': "Peso Médio",
        'col_std_dev': "Desvio Padrão",
        'col_n_records': "N° Registros",
        'col_avg_feed': "Ração Média",
        'col_total_feed': "Ração Total",
        'col_efficiency': "Eficiência",
        'chart_avg_weight_per_tank': "Peso Médio por Tanque",
        'chart_total_feed_per_tank': "Ração Total por Tanque",
        'header_processed_data': "📋 Dados Processados",
        'filter_by_tank': "Filtrar por Tanque",
        'filter_by_date': "Filtrar por Data",
        'checkbox_show_outliers': "Mostrar Outliers",
        'download_csv': "📥 Download CSV",

        # Telas e Abas
        'welcome_header': """
        <div style="text-align: center; padding: 2rem;">
            <h2>🐟 Bem-vindo ao Aquicultura Analytics Pro</h2>
            <p style="font-size: 1.2rem; color: #666;">
                Sistema avançado de análise estatística para aquicultura
            </p>
        </div>
        """,
        'features_header': "### 📋 Funcionalidades Principais",
        'features_list': """
        - **📊 Análise Estatística Avançada**: Distribuições, intervalos de confiança, testes de normalidade
        - **📈 Crescimento Temporal**: Análise de tendências e sazonalidades
        - **🔄 Correlações**: Relações entre peso, ração e eficiência
        - **🎯 Análise por Tanque**: Comparações e rankings
        - **📱 Interface Responsiva**: Fácil de usar em qualquer dispositivo
        """,
        'how_to_start_header': "### 🚀 Como Começar",
        'how_to_start_steps': """
        1. **Carregue os dados** usando o painel lateral
        2. **Configure os parâmetros** de análise
        3. **Explore os resultados** nas diferentes abas
        4. **Exporte os dados** processados
        """,
        'supported_formats_header': "### 📁 Formatos Suportados",
        'supported_formats_list': """
        - **Arquivos Excel** (.xlsx)
        - **Google Sheets** (via URL)
        - **Até 100 tanques** simultâneos
        """,
        'tip_start_upload': "💡 **Dica**: Comece carregando as planilhas 'DadosTilapias.xlsx' e 'DadosRacao.xlsx' no painel lateral",
        'tab_distribution_normality': "📊 Distribuição & Normalidade",
        'tab_growth_temporal': "📈 Crescimento Temporal",
        'tab_correlations': "🔄 Correlações",
        'tab_tank_analysis': "🎯 Análise por Tanque",
        'tab_detailed_data': "📋 Dados Detalhados",
    }
}

# Configurações de logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'aquicultura_analytics.log',
    'max_size_mb': 10,
    'backup_count': 5
}