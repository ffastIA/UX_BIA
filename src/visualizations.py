import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import io
import zipfile


class ChartGenerator:
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        self.template = "plotly_white"
        self.primary_color = "#3b82f6"
        self.secondary_color = "#1e40af"
        self.success_color = "#10b981"
        self.warning_color = "#f59e0b"
        self.danger_color = "#ef4444"

    def create_evolution_chart(self, data):
        """Gráfico de evolução temporal"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Evolução do Peso Médio", "Consumo de Ração"),
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )

        # Peso médio
        fig.add_trace(
            go.Scatter(
                x=data['data'],
                y=data['peso_medio'],
                mode='lines+markers',
                name='Peso Médio',
                line=dict(color=self.primary_color, width=3),
                marker=dict(size=6),
                hovertemplate='<b>Data:</b> %{x}<br><b>Peso:</b> %{y:.2f} kg<extra></extra>'
            ),
            row=1, col=1
        )

        # Eficiência no eixo secundário
        if 'eficiencia_alimentar' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['data'],
                    y=data['eficiencia_alimentar'],
                    mode='lines',
                    name='Eficiência',
                    line=dict(color=self.warning_color, width=2, dash='dash'),
                    yaxis='y2',
                    hovertemplate='<b>Data:</b> %{x}<br><b>Eficiência:</b> %{y:.2f}<extra></extra>'
                ),
                row=1, col=1, secondary_y=True
            )

        # Ração
        fig.add_trace(
            go.Bar(
                x=data['data'],
                y=data['total_racao'],
                name='Ração Total',
                marker_color=self.secondary_color,
                opacity=0.7,
                hovertemplate='<b>Data:</b> %{x}<br><b>Ração:</b> %{y:.1f} kg<extra></extra>'
            ),
            row=2, col=1
        )

        fig.update_layout(
            title="Evolução Temporal dos Indicadores",
            template=self.template,
            height=600,
            showlegend=True
        )

        fig.update_yaxes(title_text="Peso (kg)", row=1, col=1)
        fig.update_yaxes(title_text="Eficiência", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Ração (kg)", row=2, col=1)
        fig.update_xaxes(title_text="Data", row=2, col=1)

        return fig

    def create_weight_distribution(self, data):
        """Distribuição de pesos com curva normal"""
        fig = go.Figure()

        # Histograma
        fig.add_trace(
            go.Histogram(
                x=data['peso_medio'],
                nbinsx=20,
                name='Distribuição Observada',
                opacity=0.7,
                marker_color=self.primary_color,
                histnorm='probability density'
            )
        )

        # Curva normal teórica
        mean = data['peso_medio'].mean()
        std = data['peso_medio'].std()
        x_range = np.linspace(data['peso_medio'].min() - std, data['peso_medio'].max() + std, 100)
        y_normal = stats.norm.pdf(x_range, mean, std)

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_normal,
                mode='lines',
                name='Distribuição Normal Teórica',
                line=dict(color='red', width=3)
            )
        )

        fig.update_layout(
            title=f"Distribuição do Peso dos Peixes (μ={mean:.2f}, σ={std:.2f})",
            xaxis_title="Peso (kg)",
            yaxis_title="Densidade",
            template=self.template,
            height=400
        )

        return fig

    def create_efficiency_by_tank(self, data):
        """Eficiência por tanque"""
        if 'tanque' not in data.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="Dados de tanque não disponíveis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="Eficiência por Tanque",
                template=self.template,
                height=400
            )
            return fig

        tank_stats = data.groupby('tanque').agg({
            'eficiencia_alimentar': ['mean', 'std'],
            'peso_medio': 'mean',
            'total_racao': 'sum'
        }).round(2)

        tank_stats.columns = ['eficiencia_media', 'eficiencia_std', 'peso_medio', 'racao_total']
        tank_stats = tank_stats.reset_index()

        # Definir cores baseadas na eficiência
        colors = []
        for eff in tank_stats['eficiencia_media']:
            if eff >= 2.0:
                colors.append(self.success_color)
            elif eff >= 1.5:
                colors.append(self.warning_color)
            else:
                colors.append(self.danger_color)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=tank_stats['tanque'],
                y=tank_stats['eficiencia_media'],
                error_y=dict(
                    type='data',
                    array=tank_stats['eficiencia_std'],
                    visible=True
                ),
                marker_color=colors,
                text=tank_stats['eficiencia_media'],
                textposition='auto',
                texttemplate='%{text:.2f}',
                hovertemplate='<b>Tanque:</b> %{x}<br><b>Eficiência:</b> %{y:.2f}<br><b>Desvio:</b> %{error_y.array:.2f}<extra></extra>'
            )
        )

        # Linha de referência
        fig.add_hline(y=1.5, line_dash="dash", line_color="orange",
                      annotation_text="Meta Mínima (1.5)")

        fig.update_layout(
            title="Eficiência Alimentar por Tanque",
            xaxis_title="Tanque",
            yaxis_title="Eficiência (kg peixe/kg ração)",
            template=self.template,
            height=400
        )

        return fig

    def create_main_correlation_chart(self, data):
        """Gráfico principal de correlação"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Dados insuficientes para correlação",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="Correlação Principal", template=self.template)
            return fig

        # Scatter plot das duas principais variáveis
        x_col = 'total_racao' if 'total_racao' in numeric_cols else numeric_cols[0]
        y_col = 'peso_medio' if 'peso_medio' in numeric_cols else numeric_cols[1]

        fig = go.Figure()

        # Colorir pontos por tanque se disponível
        if 'tanque' in data.columns:
            for i, tank in enumerate(data['tanque'].unique()):
                tank_data = data[data['tanque'] == tank]
                fig.add_trace(
                    go.Scatter(
                        x=tank_data[x_col],
                        y=tank_data[y_col],
                        mode='markers',
                        name=f'Tanque {tank}',
                        marker=dict(
                            size=8,
                            color=self.color_palette[i % len(self.color_palette)],
                            opacity=0.7
                        ),
                        hovertemplate=f'<b>Tanque:</b> {tank}<br><b>{x_col}:</b> %{{x:.2f}}<br><b>{y_col}:</b> %{{y:.2f}}<extra></extra>'
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=data[x_col],
                    y=data[y_col],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=self.primary_color,
                        opacity=0.6
                    ),
                    name='Dados',
                    hovertemplate=f'<b>{x_col}:</b> %{{x:.2f}}<br><b>{y_col}:</b> %{{y:.2f}}<extra></extra>'
                )
            )

        # Linha de tendência
        if len(data) > 1:
            z = np.polyfit(data[x_col], data[y_col], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(data[x_col].min(), data[x_col].max(), 100)
            y_trend = p(x_trend)

            fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=y_trend,
                    mode='lines',
                    name='Tendência',
                    line=dict(color='red', dash='dash', width=2),
                    hovertemplate='Linha de Tendência<extra></extra>'
                )
            )

            # Calcular R²
            correlation = np.corrcoef(data[x_col], data[y_col])[0, 1]
            r2 = correlation ** 2

            fig.update_layout(
                title=f"Correlação: {x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()} (R² = {r2:.3f})",
            )
        else:
            fig.update_layout(
                title=f"Correlação: {x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}"
            )

        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            template=self.template,
            height=400
        )

        return fig

    def create_histogram_with_normal(self, data):
        """Histograma com curva normal sobreposta"""
        fig = go.Figure()

        # Histograma
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=25,
                name='Dados',
                opacity=0.7,
                marker_color=self.primary_color,
                histnorm='probability density'
            )
        )

        # Curva normal
        mean = np.mean(data)
        std = np.std(data)
        x_range = np.linspace(data.min() - 2 * std, data.max() + 2 * std, 100)
        y_normal = stats.norm.pdf(x_range, mean, std)

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_normal,
                mode='lines',
                name='Normal Teórica',
                line=dict(color='red', width=3)
            )
        )

        fig.update_layout(
            title=f"Distribuição com Curva Normal (μ={mean:.3f}, σ={std:.3f})",
            xaxis_title="Valor",
            yaxis_title="Densidade",
            template=self.template,
            height=400
        )

        return fig

    def create_boxplot(self, data):
        """Boxplot para análise de outliers"""
        fig = go.Figure()

        fig.add_trace(
            go.Box(
                y=data,
                name='Distribuição',
                boxpoints='outliers',
                marker_color=self.primary_color,
                line_color=self.secondary_color
            )
        )

        fig.update_layout(
            title="Boxplot - Análise de Outliers",
            yaxis_title="Valor",
            template=self.template,
            height=400,
            showlegend=False
        )

        return fig

    def create_correlation_matrix(self, correlation_matrix):
        """Heatmap da matriz de correlação"""
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlação: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title="Matriz de Correlação",
            template=self.template,
            height=500,
            xaxis_title="Variáveis",
            yaxis_title="Variáveis"
        )

        return fig

    def create_regression_plot(self, x, y, regression_results):
        """Gráfico de regressão com diagnósticos"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Regressão Linear",
                "Resíduos vs Valores Ajustados",
                "Q-Q Plot dos Resíduos",
                "Distribuição dos Resíduos"
            ),
            specs=[[{"colspan": 2}, None],
                   [{}, {}]]
        )

        # Gráfico principal de regressão
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode='markers',
                name='Dados',
                marker=dict(color=self.primary_color, opacity=0.6),
                hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Linha de regressão
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = regression_results['slope'] * x_line + regression_results['intercept']

        fig.add_trace(
            go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                name=f'Regressão (R²={regression_results["r2"]:.3f})',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )

        # Resíduos vs valores ajustados
        if len(regression_results['predictions']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=regression_results['predictions'],
                    y=regression_results['residuals'],
                    mode='markers',
                    name='Resíduos',
                    marker=dict(color=self.secondary_color, opacity=0.6),
                    showlegend=False
                ),
                row=2, col=1
            )

            # Linha zero
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)

            # Q-Q plot dos resíduos
            residuals_sorted = np.sort(regression_results['residuals'])
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals_sorted)))

            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=residuals_sorted,
                    mode='markers',
                    name='Q-Q Resíduos',
                    marker=dict(color=self.warning_color, opacity=0.6),
                    showlegend=False
                ),
                row=2, col=2
            )

            # Linha de referência Q-Q
            min_val = min(theoretical_quantiles.min(), residuals_sorted.min())
            max_val = max(theoretical_quantiles.max(), residuals_sorted.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    showlegend=False
                ),
                row=2, col=2
            )

        fig.update_layout(
            title="Análise de Regressão e Diagnósticos",
            template=self.template,
            height=600
        )

        return fig

    def create_water_quality_heatmap(self, data):
        """Heatmap de qualidade da água usando dados REAIS da planilha"""
        if not all(col in data.columns for col in ['temperatura_media', 'ph_medio', 'o2_medio']):
            fig = go.Figure()
            fig.add_annotation(
                text="Dados de qualidade da água não disponíveis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="Qualidade da Água - Dados Reais",
                template=self.template,
                height=400
            )
            return fig

        # Preparar dados para heatmap usando valores REAIS
        water_data = data[['data', 'temperatura_media', 'ph_medio', 'o2_medio']].copy()
        water_data['data_str'] = water_data['data'].dt.strftime('%Y-%m-%d')

        # Mostrar valores reais, não normalizados
        fig = go.Figure(data=go.Heatmap(
            z=water_data[['temperatura_media', 'ph_medio', 'o2_medio']].T.values,
            x=water_data['data_str'],
            y=['Temperatura (°C)', 'pH', 'O2 (mg/L)'],
            colorscale='RdYlGn',
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>Data: %{x}<br>Valor: %{z:.2f}<extra></extra>',
            text=water_data[['temperatura_media', 'ph_medio', 'o2_medio']].T.round(1).values,
            texttemplate="%{text}",
            textfont={"size": 10}
        ))

        fig.update_layout(
            title="Qualidade da Água - Dados Reais da Planilha",
            xaxis_title="Data",
            yaxis_title="Parâmetros Ambientais",
            template=self.template,
            height=400
        )

        return fig

    def create_tank_status_chart(self, data):
        """Gráfico de status dos tanques"""
        if 'tanque' not in data.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="Dados de tanque não disponíveis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Calcular status por tanque
        tank_status = data.groupby('tanque').agg({
            'eficiencia_alimentar': 'mean',
            'peso_medio': 'mean'
        }).reset_index()

        # Definir status
        tank_status['status'] = tank_status['eficiencia_alimentar'].apply(
            lambda x: 'Excelente' if x > 2.0 else 'Bom' if x > 1.5 else 'Atenção' if x > 1.0 else 'Crítico'
        )

        status_colors = {
            'Excelente': self.success_color,
            'Bom': self.primary_color,
            'Atenção': self.warning_color,
            'Crítico': self.danger_color
        }

        fig = go.Figure()

        for status in tank_status['status'].unique():
            status_data = tank_status[tank_status['status'] == status]
            fig.add_trace(
                go.Scatter(
                    x=status_data['tanque'],
                    y=status_data['eficiencia_alimentar'],
                    mode='markers',
                    name=status,
                    marker=dict(
                        size=15,
                        color=status_colors[status]
                    ),
                    hovertemplate=f'<b>Status:</b> {status}<br><b>Tanque:</b> %{{x}}<br><b>Eficiência:</b> %{{y:.2f}}<extra></extra>'
                )
            )

        fig.update_layout(
            title="Status dos Tanques por Eficiência",
            xaxis_title="Tanque",
            yaxis_title="Eficiência Alimentar",
            template=self.template,
            height=400
        )

        return fig

    def create_recent_trends(self, data):
        """Tendências recentes (últimos 7 dias)"""
        if len(data) < 7:
            recent_data = data
        else:
            recent_data = data.tail(7)

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Tendência do Peso", "Tendência da Eficiência"),
            vertical_spacing=0.15
        )

        # Peso
        fig.add_trace(
            go.Scatter(
                x=recent_data['data'],
                y=recent_data['peso_medio'],
                mode='lines+markers',
                name='Peso Médio',
                line=dict(color=self.primary_color, width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )

        # Eficiência
        fig.add_trace(
            go.Scatter(
                x=recent_data['data'],
                y=recent_data['eficiencia_alimentar'],
                mode='lines+markers',
                name='Eficiência',
                line=dict(color=self.warning_color, width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )

        fig.update_layout(
            title="Tendências Recentes (Últimos Registros)",
            template=self.template,
            height=500,
            showlegend=False
        )

        return fig

    def create_qq_plot(self, data):
        """Q-Q plot para teste de normalidade"""
        sorted_data = np.sort(data)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))

        fig = go.Figure()

        # Pontos Q-Q
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sorted_data,
                mode='markers',
                name='Dados',
                marker=dict(color=self.primary_color, opacity=0.6)
            )
        )

        # Linha de referência
        min_val = min(theoretical_quantiles.min(), sorted_data.min())
        max_val = max(theoretical_quantiles.max(), sorted_data.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Linha Teórica',
                line=dict(color='red', dash='dash', width=2)
            )
        )

        fig.update_layout(
            title="Q-Q Plot - Teste de Normalidade",
            xaxis_title="Quantis Teóricos",
            yaxis_title="Quantis Observados",
            template=self.template,
            height=400
        )

        return fig

    def create_outlier_plot(self, data, column, outliers):
        """Gráfico de outliers"""
        fig = go.Figure()

        # Dados normais
        normal_indices = [i for i in range(len(data)) if i not in outliers['indices']]

        fig.add_trace(
            go.Scatter(
                x=normal_indices,
                y=data[column].iloc[normal_indices],
                mode='markers',
                name='Dados Normais',
                marker=dict(color=self.primary_color, opacity=0.6)
            )
        )

        # Outliers
        if outliers['n_outliers'] > 0:
            fig.add_trace(
                go.Scatter(
                    x=outliers['indices'],
                    y=outliers['values'],
                    mode='markers',
                    name='Outliers',
                    marker=dict(color=self.danger_color, size=10, symbol='x')
                )
            )

        fig.update_layout(
            title=f"Detecção de Outliers - {column}",
            xaxis_title="Índice",
            yaxis_title="Valor",
            template=self.template,
            height=400
        )

        return fig

    def create_regression_diagnostics(self, x, y, regression_results):
        """Diagnósticos de regressão"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Resíduos vs Ajustados",
                "Q-Q dos Resíduos",
                "Scale-Location",
                "Resíduos vs Leverage"
            )
        )

        if len(regression_results['predictions']) > 0:
            predictions = regression_results['predictions']
            residuals = regression_results['residuals']

            # Resíduos vs Ajustados
            fig.add_trace(
                go.Scatter(
                    x=predictions,
                    y=residuals,
                    mode='markers',
                    marker=dict(color=self.primary_color, opacity=0.6),
                    showlegend=False
                ),
                row=1, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

            # Q-Q dos resíduos
            residuals_sorted = np.sort(residuals)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals_sorted)))

            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=residuals_sorted,
                    mode='markers',
                    marker=dict(color=self.secondary_color, opacity=0.6),
                    showlegend=False
                ),
                row=1, col=2
            )

            # Scale-Location
            sqrt_abs_residuals = np.sqrt(np.abs(residuals))
            fig.add_trace(
                go.Scatter(
                    x=predictions,
                    y=sqrt_abs_residuals,
                    mode='markers',
                    marker=dict(color=self.warning_color, opacity=0.6),
                    showlegend=False
                ),
                row=2, col=1
            )

            # Resíduos vs Leverage (simplificado)
            leverage = np.ones(len(residuals)) / len(residuals)  # Simplificado
            fig.add_trace(
                go.Scatter(
                    x=leverage,
                    y=residuals,
                    mode='markers',
                    marker=dict(color=self.success_color, opacity=0.6),
                    showlegend=False
                ),
                row=2, col=2
            )

        fig.update_layout(
            title="Diagnósticos de Regressão",
            template=self.template,
            height=600
        )

        return fig

    def export_all_charts(self, data):
        """Exporta todos os gráficos em um arquivo ZIP"""
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Lista de gráficos para exportar
            charts = [
                ('evolucao_temporal', self.create_evolution_chart(data)),
                ('distribuicao_peso', self.create_weight_distribution(data)),
                ('eficiencia_tanque', self.create_efficiency_by_tank(data)),
                ('correlacao_principal', self.create_main_correlation_chart(data)),
                ('matriz_correlacao', self.create_correlation_matrix(data.select_dtypes(include=[np.number]).corr())),
                ('qualidade_agua', self.create_water_quality_heatmap(data)),
                ('status_tanques', self.create_tank_status_chart(data)),
                ('tendencias_recentes', self.create_recent_trends(data))
            ]

            for chart_name, fig in charts:
                # Exportar como HTML
                html_str = fig.to_html(include_plotlyjs='cdn')
                zip_file.writestr(f'{chart_name}.html', html_str)

                # Exportar como PNG (requer kaleido)
                try:
                    img_bytes = fig.to_image(format="png", width=1200, height=800)
                    zip_file.writestr(f'{chart_name}.png', img_bytes)
                except:
                    pass  # Se kaleido não estiver instalado, pula PNG

        return zip_buffer.getvalue()

    # Métodos de compatibilidade com versão anterior
    def create_gaussian_curve(self, data, title="Distribuição Normal"):
        """Mantém compatibilidade"""
        return self.create_histogram_with_normal(data)

    def create_growth_chart(self, data, growth_analysis):
        """Mantém compatibilidade"""
        return self.create_evolution_chart(data)

    def create_correlation_heatmap(self, correlation_matrix):
        """Mantém compatibilidade"""
        return self.create_correlation_matrix(correlation_matrix)

    def create_comparative_histograms(self, data):
        """Mantém compatibilidade"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Distribuição do Peso", "Distribuição da Ração")
        )

        fig.add_trace(
            go.Histogram(x=data['peso_medio'], name='Peso', marker_color=self.primary_color),
            row=1, col=1
        )

        fig.add_trace(
            go.Histogram(x=data['total_racao'], name='Ração', marker_color=self.secondary_color),
            row=1, col=2
        )

        fig.update_layout(
            title="Distribuições Comparativas",
            template=self.template,
            height=400,
            showlegend=False
        )

        return fig

    def create_scatter_plot(self, data, x_col, y_col, title):
        """Mantém compatibilidade"""
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='markers',
                marker=dict(color=self.primary_color, opacity=0.6)
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            template=self.template,
            height=400
        )

        return fig

    def create_time_series(self, data, x_col, y_col, title):
        """Mantém compatibilidade"""
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='lines+markers',
                line=dict(color=self.primary_color, width=2)
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            template=self.template,
            height=400
        )

        return fig

    def create_tank_comparison(self, data, tank_col, metric_col, title):
        """Mantém compatibilidade"""
        if tank_col not in data.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="Dados de tanque não disponíveis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        tank_stats = data.groupby(tank_col)[metric_col].mean().reset_index()

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=tank_stats[tank_col],
                y=tank_stats[metric_col],
                marker_color=self.primary_color
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title=tank_col,
            yaxis_title=metric_col,
            template=self.template,
            height=400
        )

        return fig

    def create_environmental_evolution(self, data):
        """Evolução dos parâmetros ambientais"""
        env_columns = ['temperatura_media', 'ph_medio', 'o2_medio']
        available_env = [col for col in env_columns if col in data.columns]

        if not available_env:
            fig = go.Figure()
            fig.add_annotation(
                text="Dados ambientais não disponíveis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        fig = go.Figure()

        colors = [self.danger_color, self.primary_color, self.success_color]
        names = ['Temperatura (°C)', 'pH', 'O2 (mg/L)']

        for i, col in enumerate(available_env):
            fig.add_trace(
                go.Scatter(
                    x=data['data'],
                    y=data[col],
                    mode='lines+markers',
                    name=names[i],
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=6)
                )
            )

        fig.update_layout(
            title="Evolução dos Parâmetros Ambientais",
            xaxis_title="Data",
            yaxis_title="Valor",
            template=self.template,
            height=500
        )

        return fig

    def create_performance_dashboard(self, data):
        """Dashboard de performance integrado"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Peso vs Eficiência",
                "Ração vs Crescimento",
                "Qualidade da Água",
                "Performance por Tanque"
            ),
            specs=[[{}, {}],
                   [{"type": "heatmap"}, {}]]
        )

        # Peso vs Eficiência
        fig.add_trace(
            go.Scatter(
                x=data['peso_medio'],
                y=data['eficiencia_alimentar'],
                mode='markers',
                marker=dict(color=self.primary_color, opacity=0.6),
                name='Peso vs Eficiência'
            ),
            row=1, col=1
        )

        # Ração vs Crescimento (simplificado)
        if len(data) > 1:
            growth_rate = data['peso_medio'].diff().fillna(0)
            fig.add_trace(
                go.Scatter(
                    x=data['total_racao'],
                    y=growth_rate,
                    mode='markers',
                    marker=dict(color=self.secondary_color, opacity=0.6),
                    name='Ração vs Crescimento'
                ),
                row=1, col=2
            )

        # Performance por tanque
        if 'tanque' in data.columns:
            tank_perf = data.groupby('tanque')['eficiencia_alimentar'].mean()
            fig.add_trace(
                go.Bar(
                    x=tank_perf.index,
                    y=tank_perf.values,
                    marker_color=self.success_color,
                    name='Performance por Tanque'
                ),
                row=2, col=2
            )

        fig.update_layout(
            title="Dashboard de Performance Integrado",
            template=self.template,
            height=700,
            showlegend=False
        )

        return fig