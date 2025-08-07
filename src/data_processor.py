import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from src.weight_estimator import WeightEstimator


class DataProcessor:
    """
    Classe responsável por processar, limpar, integrar e calcular métricas
    a partir de dados brutos de peso de peixes e consumo de ração.
    """

    def __init__(self, correction_factor: float = 1.0):
        """
        Inicializa o processador de dados.
        """
        self.correction_factor = correction_factor
        self.weight_estimator = WeightEstimator(correction_factor=self.correction_factor)

    def process_integrated_data(self, raw_fish_df: pd.DataFrame, raw_feed_df: pd.DataFrame,
                                start_date: datetime.date, end_date: datetime.date,
                                selected_tanks: list) -> pd.DataFrame:
        """
        Função principal para processar e integrar dados brutos de peixes e ração.
        """
        try:
            st.info("🔄 Iniciando processamento dos dados...")

            # Limpeza e Validação dos Dados Brutos
            fish_clean = self._clean_fish_data(raw_fish_df.copy())
            feed_clean = self._clean_feed_data(raw_feed_df.copy())

            if fish_clean is None or feed_clean is None:
                st.error("❌ Falha na limpeza dos dados. Verifique o formato dos arquivos.")
                return None

            # Filtragem por Período de Tempo
            st.write(f"Filtrando dados de {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}...")
            fish_filtered = self._filter_by_date(fish_clean, start_date, end_date)
            feed_filtered = self._filter_by_date(feed_clean, start_date, end_date)

            st.write(f"**Dados após filtro de data:**")
            st.write(f"- Peixes: {len(fish_filtered)} registros")
            st.write(f"- Ração: {len(feed_filtered)} registros")

            if fish_filtered.empty or feed_filtered.empty:
                st.warning("⚠️ Nenhum dado encontrado após o filtro de período selecionado.")
                return pd.DataFrame()

            # Filtragem por Tanques Selecionados
            if selected_tanks and 'tanque' in feed_filtered.columns:
                feed_filtered = feed_filtered[feed_filtered['tanque'].isin(selected_tanks)]
                st.write(f"**Dados de ração após filtro de tanques:** {len(feed_filtered)} registros")

            if feed_filtered.empty:
                st.warning("⚠️ Nenhum dado de ração encontrado após o filtro de tanques selecionado.")
                return pd.DataFrame()

            # Integração dos Dados
            integrated_data = self._integrate_data(fish_filtered, feed_filtered)

            if integrated_data is not None and not integrated_data.empty:
                st.success("✅ Processamento concluído com sucesso!")
                st.write(f"**Dados finais integrados:** {len(integrated_data)} registros")
                return integrated_data
            else:
                st.error("❌ A integração dos dados resultou em um DataFrame vazio ou nulo.")
                return pd.DataFrame()

        except Exception as e:
            st.error(f"⚠️ Erro no processamento de dados: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None

    def _clean_fish_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpa e padroniza os dados brutos dos peixes.
        """
        try:
            st.write("🐟 Processando dados dos peixes...")
            st.write(f"**Colunas originais:** {list(df.columns)}")

            # Validação de Colunas Essenciais
            expected_columns = ['data', 'largura', 'altura']
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                st.error(f"❌ Colunas essenciais ausentes nos dados dos peixes: {missing_columns}")
                return None

            # Conversão de Tipos de Dados
            df['data'] = pd.to_datetime(df['data'], errors='coerce')
            df['largura'] = pd.to_numeric(df['largura'], errors='coerce')
            df['altura'] = pd.to_numeric(df['altura'], errors='coerce')

            # Processamento de Dados Ambientais (se existirem)
            environmental_cols = ['temperatura', 'ph', 'o2', 'hora']
            for col in environmental_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Criação da Coluna 'datetime' Combinada
            if 'hora' in df.columns and not df['hora'].isnull().all():
                try:
                    df['datetime'] = pd.to_datetime(
                        df['data'].dt.strftime('%Y-%m-%d') + ' ' + df['hora'].astype(str).str.zfill(4),
                        format='%Y-%m-%d %H%M',
                        errors='coerce'
                    )
                except Exception:
                    df['datetime'] = df['data']
                    st.warning("⚠️ Não foi possível combinar 'data' e 'hora'.")
            else:
                df['datetime'] = df['data']

            # Remoção de Linhas com Dados Inválidos
            original_len = len(df)
            df = df.dropna(subset=['data', 'largura', 'altura', 'datetime'])
            df = df[(df['largura'] > 0) & (df['altura'] > 0)]

            st.write(f"**Linhas removidas:** {original_len - len(df)}")
            st.write(f"**Linhas válidas:** {len(df)}")

            if len(df) == 0:
                st.error("❌ Nenhum dado de peixe válido após limpeza!")
                return None

            # Estimativa de Peso
            st.write("🔄 Calculando peso estimado...")

            # CORREÇÃO: Usar o método correto para estimar peso
            df['peso_estimado'] = df.apply(
                lambda row: self.weight_estimator.estimate_weight_advanced(
                    row['largura'],
                    row['altura'],
                    row.get('temperatura'),
                    row.get('ph'),
                    row.get('o2')
                ), axis=1
            )

            st.write(f"**Peso estimado - Estatísticas:**")
            st.write(f"- Min: {df['peso_estimado'].min():.3f} kg")
            st.write(f"- Max: {df['peso_estimado'].max():.3f} kg")
            st.write(f"- Média: {df['peso_estimado'].mean():.3f} kg")

            return df

        except Exception as e:
            st.error(f"❌ Erro na limpeza dos dados dos peixes: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None

    def _clean_feed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpa e padroniza os dados brutos de ração.
        """
        try:
            st.write("🍽️ Processando dados de ração...")
            st.write(f"**Colunas de ração:** {list(df.columns)}")

            # Validação de Colunas Essenciais
            expected_columns = ['data', 'peso']
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                st.error(f"❌ Colunas essenciais ausentes nos dados de ração: {missing_columns}")
                return None

            # Conversão de Tipos de Dados
            df['data'] = pd.to_datetime(df['data'], errors='coerce')
            df['peso'] = pd.to_numeric(df['peso'], errors='coerce')

            # Remoção de Linhas com Dados Inválidos
            original_len = len(df)
            df = df.dropna(subset=['data', 'peso'])
            df = df[df['peso'] >= 0]

            st.write(f"**Linhas de ração removidas:** {original_len - len(df)}")
            st.write(f"**Linhas de ração válidas:** {len(df)}")

            if len(df) == 0:
                st.error("❌ Nenhum dado de ração válido após limpeza!")
                return None

            # Agrupamento de Dados de Ração
            if 'tanque' in df.columns:
                df_grouped = df.groupby(['data', 'tanque'])['peso'].sum().reset_index()
            else:
                df_grouped = df.groupby('data')['peso'].sum().reset_index()
                df_grouped['tanque'] = 'Tanque Único'

            df_grouped = df_grouped.rename(columns={'peso': 'total_racao'})

            st.write("**Dados de ração agrupados:**")
            st.dataframe(df_grouped.head())

            return df_grouped

        except Exception as e:
            st.error(f"❌ Erro na limpeza dos dados de ração: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None

    def _filter_by_date(self, df: pd.DataFrame, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
        """
        Filtra um DataFrame por um intervalo de datas.
        """
        if df.empty:
            return df
        df['data'] = pd.to_datetime(df['data'])
        mask = (df['data'].dt.date >= start_date) & (df['data'].dt.date <= end_date)
        return df[mask].copy()

    def _integrate_data(self, fish_df: pd.DataFrame, feed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Integra os DataFrames de peixes processados e ração agrupada.
        """
        try:
            st.write("🔗 Integrando dados...")

            # Agrupamento dos Dados de Peixes
            agg_dict = {
                'peso_estimado': ['mean', 'std', 'count', 'sum', 'min', 'max'],
                'largura': ['mean', 'std'],
                'altura': ['mean', 'std']
            }

            # Adiciona colunas ambientais se existirem
            for col in ['temperatura', 'ph', 'o2']:
                if col in fish_df.columns:
                    agg_dict[col] = 'mean'

            fish_grouped = fish_df.groupby(['data', 'tanque']).agg(agg_dict).reset_index()

            # Achatar nomes de colunas
            new_columns = ['data', 'tanque', 'peso_medio', 'peso_std', 'n_peixes', 'peso_total',
                           'peso_min', 'peso_max', 'largura_media', 'largura_std', 'altura_media', 'altura_std']

            # Adiciona colunas ambientais
            for col in ['temperatura', 'ph', 'o2']:
                if col in fish_df.columns:
                    new_columns.append(f'{col}_medio')

            # CORREÇÃO: Ajustar o número de colunas
            fish_grouped.columns = new_columns[:len(fish_grouped.columns)]

            st.write(f"**Peixes agrupados:** {len(fish_grouped)} registros")

            # Merge com Dados de Ração
            integrated = pd.merge(fish_grouped, feed_df, on=['data', 'tanque'], how='left')
            integrated['total_racao'] = integrated['total_racao'].fillna(0)

            # Cálculo da Eficiência Alimentar
            integrated['eficiencia_alimentar'] = integrated.apply(
                lambda row: row['peso_total'] / row['total_racao'] if row['total_racao'] > 0 else 0,
                axis=1
            )

            # Cálculo do Consumo Per Capita
            integrated['consumo_per_capita'] = integrated.apply(
                lambda row: row['total_racao'] / row['n_peixes'] if row['n_peixes'] > 0 else 0,
                axis=1
            )

            # Preenche NaNs
            integrated = integrated.fillna(0)

            st.write("**Dados integrados finais (amostra):**")
            st.dataframe(integrated.head())

            return integrated

        except Exception as e:
            st.error(f"❌ Erro na integração dos dados: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None