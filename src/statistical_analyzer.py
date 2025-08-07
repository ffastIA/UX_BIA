# src/statistical_analyzer.py

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """
    Classe para realizar diversas análises estatísticas em conjuntos de dados.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Inicializa a classe StatisticalAnalyzer.
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def comprehensive_statistics(self, data: pd.Series) -> dict:
        """
        Calcula um conjunto completo de estatísticas descritivas.
        """
        data_clean = pd.Series(data).dropna()

        if len(data_clean) == 0:
            return self._empty_stats()

        stats_dict = {
            'count': len(data_clean),
            'mean': np.mean(data_clean),
            'median': np.median(data_clean),
            'std': np.std(data_clean, ddof=1),
            'variance': np.var(data_clean, ddof=1),
            'min': np.min(data_clean),
            'max': np.max(data_clean),
            'q25': np.percentile(data_clean, 25),  # CORREÇÃO: usar q25 em vez de q1
            'q75': np.percentile(data_clean, 75),  # CORREÇÃO: usar q75 em vez de q3
            'iqr': np.percentile(data_clean, 75) - np.percentile(data_clean, 25),
            'range': np.max(data_clean) - np.min(data_clean),
            'skewness': stats.skew(data_clean),
            'kurtosis': stats.kurtosis(data_clean),
            'cv': np.std(data_clean, ddof=1) / np.mean(data_clean) if np.mean(data_clean) != 0 else np.inf,
            'confidence_level': self.confidence_level  # CORREÇÃO: Adicionar confidence_level
        }

        # Cálculo do Intervalo de Confiança
        if len(data_clean) > 1:
            ci = stats.t.interval(
                self.confidence_level,
                len(data_clean) - 1,
                loc=stats_dict['mean'],
                scale=stats.sem(data_clean)
            )
            stats_dict['ci_lower'] = ci[0]
            stats_dict['ci_upper'] = ci[1]
            stats_dict['margin_error'] = (ci[1] - ci[0]) / 2
        else:
            stats_dict['ci_lower'] = stats_dict['mean']
            stats_dict['ci_upper'] = stats_dict['mean']
            stats_dict['margin_error'] = 0

        return stats_dict

    def _empty_stats(self) -> dict:
        """
        Retorna um dicionário de estatísticas com valores NaN.
        """
        return {key: np.nan for key in [
            'count', 'mean', 'median', 'std', 'variance', 'min', 'max',
            'q25', 'q75', 'iqr', 'range', 'skewness', 'kurtosis', 'cv',
            'ci_lower', 'ci_upper', 'margin_error', 'confidence_level'
        ]}

    def advanced_normality_tests(self, data: pd.Series) -> dict:
        """
        Realiza múltiplos testes de normalidade.
        """
        data_clean = pd.Series(data).dropna()
        results = {}

        # Teste de Shapiro-Wilk
        if len(data_clean) > 0 and len(data_clean) <= 5000:
            try:
                stat, p_val = stats.shapiro(data_clean)
                results['shapiro'] = {
                    'statistic': stat,
                    'p_value': p_val,
                    'is_normal': p_val > self.alpha
                }
            except Exception:
                results['shapiro'] = {'statistic': np.nan, 'p_value': np.nan, 'is_normal': False}

        # Teste de Anderson-Darling
        if len(data_clean) > 0:
            try:
                result = stats.anderson(data_clean, dist='norm')
                results['anderson'] = {
                    'statistic': result.statistic,
                    'critical_values': result.critical_values,
                    'significance_levels': result.significance_level
                }
            except Exception:
                results['anderson'] = {'statistic': np.nan}

        # Teste de Kolmogorov-Smirnov
        if len(data_clean) > 0:
            try:
                stat, p_val = stats.kstest(data_clean, 'norm',
                                          args=(np.mean(data_clean), np.std(data_clean)))
                results['ks'] = {
                    'statistic': stat,
                    'p_value': p_val,
                    'is_normal': p_val > self.alpha
                }
            except Exception:
                results['ks'] = {'statistic': np.nan, 'p_value': np.nan, 'is_normal': False}

        return results

    def detect_outliers(self, data: pd.Series, method: str = 'iqr') -> dict:
        """
        Detecta outliers em uma série de dados.
        """
        data_clean = pd.Series(data).dropna()

        if len(data_clean) == 0:
            return {'indices': [], 'values': [], 'n_outliers': 0, 'percentage': 0}

        outliers = pd.Series([False] * len(data_clean), index=data_clean.index)

        if method == 'iqr':
            Q1 = np.percentile(data_clean, 25)
            Q3 = np.percentile(data_clean, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (data_clean < lower_bound) | (data_clean > upper_bound)

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data_clean))
            outliers = z_scores > 3

        elif method == 'modified_zscore':
            median = np.median(data_clean)
            mad = np.median(np.abs(data_clean - median))
            if mad == 0:
                outliers = pd.Series([False] * len(data_clean))
            else:
                modified_z_scores = 0.6745 * (data_clean - median) / mad
                outliers = np.abs(modified_z_scores) > 3.5

        return {
            'indices': np.where(outliers)[0].tolist(),
            'values': data_clean[outliers].tolist(),
            'n_outliers': np.sum(outliers),
            'percentage': (np.sum(outliers) / len(data_clean)) * 100 if len(data_clean) > 0 else 0
        }

    def correlation_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula a matriz de correlação.
        """
        return data.corr()

    def regression_analysis(self, x: pd.Series, y: pd.Series) -> dict:
        """
        Realiza análise de regressão linear simples.
        """
        x_clean = pd.Series(x).dropna()
        y_clean = pd.Series(y).dropna()

        min_len = min(len(x_clean), len(y_clean))
        x_clean = x_clean.iloc[:min_len]
        y_clean = y_clean.iloc[:min_len]

        if len(x_clean) < 2:
            return self._empty_regression()

        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        y_pred = slope * x_clean + intercept

        r2 = r_value ** 2
        if len(x_clean) - 2 > 0:
            r2_adj = 1 - (1 - r2) * (len(x_clean) - 1) / (len(x_clean) - 2)
        else:
            r2_adj = np.nan

        return {
            'slope': slope,
            'intercept': intercept,
            'r2': r2,
            'r2_adj': r2_adj,
            'p_value': p_value,
            'std_err': std_err,
            'predictions': y_pred,
            'residuals': y_clean - y_pred,
            'X': x_clean,  # CORREÇÃO: Adicionar X e y para compatibilidade
            'y': y_clean
        }

    def _empty_regression(self) -> dict:
        """
        Retorna um dicionário de resultados de regressão vazio.
        """
        return {
            'slope': np.nan,
            'intercept': np.nan,
            'r2': np.nan,
            'r2_adj': np.nan,
            'p_value': np.nan,
            'std_err': np.nan,
            'predictions': np.array([]),
            'residuals': np.array([]),
            'X': pd.Series([]),
            'y': pd.Series([])
        }