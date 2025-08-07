import numpy as np
import pandas as pd


class WeightEstimator:
    """
    Classe para estimativa de peso de peixes baseada nas dimensões de um 'bounding box'
    (largura e altura) e em parâmetros ambientais.
    """

    def __init__(self, correction_factor=1.0):
        """
        Inicializa a classe WeightEstimator.

        Parâmetros:
            correction_factor (float): Fator de correção geral que pode ser aplicado
                                       à estimativa final do peso.
        """
        self.correction_factor = correction_factor

        # PARÂMETROS PARA CALIBRAÇÃO
        self.CALIBRATION_PARAMS = {
            'densidade_peixe': 1.05,
            'fator_forma': 0.7,
            'fator_profundidade': 0.6,
            'coef_alometrico': 3.2,
            'intercepto': 0.001
        }

    def estimate_weight_basic(self, largura: float, altura: float) -> float:
        """
        Estimativa básica de peso de um peixe usando uma combinação de
        estimativa volumétrica e relação alométrica.
        """
        params = self.CALIBRATION_PARAMS

        # Método 1: Estimativa de Peso Baseada no Volume
        profundidade_est = largura * params['fator_profundidade']
        volume_cm3 = largura * altura * profundidade_est * params['fator_forma']
        volume_litros = volume_cm3 / 1000
        peso_volumetrico = volume_litros * params['densidade_peixe']

        # Método 2: Estimativa de Peso Baseada em Relação Alométrica
        comprimento_equiv = np.sqrt(largura * altura)
        peso_alometrico = (
                params['intercepto'] *
                np.power(comprimento_equiv, params['coef_alometrico'])
        )

        # Combinação dos Métodos
        peso_final = (0.6 * peso_volumetrico + 0.4 * peso_alometrico)
        peso_final *= self.correction_factor
        peso_final = np.clip(peso_final, 0.01, 100.0)

        return peso_final

    def estimate_weight_advanced(self, largura, altura, temperatura=None, ph=None, o2=None):
        """
        Estimativa avançada de peso considerando fatores ambientais.

        CORREÇÃO: Este método agora aceita tanto valores únicos quanto Series do pandas.
        """
        # Se receber Series do pandas, aplica elemento por elemento
        if isinstance(largura, pd.Series):
            return largura.combine(altura, lambda l, a: self._estimate_single_advanced(
                l, a,
                temperatura.loc[largura.index] if isinstance(temperatura, pd.Series) else temperatura,
                ph.loc[largura.index] if isinstance(ph, pd.Series) else ph,
                o2.loc[largura.index] if isinstance(o2, pd.Series) else o2
            ))
        else:
            return self._estimate_single_advanced(largura, altura, temperatura, ph, o2)

    def _estimate_single_advanced(self, largura, altura, temperatura=None, ph=None, o2=None):
        """
        Estimativa avançada para um único peixe.
        """
        # Calcula o peso base
        peso_base = self.estimate_weight_basic(largura, altura)

        # Fatores de correção ambiental
        fator_temp = 1.0
        fator_ph = 1.0
        fator_o2 = 1.0

        # Fator de Correção para Temperatura
        if temperatura is not None and not pd.isna(temperatura):
            temp_otima = 26.5
            desvio_temp = abs(temperatura - temp_otima)
            fator_temp = max(0.8, 1.0 - (desvio_temp * 0.02))

        # Fator de Correção para pH
        if ph is not None and not pd.isna(ph):
            ph_otimo = 7.5
            desvio_ph = abs(ph - ph_otimo)
            fator_ph = max(0.9, 1.0 - (desvio_ph * 0.05))

        # Fator de Correção para O2
        if o2 is not None and not pd.isna(o2):
            if o2 >= 5:
                fator_o2 = 1.0
            else:
                fator_o2 = max(0.7, o2 / 5.0)

        peso_corrigido = peso_base * fator_temp * fator_ph * fator_o2
        return peso_corrigido