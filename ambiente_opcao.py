import gymnasium as gym
from gymnasium import spaces
import numpy as np
import QuantLib as ql

class AmbienteOpcao(gym.Env):
    """
    Ambiente customizado de Gymnasium para simulação de preficiação
    e hedge de uma opção europeia.
    """

    def __init__(self, S0, K, r, sigma, T):

        # --- Parâmetros da Simulação ---
        self.S0 = S0 # Preço inicial do ativo.
        self.K = K # Preço de exercício (Strike).
        self.r = r # Taxa de juros livres de risco.
        self.sigma = sigma # Volatilidade do ativo.
        self.T = T # tempo até o vencimento em dias.

        # --- Definição dos Espaço de Ação e Observação. ---
        # TODO:

        # --- Configuração do Processo de Simulação (QuantLib) ---
        # TODO: Configura os processo de BSM aqui.


    def _get_obs(self):
        """Retorna a observação do estado atual.""" 
        # TODO: Implementar a lógica de discretização do estado.
        pass

    def _get_info(self):
        "Retorna informações auxiliares sobre o ambiente."
        # TODO: Pode retornar o preço atual, etc.
        pass

    def reset(self, seed=None, option=None):
        """Reseta o ambiente para um estado inicial."""
        # TODO: Gerar uma nova trajeatória de preços e resetar o tempo.
        observation = self.get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        """Executa um passo no ambiente a partir de uma ação."""
        # TODO: A lógica principal acontece aqui

        return self._get_obs(), 0, False, False, self._get_info()