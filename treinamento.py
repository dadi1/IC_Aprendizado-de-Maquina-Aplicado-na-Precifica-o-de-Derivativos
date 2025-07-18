import numpy as np
import matplotlib.pyplot as plt

from ambiente_opcao import AmbienteOpcao
from agente_qlbs import AgenteQLBS


# --- Hiperparãmetros do treinamento ---
# Estes valores são ajustáveis.
EPISODES = 20000
LEARNING_RATE = 0.1         # Alpha
DISCOUNT_FACTOR = 0.95      # Gamma.
EPSILON = 1.0               # Taxa inicial de exploração.
EPSILON_DECAY = 0.9999      # Fator de decaimento de epsilon.
MIN_EPSILON = 0.01          # Valor mínimo para o epsilon.

# --- Inicialização ---
# Criar o ambiente.
env = AmbienteOpcao()
# Cria o agente.
agente = AgenteQLBS(
    learning_rate=LEARNING_RATE,
    discount_factor=DISCOUNT_FACTOR,
    epsilon=EPSILON,
    epsilon_decay=EPSILON_DECAY,
    min_epsilon=MIN_EPSILON,
    observation_space_dims=env.observation_space,
    action_space_size=env.action_space
)