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

# List que armazena as recompensas totai de cada episódio para plotagem.
total_rewards = []

# --- Traço de treinamento ---
print("Iniciando o treinamento!!!!")
for episode in range(EPISODES):
    state, info = env.reset()
    state = tuple(state) # Converte o estado para uma tupla e ser usado como índice.

    terminated = False
    total_reward = 0
    
    while not terminated:
        # 1. Agente escolhe uma ação.
        action = agente.choose_action(state)

        # 2. Ambiente executa a ação e retorna
        next_state, reward, terminated, truncated, infor = env.step(action)

        # 3. Agente aprende com a experiência.
        agente.learn(state, action, reward, next_state)

        # 4. Atualiza o estado atual.
        state = next_state
        total_reward += reward


    # Decaimento do Epsilon após cada episódio
    agente.epsilon = max(agente.min_epsilon, agente.epsilon * agente.epsilon_decay)
    
    # Armazena a recompensa total para análise
    total_rewards.append(total_reward)

    # Imprime o progresso a cada 1000 episódios
    if (episode + 1) % 1000 == 0:
        avg_reward = np.mean(total_rewards[-1000:])
        print(f"Episódio: {episode + 1}/{EPISODES} | Recompensa Média (últimos 1000): {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f}")

print("\nTreinamento concluído!")

# --- Salvar a Tabela Q ---
np.save("q_table_qlbs.npy", agente.q_table)
print("Tabela Q salva em 'q_table_qlbs.npy'")

# --- Plotar os Resultados ---
plt.figure(figsize=(12, 6))
plt.plot(total_rewards, label='Recompensa por Episódio')
# Calcula uma média móvel para visualizar a tendência de aprendizado
moving_avg = np.convolve(total_rewards, np.ones(100)/100, mode='valid')
plt.plot(moving_avg, label='Média Móvel (100 episódios)', color='red')
plt.title('Recompensa por Episódio Durante o Treinamento')
plt.xlabel('Episódio')
plt.ylabel('Recompensa Total')
plt.legend()
plt.grid(True)
plt.show()
