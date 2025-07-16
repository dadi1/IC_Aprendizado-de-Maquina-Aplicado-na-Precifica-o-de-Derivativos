from ambiente_opcao import AmbienteOpcao

# Instância de simulação do ambiente.
env = AmbienteOpcao(S0=100, K=100, T=30)

# Reseta o ambiente par o inicio de um episódio.
observation, info = env.reset()
print(f"Estado Inicial: {observation}, Info: {info}")

done = False
total_reward = 0

# Loop de um episodio
while not done:

    # Escole uma ação aleatória
    action = env.action_sample()

    # Executa a ação aleatória no ambiente.
    observation, reward, terminated, truncated, info = env.step()

    done = terminated or truncated
    total_reward += reward

    print(f"Passo: {env.current_step}, Ação: {action}, Estado: {observation}, Recompensa: {reward:.2f}, Info: {infor}")

print(f"\n Simulação finalizada! Recompensa Total: {total_reward:.2f}")

env.close()

