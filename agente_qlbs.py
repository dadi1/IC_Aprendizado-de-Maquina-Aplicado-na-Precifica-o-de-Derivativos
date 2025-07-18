import numpy as np

class AgenteQLBS:
    """ Classe para """

    def __init__(self, 
                 learning_rate, 
                 discount_factor, 
                 epsilon, 
                 epsilon_decay,
                 min_epsilon,
                 observation_space_dims,
                 action_space_size):
        """
        Inicialização do Agente de Q-Learning.
        
        Parâmetros:
        - learning_rate (alpha) : A taxa na qual o agente aprende.
        - discount_factor (gamma) : O peso dado a recompensas futuras.
        - epsilon : A taxa inicial de exploração (escolha das ações aleatórias).
        - epsilon_decay : O fator de decaimento de epsilon a cada episódio.
        - min_epsion: A taxa mínima de exploração.
        - observation_space_dims : As dimensões do espaço de obersvação (do ambiente)
        - action_space_size : O numero de ações possíveis.
        """

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        q_table_dims = tuple(observation_space_dims) + (action_space_size,)
        self.q_table = np.zeros(q_table_dims)

        def choose_action(self, state):
            """Escolhe uma ação usando a política Epsilon-Greedy"""

            # Exploração: Escolhe uma ação aleatória com probabilidade episilon.
            if np.random.random() < self.epsilon:
                return np.random.randint(0, self.q_table.shape[-1])
            # Exploração: Escolhe a melhor ação conhecida.
            else:
                return np.argmax(self.q_table[state])
            
        
        def learn(self, state, action, reward, next_state):
            """Atualiza a tabela Q usando a equação de Bellman."""

            # O estado já uma tupla de índices, então pode ser usado diretamente.
            old_q_value = self.q_table[state][action]

            # Encontra o Q máximo para o próximo estado (melhor ação futura)
            next_max_q = np.max(self.q_table[state])

            # Calcula o novo valor Q usando a fóruma da atualização.
            # target = recompensa_imediata + gamma * valor_futuro_esperado.
            target = reward + self.dicount_factor * next_max_q

            # Calcula o novo Q valor.
            # Q_valor_novo = (1 - taxa de aprendizagem) * Q_valor_antigo + taxa_de_aprindizagem * target.
            new_q_value = (1 - self.learing_rate) * old_q_value + self.learning_rate * target

            # Atualiza a tabela Q com o novo valor.
            self.q_table[state][action] = new_q_value
            