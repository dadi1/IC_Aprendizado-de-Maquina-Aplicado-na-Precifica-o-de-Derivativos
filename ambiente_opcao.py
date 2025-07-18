import gymnasium as gym
from gymnasium import spaces
import numpy as np
import QuantLib as ql

class AmbienteOpcao(gym.Env):
    """
    Ambiente customizado de Gymnasium para simulação de preficiação
    e hedge de uma opção europeia.
    """

    def __init__(self, S0=100, K=100, r=0.05, sigma=0.2, T=30):
        super(AmbienteOpcao, self).__init__()

        # --- Parâmetros da Simulação ---
        self.S0 = S0 # Preço inicial do ativo.
        self.K = K # Preço de exercício (Strike).
        self.r = r # Taxa de juros livres de risco.
        self.sigma = sigma # Volatilidade do ativo.
        self.T = T # tempo até o vencimento em dias.

        self.transaction_cost = 0.001 # 1% sobre o valor de transição.

        # Posição começa zerada.
        self.stock_position = 0

        # Caixa começa zerado.
        self.cash_balance = 0

        # --- Definição dos Espaço de Ação e Observação. ---
        # Ações: 0=vender, 1=Manter, 2=Comprar unidade do ativo para hedge.
        self.action_space = spaces.Discrete(3)

        self.num_price_bins = 100
        self.price_bins = np.linspace(self.S0 * 0.5, self.S0 * 1.5, self.num_price_bins)

        self.observation_space = spaces.MultiDiscrete([self.num_price_bins, self.T + 1])


        # --- Configuração do Processo de Simulação (QuantLib) ---
        self.timestep = 1
        self.day_count = ql.Actual365Fixed()
        self.calendar = ql.Brazil()
        
        self.calculation_date = self.calendar.adjust(ql.Date.todaysDate())
        ql.Settings.instance().evaluationDate = self.calculation_date

        
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.S0))
        risk_free_rate = ql.FlatForward(self.calculation_date, self.r, self.day_count)
        rate_ts = ql.YieldTermStructureHandle(risk_free_rate)
        volatility = ql.BlackConstantVol(self.calculation_date, self.calendar, self.sigma, self.day_count)
        vol_ts = ql.BlackVolTermStructureHandle(volatility)

        self.bsm_process = ql.BlackScholesMertonProcess(spot_handle, rate_ts, rate_ts, vol_ts)

        # --- Variaveis do processo para cada tempo T. ---
        self.current_step = 0
        self.stock_path = []


    def _get_obs(self):
        """Retorna a observação do estado atual.""" 
        price_idx = min(self.current_step, len(self.stock_path) - 1)
        price = self.stock_path[price_idx]

        time_to_maturity = self.T - self.current_step

        price_bin = np.digitize(price, self.price_bins) - 1
        # Garante que o bin esteja sempre dentro dos limites [0, num_precis_bins -1]
        price_bin = max(0, min(price_bin, self.num_price_bins -1))

        return np.array([price_bin, time_to_maturity])
        

    def _get_info(self):
        "Retorna informações auxiliares sobre o ambiente."
        price_idx = min(self.current_step, len(self.stock_path) - 1)
        return {"current_price": self.stock_path[price_idx]}

    def reset(self, seed=None, option=None):
        super().reset(seed=seed)

        """Reseta o ambiente para um estado inicial."""
        times = np.linspace(0.0, self.T / 365.0, self.T + 1)
        time_grid = ql.TimeGrid(list(times), len(times))

        # Gerador de números aleatórios.
        rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(self.T, ql.UniformRandomGenerator(seed=np.random.randint(0,1000))))
        path_generator = ql.GaussianPathGenerator(self.bsm_process, time_grid, rng, False)

        path = path_generator.next().value()
        self.stock_path = np.array(list(path))

        # reset do estado interno.
        self.current_step = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        """Executa um passo no ambiente a partir de uma ação."""
        
        # 1. Verifica se o episódio JÁ está no fim ANTES de fazer qualquer coisa.
        # A última ação possível é no penúltimo dia (T-1).
        terminated = self.current_step >= self.T

        if terminated:
            # Se já terminou, não faz sentido executar um novo passo.
            # Apenas retornamos a observação final com recompensa 0.
            # Isso evita que o 'current_step' seja incrementado além do limite.
            return self._get_obs(), 0, True, False, self._get_info()

        # 2. Pega o preço atual antes de avançar no tempo.
        current_price = self.stock_path[self.current_step]

        # 3. Executa a ação do agente no portifólio.
        if action == 2: # Ação de Compra.
            self.stock_position += 1
            self.cash_balance -= current_price * (1 + self.transaction_cost)
        elif action == 0: # Ação de venda.
            self.stock_position -= 1
            self.cash_balance += current_price * (1 - self.transaction_cost)

        # 4. Se não terminou, avança para o próximo dia.
        self.current_step += 1

        # 5. Recompensa só é dada no final do episódio.
        reward = 0

        # A verificação de terminação agora é se o NOVO passo é o final.
        if self.current_step >= self.T:

            # Preço final da posição.
            final_price = self.stock_path[self.current_step]

            # Liquida a posição em ações e soma ao caixa.
            final_portfolio_value = self.cash_balance + self.stock_position * final_price

            # Calcula o Payoff da opção. 
            option_payoff = max(final_price - self.K, 0)
            
            # O objetivo é encontrar o valor do portfólio seja igual ao payoff.
            # A diferença é o erro de hedge.
            hedge_pnl = final_portfolio_value + option_payoff
            
            # Recompensas negativas penalizam o desvio.
            reward = -(hedge_pnl**2)
        
        observation = self._get_obs()
        info = self._get_info()
        truncated = False

        # O episódio termina se o passo atual for o último dia.
        return observation, reward, self.current_step >= self.T, truncated, info