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

        # --- Parâmetros da Simulação ---
        self.S0 = S0 # Preço inicial do ativo.
        self.K = K # Preço de exercício (Strike).
        self.r = r # Taxa de juros livres de risco.
        self.sigma = sigma # Volatilidade do ativo.
        self.T = T # tempo até o vencimento em dias.

        # --- Definição dos Espaço de Ação e Observação. ---
        # Ações: 0=vender, 1=Manter, 2=Comprar unidade do ativo para hedge.
        self.acition_space = spaces.Discrete(3)

        self.num_price_bins = 100
        self.price_bins = np.linspace(self.S0 * 0.5, self.S0 * 1.5, self.num_price_bins)

        self.observation_space = spaces.MultiDiscrete([self.num_price_bins, self.T_days + 1])


        # --- Configuração do Processo de Simulação (QuantLib) ---
        self.timestep = 1
        self.day_count = ql.Actual365Fixed()
        self.calender = ql.Brazil()
        
        self.calculation_date = self.calender.adjust(ql.Date.todaysDate())
        ql.Setting.instance().evaluationDate = self.calculation_date

        # TODO: Configura os processo de BSM aqui.
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.S0))
        risk_free_rate = ql.FlatForward(self.calculation_date, self.r, self.day_count)
        rate_ts = ql.YieldTermStructureHandle(risk_free_rate)
        volatility = ql.BlackConstantVol(self.calculation_date, self.calender, self.sigma, self.day_count)
        vol_ts = ql.BlackVolTermStructureHandle(volatility)

        self.bsm_process = ql.BlackScholesMertonProcess(spot_handle, rate_ts, rate_ts, vol_ts)

        # --- Variaveis do processo para cada tempo T. ---
        self.current_step = 0
        self.stock_path = []


    def _get_obs(self):
        """Retorna a observação do estado atual.""" 
        price = self.stock_path[self.current_step]
        time_to_maturity = self.T - self.current_step

        price_bin = np.digitize(price, self.price_bins) - 1
        # Garante que o bin esteja sempre dentro dos limites [0, num_precis_bins -1]
        price_bin = max(0, min(price_bin, self.num_price_bins -1))

        return np.array([price_bin, time_to_maturity])
        

    def _get_info(self):
        "Retorna informações auxiliares sobre o ambiente."
        
        return {"current_price": self.stock_path[self.current_step]}

    def reset(self, seed=None, option=None):
        """Reseta o ambiente para um estado inicial."""
        times = np.linspace(0.0, self.T / 365.0, self.T + 1)
        time_grid = ql.TimeGrid(list(times), len(times))

        # Gerador de números aleatórios.
        rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomGenrator(self.T, ql.UniformRandomGenerator(seed=np.random.randint(0,1000))))
        path_generator = ql.GaussianPathGenerator(self.bsm_process, time_grid, rng, False)

        path = path_generator.next().value()

        # reset do estado interno.
        self.current_step = 0

        observation = self.get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        """Executa um passo no ambiente a partir de uma ação."""
        self.currente_step += 1

        # verifica se o evento terminou.
        terminated = self.current_step >= self.T

        # Lógica de recompensa
        # Recompensa é calculada com base nos custos de hedge.
        # Por enquanto, a recompensa é apenas calculada no final do evento.
        reward = 0
        if terminated:
            # No final a recompensa é baseada no payoff da opção
            final_price = self.stock_path[self.current_step]
            payoff = max(final_price - self.K, 0)

            # A recompensa será o inverso do payoff.
            reward = - payoff
        
        observation = self._get_obs()
        info = self._get_info()
        truncated = False # Não há truncamento.


        return observation, reward, terminated, truncated, info
    