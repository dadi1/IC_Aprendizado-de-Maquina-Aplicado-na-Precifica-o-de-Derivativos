import gymnasium as gym
from gymnasium import spaces
import numpy as np
import QuantLib as ql

class AmbienteOpcao(gym.Env):
    """
    Ambiente customizado de Gymnasium para simulação de precificação
    e hedge de uma opção europeia.
    """

    def __init__(self, S0=100, K=100, r=0.05, sigma=0.2, T=30):
        super(AmbienteOpcao, self).__init__()

        # --- Parâmetros da Simulação ---
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.transaction_cost = 0.001

        # --- Estado do Portfólio ---
        self.stock_position = 0
        self.cash_balance = 0

        # --- Definição dos Espaço de Ação e Observação ---
        self.action_space = spaces.Discrete(3)

        self.num_price_bins = 100
        self.price_bins = np.linspace(self.S0 * 0.5, self.S0 * 1.5, self.num_price_bins)
        
        # Discretizar a posição em ações e o delta também.
        self.max_stocks = 10
        self.num_position_bins = self.max_stocks * 2 + 1
        
        self.num_delta_bins = 20

        self.observation_space = spaces.MultiDiscrete([
            self.num_price_bins,
            self.T + 1,
            self.num_position_bins,
            self.num_delta_bins
        ])

        # --- Configuração do Processo de Simulação (QuantLib) ---
        self.day_count = ql.Actual365Fixed()
        self.calendar = ql.Brazil()
        self.calculation_date = self.calendar.adjust(ql.Date.todaysDate())
        ql.Settings.instance().evaluationDate = self.calculation_date

        spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.S0))
        risk_free_rate = ql.FlatForward(self.calculation_date, self.r, self.day_count)
        self.rate_ts = ql.YieldTermStructureHandle(risk_free_rate)
        volatility = ql.BlackConstantVol(self.calculation_date, self.calendar, self.sigma, self.day_count)
        self.vol_ts = ql.BlackVolTermStructureHandle(volatility)

        self.bsm_process = ql.BlackScholesMertonProcess(spot_handle, self.rate_ts, self.rate_ts, self.vol_ts)

        self.maturity_date = self.calendar.advance(self.calculation_date, ql.Period(self.T, ql.Days))
        
        # Calculo das gregas
        option_payoff = ql.PlainVanillaPayoff(ql.Option.Call, self.K)
        exercise = ql.EuropeanExercise(self.maturity_date)
        self.vanilla_option = ql.VanillaOption(option_payoff, exercise)

        # --- Variáveis de estado do episódio ---
        self.current_step = 0
        self.stock_path = []

    def _get_obs(self):
        """Retorna a observação do estado atual, incluindo o delta do BSM."""
        price_idx = min(self.current_step, len(self.stock_path) - 1)
        price = self.stock_path[price_idx]
        time_to_maturity_days = self.T - self.current_step

        price_bin = np.digitize(price, self.price_bins) - 1
        price_bin = max(0, min(price_bin, self.num_price_bins - 1))

        position_bin = int(self.stock_position + self.max_stocks)
        position_bin = max(0, min(position_bin, self.num_position_bins - 1))

        bsm_delta = self._calculate_bsm_delta(price, time_to_maturity_days)
        delta_bin = int(bsm_delta * (self.num_delta_bins -1)) # Mapeia delta (0 a 1) para bins
        delta_bin = max(0, min(delta_bin, self.num_delta_bins - 1))

        return np.array([price_bin, time_to_maturity_days, position_bin, delta_bin])

    def _get_info(self):
        """Retorna informações auxiliares sobre o ambiente."""
        price_idx = min(self.current_step, len(self.stock_path) - 1)
        return {"current_price": self.stock_path[price_idx]}

    def _calculate_bsm_delta(self, current_price, days_to_maturity):
        """Calcula o Delta teórico do BSM para o estado atual."""
        if days_to_maturity == 0:
            return 1.0 if current_price > self.K else 0.0

        calc_date = self.calendar.advance(self.maturity_date, -days_to_maturity, ql.Days)
        ql.Settings.instance().evaluationDate = calc_date
        
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(current_price))
        bsm_process_today = ql.BlackScholesMertonProcess(spot_handle, self.rate_ts, self.rate_ts, self.vol_ts)
        
        self.vanilla_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process_today))
        
        delta = self.vanilla_option.delta()
        return delta if not np.isnan(delta) else 0.0

    def reset(self, seed=None, options=None):
        """Reseta o ambiente para um estado inicial."""
        super().reset(seed=seed)

        times = np.linspace(0.0, self.T / 365.0, self.T + 1)
        time_grid = ql.TimeGrid(list(times), len(times))

        rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(self.T, ql.UniformRandomGenerator(seed=np.random.randint(0,10000))))
        path_generator = ql.GaussianPathGenerator(self.bsm_process, time_grid, rng, False)

        path = path_generator.next().value()
        self.stock_path = np.array(list(path))

        self.current_step = 0
        self.stock_position = 0
        self.cash_balance = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Executa um passo no ambiente a partir de uma ação."""
        terminated = self.current_step >= self.T
        if terminated:
            return self._get_obs(), 0, True, False, self._get_info()

        current_price = self.stock_path[self.current_step]

        if action == 2:
            self.stock_position += 1
            self.cash_balance -= current_price * (1 + self.transaction_cost)
        elif action == 0:
            self.stock_position -= 1
            self.cash_balance += current_price * (1 - self.transaction_cost)

        self.current_step += 1
        
        reward = 0
        if self.current_step >= self.T:
            final_price = self.stock_path[self.current_step]
            final_portfolio_value = self.cash_balance + self.stock_position * final_price
            option_payoff = max(final_price - self.K, 0)
            hedge_pnl = final_portfolio_value + option_payoff
            reward = -(hedge_pnl**2)
        
        observation = self._get_obs()
        info = self._get_info()
        truncated = False

        return observation, reward, self.current_step >= self.T, truncated, info