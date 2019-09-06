"""Nani."""


class Indicators():
    """Nani."""

    @staticmethod
    def stochastic_oscilator(data, window_size=3):
        """Nani."""
        indicator_values = []
        for i in range(window_size, len(data)+1):
            window = data[i-window_size:i]
            now = window[-1]
            lowest = min(window)
            highest = max(window)
            value = (now - lowest) / (highest - lowest)
            indicator_values.append(value)

        return indicator_values

    @staticmethod
    def rate_of_change(data, window_size=3):
        """Nani."""
        indicator_values = []

        for i in range(window_size, len(data)+1):
            window = data[i-window_size:i]
            now = window[-1]
            first = window[0]
            value = now/first
            indicator_values.append(value)

        return indicator_values

    @staticmethod
    def momentum(data, window_size=3):
        """Nani."""
        indicator_values = []
        for i in range(window_size, len(data)+1):
            window = data[i-window_size:i]
            now = window[-1]
            first = window[0]
            value = now-first
            indicator_values.append(value)

        return indicator_values

    @staticmethod
    def moving_avg(data, window_size=3):
        """Nani."""
        indicator_values = []

        for i in range(window_size, len(data)+1):
            window = data[i-window_size:i]
            data_sum = 0
            for j in range(0, window_size):
                data_sum = data_sum + window[j]
            indicator_values.append(data_sum/window_size)

        return indicator_values

    @classmethod
    def mso(cls, data, window_size=3):
        """Nani."""
        indicator_values = []

        stochastic_oscilator = cls.stochastic_oscilator(data, window_size)
        indicator_values = cls.moving_avg(stochastic_oscilator, window_size)

        return indicator_values

    @classmethod
    def sso(cls, data, window_size=3):
        """Nani."""
        indicator_values = []

        mean_sqr_error = cls.mso(data, window_size)
        indicator_values = cls.moving_avg(mean_sqr_error, window_size)

        return indicator_values

    @classmethod
    def moving_var(cls, data, window_size=3):
        """Nani."""
        indicator_values = []

        avgs = cls.moving_avg(data, 3)
        for i in range(window_size, len(data)+1):
            window = data[i-window_size:i]
            data_sum = 0
            for j in range(0, window_size):
                data_sum += (window[j]-avgs[i-window_size])**2
            indicator_values.append(data_sum/window_size)

        return indicator_values

    @classmethod
    def mv_ratio(cls, data, window_size=3):
        """Nani."""
        indicator_values = []

        data_vars = cls.moving_var(data, 3)
        for i in range(window_size, len(data_vars)+1):
            window = data_vars[i-window_size:i]
            mv_t = window[-1]
            mv_i = window[0]
            mvr = (mv_t**2) / (mv_i**2)
            indicator_values.append(mvr)

        return indicator_values

    @classmethod
    def exponencial_movie_avg(cls, data, window_size=3):
        """Nani."""
        indicators_values = []
        k = 2/(window_size + 1)

        indicators_values.append(cls.moving_avg(data, window_size)[0])

        for i in range(window_size, len(data)):
            value = data[i] * k + indicators_values[-1] * (1-k)
            indicators_values.append(value)

        return indicators_values

    @classmethod
    def macd(cls, data, window_size_1=3, window_size_2=4):
        """Nani."""
        indicators_values = []

        ema1 = cls.exponencial_movie_avg(data, window_size_1)
        ema2 = cls.exponencial_movie_avg(data, window_size_2)

        for i, element in enumerate(ema2):
            value = ema1[i] - element
            indicators_values.append(value)

        return indicators_values
