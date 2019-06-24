def stochastic_oscilator(data, window_size=3):

    indicator_values = []
    for i in range(window_size, len(data)+1):
        window = data[i-window_size:i]
        now = window[-1]
        lowest = min(window)
        highest = max(window)
        value = (now - lowest) / (highest - lowest)
        indicator_values.append(value)

    return indicator_values


def rate_of_change(data, window_size=3):
    indicator_values = []

    for i in range(window_size, len(data)+1):
        window = data[i-window_size:i]
        now = window[-1]
        first = window[0]
        value = now/first
        indicator_values.append(value)

    return indicator_values


def momentum(data, window_size=3):
    indicator_values = []
    for i in range(window_size, len(data)+1):
        window = data[i-window_size:i]
        now = window[-1]
        first = window[0]
        value = now-first
        indicator_values.append(value)

    return indicator_values

def moving_avg(data, window_size=3):
    indicator_values = []

    for i in range(window_size, len(data)+1):
        window = data[i-window_size:i]
        sum = 0
        for j in range(0, window_size):
            sum = sum + window[j]
        indicator_values.append(sum/window_size)

    return indicator_values

def mso(data, window_size=3):
    indicator_values = []

    SO = stochastic_oscilator(data, window_size)
    indicator_values = moving_avg(SO, window_size)

    return indicator_values


def sso(data, window_size=3):
    indicator_values = []

    MSO = mso(data, window_size)
    indicator_values = moving_avg(MSO, window_size)

    return indicator_values

def moving_var(data, window_size=3):
    indicator_values = []

    avgs = moving_avg(data, 3)
    for i in range(window_size, len(data)+1):
        window = data[i-window_size:i]
        sum = 0
        for j in range(0, window_size):
            sum += (window[j]-avgs[i-window_size])**2
        indicator_values.append(sum/window_size)


    print(indicator_values)
    return indicator_values

def mv_ratio(data, window_size=3):
    indicator_values = []

    vars = moving_var(data, 3)
    for i in range(window_size, len(vars)+1):
        window = vars[i-window_size:i]
        mv_t = window[-1]
        mv_i = window[0]
        mvr = (mv_t**2) / (mv_i**2)
        indicator_values.append(mvr)

    return indicator_values

def exponencial_movie_avg(data, window_size=3):
    indicators_values = []
    k = 2/(window_size + 1)

    indicators_values.append(moving_avg(data, window_size)[0])

    for i in range(window_size, len(data)):
        value = data[i] * k + indicators_values[-1] * (1-k)
        indicators_values.append(value)

    return indicators_values

def macd(data, window_size_1=3, window_size_2=4):
    indicators_values = []

    ema1 = exponencial_movie_avg(data, window_size_1)
    ema2 = exponencial_movie_avg(data, window_size_2)

    for i in range(0, len(ema2)):
        value = ema1[i] - ema2[i]
        indicators_values.append(value)

    return indicators_values


# d = [10, 9, 6, 7, 8, 11, 30]

# print("stochastic_oscilator")
# print(stochastic_oscilator(d, 3))

# print("mso")
# print(mso(d, 3))

# print("sso")
# print(sso(d, 3))

# print("rate_of_change")
# print(rate_of_change(d, 3))

# print("momentum")
# print(momentum(d, 3))

# print("moving avg")
# print(moving_avg(d, 3))

# print("moving var")
# print(moving_var(d, 3))

# print("mv ratio")
# print(mv_ratio(d, 3))

# print("exponencial movie avg")
# print(exponencial_movie_avg(d, 3))

# print("macd")
# print(macd(d, 3, 4))
