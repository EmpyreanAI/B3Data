def stochastic_oscilator(data, window=14):
	indicator_values = []

	for i in xrange(window, len(data)):
		lowest = min(data[i-window:i])
		highest = max(data[i-window:i])

		value = (data[i] - lowest) / (highest - lowest) 
		result.append(value)
