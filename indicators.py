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

def mso(data, window_size=3):
	indicator_values = []

	SO = stochastic_oscilator(data, window_size)
	for i in range(window_size, len(data)+1):
		value = sum(SO[i-window_size:i])/window_size
		indicator_values.append(value)

	return indicator_values

d = [10, 9, 6, 7, 8, 11, 30]

print("stochastic_oscilator")
print(stochastic_oscilator(d, 4))

print("mso")
print(mso(d, 4))

print("rate_of_change")
print(rate_of_change(d, 4))

print("momentum")
print(momentum(d, 4))