
def traverse(inputs):
	Out = []
	for input in inputs:
		multiple = int(input / 5)
		limit_low = multiple * 5
		limit_up = limit_low + 5
		middle = float((limit_low + limit_up) / 2)

		if input > middle:
			out = limit_up
		else:
			out = limit_low
		Out.append(out)

	return Out