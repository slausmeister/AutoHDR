def _ema(list):
    alpha = 2 / (len(list) + 1)  # Calculate the smoothing factor
    ema_value = list[0]  # Initialize the EMA with the first value in the list

    for i in range(1, len(list)):
        ema_value = alpha * list[i] + (1 - alpha) * ema_value

    return ema_value
