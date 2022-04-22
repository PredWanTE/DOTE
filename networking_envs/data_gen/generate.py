import numpy as np


# returns extrapolated TM series between two TMs
def extrapolate(tm1, tm2, steps_mean, steps_var):
    num_steps = max(0, int(np.random.normal(steps_mean, steps_var) + 0.5))
    epsilons = sorted([np.random.rand() for _ in range(num_steps)], reverse=True)
    return [tm1 * eps + (1-eps) * tm2 for eps in epsilons]


# creates new series of TMs using extrapolation
def generate_series(tm_generator, g, props): #series_len, extrapulate=True):
    series_ = [tm_generator(g, props) for _ in range(props.series_len)]
    if props.extrapulate_tm_series:
        series = []
        for i in range(len(series_)-1):
            series.append(series_[i])
            series += extrapolate(series_[i], series_[i+1], props.extrapulate_mean, props.extrapulate_var)
        series.append(series_[-1])
    else:
        series = series_
    return series
