"""Variance-matched guidance: choose the guidance weight of a diffusion model without sampling.

    from vmg import solve, moments, models
    m = moments.reduce(*moments.load(paths))
    w = solve.fit_global(m)[0]           # eq. wconstant
    w_t = solve.fit_schedule(m)["w"]     # eq. wstar
    w_mse = solve.w_mse(m)               # eq. wmse

The command-line entry points are compute_w.py and compute_wmse.py at the repository root.
"""
