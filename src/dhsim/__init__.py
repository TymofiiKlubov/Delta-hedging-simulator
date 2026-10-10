"""dhsim: a delta-hedging simulator.

Sell a European option at its Black-Scholes price, delta-hedge it on simulated paths and
attribute the P&L. The package is organised as:

* :mod:`dhsim.bs`          Black-Scholes prices and Greeks (vectorised)
* :mod:`dhsim.paths`       GBM, Merton jump-diffusion and Heston path simulators
* :mod:`dhsim.hedging`     the hedging engine and P&L attribution
* :mod:`dhsim.theory`      closed-form / leading-order results the simulation is checked against
* :mod:`dhsim.experiments` the five experiments behind the CLI
"""

__version__ = "0.1.0"
