# Quantitative Finance

### Diffusion
Libraries for stochastic processes simulation and visualization including:
* Ito diffusion : Brownian motion, Geometric Brownian motion, Vasicek, CIR...
* Jump processes : Ito diffusion driven by a Levy process i.e with a jump component with a given intensity and jump size distribution
* Multidimensional processes, stochastic volatility diffusions (SABR...)
* Fractional Brownian motion, Karhunen-Loeve expansion, fractional diffusions

### Hurst
Estimation of Hurst exponent, a measure of geometric roughness of time series that encapsulates local mean-reversion or trend-following behaviours. Estimation on both simulated and real-world financial data at order book level (stocks, Bund, BTC...)

### Implied Vol
Toy model for an option volatility marking tool using SVI parametrization.

### Jump Models
Fit a Ornstein-Uhlenbeck process (potentially with Laplace jumps) on historical data using the generalized methods of moments on the characteristic functon.

### Lasso selection
Sparse replication of portfolio using Lasso variable selection.

### Lobster 
Reader for Lobster order book data (https://lobsterdata.com/)

### PCA
Study of main drivers for UST curve moves using PCA.

### Vol Models
Libraries for option volatility smile models including:
* Several models: SVI, displaced LN, SABR and extensions, local vol SABR...
* Implied volatility expansion, extended Hagan formula, Monte Carlo simulations
* Fitting routine to quoted implied volatility

### cleaning_correlation_matrix
Quick study of several spectral methods to improve empirical estimation of correlation matrices

### low_beta_strategy
Quick study of the performance of low vs high sectorial beta portfolios. Low beta sectors are less correlated with the index thus are more robust to crash periods and can potentially generate higher alpha.
