import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class PortfolioOptimization:
    def __init__(self, tsla_data, bnd_data, spy_data):
        """
        Initializes the PortfolioOptimization class with data for each asset.
        
        :param tsla_data: Forecasted data for Tesla stock.
        :param bnd_data: Forecasted data for Vanguard Total Bond Market ETF.
        :param spy_data: Forecasted data for S&P 500 ETF.
        """
        # Convert 'Close' column to numeric, in case there are any non-numeric values
        tsla_data['Close'] = pd.to_numeric(tsla_data['Close'], errors='coerce')
        bnd_data['Close'] = pd.to_numeric(bnd_data['Close'], errors='coerce')
        spy_data['Close'] = pd.to_numeric(spy_data['Close'], errors='coerce')

        # Merge the dataframes based on 'Date' and set index as 'Date'
        self.df = pd.merge(tsla_data[['Date', 'Close']], bnd_data[['Date', 'Close']], on='Date', how='inner')
        self.df = pd.merge(self.df, spy_data[['Date', 'Close']], on='Date', how='inner')

        # Rename columns
        self.df.columns = ['Date', 'TSLA', 'BND', 'SPY']
        self.df.set_index('Date', inplace=True)

        # Convert the Close columns to numeric just to be safe
        self.df = self.df.apply(pd.to_numeric, errors='coerce')

        # Calculate daily returns
        self.returns = self.df.pct_change().dropna()

        # Initialize the annualized returns and covariance matrix
        self.annualized_returns = self.calculate_annualized_returns()
        self.cov_matrix = self.calculate_cov_matrix()

    def calculate_annualized_returns(self):
        """Calculate and return the annualized returns for each asset."""
        average_daily_returns = self.returns.mean()
        return average_daily_returns * 252  # Annualize the returns assuming 252 trading days

    def calculate_cov_matrix(self):
        """Calculate and return the annualized covariance matrix."""
        return self.returns.cov() * 252  # Annualize the covariance matrix

    def optimize_portfolio(self):
        """
        Optimizes the portfolio weights to maximize the Sharpe ratio.
        
        :return: The optimal portfolio weights that maximize the Sharpe ratio.
        """
        # Initial guess for the portfolio weights (equal weights)
        initial_weights = np.array([1/3, 1/3, 1/3])
        
        # Constraints: the sum of weights should be 1
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        
        # Bounds for the weights: each weight should be between 0 and 1
        bounds = [(0, 1) for _ in range(len(self.annualized_returns))]
        
        # Optimize portfolio
        optimal_weights = minimize(self.negative_sharpe_ratio, initial_weights, args=(self.annualized_returns, self.cov_matrix),
                                   method='SLSQP', bounds=bounds, constraints=constraints)
        return optimal_weights.x

    def negative_sharpe_ratio(self, weights, annualized_returns, cov_matrix):
        """
        The objective function for portfolio optimization, which minimizes the negative Sharpe ratio.
        
        :param weights: Portfolio weights for each asset.
        :param annualized_returns: Annualized returns of the assets.
        :param cov_matrix: Covariance matrix of the asset returns.
        :return: Negative Sharpe ratio (we minimize this to maximize the Sharpe ratio).
        """
        portfolio_return = np.dot(weights, annualized_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -portfolio_return / portfolio_volatility  # We negate the Sharpe Ratio to minimize

    def calculate_portfolio_performance(self, weights):
        """
        Calculates the expected return, volatility (risk), and Sharpe ratio of the portfolio.
        
        :param weights: The weights for each asset in the portfolio.
        :return: A tuple containing the portfolio's expected return, risk, and Sharpe ratio.
        """
        portfolio_return = np.dot(weights, self.annualized_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        risk_free_rate = 0  # Assume 0% risk-free rate for simplicity
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def visualize_risk_return(self):
        """
        Visualizes the risk-return trade-off for the portfolio.
        """
        portfolio_returns = []
        portfolio_volatilities = []

        for w1 in np.linspace(0, 1, 100):
            for w2 in np.linspace(0, 1-w1, 100):
                w3 = 1 - w1 - w2
                weights = np.array([w1, w2, w3])
                portfolio_returns.append(np.dot(weights, self.annualized_returns))
                portfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))))

        # Avoid division by zero for color scaling in scatter plot
        portfolio_returns = np.array(portfolio_returns)
        portfolio_volatilities = np.array(portfolio_volatilities)
        sharpe_ratios = portfolio_returns / portfolio_volatilities
        sharpe_ratios = np.nan_to_num(sharpe_ratios)  # Replace NaN values with 0

        # Plot the risk-return trade-off
        plt.figure(figsize=(10, 6))
        plt.scatter(portfolio_volatilities, portfolio_returns, c=sharpe_ratios, cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility (Risk)')
        plt.ylabel('Return')
        plt.title('Risk-Return Trade-Off')
        plt.show()