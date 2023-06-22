import numpy as np

class MultivariateNormalConjugatePrior:
    def __init__(self, mu0, Sigma0, Sigma):
        """  
        :param mu0: Prior mean
        :param Sigma0: Prior covariance matrix
        :param Sigma: Likelihood covariance matrix
        """
        
        self.mu0 = mu0
        self.Sigma0 = Sigma0
        self.Lambda0 = np.linalg.inv(Sigma0)  # Precision matrix
        
        self.Sigma = Sigma
        self.Lambda = np.linalg.inv(Sigma)
        
        self.mu_list = [self.mu0]
        self.Sigma_list = [self.Sigma0]

    def update(self, x):
        """ Update posterior for new data x.
        x should be of shape (n, d), where n is number of datapoints
        and d is dimensionality of MVN
        """
        n = x.shape[0]  
        x_bar = np.mean(x, axis=0)  # Sample mean 
        self.Sigma0 = np.linalg.inv(self.Lambda0 + n * self.Lambda)
        self.mu0 = self.Sigma0 @ (self.Lambda0 @ self.mu0 + n * self.Lambda @ x_bar)
        self.Lambda0 = np.linalg.inv(self.Sigma0)
        
        self.mu_list.append(self.mu0)
        self.Sigma_list.append(self.Sigma0)        