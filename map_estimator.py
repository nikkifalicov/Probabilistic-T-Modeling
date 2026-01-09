import numpy as np
import scipy.stats
from sklearn.preprocessing import PolynomialFeatures


class MAPEstimator():

    def __init__(self, feature_transformer=None, alpha=1.0, beta=1.0):
        self.feature_transformer = feature_transformer
        self.alpha = float(alpha)
        self.beta = float(beta)

        # State that is adjusted by calls to 'fit'
        self.w_map_M = 0.0

    def get_params(self, deep=False):
        ''' Obtain key attributes for this object as a dictionary

        Needed for use with sklearn CV functionality

        Returns
        -------
        param_dict : dict
        '''
        return {'alpha': self.alpha, 'beta': self.beta, 'feature_transformer': self.feature_transformer}

    def set_params(self, **params_dict):
        ''' Set key attributes of this object from provided dictionary

        Needed for use with sklearn CV functionality

        Returns
        -------
        self. Internal attributes updated.
        '''
        for param_name, value in params_dict.items():
            setattr(self, param_name, value)
        return self

    def fit(self, x_ND, t_N):
        ''' Fit this estimator to provided training data

        Args
        ----
        x_ND : 2D array, shpae (N, D)
            Each row is a 'raw' feature vector of size D
            D is same as self.feature_transformer.input_dim
        t_N : 1D array, shape (N,)
            Outputs for regression

        Returns
        -------
        self. Internal attributes updated.
        '''
        Phi_NM = self.feature_transformer.fit_transform(x_ND)
        M = self.feature_transformer.n_output_features_  # num features
        Phi_transpose = Phi_NM.transpose()
        phi_phi_t = np.matmul(Phi_transpose, Phi_NM)
        I_m = np.identity(M)

        inner_term = phi_phi_t + (self.alpha/self.beta)*I_m

        matrix_product = np.matmul(
            np.linalg.inv(inner_term), Phi_transpose)

        self.w_map_M = np.matmul(matrix_product, t_N)

        return self

    def predict(self, x_ND):
        ''' Make predictions of output value for each provided input feature vectors

        Args
        ----
        x_ND : 2D array, shape (N, D)
            Each row is a 'raw' feature vector of size D
            D is same as self.feature_transformer.input_dim

        Returns
        -------
        t_est_N : 1D array, size (N,)
            Each entry at index n is prediction given features in row n
        '''
        phi_NM = self.feature_transformer.transform(x_ND)
        N, M = phi_NM.shape
        # compute mean
        # print(self.w_map_M.transpose().shape)
        # print(phi_NM.shape)
        result = np.matmul(phi_NM, self.w_map_M.transpose())

        return result

    def predict_variance(self, x_ND):
        ''' Produce predictive variance at each input feature

        Args
        ----
        x_ND : 2D array, shape (N, D)
            Each row is a 'raw' feature vector of size D
            D is same as self.feature_transformer.input_dim

        Returns
        -------
        t_var_N : 1D array, size (N,)
            Each entry at index n is variance given features in row n
        '''
        phi_NM = self.feature_transformer.transform(x_ND)
        N, M = phi_NM.shape

        # TODO compute variance
        # return the beta^-1 but for each of the n entries
        result = np.ones(N)
        result = result*(1/self.beta)
        return result

    def score(self, x_ND, t_N):
        ''' Compute the average log probability of provided dataset

        Assumes w is set to MAP value (internal attribute).
        Assumes Normal iid likelihood with precision \beta.

        Args
        ----
        x_ND : 2D array, shape (N, D)
            Each row is a 'raw' feature vector of size D
            D is same as self.feature_transformer.input_dim
        t_N : 1D array, shape (N,)
            Outputs for regression

        Returns
        -------
        avg_log_proba : float
        '''
        N = x_ND.shape[0]
        mean_N = self.predict(x_ND)
        total_log_proba = scipy.stats.norm.logpdf(
            t_N, mean_N, 1.0/np.sqrt(self.beta))
        return np.sum(total_log_proba) / N
