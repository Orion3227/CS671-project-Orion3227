from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
import logging

class GaussianDistribution:
    """
    Data should be in the format (points x spaces)
    So if we have 10 datapoints in a 3dim space --> data.shape == (10, 3)
    """

    def __init__(self, data=None):
        self.mu = 0
        self.std = 1
        if data is not None:
           self.set_data(data)

    def set_data(self, data):
        if data.ndim == 1:
            data = np.reshape(data, (data.shape[0], 1))
        elif data.ndim != 2:
            raise ValueError("List has to be either one or two dimensional")

        self.mu = np.mean(data, axis=0)
        self.std = np.cov(data.T)
        self.data = data
        print(self.data)
        print("Updated Gaussian Distribution to:\n\tmu: {}\n\tstd: {}".format(self.mu, self.std))

    @property
    def dimension(self):
        return self.data.shape[1]

    @property
    def pdf(self):
        rv = multivariate_normal(self.mu, self.std)
        def pdf(x):
            return rv.pdf(x)
        return pdf

    def add_to_plot_1d(self, ax, dim=0, color="C0"):
        ax.scatter(self.data[:, dim], np.zeros((self.data.shape[0])), color=color)
        # TODO add code for Gaussian visualization
        return ax

    def add_to_plot_2d(self, ax, dim=(0, 1), color="C0"):
        print(len(self.data[:, dim[0]]))
        ax.scatter(self.data[:, dim[0]], self.data[:, dim[1]])
        # TODO add code for Gaussian visualization
        return ax

    def plot_1d(self, dim=0, color="C0"):
        fig, ax = plt.subplots(1)
        ax = self.add_to_plot_1d(ax)
        fig.show()

    def plot_2d(self, dim=(0, 1), color="C0"):
        fig, ax = plt.subplots(1)
        ax = self.add_to_plot_2d(ax)
        fig.show()


def KullbackLeiberDivergenceSamples(pdf_0, pdf_1, dimension, range=[-10, 10]):
    pass

def KullbackLeiberDivergenceSingleVarianteGaussians(gaussian_0, gaussian_1):
    return 0

def KullbackLeiberDivergenceMultiVarianteGaussians(gaussian_0, gaussian_1):
    if gaussian_1.dimension != gaussian_0.dimension:
        raise ValueError("Gaussians need to have the same dimension")
    if gaussian_0.dimension < 2:
        return KullbackLeiberDivergenceSingleVarianteGaussians(gaussian_0, gaussian_1)

    mu_0 = gaussian_0.mu
    mu_1 = gaussian_1.mu
    sigma_0 = gaussian_0.std
    sigma_1 = gaussian_1.std
    sigma_1_inv = np.linalg.pinv(sigma_1)

    divergence = 1/2 * (np.trace(sigma_1_inv.dot(sigma_0)) 
                        + (mu_1 - mu_0).T.dot(sigma_1_inv).dot(mu_1 - mu_0)
                        - gaussian_0.dimension 
                        + np.log(np.linalg.norm(sigma_1)/np.linalg.norm(sigma_0)))
    return divergence   
   
def JensenShannonDivergenceMultiVarianteGaussians(gaussian_0, gaussian_1):
    #https://math.stackexchange.com/questions/60911/multivariate-normal-difference-distribution
    pass

if __name__ == "__main__":
    data_1d = np.array([-1.0 , -0.4, -0.2, 0, 0.1, 0.3, 0.5])
    data_2d = np.array([[-1.0, 0.5], [-0.5, 0.3], [-0.1, 0.2], [0.2, -0.4]])
    data_5d = np.array([[0.1, 0.3, 0.4, 0.8, 0.9],
                        [3.2, 2.4, 2.4, 0.1, 5.5],
                        [10., 8.2, 4.3, 2.6, 0.9]])

    gaussian_distribution_1d = GaussianDistribution(data_1d)
    gaussian_distribution_1d.plot_1d(dim=0)

    # print(KullbackLeiberDivergenceMultiVarianteGaussians(gaussian_distribution_1d, gaussian_distribution_1d))

    gaussian_distribution_2d = GaussianDistribution(data_2d)
    gaussian_distribution_2d.plot_2d(dim=(0, 1))

    kl_2d_self = KullbackLeiberDivergenceMultiVarianteGaussians(gaussian_distribution_2d, gaussian_distribution_2d)
    print("Kullbackleiber Divergence 2D for self: {}".format(kl_2d_self))

    gaussian_distribution_5d = GaussianDistribution(data_5d)
    gaussian_distribution_5d.plot_2d(dim=(2,3))

    pdf_1d = gaussian_distribution_1d.pdf
    
    input("Finish?")