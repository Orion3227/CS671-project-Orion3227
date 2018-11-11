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
        data = np.array(data)
        if data.ndim == 1:
            data = np.reshape(data, (data.shape[0], 1))
        elif data.ndim != 2:
            raise ValueError("List has to be either one or two dimensional, found {}".format(data.ndim))

        self.mu = np.mean(data, axis=0)
        self.std = np.cov(data.T)
        self.data = data
        # print(self.data)
        print("Updated Gaussian Distribution to:\n\tmu: {}\n\tstd: {}".format(self.mu, self.std))

    @property
    def dimension(self):
        return self.mu.shape[0]

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

    def mahalanobis_distance(self, sample):
        return np.sqrt((sample - self.mu).T.dot(np.linalg.pinv(self.std)).dot(sample - self.mu))

    def is_in_distribution(self, sample, treshold=0.3):
        distance = self.mahalanobis_distance(sample)
        print("Mahalanobis Distance: {}".format(distance))
        norm_distance = distance / (distance + 1)
        print("Normalized: {}".format(norm_distance))
        return distance < treshold       


def KullbackLeiberDivergenceSamples(pdf_0, pdf_1, dimension, range=[-10, 10]):
    pass

def KullbackLeiberDivergenceSingleVarianteGaussians(gaussian_0, gaussian_1):
    pass

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
    # https://www.tensorflow.org/api_docs/python/tf/contrib/distributions/MultivariateNormalFullCovariance#kl_divergence
    # We might reimplement this in Tensorflow?
    
    #https://math.stackexchange.com/questions/60911/multivariate-normal-difference-distribution
    #https://math.stackexchange.com/questions/275648/multiplication-of-a-random-variable-with-constant
    # Implementation with Monte Carlo Sampling:
    #https://stats.stackexchange.com/questions/345915/trying-to-implement-the-jensen-shannon-divergence-for-multivariate-gaussians
    gaussian_m = GaussianDistribution()
    gaussian_m.mu = 1/2 * gaussian_0.mu + 1/2 * gaussian_1.mu
    # I'm not sure if you have to take the quadratic term here
    # According to the matrix cookbook 8.1.4 - Equation 355 we have
    # A = B = 1/2
    # So this holds true only if distributions are independent!
    gaussian_m.std = (1/2) ** 2 * gaussian_0.std + (1/2) ** 2 * gaussian_1.std
    # gaussian_m.std = (1/2) * gaussian_0.std + (1/2) * gaussian_1.std

    KL_0 = KullbackLeiberDivergenceMultiVarianteGaussians(gaussian_0, gaussian_m)
    KL_1 = KullbackLeiberDivergenceMultiVarianteGaussians(gaussian_1, gaussian_m)

    divergence = 1/2 * KL_0 + 1/2 * KL_1
    return divergence


if __name__ == "__main__":
    data_1d = np.array([-1.0 , -0.4, -0.2, 0, 0.1, 0.3, 0.5])
    data_2d = np.array([[-1.0, 0.5], [-0.5, 0.3], [-0.1, 0.2], [0.2, -0.4]])
    data_5d = np.array([[0.15, 0.32, 0.42, 0.89, 0.91],
                        [3.2, 9.4, 3.1, 5.12, 9.5],
                        [10.3, 8.2, 4.3, 0.6, 1.6],
                        [10.6, 8.9, 4.0, 1.6, 1.12],
                        [10.7, 3.2, 3.8, 1.2, 4.13],
                        [10.8, 0.2, 4.6, 0.9, 1.3]])

    gaussian_distribution_1d = GaussianDistribution(data_1d)
    gaussian_distribution_1d.plot_1d(dim=0)

    # print(KullbackLeiberDivergenceMultiVarianteGaussians(gaussian_distribution_1d, gaussian_distribution_1d))

    gaussian_distribution_2d = GaussianDistribution(data_2d)
    gaussian_distribution_2d.plot_2d(dim=(0, 1))

    kl_2d_self = KullbackLeiberDivergenceMultiVarianteGaussians(gaussian_distribution_2d, gaussian_distribution_2d)
    print("Kullbackleiber Divergence 2D for self: {}".format(kl_2d_self))
    js_2d_self = JensenShannonDivergenceMultiVarianteGaussians(gaussian_distribution_2d, gaussian_distribution_2d)
    print("Jensen Shannon Divergence 2D for self: {}".format(js_2d_self))

    gaussian_distribution_5d = GaussianDistribution(data_5d)
    gaussian_distribution_5d.plot_2d(dim=(2,3))

    pdf_1d = gaussian_distribution_1d.pdf
    pdf_2d = gaussian_distribution_2d.pdf
    print("PDF 2D Mu: {}".format(pdf_2d(gaussian_distribution_2d.mu)))

    print(gaussian_distribution_5d.is_in_distribution(gaussian_distribution_5d.mu))
    print(gaussian_distribution_5d.is_in_distribution(gaussian_distribution_5d.mu + np.array([5, 5, 5, 5, 5])))
    
    input("Finish?")