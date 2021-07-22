import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from tqdm import tqdm

class GMM:
    def __init__(self, x):
        self.x = x
        self.pts = x.shape[0]
        self.k, self.w, self.pi, self.mu, self.sigma = None, None, None, None, None

    # TODO:应当更新w, pi, mu, sigma的写法，使之为更迅速的矩阵乘法
    def update_w(self, k, mu, sigma, pi):
        w = np.zeros((self.x.shape[0], k))
        for j in range(k):
            w[:, j] = pi[j] * multivariate_normal.pdf(
                self.x,
                mu[j],
                np.diag([sigma[j, _, _] for _ in range(sigma.shape[-1])])
            )
            # 注意这里的sigma在写代码时应该用var代替，否则难以满足其正定条件
        # print(np.sum(w, axis=1).reshape(-1, 1))
        w /= np.sum(w, axis=1).reshape(-1, 1)
        return w

    def update_pi(self, k, w):
        pi = np.zeros(k)
        pi += np.sum(w, axis=0)
        pi /= np.sum(w)
        return pi

    def update_mu(self, k, w):
        mu = np.zeros((k, self.x.shape[-1]))
        for j in range(k):
            mu[j] += np.dot(w[:, j].T, self.x)   # [1, n] * [n, 2]
        mu /= np.sum(w, axis=0).reshape(-1, 1)
        return mu

    def update_sigma(self, k, w, mu):
        sigma = np.zeros((k, self.x.shape[-1], self.x.shape[-1]))
        for j in range(k):
            sigma[j] += np.dot((w[:, j].reshape(-1, 1) * (self.x - mu[j])).T, (self.x - mu[j]))
            sigma[j] /= np.sum(w[:, j])
        return sigma

    def get_llh(self, k, pi, mu, sigma):
        pdfs = np.zeros((self.x.shape[0], k))
        for j in range(k):
            pdfs[:, j] = pi[j] * multivariate_normal.pdf(
                self.x,
                mu[j],
                np.diag([sigma[j, _, _] for _ in range(sigma.shape[-1])])
            )
        return - np.mean(np.log(pdfs.sum(axis=-1)))

    def learn(self, k, num_iter):
        '''
        k: num of component models
        '''
        mu = np.random.randint(100, 200, (k, self.x.shape[-1]))
        sigma = np.random.randint(70, 1000, (k, self.x.shape[-1], self.x.shape[-1]))
        w = np.ones((self.pts, k)) / k     # w[i][j]代表第i个样本点属于第j个分量模型的概率，初始为等概率
        pi = np.sum(w, axis=0) / np.sum(w)      # p[j]代表选择第j个分量模型的概率，初始为等概率
        llh_ls = []
        for iter in tqdm(range(num_iter)):
            llh_ls.append(self.get_llh(k, pi, mu, sigma))
            # print(f'[{iter}]: -loglikelyhood: {llh_ls[-1]}')
            if len(llh_ls) > 1 and llh_ls[-1] == llh_ls[-2]:
                break
            w = self.update_w(k, mu, sigma, pi)
            pi = self.update_pi(k, w)
            mu = self.update_mu(k, w)
            sigma = self.update_sigma(k, w, mu)

        self.k, self.w, self.pi, self.mu, self.sigma = k, w, pi, mu, sigma
        print('模型已收敛')

    def predict(self, x):
        pdfs = np.zeros((x.shape[0], self.k))
        for j in range(self.k):
            pdfs[:, j] = self.pi[j] * multivariate_normal.pdf(
                x,
                self.mu[j],
                np.diag([self.sigma[j, _, _] for _ in range(self.sigma.shape[-1])])
            )
        # 注意这个pdfs不是单独某个分量模型的pdf，而是整个模型带有pi的pdf
        return pdfs.max(axis=1)
