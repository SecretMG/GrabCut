import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

'--- 注意：使用时要精心调节协方差矩阵，通常可将其设置的大一些，否则由于pdf过小，导致求解参数的过程中分母为0，出现nan和inf'

class GMM:
    def __init__(self, mu, sigma, pts, k):
        '''这里传递的是真实值，而非迭代的初始值。（GMM的类属性都是真实值而非计算值）
        mu: [k, d]
        sigma: [k, d, d]
        pts: num of points in each sample set
        k: num of component models
        '''
        self.mu, self.sigma, self.pts = mu, sigma, pts
        self.k = k
        self.x0, self.x1, self.x2, self.x = self.get_x(mu, sigma, pts)


    def get_x(self, mu, sigma, pts):
        x0 = np.random.multivariate_normal(mu[0], sigma[0], pts[0])
        x1 = np.random.multivariate_normal(mu[1], sigma[1], pts[1])
        x2 = np.random.multivariate_normal(mu[2], sigma[2], pts[2])
        # 生成三簇数据，每簇100个样本点
        x = np.vstack((x0, x1, x2))     # [300+600+900, 2]
        return x0, x1, x2, x

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
        return - np.mean(np.log(pdfs.sum(axis=1)))

    def learn(self, k, num_epoch):
        '''
        k: num of component models
        '''
        mu = np.random.randint(1, 5, (k, self.x.shape[-1]))
        sigma = np.random.randint(1, 5, (k, self.x.shape[-1], self.x.shape[-1]))
        w = np.ones((sum(self.pts), k)) / k     # w[i][j]代表第i个样本点属于第j个分量模型的概率，初始为等概率
        pi = np.sum(w, axis=0) / np.sum(w)      # p[j]代表选择第j个分量模型的概率，初始为等概率
        llh_ls = []
        for epoch in range(num_epoch):
            llh_ls.append(self.get_llh(k, pi, mu, sigma))
            print(f'[{epoch}]: -loglikelyhood: {llh_ls[-1]}')
            if len(llh_ls) > 1 and llh_ls[-1] == llh_ls[-2]:
                self.show(k, mu, sigma)
                break
            elif epoch % 10 == 0:
                self.show(k, mu, sigma)
            w = self.update_w(k, mu, sigma, pi)
            pi = self.update_pi(k, w)
            mu = self.update_mu(k, w)
            sigma = self.update_sigma(k, w, mu)


    def show(self, k, mu, sigma):
        plt.figure(figsize=(10, 8))
        plt.axis((-5, 15, -5, 15))
        def true_plot():
            '--- 展示真实数据类'
            plt.scatter(self.x0[:, 0], self.x0[:, 1], s=5)
            plt.scatter(self.x1[:, 0], self.x1[:, 1], s=5)
            plt.scatter(self.x2[:, 0], self.x2[:, 1], s=5)
        def calc_plot():
            '--- 展示计算分类'
            colors = ['r', 'g', 'b']
            plt.scatter(self.x[:, 0], self.x[:, 1], s=5)
            ax = plt.gca()
            for cur in range(self.k):
                # 绘制真实高斯分布流形
                args = {
                    'facecolor': 'None',        # 默认有颜色填充，将其更改为无填充
                    'edgecolor': colors[cur],   # 默认为无色，将其更改为有颜色
                    'linewidth': 2,             # 线条粗细
                    'linestyle': ':'            # 默认为实线，设置为虚线
                }
                conv = self.sigma[cur]
                _, v = np.linalg.eig(conv)  # 计算协方差矩阵的特征向量
                angle = np.rad2deg(np.arccos(v[0, 0]))  # 计算应该偏转的角度
                e = Ellipse(self.mu[cur], 3 * self.sigma[cur][0][0], 3 * self.sigma[cur][1][1], angle, **args)
                # 椭圆中心坐标，宽度，高度，绘画参数
                ax.add_patch(e)
            for cur in range(k):
                # 绘制计算高斯分布流形
                args = {
                    'facecolor': 'None',        # 默认有颜色填充，将其更改为无填充
                    'edgecolor': 'r',           # 默认为无色，将其更改为有颜色
                    'linewidth': 2,             # 线条粗细
                }
                conv = sigma[cur]
                _, v = np.linalg.eig(conv)  # 计算协方差矩阵的特征向量
                angle = np.rad2deg(np.arccos(v[0, 0]))  # 计算应该偏转的角度
                e = Ellipse(mu[cur], 3 * sigma[cur][0][0], 3 * sigma[cur][1][1], angle, **args)
                ax.add_patch(e)     # 绘制高斯分布（椭圆）
        calc_plot()
        plt.show()

def main():
    mu_hat = np.array(
        (
            [2.5, 8],
            [8, 2.5],
            [10, 10]
        )
    )   # [3, 2]
    sigma_hat = np.array(
        (
            [[2, 2], [2, 4]],
            [[4, 2], [2, 2]],
            [[2, 0], [0, 2]]
        )
    )   # [3, 2, 2]
    pts_hat = [300, 600, 900]
    k = 3
    gmm = GMM(mu_hat, sigma_hat, pts_hat, k)
    gmm.learn(k=3, num_epoch=100)


if __name__ == '__main__':
    main()