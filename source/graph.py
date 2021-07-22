import numpy as np
from tqdm import tqdm
from time import time

from GMM import GMM
from Dinic import Dinic
from utils.args import args

class GCGraph:
    def __init__(self, input, left_top, right_bottom, fore, back):
        self.input = input.astype(int)  # 破除像素值限制
        self.ord2id = {}
        self.id2ord = []
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                self.ord2id[(i, j)] = i * input.shape[1] + j
                self.id2ord.append((i, j))
        self.foreground = []    # 存的是idx，记录求出的点集
        self.background = []

        # 将矩形框初始化为前景
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                if i in range(left_top[0], right_bottom[0]) and j in range(left_top[1], right_bottom[1]):
                    self.foreground.append(self.ord2id[(i, j)])
                else:
                    self.background.append(self.ord2id[(i, j)])

        # 设置用户规定的部分
        self.fore = fore
        self.back = back
        self.set_foreground(fore)
        self.set_background(back)

    def learn_GMM(self):
        '--- 将目前这些数据喂给两个GMM，进行迭代收敛，计算出其pdf作为点代价'
        fore_x = np.zeros((len(self.foreground), 3))
        back_x = np.zeros((len(self.background), 3))
        for i, idx in enumerate(self.foreground):
            fore_x[i] = self.input[self.id2ord[idx]]
        for i, idx in enumerate(self.background):
            back_x[i] = self.input[self.id2ord[idx]]
        self.fore_model = GMM(fore_x)
        self.back_model = GMM(back_x)
        print('learn foreground GMM')
        self.fore_model.learn(args.num_k, args.num_iter)
        print('learn background GMM')
        self.back_model.learn(args.num_k, args.num_iter)

    def test_GMM(self, left_top, right_bottom):
        x = np.zeros((self.input.shape[0] * self.input.shape[1], 3))
        # 这里其实可以只计算框内的p，但是计算量也不大就全计算了
        for i in range(self.input.shape[0]):
            for j in range(self.input.shape[1]):
                x[i*self.input.shape[1] + j] = self.input[i][j]
        ans = self.input
        p_fore = self.fore_model.predict(x)
        p_back = self.back_model.predict(x)
        for i in range(ans.shape[0]):
            for j in range(ans.shape[1]):
                if i in range(left_top[0], right_bottom[0]) and j in range(left_top[1], right_bottom[1]):
                    if p_back[i * ans.shape[1] + j] > p_fore[i * ans.shape[1] + j]:
                        ans[i][j] = (255, 255, 255)
                else:
                    ans[i][j] = (255, 255, 255)
        return ans

    def build(self, left_top, right_bottom):
        points, edges = [], []
        print('建图')
        '--- 建立点集'
        points.append(self.input.shape[0] * self.input.shape[1])      # 前景点
        points.append(self.input.shape[0] * self.input.shape[1] + 1)  # 后景点
        for i in range(left_top[0], right_bottom[0]):
            for j in range(left_top[1], right_bottom[1]):
                points.append(self.ord2id[(i, j)])
        '--- 建立边集'
        if args.n_ways == 4:
            neighbor = [
                (1, 0), (0, 1)
            ]
        elif args.n_ways == 8:
            neighbor = [
                (1, 0), (0, 1), (1, 1), (1, -1)
            ]   # 因为是无向边，每个点只扩展一半的邻居即可
        # 先计算diff的期望，用于在公式中使用
        mean_of_diff, cnt = 0, 0
        for i in range(left_top[0], right_bottom[0]):
            for j in range(left_top[1], right_bottom[1]):
                me = (i, j)
                for nb in neighbor:
                    nxt = (me[0] + nb[0], me[1] + nb[1])
                    if nxt[0] in range(left_top[0], right_bottom[0]) and nxt[1] in range(left_top[1], right_bottom[1]):
                        mean_of_diff += np.sum(
                            (self.input[me] - self.input[nxt]) * (self.input[me] - self.input[nxt])
                        )
                        cnt += 1
        mean_of_diff /= cnt     # 求期望
        mean_of_diff *= 2
        mean_of_diff = 1 / mean_of_diff

        # 逐点计算各边权重
        for i in tqdm(range(left_top[0], right_bottom[0])):
            for j in range(left_top[1], right_bottom[1]):
                me = (i, j)
                # 首先计算边界项
                for nb in neighbor:
                    nxt = (me[0] + nb[0], me[1] + nb[1])
                    dist = np.sqrt(nb[0] * nb[0] + nb[1] * nb[1])
                    if nxt[0] in range(left_top[0], right_bottom[0]) and nxt[1] in range(left_top[1], right_bottom[1]):
                        # 由于图片数据的非负性，需要手动操作
                        w = np.sum(
                            (self.input[me] - self.input[nxt]) * (self.input[me] - self.input[nxt])
                        )
                        w *= - mean_of_diff
                        edges.append(
                            (
                                self.ord2id[me],
                                self.ord2id[nxt],
                                int(args.gamma * (1 / dist) * np.exp(w)),
                            )
                        )
                # 其次计算区域项
                if me in self.fore:     # 设置用户规定的部分
                    p_fore = args.maxValidFloat
                    p_back = 0
                elif me in self.back:
                    p_back = args.maxValidFloat
                    p_fore = 0
                else:
                    pdf_fore = self.fore_model.predict(self.input[me].reshape(-1, self.input.shape[-1]))[0]
                    pdf_back = self.back_model.predict(self.input[me].reshape(-1, self.input.shape[-1]))[0]
                    p_fore = - np.log(pdf_back)
                    p_back = - np.log(pdf_fore)

                edges.append(
                    (
                        self.input.shape[0] * self.input.shape[1],
                        self.ord2id[me],
                        int(p_fore)
                    )
                )
                edges.append(
                    (
                        self.input.shape[0] * self.input.shape[1] + 1,
                        self.ord2id[me],
                        int(p_back)
                    )
                )
        cutter = Dinic(points, edges, self.input.shape[0] * self.input.shape[1] + 2)
        print('网络流计算中...')
        pre = time()
        cutter.solve(self.input.shape[0] * self.input.shape[1], self.input.shape[0] * self.input.shape[1] + 1)
        print(f'网络流计算完成，spending time={int(time() - pre)}s')
        self.foreground = cutter.get_foreground(self.input.shape[0] * self.input.shape[1])    # 用于下一轮迭代
        self.background = cutter.get_background(self.input.shape[0] * self.input.shape[1] + 1)

    def predict(self):
        print('像素分类中...')
        pre = time()
        ans = np.zeros_like(self.input)
        for i in range(self.input.shape[0]):
            for j in range(self.input.shape[1]):
                if self.ord2id[(i, j)] in self.foreground:
                    ans[i][j] = self.input[i][j]
                else:
                    ans[i][j] = (255, 255, 255)
        print(f'像素分类完成，spending time={int(time() - pre)}s')
        return ans

    def set_foreground(self, fore):
        for ord in fore:
            idx = self.ord2id[ord]
            if idx not in self.foreground:
                self.foreground.append(idx)
                self.background.remove(idx)

    def set_background(self, back):
        for ord in back:
            idx = self.ord2id[ord]
            if idx not in self.background:
                self.background.append(idx)
                self.foreground.remove(idx)

