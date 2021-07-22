import cv2 as cv
import numpy as np
from utils.args import args
from graph import GCGraph


class drawer:
    def __init__(self, source):
        self.input = cv.imread(source)
        self.script = self.input.copy()     # 仅用于展示画布
        self.output = np.zeros_like(self.input)
        self.fore = []
        self.back = []
        self.finished = False

        self.mode = 0
        '--- mode=0：等待用户绘制矩形框'
        self.pressed = False
        self.left_top = (0, 0)
        self.right_bottom = (0, 0)
        '--- mode=1：等待用户绘制前景'
        '--- mode=2：等待用户绘制后景'
        '--- mode=-1：结束'

    def draw(self):
        cv.imshow('input', self.script)
        cv.imshow('output', self.output)
        if self.mode == 0:
            cv.setMouseCallback('input', self.get_boader)
        elif self.mode == 1:
            cv.setMouseCallback('input', self.get_foreground)
        elif self.mode == 2:
            cv.setMouseCallback('input', self.get_background)
        elif self.mode == -1:
            self.finished = True

        k = cv.waitKey()
        if k == ord('\r') or k == ord('\n'):
            self.solve()    # 用户敲击回车，表示确认进行操作
        elif k == ord('\b'):
            self.mode = -1
        elif k == ord('f'):
            self.mode = 1
        elif k == ord('b'):
            self.mode = 2

    def get_boader(self, event, y, x, _, __):
        img = self.script.copy()
        if event == 1:
            # 按下鼠标左键
            self.pressed = True
            self.left_top = (x, y)
        elif event == 4:
            # 释放鼠标左键
            self.pressed = False
            self.right_bottom = (x, y)
        elif self.pressed and event == 0:
            cv.rectangle(img, (self.left_top[1], self.left_top[0]), (y, x), (0, 0, 255), 2)
            # 注意是BGR颜色空间，注意其y、x的顺序
            cv.imshow('input', img)

    def get_foreground(self, event, y, x, _, __):
        if event == 1:
            # 按下鼠标左键
            self.pressed = True
        elif event == 4:
            # 释放鼠标左键
            self.pressed = False
        elif self.pressed and event == 0:
            neighbor = [
                (1, 0), (0, 1), (1, 1), (1, -1),
                (-1, 0), (0, -1), (-1, -1), (-1, 1)
            ]
            self.script[x, y] = (255, 0, 0)
            self.fore.append((x, y))
            for nb in neighbor:
                self.script[x + nb[0], y + nb[1]] = (255, 0, 0)
                self.fore.append((x + nb[0], y + nb[1]))
            # 注意是BGR颜色空间，注意其y、x的顺序
            cv.imshow('input', self.script)

    def get_background(self, event, y, x, _, __):
        if event == 1:
            # 按下鼠标左键
            self.pressed = True
        elif event == 4:
            # 释放鼠标左键
            self.pressed = False
        elif self.pressed and event == 0:
            neighbor = [
                (1, 0), (0, 1), (1, 1), (1, -1),
                (-1, 0), (0, -1), (-1, -1), (-1, 1)
            ]
            self.script[x, y] = (0, 255, 0)
            self.back.append((x, y))
            for nb in neighbor:
                self.script[x + nb[0], y + nb[1]] = (0, 255, 0)
                self.back.append((x + nb[0], y + nb[1]))
            # 注意是BGR颜色空间，注意其y、x的顺序
            cv.imshow('input', self.script)

    def solve(self):
        '--- 确认输入的矩形框以及前景后景，并将其保持展示'
        cv.rectangle(self.script, (self.left_top[1], self.left_top[0]), (self.right_bottom[1], self.right_bottom[0]), (0, 0, 255), 2)  # 注意是GBR颜色空间
        graph = GCGraph(self.input, self.left_top, self.right_bottom, self.fore, self.back)

        '--- 进行多轮计算，每次展示结果'
        for epoch in range(args.num_epoch):
            print(f'开始第{epoch + 1}/{args.num_epoch}轮GrabCut')
            graph.learn_GMM()
            # self.output = graph.test_GMM(self.left_top, self.right_bottom).astype(np.uint8)
            graph.build(self.left_top, self.right_bottom)
            self.output = graph.predict().astype(np.uint8)  # 转换回图片形式
            cv.imshow(f'output{epoch+1}', self.output)
        print('等待下一步操作')


def main():
    gc_client = drawer(args.source)
    while True:
        gc_client.draw()
        if gc_client.finished:
            break


if __name__ == '__main__':
    main()
