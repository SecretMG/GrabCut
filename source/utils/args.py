import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', '-s',
    default='imgs/human.jpg',
    help='使用的图片'
)
parser.add_argument(
    '--num_k', '-nk',
    default=5,
    help='GMM的分量模型个数'
)
parser.add_argument(
    '--num_iter', '-ni',
    default=1000,
    help='GMM最大迭代次数'
)
parser.add_argument(
    '--gamma', '-g',
    default=5,
    help='边权Region和Boarder的平衡因子'
)
parser.add_argument(
    '--n_ways', '-nw',
    default=8,
    choices=[8, 4],
    help='计算边界项时使用的邻居数'
)
parser.add_argument(
    '-num_epoch', '-ne',
    default=5,
    help='进行全部流程的次数'
)
parser.add_argument(
    '--maxValidFloat', '-mVF',
    default=1000000,
)
args = parser.parse_args()