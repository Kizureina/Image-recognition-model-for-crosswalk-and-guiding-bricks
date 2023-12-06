import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l
import matplotlib.pyplot as plt


def use_svg_display():  # @save
    """使用svg格式在Jupyter中显示绘图"""
    backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):  # @save
    """设置matplotlib的图表大小"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize


# @save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
        axes.grid()


# @save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []
    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
    else:
        axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def f(x):
    return 3 * x ** 2 - 4 * x

def draw_comprae():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    y1 = [0.227, 0.227, 0.803, 0.879, 0.864, 0.894, 0.879, 0.909, 0.894, 0.909, 0.894, 0.879, 0.939, 0.939, 0.939, 0.939, 0.924, 0.909, 0.909, 0.939]

    y2 = [0.682, 0.303, 0.833, 0.894, 0.939, 0.864, 0.864, 0.939, 0.924, 0.955, 0.909, 0.955, 0.879, 0.939, 0.939,
          0.939, 0.939, 0.955, 0.939, 0.955]

    y3 = [0.879, 0.955, 0.955, 0.955, 0.955, 0.955, 0.97, 0.97, 0.985, 0.97, 0.97, 0.97, 0.985, 0.985, 0.985, 0.985, 0.985, 0.985, 0.97, 0.985]

    # 绘制折线图
    plt.plot(x, y1, label='Transfer Learning', color="blue")
    plt.plot(x, y2, label='Transfer Learning with freeze features weights', color="green")
    plt.plot(x, y3, label='Transfer Learning with deleting classifier weights', color="red")

    # 添加标题和图例
    plt.title('MobileNetV2 Accuracy Results')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(1, 21, 1))  # 设置 x 轴的刻度为每1个单位显示一个标签
    plt.ylabel('Accuracy')
    plt.legend()


def draw_accuracy():
    # 给出三个函数的坐标
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    y1 = [0.682, 0.682, 0.742, 0.742, 0.758, 0.758, 0.742, 0.742, 0.758, 0.742, 0.773, 0.773, 0.773, 0.833, 0.833,
          0.894, 0.894, 0.879, 0.909, 0.894]
    # 使用transfer learning
    # y2 = [0.970, 0.955, 0.985, 0.939, 0.970, 0.970, 0.985, 1.0, 0.985, 0.985, 0.985, 0.970, 0.955, 0.970, 1.000, 0.970, 1.000, 0.985, 0.970, 0.955]
    y2 = [0.848, 0.788, 0.894, 0.833, 0.894, 0.788, 0.924, 0.894, 0.924, 0.985, 0.879, 0.939, 0.924, 0.955, 0.924, 0.939,
          0.955, 0.924, 0.939, 0.894]
    # y3 = [0.682, 0.303, 0.833, 0.894, 0.939, 0.864, 0.864, 0.939, 0.924, 0.955, 0.909, 0.955, 0.879, 0.939, 0.939,
    #       0.939, 0.939, 0.955, 0.939, 0.955]
    y3 = [0.879, 0.955, 0.955, 0.955, 0.955, 0.955, 0.97, 0.97, 0.985, 0.97, 0.97, 0.97, 0.985, 0.985, 0.985, 0.985, 0.985, 0.985, 0.97, 0.985]



    # 绘制折线图
    plt.plot(x, y1, label='CNN')
    plt.plot(x, y2, label='ResNet', color="green")
    plt.plot(x, y3, label='MobileNetV2', color="red")

    # 添加标题和图例
    plt.title('Accuracy Results')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(1, 21, 1))  # 设置 x 轴的刻度为每1个单位显示一个标签
    plt.ylabel('Accuracy')
    plt.legend()


def draw_loss():
    # 给出三个函数的坐标
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    y1 = [2.257, 1.364, 0.760, 0.578, 0.527, 0.513, 0.495, 0.481, 0.458, 0.438, 0.411, 0.376, 0.348, 0.333, 0.317,
          0.281, 0.256, 0.256, 0.232, 0.227]
    y2 = [0.75, 0.257, 0.164, 0.248, 0.174, 0.126, 0.105, 0.096, 0.136, 0.121, 0.093, 0.119, 0.118, 0.116, 0.106, 0.122,
          0.081, 0.086, 0.076, 0.128]
    y3 = [1.391, 0.979, 0.829, 0.752, 0.704, 0.654, 0.617, 0.591, 0.565, 0.532, 0.506, 0.490, 0.470, 0.473, 0.455,
          0.443, 0.445, 0.411, 0.395, 0.401]

    # 绘制折线图
    plt.plot(x, y1, label='CNN')
    plt.plot(x, y2, label='ResNet', color="green")
    plt.plot(x, y3, label='MobileNetV2', color="red")

    # 添加标题和图例
    plt.title('Loss Results')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(1, 21, 1))  # 设置 x 轴的刻度为每1个单位显示一个标签
    plt.ylabel('Loss')
    plt.legend()


if __name__ == '__main__':
    # x = np.arange(0, 3, 0.1)
    # plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
    # draw_loss()
    draw_accuracy()
    # draw_comprae()
    # 显示图形
    plt.show()
'''
cpu
train loss: 100%[**************************************************->]0.7795
[epoch 1] train_loss: 1.108  test_accuracy: 0.682
train loss: 100%[**************************************************->]0.5229
[epoch 2] train_loss: 0.685  test_accuracy: 0.303
train loss: 100%[**************************************************->]0.6812
[epoch 3] train_loss: 0.528  test_accuracy: 0.833
train loss: 100%[**************************************************->]0.9712
[epoch 4] train_loss: 0.456  test_accuracy: 0.894
train loss: 100%[**************************************************->]0.7288
[epoch 5] train_loss: 0.395  test_accuracy: 0.939
train loss: 100%[**************************************************->]0.2504
[epoch 6] train_loss: 0.397  test_accuracy: 0.864
train loss: 100%[**************************************************->]0.1355
[epoch 7] train_loss: 0.362  test_accuracy: 0.864
train loss: 100%[**************************************************->]0.6067
[epoch 8] train_loss: 0.375  test_accuracy: 0.939
train loss: 100%[**************************************************->]0.3551
[epoch 9] train_loss: 0.401  test_accuracy: 0.924
train loss: 100%[**************************************************->]0.2509
[epoch 10] train_loss: 0.315  test_accuracy: 0.955
train loss: 100%[**************************************************->]0.4552
[epoch 11] train_loss: 0.317  test_accuracy: 0.909
train loss: 100%[**************************************************->]0.3902
[epoch 12] train_loss: 0.312  test_accuracy: 0.955
train loss: 100%[**************************************************->]0.2409
[epoch 13] train_loss: 0.311  test_accuracy: 0.879
train loss: 100%[**************************************************->]0.2918
[epoch 14] train_loss: 0.340  test_accuracy: 0.939
train loss: 100%[**************************************************->]0.1300
[epoch 15] train_loss: 0.268  test_accuracy: 0.939
train loss: 100%[**************************************************->]0.4168
[epoch 16] train_loss: 0.280  test_accuracy: 0.939
train loss: 100%[**************************************************->]0.2651
[epoch 17] train_loss: 0.321  test_accuracy: 0.939
train loss: 100%[**************************************************->]0.0976
[epoch 18] train_loss: 0.281  test_accuracy: 0.955
train loss: 100%[**************************************************->]0.2992
[epoch 19] train_loss: 0.269  test_accuracy: 0.939
train loss: 100%[**************************************************->]0.4147
[epoch 20] train_loss: 0.261  test_accuracy: 0.955
Finished Training

resnet:
cpu
train loss: 100%[**************************************************->]0.4622
[epoch 1] train_loss: 2.235  test_accuracy: 0.848
train loss: 100%[**************************************************->]1.0632
[epoch 2] train_loss: 0.527  test_accuracy: 0.788
train loss: 100%[**************************************************->]0.1784
[epoch 3] train_loss: 0.375  test_accuracy: 0.894
train loss: 100%[**************************************************->]0.6974
[epoch 4] train_loss: 0.364  test_accuracy: 0.833
train loss: 100%[**************************************************->]0.3114
[epoch 5] train_loss: 0.350  test_accuracy: 0.894
train loss: 100%[**************************************************->]0.3941
[epoch 6] train_loss: 0.315  test_accuracy: 0.788
train loss: 100%[**************************************************->]0.3023
[epoch 7] train_loss: 0.268  test_accuracy: 0.924
train loss: 100%[**************************************************->]0.2273
[epoch 8] train_loss: 0.259  test_accuracy: 0.894
train loss: 100%[**************************************************->]0.2892
[epoch 9] train_loss: 0.337  test_accuracy: 0.924
train loss: 100%[**************************************************->]0.0699
[epoch 10] train_loss: 0.258  test_accuracy: 0.879
train loss: 100%[**************************************************->]0.2550
[epoch 11] train_loss: 0.212  test_accuracy: 0.939
train loss: 100%[**************************************************->]0.1141
[epoch 12] train_loss: 0.268  test_accuracy: 0.924
train loss: 100%[**************************************************->]0.3190
[epoch 13] train_loss: 0.265  test_accuracy: 0.955
train loss: 100%[**************************************************->]0.1073
[epoch 14] train_loss: 0.224  test_accuracy: 0.924
train loss: 100%[**************************************************->]0.2021
[epoch 15] train_loss: 0.193  test_accuracy: 0.939
train loss: 100%[**************************************************->]0.1950
[epoch 16] train_loss: 0.226  test_accuracy: 0.955
train loss: 100%[**************************************************->]0.2538
[epoch 17] train_loss: 0.195  test_accuracy: 0.924
train loss: 100%[**************************************************->]0.0747
[epoch 18] train_loss: 0.224  test_accuracy: 0.939
train loss: 100%[**************************************************->]0.1821
[epoch 19] train_loss: 0.238  test_accuracy: 0.894
train loss: 100%[**************************************************->]0.3500
[epoch 20] train_loss: 0.177  test_accuracy: 0.939
Finished Training

使用预训练模型的分类器的mobilenet：
cpu
train loss: 100%[**************************************************->]0.7251
[epoch 1] train_loss: 1.124  test_accuracy: 0.227
train loss: 100%[**************************************************->]0.5565
[epoch 2] train_loss: 0.684  test_accuracy: 0.227
train loss: 100%[**************************************************->]0.3704
[epoch 3] train_loss: 0.513  test_accuracy: 0.803
train loss: 100%[**************************************************->]0.4103
[epoch 4] train_loss: 0.464  test_accuracy: 0.879
train loss: 100%[**************************************************->]0.8382
[epoch 5] train_loss: 0.426  test_accuracy: 0.864
train loss: 100%[**************************************************->]0.6873
[epoch 6] train_loss: 0.384  test_accuracy: 0.894
train loss: 100%[**************************************************->]0.2958
[epoch 7] train_loss: 0.345  test_accuracy: 0.879
train loss: 100%[**************************************************->]0.3603
[epoch 8] train_loss: 0.359  test_accuracy: 0.909
train loss: 100%[**************************************************->]0.2080
[epoch 9] train_loss: 0.341  test_accuracy: 0.894
train loss: 100%[**************************************************->]0.2716
[epoch 10] train_loss: 0.339  test_accuracy: 0.909
train loss: 100%[**************************************************->]0.2553
[epoch 11] train_loss: 0.332  test_accuracy: 0.894
train loss: 100%[**************************************************->]0.1304
[epoch 12] train_loss: 0.355  test_accuracy: 0.879
train loss: 100%[**************************************************->]0.2897
[epoch 13] train_loss: 0.287  test_accuracy: 0.939
train loss: 100%[**************************************************->]0.3360
[epoch 14] train_loss: 0.311  test_accuracy: 0.939
train loss: 100%[**************************************************->]0.3291
[epoch 15] train_loss: 0.284  test_accuracy: 0.939
train loss: 100%[**************************************************->]0.3417
[epoch 16] train_loss: 0.272  test_accuracy: 0.939
train loss: 100%[**************************************************->]0.1098
[epoch 17] train_loss: 0.322  test_accuracy: 0.924
train loss: 100%[**************************************************->]0.1790
[epoch 18] train_loss: 0.268  test_accuracy: 0.909
train loss: 100%[**************************************************->]0.7495
[epoch 19] train_loss: 0.287  test_accuracy: 0.909
train loss: 100%[**************************************************->]0.4220
[epoch 20] train_loss: 0.304  test_accuracy: 0.939
Finished Training

'''