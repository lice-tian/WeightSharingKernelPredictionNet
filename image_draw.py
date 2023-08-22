import matplotlib.pyplot as plt
import numpy as np

def scale_eval():
    # 准备数据
    all_eval = np.load('data/evaluation/scale-eval.npz')
    scales = all_eval['arr_0']
    psnr_classroom = all_eval['arr_1']
    time_classroom = all_eval['arr_2']

    psnr_livingroom = all_eval['arr_3']
    time_livingroom = all_eval['arr_4']

    psnr_sanmiguel = all_eval['arr_5']
    time_sanmiguel = all_eval['arr_6']

    # 正确显示中文和负号
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 有三种类型的数据，n设置为3
    total_width, n = 0.8, 3
    # 每种类型的柱状图宽度
    width = total_width / n

    # 重新设置x轴的坐标
    scales = scales - (total_width - width) / 2

    # 画图，plt.bar()可以画柱状图
    plt.bar(scales, psnr_classroom, width=width, label="classroom")
    plt.bar(scales + width, psnr_livingroom, width=width, label="living-room")
    plt.bar(scales + 2 * width, psnr_sanmiguel, width=width, label="san-miguel")
    # 设置图片名称
    plt.title("scale对psnr的影响")
    # 设置x轴标签名
    plt.xlabel("scales")
    # 设置y轴标签名
    plt.ylabel("psnr")
    # 显示图例
    plt.legend()

    plt.savefig(fname="./data/evaluation/pictures/scale-psnr.png")
    # 显示
    plt.show()

    # 画图，plt.bar()可以画柱状图
    plt.bar(scales, time_classroom, width=width, label="classroom")
    plt.bar(scales + width, time_livingroom, width=width, label="living-room")
    plt.bar(scales + 2 * width, time_sanmiguel, width=width, label="san-miguel")
    # 设置图片名称
    plt.title("scale对time的影响")
    # 设置x轴标签名
    plt.xlabel("scales")
    # 设置y轴标签名
    plt.ylabel("time/(ms)")
    # 显示图例
    plt.legend()

    plt.savefig(fname="./data/evaluation/pictures/scale-time.png")

    # 显示
    plt.show()


def kernel_size_eval():
    # 准备数据
    all_eval = np.load('data/evaluation/kernel-size-eval.npz')
    k = all_eval['arr_0']
    psnr_classroom = all_eval['arr_1']
    time_classroom = all_eval['arr_2']

    psnr_livingroom = all_eval['arr_3']
    time_livingroom = all_eval['arr_4']

    psnr_sanmiguel = all_eval['arr_5']
    time_sanmiguel = all_eval['arr_6']

    # 正确显示中文和负号
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 有三种类型的数据，n设置为3
    total_width, n = 0.8, 3
    # 每种类型的柱状图宽度
    width = total_width / n

    # 重新设置x轴的坐标
    k = k - (total_width - width) / 2

    # 画图，plt.bar()可以画柱状图
    plt.bar(k, psnr_classroom, width=width, label="classroom")
    plt.bar(k + width, psnr_livingroom, width=width, label="living-room")
    plt.bar(k + 2 * width, psnr_sanmiguel, width=width, label="san-miguel")
    # 设置图片名称
    plt.title("kernel size对psnr的影响")
    # 设置x轴标签名
    plt.xlabel("k")
    # 设置y轴标签名
    plt.ylabel("psnr")
    # 显示图例
    plt.legend()

    plt.savefig(fname="./data/evaluation/pictures/kernel-size-psnr.png")
    # 显示
    plt.show()

    # 画图，plt.bar()可以画柱状图
    plt.bar(k, time_classroom, width=width, label="classroom")
    plt.bar(k + width, time_livingroom, width=width, label="living-room")
    plt.bar(k + 2 * width, time_sanmiguel, width=width, label="san-miguel")
    # 设置图片名称
    plt.title("kernel size对time的影响")
    # 设置x轴标签名
    plt.xlabel("k")
    # 设置y轴标签名
    plt.ylabel("time/(ms)")
    # 显示图例
    plt.legend()

    plt.savefig(fname="./data/evaluation/pictures/kernel-size-time.png")

    # 显示
    plt.show()


def se_block_eval():
    # 准备数据
    all_eval = np.load('data/evaluation/se-block-eval.npz')
    k = all_eval['arr_0']
    psnr_classroom = all_eval['arr_1']
    ssim_classroom = all_eval['arr_2']

    psnr_livingroom = all_eval['arr_3']
    ssim_livingroom = all_eval['arr_4']

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.plot(k, psnr_classroom, label="classroom")
    plt.plot(k, psnr_livingroom, label="living-room")
    plt.title("se-block对psnr的影响")
    plt.xlabel("k")
    # 设置y轴标签名
    plt.ylabel("psnr")
    plt.legend()
    plt.savefig(fname="./data/evaluation/pictures/se-block-psnr.png")
    plt.show()

    plt.plot(k, ssim_classroom, label="classroom")
    plt.plot(k, ssim_livingroom, label="living-room")
    plt.title("se-block对ssim的影响")
    plt.xlabel("k")
    # 设置y轴标签名
    plt.ylabel("psnr")
    plt.legend()
    plt.savefig(fname="./data/evaluation/pictures/se-block-ssim.png")
    plt.show()
