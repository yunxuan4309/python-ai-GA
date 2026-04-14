

import numpy as np
import time
from genetic_algorithm import rastrigin, genetic_algorithm  # 复用目标函数

# ================== 穷举法 (网格搜索) ==================
def exhaustive_search(x_range=(-5, 5), y_range=(-5, 5), step=0.05):
    """
    网格搜索寻找最小值
    step: 网格步长
    """
    print("\n" + "=" * 60)
    print("穷举法 (网格搜索) 开始")
    print(f"搜索范围: x∈{x_range}, y∈{y_range}, 步长={step}")
    print("=" * 60)

    start_time = time.time()

    x_vals = np.arange(x_range[0], x_range[1] + step, step)
    y_vals = np.arange(y_range[0], y_range[1] + step, step)

    total_points = len(x_vals) * len(y_vals)
    best_val = float('inf')
    best_x, best_y = 0.0, 0.0
    count = 0

    for x in x_vals:
        for y in y_vals:
            val = rastrigin(x, y)
            if val < best_val:
                best_val = val
                best_x, best_y = x, y
            count += 1
            # 每10万次输出进度（避免刷屏）
            if count % 100000 == 0:
                print(f"已搜索 {count} 个点，当前最优值 = {best_val:.8f}")

    end_time = time.time()

    print("\n穷举法完成")
    print(f"总搜索点数: {total_points}")
    print(f"最优解: x = {best_x:.8f}, y = {best_y:.8f}")
    print(f"最优函数值: f(x,y) = {best_val:.8f}")
    print(f"理论最小值: 0.0")
    print(f"误差: {best_val:.8f}")
    print(f"运行耗时: {end_time - start_time:.4f} 秒")
    print("=" * 60)

    return best_x, best_y, best_val, total_points, end_time - start_time

# ================== 随机搜索 ==================
def random_search(x_range=(-5, 5), y_range=(-5, 5), max_iter=50000):
    """
    随机搜索：在范围内随机生成点，记录最小值
    max_iter: 最大随机搜索次数
    """
    print("\n" + "=" * 60)
    print("随机搜索开始")
    print(f"搜索范围: x∈{x_range}, y∈{y_range}, 最大迭代次数={max_iter}")
    print("=" * 60)

    start_time = time.time()

    best_val = float('inf')
    best_x, best_y = 0.0, 0.0

    for i in range(max_iter):
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        val = rastrigin(x, y)
        if val < best_val:
            best_val = val
            best_x, best_y = x, y
        # 每5000次输出一次信息
        if (i+1) % 5000 == 0:
            print(f"迭代 {i+1:6d}: 当前最优值 = {best_val:.8f}")

    end_time = time.time()

    print("\n随机搜索完成")
    print(f"总迭代次数: {max_iter}")
    print(f"最优解: x = {best_x:.8f}, y = {best_y:.8f}")
    print(f"最优函数值: f(x,y) = {best_val:.8f}")
    print(f"理论最小值: 0.0")
    print(f"误差: {best_val:.8f}")
    print(f"运行耗时: {end_time - start_time:.4f} 秒")
    print("=" * 60)

    return best_x, best_y, best_val, max_iter, end_time - start_time

# ================== 对比主程序 ==================
if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# 多变量函数优化算法对比实验")
    print("# 目标函数: Rastrigin 函数 (理论最小值 0.0)")
    print("#" * 60)

    # 1. 运行遗传算法
    print("\n>>> 正在运行遗传算法...")
    ga_start = time.time()
    ga_best_chrom, ga_best_val, ga_history = genetic_algorithm(
        generations=200, pop_size=20, chrom_len=40, pc=0.7, pm=0.02
    )
    ga_time = time.time() - ga_start
    # 解码最优坐标
    from genetic_algorithm import decode
    ga_best_x, ga_best_y = decode(ga_best_chrom)

    # 2. 运行穷举法
    print("\n>>> 正在运行穷举法...")
    ex_best_x, ex_best_y, ex_best_val, ex_points, ex_time = exhaustive_search(step=0.05)

    # 3. 运行随机搜索
    print("\n>>> 正在运行随机搜索...")
    rs_best_x, rs_best_y, rs_best_val, rs_iters, rs_time = random_search(max_iter=50000)

    # 4. 汇总对比表格
    print("\n" + "=" * 70)
    print("算法对比结果汇总")
    print("=" * 70)
    print(f"{'算法':<15}{'最优 x':<12}{'最优 y':<12}{'最优值':<15}{'耗时(秒)':<12}{'评估次数':<12}")
    print("-" * 70)
    print(f"{'遗传算法':<15}{ga_best_x:<12.6f}{ga_best_y:<12.6f}{ga_best_val:<15.8f}{ga_time:<12.4f}{200*20:<12}")
    print(f"{'穷举法':<15}{ex_best_x:<12.6f}{ex_best_y:<12.6f}{ex_best_val:<15.8f}{ex_time:<12.4f}{ex_points:<12}")
    print(f"{'随机搜索':<15}{rs_best_x:<12.6f}{rs_best_y:<12.6f}{rs_best_val:<15.8f}{rs_time:<12.4f}{rs_iters:<12}")
    print("=" * 70)

    # 结论提示
    print("\n【遗传算法特点总结】")
    print("优点: 全局搜索能力强，不依赖梯度信息，适合复杂非线性问题。")
    print("缺点: 参数较多(种群大小、交叉/变异概率等)，收敛速度可能较慢，有时会陷入局部最优。")
    print("穷举法精度依赖步长，但计算量随维度指数增长。随机搜索简单但效率较低。")