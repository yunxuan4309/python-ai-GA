"""
遗传算法求解 Rastrigin 函数最小值
文件名：genetic_algorithm.py
"""

import numpy as np
import time

# ================== 目标函数 ==================
def rastrigin(x, y):
    """Rastrigin 函数，理论最小值 0 在 (0,0)"""
    return x**2 + y**2 - 10 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y)) + 20

# ================== 编码与解码 ==================
def encode(x, y, bits=20, x_range=(-5, 5), y_range=(-5, 5)):
    """将 x, y 编码为二进制染色体，各占 bits 位"""
    x_norm = (x - x_range[0]) / (x_range[1] - x_range[0])
    y_norm = (y - y_range[0]) / (y_range[1] - y_range[0])
    x_int = int(round(x_norm * (2**bits - 1)))
    y_int = int(round(y_norm * (2**bits - 1)))
    chrom = format(x_int, f'0{bits}b') + format(y_int, f'0{bits}b')
    return chrom

def decode(chrom, bits=20, x_range=(-5, 5), y_range=(-5, 5)):
    """将二进制染色体解码为 x, y"""
    x_bin = chrom[:bits]
    y_bin = chrom[bits:]
    x_int = int(x_bin, 2)
    y_int = int(y_bin, 2)
    x = x_range[0] + (x_int / (2**bits - 1)) * (x_range[1] - x_range[0])
    y = y_range[0] + (y_int / (2**bits - 1)) * (y_range[1] - y_range[0])
    return x, y

# ================== 适应度函数 ==================
def fitness(chrom):
    """适应度函数：值越小适应度越高，取倒数"""
    x, y = decode(chrom)
    val = rastrigin(x, y)
    return 1.0 / (1.0 + val)

# ================== 种群初始化 ==================
def init_population(pop_size, chrom_len):
    """随机生成长度为 chrom_len 的二进制种群"""
    population = []
    for _ in range(pop_size):
        chrom = ''.join(np.random.choice(['0', '1']) for _ in range(chrom_len))
        population.append(chrom)
    return population

# ================== 选择算子 (轮盘赌) ==================
def selection(population, fitness_values):
    """轮盘赌选择，返回新种群"""
    probs = np.array(fitness_values) / np.sum(fitness_values)
    cum_probs = np.cumsum(probs)
    new_pop = []
    for _ in range(len(population)):
        r = np.random.rand()
        idx = np.searchsorted(cum_probs, r)
        new_pop.append(population[idx])
    return new_pop

# ================== 交叉算子 (单点交叉) ==================
def crossover(population, pc=0.7):
    """对种群进行交叉操作"""
    half = len(population) // 2
    fathers = population[:half]
    mothers = population[half:]
    np.random.shuffle(fathers)
    np.random.shuffle(mothers)
    offspring = []
    for f, m in zip(fathers, mothers):
        if np.random.rand() < pc:
            point = np.random.randint(1, len(f)-1)
            s1 = f[:point] + m[point:]
            s2 = m[:point] + f[point:]
        else:
            s1, s2 = f, m
        offspring.extend([s1, s2])
    return offspring

# ================== 变异算子 ==================
def mutation(population, pm=0.02):
    """对种群进行变异操作"""
    for i in range(len(population)):
        chrom = list(population[i])
        for j in range(len(chrom)):
            if np.random.rand() < pm:
                chrom[j] = '1' if chrom[j] == '0' else '0'
        population[i] = ''.join(chrom)
    return population

# ================== 遗传算法主流程 ==================
def genetic_algorithm(generations=200, pop_size=20, chrom_len=40, pc=0.7, pm=0.02):
    """运行遗传算法并返回最优解及历史记录"""
    print("=" * 60)
    print("遗传算法开始运行")
    print(f"参数设置: 种群大小={pop_size}, 进化代数={generations}")
    print(f"交叉概率={pc}, 变异概率={pm}, 染色体长度={chrom_len}")
    print("=" * 60)

    # 初始化种群
    population = init_population(pop_size, chrom_len)

    best_solution = None
    best_fitness = 0.0
    best_value = float('inf')
    history = []

    start_time = time.time()

    for gen in range(generations):
        # 计算适应度
        fits = [fitness(ch) for ch in population]

        # 记录当前代最优
        max_fit_idx = np.argmax(fits)
        current_best_fitness = fits[max_fit_idx]
        current_best_chrom = population[max_fit_idx]
        x_cur, y_cur = decode(current_best_chrom)
        current_value = rastrigin(x_cur, y_cur)

        history.append(current_value)

        # 更新全局最优
        if current_value < best_value:
            best_value = current_value
            best_fitness = current_best_fitness
            best_solution = current_best_chrom

        # 每10代输出一次信息
        if gen % 20 == 0 or gen == generations - 1:
            print(f"第 {gen:3d} 代: 最优函数值 = {current_value:.8f}, 坐标 = ({x_cur:.6f}, {y_cur:.6f})")

        # 遗传操作
        population = selection(population, fits)
        population = crossover(population, pc)
        population = mutation(population, pm)

    end_time = time.time()

    # 最终解码
    best_x, best_y = decode(best_solution)

    print("\n" + "=" * 60)
    print("遗传算法运行结束")
    print(f"全局最优解: x = {best_x:.8f}, y = {best_y:.8f}")
    print(f"最优函数值: f(x,y) = {best_value:.8f}")
    print(f"理论最小值: 0.0")
    print(f"误差: {best_value:.8f}")
    print(f"运行耗时: {end_time - start_time:.4f} 秒")
    print("=" * 60)

    return best_solution, best_value, history

# ================== 主程序入口 ==================
if __name__ == "__main__":
    # 运行遗传算法
    best_chrom, best_val, history = genetic_algorithm(
        generations=200,
        pop_size=20,
        chrom_len=40,
        pc=0.7,
        pm=0.02
    )

    # 可选：绘制收敛曲线（如不需要绘图可注释掉）
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(history, 'b-', linewidth=1)
        plt.axhline(y=0, color='r', linestyle='--', label='理论最小值 0')
        plt.xlabel('进化代数')
        plt.ylabel('最优函数值')
        plt.title('遗传算法收敛曲线 (Rastrigin 函数)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    except ImportError:
        print("\nmatplotlib 未安装，跳过绘图。")