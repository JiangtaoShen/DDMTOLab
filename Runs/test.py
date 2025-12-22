import numpy as np
from Problems.RWP.SOPM_MTMO.sopm import SOPMMTMO

# 创建问题实例
sopm = SOPMMTMO()
problem = sopm.P2()

# 5个固定测试点（每个任务30维，范围[0, 90]）
np.random.seed(123)  # 固定随机种子以便复现

# 测试点1: 均匀分布在[0, 90]
X1 = np.linspace(90, 0, 30).reshape(1, -1)  # 从90递减到0

# 测试点2: 随机点
X2 = np.array([[87.5, 85.2, 82.8, 80.5, 78.1, 75.8, 73.4, 71.1, 68.7, 66.4,
                64.0, 61.7, 59.3, 57.0, 54.6, 52.3, 49.9, 47.6, 45.2, 42.9,
                40.5, 38.2, 35.8, 33.5, 31.1, 28.8, 26.4, 24.1, 21.7, 19.4]])

# 测试点3: 另一组随机点
X3 = np.array([[89.0, 86.8, 84.5, 82.3, 80.0, 77.8, 75.5, 73.3, 71.0, 68.8,
                66.5, 64.3, 62.0, 59.8, 57.5, 55.3, 53.0, 50.8, 48.5, 46.3,
                44.0, 41.8, 39.5, 37.3, 35.0, 32.8, 30.5, 28.3, 26.0, 23.8]])

# 测试点4: 小角度范围
X4 = np.array([[48.0, 46.5, 45.0, 43.5, 42.0, 40.5, 39.0, 37.5, 36.0, 34.5,
                33.0, 31.5, 30.0, 28.5, 27.0, 25.5, 24.0, 22.5, 21.0, 19.5,
                18.0, 16.5, 15.0, 13.5, 12.0, 10.5, 9.0, 7.5, 6.0, 4.5]])

# 测试点5: 大角度范围
X5 = np.array([[90.0, 88.0, 86.0, 84.0, 82.0, 80.0, 78.0, 76.0, 74.0, 72.0,
                70.0, 68.0, 66.0, 64.0, 62.0, 60.0, 58.0, 56.0, 54.0, 52.0,
                50.0, 48.0, 46.0, 44.0, 42.0, 40.0, 38.0, 36.0, 34.0, 32.0]])

# 合并所有测试点
X_test = np.vstack([X1, X2, X3, X4, X5])

print("=" * 80)
print("SOPM_MTMO P2 测试 - Python实现")
print("=" * 80)
print(f"\n测试点数量: {X_test.shape[0]}")
print(f"决策变量维度: {X_test.shape[1]}")
print(f"问题任务数: {len(problem.tasks)}")

# 使用 evaluate_tasks 方法同时评估3个任务
print(f"\n{'=' * 80}")
print("使用 evaluate_tasks 方法同时评估所有任务")
print(f"{'=' * 80}")

# 为每个任务准备输入数据（3个任务使用相同的测试点）
X_list = [X_test, X_test, X_test]
task_indices = [0, 1, 2]

# 同时评估3个任务
objs_list, cons_list = problem.evaluate_tasks(task_indices=task_indices, X_list=X_list)

# 显示每个任务的结果
task_names = ['9-level', '11-level', '13-level']
for task_idx, (objs, cons) in enumerate(zip(objs_list, cons_list)):
    print(f"\n{'=' * 80}")
    print(f"任务 {task_idx + 1} ({task_names[task_idx]} Inverter)")
    print(f"{'=' * 80}")
    print(f"objs shape: {objs.shape}, cons shape: {cons.shape}")

    print(f"\n目标函数值:")
    print("  Point | Obj1 (THD)      | Obj2 (Fund Dev)")
    print("  " + "-" * 50)
    for i in range(objs.shape[0]):
        print(f"  {i + 1:5d} | {objs[i, 0]:15.10f} | {objs[i, 1]:15.10f}")

    print(f"\n约束违反值:")
    print(f"  总约束数: {cons.shape[1]}")

    # 统计约束违反情况
    for i in range(cons.shape[0]):
        num_violations = np.sum(cons[i, :] > 0)
        max_violation = np.max(cons[i, :])
        sum_violations = np.sum(cons[i, :])
        print(f"  Point {i + 1}: violations={num_violations:2d}, max={max_violation:.6f}, sum={sum_violations:.6f}")

# 汇总统计
print(f"\n{'=' * 80}")
print("汇总统计")
print(f"{'=' * 80}")
for i, (objs, cons) in enumerate(zip(objs_list, cons_list)):
    print(f"\n任务{i + 1} ({task_names[i]} Inverter):")
    print(f"  目标函数 shape: {objs.shape}")
    print(f"  约束函数 shape: {cons.shape}")
    print(f"  Obj1 (THD) 范围: [{np.min(objs[:, 0]):.6f}, {np.max(objs[:, 0]):.6f}]")
    print(f"  Obj2 (Fund) 范围: [{np.min(objs[:, 1]):.6f}, {np.max(objs[:, 1]):.6f}]")

    num_feasible = np.sum(np.all(cons <= 0, axis=1))
    print(f"  可行解数量: {num_feasible}/{cons.shape[0]}")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)

# 保存测试数据供MATLAB验证
np.savetxt('test_points_sopm_p2.txt', X_test, fmt='%.10f', delimiter=',')
print("\n测试点已保存到: test_points_sopm_p2.txt")