import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import sys
import glob
import argparse

plt.rcParams.update({'font.size': 16})

# 更新颜色映射，可以根据需要添加更多算法
colors_map = {
    'IC3Net': '#fca503',
    'CommNet': '#b0b0b0',
    'TarMAC-IC3Net': '#b700ff',
    'GA-Comm': '#77ab3f',
    'MAGIC': '#0040ff',  # 您的方法
    'MAGIC w/o Scheduler': '#ff6373'
}

def read_file(vec, file_name, term):
    print(f"读取文件: {file_name}")
    with open(file_name, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            return vec

        for idx, line in enumerate(lines):
            if term not in line:
                continue
            epoch_idx = idx
            epoch_line = line
            while 'Epoch' not in epoch_line and epoch_idx > 0:
                epoch_idx -= 1
                epoch_line = lines[epoch_idx]

            if 'Epoch' not in epoch_line:
                continue

            epoch = int(epoch_line.split(' ')[1].split('\t')[0])

            # 提取指标值
            if term in ['Reward', 'Success', 'Add-Rate', 'Steps-Taken']:
                value_part = line.split(': ')[1].strip()
                if '[' in value_part and ']' in value_part:
                    # 处理数组形式的值
                    left_bracket = value_part.find('[')
                    right_bracket = value_part.find(']')
                    value_str = value_part[left_bracket + 1:right_bracket]
                    values = np.fromstring(value_str, dtype=float, sep=' ')
                    value = values.mean()
                else:
                    # 处理单个数值
                    value = float(value_part)
                
                while epoch > len(vec):
                    vec.append([])
                if epoch <= len(vec):
                    vec[epoch - 1].append(value)

    return vec

def parse_plot(files, term='Reward', save_path=None):
    coll = dict()
    
    for fname in files:
        # 确定算法标签
        if 'ic3net' in fname.lower() and not 'tar' in fname.lower():
            label = 'IC3Net'
        elif 'commnet' in fname.lower():
            label = 'CommNet'
        elif 'magic' in fname.lower() or 'gcomm' in fname.lower():
            if 'scheduler' in fname.lower() or 'complete' in fname.lower():
                label = 'MAGIC w/o Scheduler'
            else:
                label = 'MAGIC'
        else:
            # 从文件名提取算法名称
            # 从文件名提取算法名称
            label = fname.split('/')[-1].replace('.log', '')
            
        if label not in coll:
            coll[label] = []

        coll[label] = read_file(coll[label], fname, term)

    plt.figure(figsize=(10, 6))
    
    for label in coll.keys():
        if not coll[label]:
            print(f"警告：{label}没有数据")
            continue
            
        # 计算均值和标准差
        epochs = min(2000, len(coll[label]))  # 最多显示1000个epoch
        mean_values = []
        max_values = []
        min_values = []

        for i in range(epochs):
            if i < len(coll[label]) and coll[label][i]:
                mean = np.mean(coll[label][i])
                if term == 'Success':
                    mean *= 100  # 转换为百分比
                mean_values.append(mean)
                
                std = np.std(coll[label][i]) / np.sqrt(len(coll[label][i]))
                if term == 'Success':
                    std *= 100
                std = min(std, 20)  # 限制标准差范围
                
                max_values.append(mean + std)
                min_values.append(mean - std)
            else:
                # 处理缺失数据
                if mean_values:
                    mean_values.append(mean_values[-1])
                    max_values.append(max_values[-1])
                    min_values.append(min_values[-1])
                else:
                    mean_values.append(0)
                    max_values.append(0)
                    min_values.append(0)

        # 打印统计信息
        print(f"\n{label}算法的{term}指标统计:")
        print(f"最大值: {np.max(mean_values):.4f}")
        print(f"最小值: {np.min(mean_values):.4f}")
        print(f"平均值: {np.mean(mean_values):.4f}")
        
        # 绘制曲线
        color = colors_map.get(label, np.random.rand(3,))
        plt.plot(np.arange(len(mean_values)), mean_values, linewidth=2.0, label=label, color=color)
        plt.fill_between(np.arange(len(mean_values)), min_values, max_values, 
                        color=colors.to_rgba(color, alpha=0.2))

    plt.xlabel('训练轮数 (Epochs)')
    
    # 设置Y轴标签
    if term == 'Reward':
        plt.ylabel('平均奖励')
    elif term == 'Success':
        plt.ylabel('成功率 (%)')
    elif term == 'Steps-Taken':
        plt.ylabel('平均步数')
    elif term == 'Add-Rate':
        plt.ylabel('添加率')
    else:
        plt.ylabel(term)
    
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'交通路口环境{term}指标对比')
    
    if save_path:
        plt.savefig(f"{save_path}_{term}.png", dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}_{term}.png")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='绘制训练日志图表')
    parser.add_argument('log_path', type=str, help='日志文件路径(可使用通配符)')
    parser.add_argument('--term', type=str, default='Reward', help='要绘制的指标，可选值：Reward, Success, Steps-Taken, Add-Rate')
    parser.add_argument('--save', type=str, default=None, help='保存图表的路径前缀')
    
    args = parser.parse_args()
    
    files = glob.glob(args.log_path)
    # 过滤掉.pt文件
    files = list(filter(lambda x: '.pt' not in x, files))
    
    if not files:
        print(f"错误：找不到匹配的日志文件: {args.log_path}")
        sys.exit(1)
    
    print(f"找到{len(files)}个日志文件:")
    for f in files:
        print(f"- {f}")
    
    parse_plot(files, args.term, args.save)