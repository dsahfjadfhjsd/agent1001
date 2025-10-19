# -*- coding: utf-8 -*-
"""
从 plot_data_points.json 读取数据并绘制改变量图表

使用方法:
    python plot_change_from_json.py [json_file_path]
    
如果不指定路径，会查找最新的 plot_data_points.json 文件
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys
import numpy as np

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def find_latest_plot_data():
    """查找最新的 plot_data_points.json 文件"""
    timeseries_dir = Path("Output/timeseries")

    if not timeseries_dir.exists():
        print("❌ 未找到 Output/timeseries 目录")
        return None

    # 查找所有 plot_data_points.json 文件
    json_files = list(timeseries_dir.glob("*/plot_data_points.json"))

    if not json_files:
        print("❌ 未找到任何 plot_data_points.json 文件")
        return None

    # 按修改时间排序，返回最新的
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 找到最新数据文件: {latest_file}")
    return latest_file


def load_plot_data(json_path):
    """加载 plot_data_points.json 文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 成功加载数据: {len(data.get('data_points', []))} 个时间点")
        return data
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return None


def plot_change_over_time(data, output_dir=None):
    """
    绘制改变量随时间变化图

    Args:
        data: plot_data_points.json 的数据
        output_dir: 输出目录，如果为 None 则使用 json 文件所在目录
    """
    data_points = data.get('data_points', [])

    if not data_points:
        print("❌ 数据点为空，无法绘图")
        return

    # 提取数据
    times = [datetime.fromisoformat(point['datetime']) for point in data_points]
    stance_changes = [point['stance_change'] for point in data_points]
    sentiment_changes = [point['sentiment_change'] for point in data_points]

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('用户立场与情感改变量分析', fontsize=16, fontweight='bold')

    # === 子图1: 立场改变量 ===
    ax1.plot(times, stance_changes,
             marker='o', linewidth=2.5, markersize=8,
             color='#E74C3C', label='立场改变量',
             alpha=0.8)

    # 添加零线
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # 填充正负区域
    ax1.fill_between(times, stance_changes, 0,
                     where=[x >= 0 for x in stance_changes],
                     alpha=0.2, color='green', label='正向改变')
    ax1.fill_between(times, stance_changes, 0,
                     where=[x < 0 for x in stance_changes],
                     alpha=0.2, color='red', label='负向改变')

    ax1.set_ylabel('立场改变量', fontsize=12, fontweight='bold')
    ax1.set_title('立场改变量随时间变化 (负值=更反对, 正值=更支持)', fontsize=13)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 显示数值标签
    for i, (time, change) in enumerate(zip(times, stance_changes)):
        ax1.text(time, change, f'{change:+.3f}',
                 ha='center', va='bottom' if change >= 0 else 'top',
                 fontsize=9, alpha=0.7)

    # === 子图2: 情感改变量 ===
    ax2.plot(times, sentiment_changes,
             marker='s', linewidth=2.5, markersize=8,
             color='#3498DB', label='情感改变量',
             alpha=0.8)

    # 添加零线
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # 填充正负区域
    ax2.fill_between(times, sentiment_changes, 0,
                     where=[x >= 0 for x in sentiment_changes],
                     alpha=0.2, color='green', label='正向改变')
    ax2.fill_between(times, sentiment_changes, 0,
                     where=[x < 0 for x in sentiment_changes],
                     alpha=0.2, color='red', label='负向改变')

    ax2.set_xlabel('时间', fontsize=12, fontweight='bold')
    ax2.set_ylabel('情感改变量', fontsize=12, fontweight='bold')
    ax2.set_title('情感改变量随时间变化 (负值=更消极, 正值=更积极)', fontsize=13)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 显示数值标签
    for i, (time, change) in enumerate(zip(times, sentiment_changes)):
        ax2.text(time, change, f'{change:+.3f}',
                 ha='center', va='bottom' if change >= 0 else 'top',
                 fontsize=9, alpha=0.7)

    # 调整布局
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 保存图片
    if output_dir is None:
        output_dir = Path("Output/timeseries") / data.get('batch_id', 'unknown')
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "change_over_time.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 改变量图表已保存: {output_file}")

    # 显示图表
    plt.show()

    # 打印统计摘要
    print("\n" + "="*60)
    print("📊 改变量统计摘要")
    print("="*60)

    summary = data.get('summary', {})

    print(f"\n【立场改变量】")
    print(f"  平均值: {summary.get('stance_change_avg', 0):+.4f}")
    print(f"  累计变化: {summary.get('stance_change_total', 0):+.4f}")
    print(f"  最大单次变化: {max(stance_changes):+.4f}")
    print(f"  最小单次变化: {min(stance_changes):+.4f}")

    print(f"\n【情感改变量】")
    print(f"  平均值: {summary.get('sentiment_change_avg', 0):+.4f}")
    print(f"  累计变化: {summary.get('sentiment_change_total', 0):+.4f}")
    print(f"  最大单次变化: {max(sentiment_changes):+.4f}")
    print(f"  最小单次变化: {min(sentiment_changes):+.4f}")

    print(f"\n【总体趋势】")
    stance_trend = "偏向支持" if summary.get('stance_change_total', 0) > 0 else "偏向反对"
    sentiment_trend = "偏向积极" if summary.get('sentiment_change_total', 0) > 0 else "偏向消极"
    print(f"  立场趋势: {stance_trend}")
    print(f"  情感趋势: {sentiment_trend}")
    print(f"  平均参与人数: {summary.get('participant_avg', 0):.1f}")
    print(f"  总参与人次: {summary.get('participant_total', 0)}")

    print("="*60)


def plot_combined_view(data, output_dir=None):
    """
    绘制综合视图：立场/情感值 + 改变量

    Args:
        data: plot_data_points.json 的数据
        output_dir: 输出目录
    """
    data_points = data.get('data_points', [])

    if not data_points:
        print("❌ 数据点为空，无法绘图")
        return

    # 提取数据
    times = [datetime.fromisoformat(point['datetime']) for point in data_points]
    stance_values = [point['user_stance'] for point in data_points]
    sentiment_values = [point['user_sentiment'] for point in data_points]
    stance_changes = [point['stance_change'] for point in data_points]
    sentiment_changes = [point['sentiment_change'] for point in data_points]

    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('用户认知动态全景分析', fontsize=18, fontweight='bold')

    # === 左上: 立场值 ===
    ax1 = axes[0, 0]
    ax1.plot(times, stance_values,
             marker='o', linewidth=2.5, markersize=8,
             color='#E74C3C', label='立场值')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_ylabel('立场值', fontsize=11, fontweight='bold')
    ax1.set_title('立场值变化 (-1=强烈反对, +1=强烈支持)', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # === 右上: 立场改变量 ===
    ax2 = axes[0, 1]
    ax2.bar(times, stance_changes, color=['green' if x >= 0 else 'red' for x in stance_changes], alpha=0.6)
    ax2.axhline(y=0, color='black', linewidth=1.5)
    ax2.set_ylabel('立场改变量', fontsize=11, fontweight='bold')
    ax2.set_title('每轮立场改变量', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # === 左下: 情感值 ===
    ax3 = axes[1, 0]
    ax3.plot(times, sentiment_values,
             marker='s', linewidth=2.5, markersize=8,
             color='#3498DB', label='情感值')
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('时间', fontsize=11, fontweight='bold')
    ax3.set_ylabel('情感值', fontsize=11, fontweight='bold')
    ax3.set_title('情感值变化 (-1=极消极, +1=极积极)', fontsize=12)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # === 右下: 情感改变量 ===
    ax4 = axes[1, 1]
    ax4.bar(times, sentiment_changes, color=['green' if x >= 0 else 'red' for x in sentiment_changes], alpha=0.6)
    ax4.axhline(y=0, color='black', linewidth=1.5)
    ax4.set_xlabel('时间', fontsize=11, fontweight='bold')
    ax4.set_ylabel('情感改变量', fontsize=11, fontweight='bold')
    ax4.set_title('每轮情感改变量', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')

    # 调整布局
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # 保存图片
    if output_dir is None:
        output_dir = Path("Output/timeseries") / data.get('batch_id', 'unknown')
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "combined_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 综合分析图表已保存: {output_file}")

    plt.show()


def main():
    """主函数"""
    print("="*60)
    print("📊 改变量可视化工具")
    print("="*60)

    # 获取输入文件路径
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
        if not json_path.exists():
            print(f"❌ 文件不存在: {json_path}")
            return
    else:
        json_path = find_latest_plot_data()
        if json_path is None:
            return

    # 加载数据
    data = load_plot_data(json_path)
    if data is None:
        return

    # 获取输出目录
    output_dir = json_path.parent

    # 绘制改变量图
    print("\n🎨 正在绘制改变量分析图...")
    plot_change_over_time(data, output_dir)

    # 绘制综合视图
    print("\n🎨 正在绘制综合分析图...")
    plot_combined_view(data, output_dir)

    print("\n✅ 所有图表生成完成！")


if __name__ == "__main__":
    main()
