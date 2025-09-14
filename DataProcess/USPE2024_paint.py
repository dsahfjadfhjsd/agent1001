#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USPE2024统一数据集分析与可视化工具
模仿XMSU7D的分析方法，分析统一数据集的态度情感时间变化
以Trump为目标候选人的统一视角进行分析
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class USPE2024UnifiedAnalyzer:
    def __init__(self, data_dir="Data/USPE2024/integrated_data", hour_interval=24):
        """初始化USPE2024统一数据集分析工具"""
        self.data_dir = Path(data_dir)
        # 使用统一数据集文件
        self.articles_file = self.data_dir / "USPE2024_unified_articles.csv"
        self.comments_file = self.data_dir / "USPE2024_unified_comments.csv"
        self.output_dir = Path("Data/visualizations")
        self.output_dir.mkdir(exist_ok=True)

        # 时间间隔设置
        self.hour_interval = hour_interval
        print(f"⏰ 时间统计间隔: 每{hour_interval}小时")

        # 定义映射字典 - 以Trump为目标的立场
        self.stance_mapping = {"支持": 1, "中立": 0, "反对": -1}
        self.sentiment_mapping = {"积极": 1, "中立": 0, "消极": -1}

        # 加载数据
        self.load_data()

    def load_data(self):
        """加载并预处理统一数据集"""
        print("📊 正在加载USPE2024统一数据集...")

        # 加载统一文章数据
        self.articles_df = pd.read_csv(self.articles_file, encoding='utf-8-sig')
        print(f"✅ 统一文章数据: {len(self.articles_df)} 条")

        # 加载统一评论数据
        self.comments_df = pd.read_csv(self.comments_file, encoding='utf-8-sig')
        print(f"✅ 统一评论数据: {len(self.comments_df)} 条")

        # 转换时间列
        self.articles_df['datetime'] = pd.to_datetime(self.articles_df['created_date'])
        self.comments_df['datetime'] = pd.to_datetime(self.comments_df['created_date'])

        # 转换stance和sentiment为数值
        self.convert_categorical_to_numeric()

    def convert_categorical_to_numeric(self):
        """将分类变量转换为数值"""
        print("🔄 转换分类变量为数值...")

        # 处理文章数据 - 使用对Trump的立场（现在直接使用stance列）
        self.articles_df['stance_numeric'] = self.articles_df['stance'].map(self.stance_mapping)
        self.articles_df['sentiment_numeric'] = self.articles_df['sentiment'].map(self.sentiment_mapping)

        # 处理评论数据 - 使用对Trump的立场（现在直接使用stance列）
        self.comments_df['stance_numeric'] = self.comments_df['stance'].map(self.stance_mapping)
        self.comments_df['sentiment_numeric'] = self.comments_df['sentiment'].map(self.sentiment_mapping)

        # 处理缺失值，填充为0（中立）
        self.articles_df['stance_numeric'] = self.articles_df['stance_numeric'].fillna(0)
        self.articles_df['sentiment_numeric'] = self.articles_df['sentiment_numeric'].fillna(0)
        self.comments_df['stance_numeric'] = self.comments_df['stance_numeric'].fillna(0)
        self.comments_df['sentiment_numeric'] = self.comments_df['sentiment_numeric'].fillna(0)

        print("✅ 分类变量转换完成")

    def aggregate_hourly_data(self):
        """按设定的时间间隔聚合数据 - 模仿XMSU7D方法"""
        print(f"📈 按{self.hour_interval}小时间隔聚合数据...")

        # 合并文章和评论数据
        articles_subset = self.articles_df[['datetime', 'stance_numeric', 'sentiment_numeric']].copy()
        articles_subset['type'] = 'article'

        comments_subset = self.comments_df[['datetime', 'stance_numeric', 'sentiment_numeric']].copy()
        comments_subset['type'] = 'comment'

        # 按时间间隔分组
        if self.hour_interval == 1:
            articles_subset['time_group'] = articles_subset['datetime'].dt.floor('h')
            comments_subset['time_group'] = comments_subset['datetime'].dt.floor('h')
        else:
            articles_subset['time_group'] = articles_subset['datetime'].dt.floor(f'{self.hour_interval}h')
            comments_subset['time_group'] = comments_subset['datetime'].dt.floor(f'{self.hour_interval}h')

        # 合并数据
        combined_df = pd.concat([articles_subset, comments_subset], ignore_index=True)

        # 按时间间隔聚合
        hourly_stats = combined_df.groupby('time_group').agg({
            'stance_numeric': 'mean',
            'sentiment_numeric': 'mean',
            'type': 'count'
        }).reset_index()

        # 重命名列
        hourly_stats.columns = ['hour', 'avg_stance', 'avg_sentiment', 'total_count']

        # 分别统计文章和评论数量
        article_counts = articles_subset.groupby('time_group').size()
        comment_counts = comments_subset.groupby('time_group').size()

        # 合并计数信息
        hourly_stats = hourly_stats.set_index('hour')
        hourly_stats['article_count'] = article_counts
        hourly_stats['comment_count'] = comment_counts
        hourly_stats = hourly_stats.fillna(0).reset_index()

        print(f"✅ 聚合完成，共 {len(hourly_stats)} 个{self.hour_interval}小时间隔的数据")

        return hourly_stats

    def plot_sentiment_timeline(self):
        """绘制情感态度时间折线图 - 模仿XMSU7D风格"""
        print("🎨 绘制情感态度时间折线图...")

        # 获取小时级数据
        hourly_data = self.aggregate_hourly_data()

        # 创建图表 - 模仿XMSU7D的双子图布局
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'USPE2024 社交媒体情感态度分析 (每{self.hour_interval}小时)', fontsize=16, fontweight='bold')

        # 第一个子图：stance和sentiment平均值
        ax1.plot(hourly_data['hour'], hourly_data['avg_stance'],
                 marker='o', linewidth=2.5, markersize=6, label='立场(Stance)平均值',
                 color='#2E86AB', alpha=0.8)

        ax1.plot(hourly_data['hour'], hourly_data['avg_sentiment'],
                 marker='s', linewidth=2.5, markersize=6, label='情感(Sentiment)平均值',
                 color='#A23B72', alpha=0.8)

        # 添加中立线
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='中立线')

        ax1.set_title(f'每{self.hour_interval}小时立场与情感平均值变化', fontsize=14)
        ax1.set_ylabel('平均值', fontsize=12)
        ax1.set_ylim(-1.1, 1.1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 添加y轴标签说明
        ax1.text(-0.05, 1, '支持/积极', transform=ax1.transAxes,
                 verticalalignment='top', fontsize=10, color='green')
        ax1.text(-0.05, 0.5, '中立', transform=ax1.transAxes,
                 verticalalignment='center', fontsize=10, color='gray')
        ax1.text(-0.05, 0, '反对/消极', transform=ax1.transAxes,
                 verticalalignment='bottom', fontsize=10, color='red')

        # 第二个子图：总数量
        ax2.plot(hourly_data['hour'], hourly_data['total_count'],
                 marker='D', linewidth=3, markersize=6, label='总数量',
                 color='#F18F01', alpha=0.9)
        ax2.fill_between(hourly_data['hour'], hourly_data['total_count'],
                         alpha=0.3, color='#F18F01')

        ax2.set_title(f'每{self.hour_interval}小时评论+帖子总数量变化', fontsize=14)
        ax2.set_xlabel('时间', fontsize=12)
        ax2.set_ylabel('数量', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 格式化x轴时间显示 - 模仿XMSU7D的时间格式
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=0)
            # 根据时间间隔调整刻度显示密度
            tick_interval = max(1, 168 // self.hour_interval)  # 一周168小时
            if len(hourly_data) > tick_interval:
                selected_hours = hourly_data['hour'][::tick_interval]
            else:
                selected_hours = hourly_data['hour']
            ax.set_xticks(selected_hours)

            # 显示月-日格式，相同日期不重复
            labels = []
            last_date = None
            for hour in selected_hours:
                current_date = hour.strftime('%m-%d')
                if current_date != last_date:
                    labels.append(current_date)
                    last_date = current_date
                else:
                    labels.append('')
            ax.set_xticklabels(labels)

        plt.tight_layout()

        # 保存图表
        output_file = self.output_dir / f"USPE2024_unified_timeline_analysis_{self.hour_interval}h.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 图表已保存至: {output_file}")

        # 显示统计信息
        self.print_statistics(hourly_data)

        return fig, hourly_data

    def plot_detailed_breakdown(self):
        """绘制详细分解图 - 模仿XMSU7D的四象限布局"""
        print("🎨 绘制详细分解图...")

        hourly_data = self.aggregate_hourly_data()

        # 创建更详细的图表 - 模仿XMSU7D的2x2布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'USPE2024 详细分析 (每{self.hour_interval}小时)', fontsize=16, fontweight='bold')

        # 立场分析
        ax1.plot(hourly_data['hour'], hourly_data['avg_stance'],
                 marker='o', linewidth=2.5, markersize=6, color='#2E86AB')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('立场(Stance)时间变化')
        ax1.set_ylabel('平均立场值')
        ax1.set_ylim(-1.1, 1.1)
        ax1.grid(True, alpha=0.3)

        # 情感分析
        ax2.plot(hourly_data['hour'], hourly_data['avg_sentiment'],
                 marker='s', linewidth=2.5, markersize=6, color='#A23B72')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('情感(Sentiment)时间变化')
        ax2.set_ylabel('平均情感值')
        ax2.set_ylim(-1.1, 1.1)
        ax2.grid(True, alpha=0.3)

        # 文章数量
        ax3.plot(hourly_data['hour'], hourly_data['article_count'],
                 marker='o', linewidth=2.5, markersize=5, color='#1f77b4', alpha=0.8)
        ax3.set_title('文章数量时间变化')
        ax3.set_ylabel('文章数量')
        ax3.set_xlabel('时间')
        ax3.grid(True, alpha=0.3)

        # 评论数量
        ax4.plot(hourly_data['hour'], hourly_data['comment_count'],
                 marker='s', linewidth=2.5, markersize=5, color='#ff7f0e', alpha=0.8)
        ax4.set_title('评论数量时间变化')
        ax4.set_ylabel('评论数量')
        ax4.set_xlabel('时间')
        ax4.grid(True, alpha=0.3)

        # 格式化所有子图的x轴
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(axis='x', rotation=0)
            tick_interval = max(1, 168 // self.hour_interval)
            if len(hourly_data) > tick_interval:
                selected_hours = hourly_data['hour'][::tick_interval]
            else:
                selected_hours = hourly_data['hour']
            ax.set_xticks(selected_hours)

            labels = []
            last_date = None
            for hour in selected_hours:
                current_date = hour.strftime('%m-%d')
                if current_date != last_date:
                    labels.append(current_date)
                    last_date = current_date
                else:
                    labels.append('')
            ax.set_xticklabels(labels)

        plt.tight_layout()

        # 保存图表
        output_file = self.output_dir / f"USPE2024_unified_detailed_analysis_{self.hour_interval}h.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 详细分析图已保存至: {output_file}")

    def load_simulation_data(self, simulation_file_path):
        """加载时间序列模拟数据 - 为未来对比分析预留"""
        print("📊 正在加载模拟数据...")

        try:
            sim_df = pd.read_csv(simulation_file_path, encoding='utf-8-sig')
            sim_df['datetime'] = pd.to_datetime(sim_df['datetime'])
            print(f"✅ 加载模拟数据: {len(sim_df)} 个时间点")
            print(f"📅 模拟数据时间范围: {sim_df['datetime'].min()} 至 {sim_df['datetime'].max()}")
            return sim_df
        except Exception as e:
            print(f"❌ 加载模拟数据失败: {e}")
            return None

    def plot_comparison_analysis(self, simulation_file_path, photo_name="USPE2024_comparison_real_vs_simulation.png"):
        """绘制真实数据与模拟数据的对比分析 - 模仿XMSU7D原图风格"""
        print("🎨 绘制真实数据与模拟数据对比图...")

        # 加载模拟数据
        sim_data = self.load_simulation_data(simulation_file_path)
        if sim_data is None:
            return None

        # 获取真实数据的小时级聚合
        real_hourly_data = self.aggregate_hourly_data()

        # 创建对比图 - 模仿XMSU7D原图的风格
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('USPE2024时间序列用户立场情感变化分析 - 真实数据 vs 模拟数据对比',
                     fontsize=16, fontweight='bold')

        # 1. 立场对比 - 模仿原图风格
        ax1.plot(real_hourly_data['hour'], real_hourly_data['avg_stance'],
                 marker='o', linewidth=2, markersize=4, label='真实数据',
                 color='#2E86AB', alpha=0.8)
        ax1.plot(sim_data['datetime'], sim_data['user_stance'],
                 marker='s', linewidth=2, markersize=4, label='模拟数据',
                 color='#A23B72', alpha=0.8, linestyle='--')

        # 设置网格 - 更密集，模仿原图
        ax1.grid(True, which='major', alpha=0.6, linestyle='-', linewidth=0.8)
        ax1.grid(True, which='minor', alpha=0.3, linestyle='-', linewidth=0.3)
        ax1.minorticks_on()

        # 添加中立线
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)

        ax1.set_title('立场(Stance)时间变化', fontsize=14)
        ax1.set_ylabel('立场', fontsize=12)
        ax1.set_ylim(-1.00, 1.00)
        ax1.legend(loc='best', fontsize=10)

        # 设置y轴刻度，模仿原图
        ax1.set_yticks([-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00])

        # 添加y轴标签说明
        ax1.text(-0.08, 1, '支持', transform=ax1.transAxes,
                 verticalalignment='top', fontsize=9, color='black')
        ax1.text(-0.08, 0.5, '中立', transform=ax1.transAxes,
                 verticalalignment='center', fontsize=9, color='black')
        ax1.text(-0.08, 0, '反对', transform=ax1.transAxes,
                 verticalalignment='bottom', fontsize=9, color='black')

        # 2. 情感对比 - 使用相同的风格
        ax2.plot(real_hourly_data['hour'], real_hourly_data['avg_sentiment'],
                 marker='o', linewidth=2, markersize=4, label='真实数据',
                 color='#2E86AB', alpha=0.8)
        ax2.plot(sim_data['datetime'], sim_data['user_sentiment'],
                 marker='s', linewidth=2, markersize=4, label='模拟数据',
                 color='#A23B72', alpha=0.8, linestyle='--')

        # 设置网格 - 更密集，模仿原图
        ax2.grid(True, which='major', alpha=0.6, linestyle='-', linewidth=0.8)
        ax2.grid(True, which='minor', alpha=0.3, linestyle='-', linewidth=0.3)
        ax2.minorticks_on()

        # 添加中立线
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)

        ax2.set_title('情感(Sentiment)时间变化', fontsize=14)
        ax2.set_ylabel('情感', fontsize=12)
        ax2.set_ylim(-1.00, 1.00)
        ax2.legend(loc='best', fontsize=10)

        # 设置y轴刻度，模仿原图
        ax2.set_yticks([-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00])

        # 添加y轴标签说明
        ax2.text(-0.08, 1, '积极', transform=ax2.transAxes,
                 verticalalignment='top', fontsize=9, color='black')
        ax2.text(-0.08, 0.5, '中立', transform=ax2.transAxes,
                 verticalalignment='center', fontsize=9, color='black')
        ax2.text(-0.08, 0, '消极', transform=ax2.transAxes,
                 verticalalignment='bottom', fontsize=9, color='black')

        # 统一设置x轴 - 模仿原图的时间轴格式
        for ax in [ax1, ax2]:
            ax.set_xlabel('时间间隔时间(小时)', fontsize=12)
            # 设置x轴时间格式
            if hasattr(real_hourly_data['hour'].iloc[0], 'strftime'):
                # 如果是datetime对象，设置时间格式
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
                ax.xaxis.set_minor_locator(mdates.HourLocator(interval=24))
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.tick_params(axis='y', labelsize=10)

        plt.tight_layout()

        # 保存对比图
        output_file = self.output_dir / photo_name
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 对比图已保存至: {output_file}")

        # 打印对比统计
        self.print_comparison_statistics(real_hourly_data, sim_data)

        return fig, real_hourly_data, sim_data

    def print_statistics(self, hourly_data):
        """打印统计信息 - 模仿XMSU7D的统计格式"""
        print(f"\n📊 USPE2024统一数据集统计摘要:")

        # 时间范围
        start_time = self.articles_df['datetime'].min()
        end_time = self.articles_df['datetime'].max()
        print(f"时间范围: {start_time.strftime('%Y-%m-%d %H:%M')} 至 {end_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"时间间隔: 每{self.hour_interval}小时")
        print(f"数据点数量: {len(hourly_data)} 个")
        print(f"总帖子+评论: {hourly_data['total_count'].sum():.0f} 条")
        print(f"平均每{self.hour_interval}小时: {hourly_data['total_count'].mean():.1f} 条")

        # 数据来源统计
        article_sources = self.articles_df['data_source'].value_counts()
        comment_sources = self.comments_df['data_source'].value_counts()
        print(f"\n数据来源分布:")
        print(f"  文章: {dict(article_sources)}")
        print(f"  评论: {dict(comment_sources)}")

        # 立场统计
        avg_stance = hourly_data['avg_stance'].mean()
        stance_dist = self.articles_df['stance'].value_counts()
        print(f"\n立场分析 (对Trump):")
        print(f"  整体平均立场: {avg_stance:.3f}")
        print(f"  文章立场分布: {dict(stance_dist)}")
        if avg_stance > 0.1:
            print(f"  总体偏向: 支持Trump")
        elif avg_stance < -0.1:
            print(f"  总体偏向: 反对Trump")
        else:
            print(f"  总体偏向: 中立")

        # 情感统计
        avg_sentiment = hourly_data['avg_sentiment'].mean()
        sentiment_dist = self.comments_df['sentiment'].value_counts()
        print(f"\n情感分析:")
        print(f"  整体平均情感: {avg_sentiment:.3f}")
        print(f"  评论情感分布: {dict(sentiment_dist)}")
        if avg_sentiment > 0.1:
            print(f"  总体偏向: 积极")
        elif avg_sentiment < -0.1:
            print(f"  总体偏向: 消极")
        else:
            print(f"  总体偏向: 中立")

        # 峰值分析
        if len(hourly_data) > 0:
            peak_hour = hourly_data.loc[hourly_data['total_count'].idxmax(), 'hour']
            peak_count = hourly_data['total_count'].max()
            print(f"\n活跃度峰值: {peak_hour.strftime('%m-%d %H:%M')} ({peak_count:.0f}条)")

            # 立场和情感的极值
            max_stance_hour = hourly_data.loc[hourly_data['avg_stance'].idxmax(), 'hour']
            max_stance_val = hourly_data['avg_stance'].max()
            min_stance_hour = hourly_data.loc[hourly_data['avg_stance'].idxmin(), 'hour']
            min_stance_val = hourly_data['avg_stance'].min()

            print(f"\n立场极值:")
            print(f"  最支持时间: {max_stance_hour.strftime('%m-%d %H:%M')} (值: {max_stance_val:.3f})")
            print(f"  最反对时间: {min_stance_hour.strftime('%m-%d %H:%M')} (值: {min_stance_val:.3f})")

    def print_comparison_statistics(self, real_data, sim_data):
        """打印真实数据与模拟数据的对比统计 - 模仿XMSU7D格式"""
        print(f"\n📊 真实数据 vs 模拟数据对比统计:")

        # 立场对比
        real_avg_stance = real_data['avg_stance'].mean()
        sim_avg_stance = sim_data['user_stance'].mean()
        stance_diff = abs(real_avg_stance - sim_avg_stance)

        print(f"\n立场对比:")
        print(f"  真实数据平均立场: {real_avg_stance:.3f}")
        print(f"  模拟数据平均立场: {sim_avg_stance:.3f}")
        print(f"  立场差异: {stance_diff:.3f}")

        # 情感对比
        real_avg_sentiment = real_data['avg_sentiment'].mean()
        sim_avg_sentiment = sim_data['user_sentiment'].mean()
        sentiment_diff = abs(real_avg_sentiment - sim_avg_sentiment)

        print(f"\n情感对比:")
        print(f"  真实数据平均情感: {real_avg_sentiment:.3f}")
        print(f"  模拟数据平均情感: {sim_avg_sentiment:.3f}")
        print(f"  情感差异: {sentiment_diff:.3f}")

        # 参与度对比
        real_avg_count = real_data['total_count'].mean()
        if 'participant_count' in sim_data.columns:
            sim_avg_count = sim_data['participant_count'].mean()
            print(f"\n参与度对比:")
            print(f"  真实数据平均数量: {real_avg_count:.1f} 条/时间段")
            print(f"  模拟数据平均参与: {sim_avg_count:.1f} 人/时间段")

        # 总体评估
        print(f"\n📈 模拟质量评估:")
        if stance_diff < 0.1 and sentiment_diff < 0.1:
            print("  ✅ 优秀：模拟结果与真实数据高度一致")
        elif stance_diff < 0.2 and sentiment_diff < 0.2:
            print("  ✅ 良好：模拟结果与真实数据较为一致")
        elif stance_diff < 0.3 and sentiment_diff < 0.3:
            print("  ⚠️  一般：模拟结果与真实数据存在一定差异")
        else:
            print("  ❌ 需要改进：模拟结果与真实数据差异较大")

    def generate_report(self):
        """生成完整分析报告 - 模仿XMSU7D的报告生成流程"""
        print("📋 生成USPE2024统一数据集完整分析报告...")

        # 主要折线图
        fig1, hourly_data = self.plot_sentiment_timeline()

        # 详细分解图
        fig2 = self.plot_detailed_breakdown()

        print("✅ USPE2024统一数据集分析报告生成完成!")

        return hourly_data

    def aggregate_hourly_data_by_candidate(self):
        """按候选人分别聚合数据 - 新增功能"""
        print(f"📈 按候选人分别聚合数据，时间间隔: 每{self.hour_interval}小时...")

        # 分别处理Trump和Harris的数据
        trump_articles = self.articles_df[self.articles_df['data_source'] == 'Trump_Twitter'].copy()
        harris_articles = self.articles_df[self.articles_df['data_source'] == 'Harris_Twitter'].copy()

        trump_comments = self.comments_df[self.comments_df['data_source'] == 'Trump_Comments'].copy()
        harris_comments = self.comments_df[self.comments_df['data_source'] == 'Harris_Comments'].copy()

        # 处理Trump数据（支持Trump）
        trump_articles_subset = trump_articles[['datetime', 'stance_numeric', 'sentiment_numeric']].copy()
        trump_articles_subset['type'] = 'article'
        trump_comments_subset = trump_comments[['datetime', 'stance_numeric', 'sentiment_numeric']].copy()
        trump_comments_subset['type'] = 'comment'

        # 处理Harris数据（反对Trump，需要转换立场）
        harris_articles_subset = harris_articles[['datetime', 'stance_numeric', 'sentiment_numeric']].copy()
        harris_articles_subset['type'] = 'article'
        harris_comments_subset = harris_comments[['datetime', 'stance_numeric', 'sentiment_numeric']].copy()
        harris_comments_subset['type'] = 'comment'

        # 按时间间隔分组
        def group_by_time(df):
            if self.hour_interval == 1:
                df['time_group'] = df['datetime'].dt.floor('h')
            else:
                df['time_group'] = df['datetime'].dt.floor(f'{self.hour_interval}h')
            return df

        trump_articles_subset = group_by_time(trump_articles_subset)
        trump_comments_subset = group_by_time(trump_comments_subset)
        harris_articles_subset = group_by_time(harris_articles_subset)
        harris_comments_subset = group_by_time(harris_comments_subset)

        # 合并Trump数据
        trump_combined = pd.concat([trump_articles_subset, trump_comments_subset], ignore_index=True)

        # 合并Harris数据
        harris_combined = pd.concat([harris_articles_subset, harris_comments_subset], ignore_index=True)

        # 按时间间隔聚合Trump数据
        trump_stats = trump_combined.groupby('time_group').agg({
            'stance_numeric': 'mean',
            'sentiment_numeric': 'mean',
            'type': 'count'
        }).reset_index()
        trump_stats.columns = ['hour', 'avg_stance', 'avg_sentiment', 'total_count']

        # 分别统计Trump的文章和评论数量
        trump_article_counts = trump_articles_subset.groupby('time_group').size()
        trump_comment_counts = trump_comments_subset.groupby('time_group').size()
        trump_stats = trump_stats.set_index('hour')
        trump_stats['article_count'] = trump_article_counts
        trump_stats['comment_count'] = trump_comment_counts
        trump_stats = trump_stats.fillna(0).reset_index()

        # 按时间间隔聚合Harris数据
        harris_stats = harris_combined.groupby('time_group').agg({
            'stance_numeric': 'mean',
            'sentiment_numeric': 'mean',
            'type': 'count'
        }).reset_index()
        harris_stats.columns = ['hour', 'avg_stance', 'avg_sentiment', 'total_count']

        # 分别统计Harris的文章和评论数量
        harris_article_counts = harris_articles_subset.groupby('time_group').size()
        harris_comment_counts = harris_comments_subset.groupby('time_group').size()
        harris_stats = harris_stats.set_index('hour')
        harris_stats['article_count'] = harris_article_counts
        harris_stats['comment_count'] = harris_comment_counts
        harris_stats = harris_stats.fillna(0).reset_index()

        print(f"✅ Trump数据聚合完成，共 {len(trump_stats)} 个时间段")
        print(f"✅ Harris数据聚合完成，共 {len(harris_stats)} 个时间段")

        return trump_stats, harris_stats

    def plot_candidate_comparison_detailed(self):
        """绘制候选人对比的详细分析图 - 新增功能"""
        print("🎨 绘制候选人对比详细分析图...")

        # 获取按候选人分组的数据
        trump_data, harris_data = self.aggregate_hourly_data_by_candidate()

        # 创建详细对比图 - 2x2布局，每个子图包含两条线
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'USPE2024 候选人对比详细分析 (每{self.hour_interval}小时)', fontsize=16, fontweight='bold')

        # 颜色设置
        trump_color = '#FF6B6B'  # 红色系，代表Trump
        harris_color = '#4ECDC4'  # 蓝色系，代表Harris

        # 子图1: 立场对比
        if len(trump_data) > 0:
            ax1.plot(trump_data['hour'], trump_data['avg_stance'],
                     marker='o', linewidth=2.5, markersize=6,
                     label='Trump支持者', color=trump_color, alpha=0.8)

        if len(harris_data) > 0:
            ax1.plot(harris_data['hour'], harris_data['avg_stance'],
                     marker='s', linewidth=2.5, markersize=6,
                     label='Harris支持者', color=harris_color, alpha=0.8)

        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('立场(Stance)时间变化对比')
        ax1.set_ylabel('平均立场值')
        ax1.set_ylim(-1.1, 1.1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: 情感对比
        if len(trump_data) > 0:
            ax2.plot(trump_data['hour'], trump_data['avg_sentiment'],
                     marker='o', linewidth=2.5, markersize=6,
                     label='Trump支持者', color=trump_color, alpha=0.8)

        if len(harris_data) > 0:
            ax2.plot(harris_data['hour'], harris_data['avg_sentiment'],
                     marker='s', linewidth=2.5, markersize=6,
                     label='Harris支持者', color=harris_color, alpha=0.8)

        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('情感(Sentiment)时间变化对比')
        ax2.set_ylabel('平均情感值')
        ax2.set_ylim(-1.1, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 子图3: 文章数量对比
        if len(trump_data) > 0:
            ax3.plot(trump_data['hour'], trump_data['article_count'],
                     marker='o', linewidth=2.5, markersize=5,
                     label='Trump文章', color=trump_color, alpha=0.8)

        if len(harris_data) > 0:
            ax3.plot(harris_data['hour'], harris_data['article_count'],
                     marker='s', linewidth=2.5, markersize=5,
                     label='Harris文章', color=harris_color, alpha=0.8)

        ax3.set_title('文章数量时间变化对比')
        ax3.set_ylabel('文章数量')
        ax3.set_xlabel('时间')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 子图4: 评论数量对比
        if len(trump_data) > 0:
            ax4.plot(trump_data['hour'], trump_data['comment_count'],
                     marker='o', linewidth=2.5, markersize=5,
                     label='Trump评论', color=trump_color, alpha=0.8)

        if len(harris_data) > 0:
            ax4.plot(harris_data['hour'], harris_data['comment_count'],
                     marker='s', linewidth=2.5, markersize=5,
                     label='Harris评论', color=harris_color, alpha=0.8)

        ax4.set_title('评论数量时间变化对比')
        ax4.set_ylabel('评论数量')
        ax4.set_xlabel('时间')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 格式化所有子图的x轴
        all_hours = set()
        if len(trump_data) > 0:
            all_hours.update(trump_data['hour'])
        if len(harris_data) > 0:
            all_hours.update(harris_data['hour'])

        if all_hours:
            all_hours = sorted(all_hours)
            for ax in [ax1, ax2, ax3, ax4]:
                ax.tick_params(axis='x', rotation=0)
                tick_interval = max(1, len(all_hours) // 10)  # 最多显示10个刻度
                if len(all_hours) > tick_interval:
                    selected_hours = all_hours[::tick_interval]
                else:
                    selected_hours = all_hours
                ax.set_xticks(selected_hours)

                labels = []
                last_date = None
                for hour in selected_hours:
                    current_date = hour.strftime('%m-%d')
                    if current_date != last_date:
                        labels.append(current_date)
                        last_date = current_date
                    else:
                        labels.append('')
                ax.set_xticklabels(labels)

        plt.tight_layout()

        # 保存图表
        output_file = self.output_dir / f"USPE2024_candidate_comparison_detailed_{self.hour_interval}h.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 候选人对比详细分析图已保存至: {output_file}")

        # 打印对比统计
        self.print_candidate_comparison_statistics(trump_data, harris_data)

        return fig, trump_data, harris_data

    def print_candidate_comparison_statistics(self, trump_data, harris_data):
        """打印候选人对比统计信息"""
        print(f"\n📊 候选人对比统计摘要:")

        if len(trump_data) > 0:
            trump_avg_stance = trump_data['avg_stance'].mean()
            trump_avg_sentiment = trump_data['avg_sentiment'].mean()
            trump_total_count = trump_data['total_count'].sum()
            print(f"\nTrump支持者数据:")
            print(f"  平均立场: {trump_avg_stance:.3f}")
            print(f"  平均情感: {trump_avg_sentiment:.3f}")
            print(f"  总数量: {trump_total_count:.0f} 条")
            print(f"  时间段数: {len(trump_data)} 个")

        if len(harris_data) > 0:
            harris_avg_stance = harris_data['avg_stance'].mean()
            harris_avg_sentiment = harris_data['avg_sentiment'].mean()
            harris_total_count = harris_data['total_count'].sum()
            print(f"\nHarris支持者数据:")
            print(f"  平均立场: {harris_avg_stance:.3f}")
            print(f"  平均情感: {harris_avg_sentiment:.3f}")
            print(f"  总数量: {harris_total_count:.0f} 条")
            print(f"  时间段数: {len(harris_data)} 个")

        if len(trump_data) > 0 and len(harris_data) > 0:
            stance_diff = abs(trump_avg_stance - harris_avg_stance)
            sentiment_diff = abs(trump_avg_sentiment - harris_avg_sentiment)
            print(f"\n对比分析:")
            print(f"  立场差异: {stance_diff:.3f}")
            print(f"  情感差异: {sentiment_diff:.3f}")
            print(f"  数量比例 (Trump:Harris): {trump_total_count:.0f}:{harris_total_count:.0f}")

    def generate_candidate_comparison_report(self):
        """生成候选人对比分析报告 - 新增独立功能"""
        print("📋 生成USPE2024候选人对比分析报告...")

        # 生成候选人对比详细分析图
        fig, trump_data, harris_data = self.plot_candidate_comparison_detailed()

        print("✅ USPE2024候选人对比分析报告生成完成!")

        return fig, trump_data, harris_data

    def generate_comparison_report(self, simulation_file_path, photo_name="USPE2024_comparison_real_vs_simulation.png"):
        """生成对比分析报告 - 模仿XMSU7D的对比分析"""
        print("📋 生成对比分析报告...")

        # 直接生成对比分析，不重复生成基础图表
        comparison_result = self.plot_comparison_analysis(simulation_file_path, photo_name)

        if comparison_result is None:
            print("❌ 对比分析失败")
            return None, None, None

        comparison_fig, real_data, sim_data = comparison_result

        print("✅ 对比分析报告生成完成!")

        return None, real_data, sim_data


def main(hour_interval=24, start_date=None, end_date=None, simulation_file=None, photo_name="USPE2024_comparison_real_vs_simulation.png", include_candidate_comparison=False):
    """主函数 - 模仿XMSU7D的main函数结构"""
    # 创建分析工具
    analyzer = USPE2024UnifiedAnalyzer(hour_interval=hour_interval)

    result_data = {}

    if simulation_file:
        # 生成对比分析报告
        print("📊 生成包含模拟数据对比的报告...")
        hourly_data, real_data, sim_data = analyzer.generate_comparison_report(simulation_file, photo_name)
        result_data = {'hourly_data': hourly_data, 'real_data': real_data, 'sim_data': sim_data}
    else:
        # 生成常规报告
        print("📊 生成常规分析报告...")
        hourly_data = analyzer.generate_report()
        result_data['unified_data'] = hourly_data

    # 如果需要生成候选人对比分析
    if include_candidate_comparison:
        print("📊 生成候选人对比分析报告...")
        comparison_fig, trump_data, harris_data = analyzer.generate_candidate_comparison_report()
        result_data['candidate_comparison'] = {
            'figure': comparison_fig,
            'trump_data': trump_data,
            'harris_data': harris_data
        }

    return result_data


if __name__ == "__main__":
    # 可以通过修改这个参数来设置时间间隔
    hour_interval = 72  # 默认每72小时统计

    # 模拟数据文件路径（如果需要对比分析）
    simulation_file = None  # 暂时没有USPE2024的模拟数据

    # 是否包含候选人对比分析
    include_candidate_comparison = True  # 设置为True来生成候选人对比图

    print(f"🚀 开始USPE2024统一数据集分析，时间间隔: 每{hour_interval}小时")

    # 如果存在模拟数据文件则进行对比分析
    if simulation_file and Path(simulation_file).exists():
        print("📊 发现模拟数据文件，将进行对比分析")
        result_data = main(hour_interval=hour_interval, simulation_file=simulation_file,
                           photo_name="USPE2024_comparison_real_vs_simulation.png",
                           include_candidate_comparison=include_candidate_comparison)
    else:
        print("📊 进行USPE2024统一数据集常规分析")
        result_data = main(hour_interval=hour_interval,
                           include_candidate_comparison=include_candidate_comparison)

    # 关闭图表以释放内存
    plt.close('all')
