#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
态度情感时间分析工具
分析2025-04-01到2025-04-05期间的stance和sentiment变化
按小时计算平均值和总数
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Analyzer:
    def __init__(self, data_dir="Data/XMSU7D/integrated_data", hour_interval=1, start_date=None, end_date=None):
        """初始化情感分析工具"""
        self.data_dir = Path(data_dir)
        self.articles_file = self.data_dir / "XMSU7D_integrated_articles.csv"
        self.comments_file = self.data_dir / "XMSU7D_integrated_comments.csv"
        self.output_dir = Path("Data/visualizations")
        self.output_dir.mkdir(exist_ok=True)

        # 时间间隔设置
        self.hour_interval = hour_interval
        print(f"⏰ 时间统计间隔: 每{hour_interval}小时")

        # 定义映射字典
        self.stance_mapping = {"支持": 1, "中立": 0, "批判": -1}
        self.sentiment_mapping = {"积极": 1, "中立": 0, "消极": -1}

        # 加载数据
        self.load_data(start_date, end_date)

    def load_data(self, start_date_: str = None, end_date_: str = None):
        """加载并预处理数据"""
        print("📊 正在加载数据...")

        # 加载文章数据
        self.articles_df = pd.read_csv(self.articles_file, encoding='utf-8-sig')
        print(f"✅ 加载文章数据: {len(self.articles_df)} 条")

        # 加载评论数据
        self.comments_df = pd.read_csv(self.comments_file, encoding='utf-8-sig')
        print(f"✅ 加载评论数据: {len(self.comments_df)} 条")

        # 转换时间列
        self.articles_df['datetime'] = pd.to_datetime(self.articles_df['created_date'])
        self.comments_df['datetime'] = pd.to_datetime(self.comments_df['created_date'])

        # 过滤时间范围：2025-04-01 到 2025-04-20
        # start_date = pd.to_datetime('2025-04-01')
        # end_date = pd.to_datetime('2025-04-20')  # 不包含边界
        start_date = pd.to_datetime(start_date_) if start_date_ else self.articles_df['datetime'].min()
        end_date = pd.to_datetime(end_date_) if end_date_ else self.articles_df['datetime'].max()

        self.articles_df = self.articles_df[
            (self.articles_df['datetime'] >= start_date) &
            (self.articles_df['datetime'] < end_date)
        ].copy()

        self.comments_df = self.comments_df[
            (self.comments_df['datetime'] >= start_date) &
            (self.comments_df['datetime'] < end_date)
        ].copy()

        print(f"📅 过滤后文章数据: {len(self.articles_df)} 条")
        print(f"📅 过滤后评论数据: {len(self.comments_df)} 条")

        # 转换stance和sentiment为数值
        self.convert_categorical_to_numeric()

    def convert_categorical_to_numeric(self):
        """将分类变量转换为数值"""
        print("🔄 转换分类变量为数值...")

        # 处理文章数据
        self.articles_df['stance_numeric'] = self.articles_df['stance'].map(self.stance_mapping)
        self.articles_df['sentiment_numeric'] = self.articles_df['sentiment'].map(self.sentiment_mapping)

        # 处理评论数据
        self.comments_df['stance_numeric'] = self.comments_df['stance'].map(self.stance_mapping)
        self.comments_df['sentiment_numeric'] = self.comments_df['sentiment'].map(self.sentiment_mapping)

        # 处理缺失值，填充为0（中立）
        self.articles_df['stance_numeric'] = self.articles_df['stance_numeric'].fillna(0)
        self.articles_df['sentiment_numeric'] = self.articles_df['sentiment_numeric'].fillna(0)
        self.comments_df['stance_numeric'] = self.comments_df['stance_numeric'].fillna(0)
        self.comments_df['sentiment_numeric'] = self.comments_df['sentiment_numeric'].fillna(0)

        print("✅ 分类变量转换完成")

    def aggregate_hourly_data(self):
        """按设定的时间间隔聚合数据"""
        print(f"📈 按{self.hour_interval}小时间隔聚合数据...")

        # 合并文章和评论数据
        articles_subset = self.articles_df[['datetime', 'stance_numeric', 'sentiment_numeric']].copy()
        articles_subset['type'] = 'article'

        comments_subset = self.comments_df[['datetime', 'stance_numeric', 'sentiment_numeric']].copy()
        comments_subset['type'] = 'comment'

        # 使用更简单的时间分组方法
        if self.hour_interval == 1:
            # 1小时间隔使用小时取整
            articles_subset['time_group'] = articles_subset['datetime'].dt.floor('h')
            comments_subset['time_group'] = comments_subset['datetime'].dt.floor('h')
        else:
            # 多小时间隔使用自定义分组
            articles_subset['time_group'] = articles_subset['datetime'].dt.floor(f'{self.hour_interval}h')
            comments_subset['time_group'] = comments_subset['datetime'].dt.floor(f'{self.hour_interval}h')

        # 合并数据
        combined_df = pd.concat([articles_subset, comments_subset], ignore_index=True)

        # 按时间间隔聚合
        hourly_stats = combined_df.groupby('time_group').agg({
            'stance_numeric': 'mean',
            'sentiment_numeric': 'mean',
            'type': 'count'  # 总数
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

        # 调试信息：显示前几个数据点
        print("📋 前5个时间段的数据:")
        print(hourly_stats[['hour', 'article_count', 'comment_count', 'total_count']].head())

        return hourly_stats

    def plot_sentiment_timeline(self):
        """绘制情感态度时间折线图"""
        print("🎨 绘制情感态度时间折线图...")

        # 获取小时级数据
        hourly_data = self.aggregate_hourly_data()

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'2025-04-01至04-05 社交媒体情感态度分析 (每{self.hour_interval}小时)', fontsize=16, fontweight='bold')

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
        ax1.text(-0.05, 0, '批判/消极', transform=ax1.transAxes,
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

        # 格式化x轴时间显示
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=0)  # 不需要旋转
            # 根据时间间隔调整刻度显示密度
            tick_interval = max(1, 12 // self.hour_interval)  # 保证至少每12小时显示一次
            selected_hours = hourly_data['hour'][::tick_interval]
            ax.set_xticks(selected_hours)

            # 只显示日期，相同日期不重复
            labels = []
            last_date = None
            for hour in selected_hours:
                current_date = hour.strftime('%m-%d')

                if current_date != last_date:
                    # 新的日期，显示日期
                    labels.append(current_date)
                    last_date = current_date
                else:
                    # 相同日期，显示空字符串
                    labels.append('')

            ax.set_xticklabels(labels)

        plt.tight_layout()

        # 保存图表
        output_file = self.output_dir / f"timeline_analysis_{self.hour_interval}h.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 图表已保存至: {output_file}")

        # 显示统计信息
        self.print_statistics(hourly_data)

        return fig, hourly_data

    def plot_detailed_breakdown(self):
        """绘制详细分解图"""
        print("🎨 绘制详细分解图...")

        hourly_data = self.aggregate_hourly_data()

        # 创建更详细的图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'2025-04-01至04-05 详细分析 (每{self.hour_interval}小时)', fontsize=16, fontweight='bold')

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
            ax.tick_params(axis='x', rotation=0)  # 不需要旋转
            # 根据时间间隔调整刻度显示密度
            tick_interval = max(1, 12 // self.hour_interval)  # 保证至少每12小时显示一次
            selected_hours = hourly_data['hour'][::tick_interval]
            ax.set_xticks(selected_hours)

            # 只显示日期，相同日期不重复
            labels = []
            last_date = None
            for hour in selected_hours:
                current_date = hour.strftime('%m-%d')

                if current_date != last_date:
                    # 新的日期，显示日期
                    labels.append(current_date)
                    last_date = current_date
                else:
                    # 相同日期，显示空字符串
                    labels.append('')

            ax.set_xticklabels(labels)

        plt.tight_layout()

        # 保存图表
        output_file = self.output_dir / f"detailed_analysis_{self.hour_interval}h.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 详细分析图已保存至: {output_file}")

    def print_statistics(self, hourly_data):
        """打印统计信息"""
        print(f"\n📊 数据统计摘要:")
        print(f"时间范围: 2025-04-01 至 2025-04-10")
        print(f"时间间隔: 每{self.hour_interval}小时")
        print(f"数据点数量: {len(hourly_data)} 个")
        print(f"总帖子+评论: {hourly_data['total_count'].sum():.0f} 条")
        print(f"平均每{self.hour_interval}小时: {hourly_data['total_count'].mean():.1f} 条")

        # 立场统计
        avg_stance = hourly_data['avg_stance'].mean()
        print(f"\n立场分析:")
        print(f"  整体平均立场: {avg_stance:.3f}")
        if avg_stance > 0.1:
            print(f"  总体偏向: 支持")
        elif avg_stance < -0.1:
            print(f"  总体偏向: 批判")
        else:
            print(f"  总体偏向: 中立")

        # 情感统计
        avg_sentiment = hourly_data['avg_sentiment'].mean()
        print(f"\n情感分析:")
        print(f"  整体平均情感: {avg_sentiment:.3f}")
        if avg_sentiment > 0.1:
            print(f"  总体偏向: 积极")
        elif avg_sentiment < -0.1:
            print(f"  总体偏向: 消极")
        else:
            print(f"  总体偏向: 中立")

        # 峰值分析
        peak_hour = hourly_data.loc[hourly_data['total_count'].idxmax(), 'hour']
        peak_count = hourly_data['total_count'].max()
        print(f"\n活跃度峰值: {peak_hour} ({peak_count:.0f}条)")

        # 立场和情感的极值
        max_stance_hour = hourly_data.loc[hourly_data['avg_stance'].idxmax(), 'hour']
        max_stance_val = hourly_data['avg_stance'].max()
        min_stance_hour = hourly_data.loc[hourly_data['avg_stance'].idxmin(), 'hour']
        min_stance_val = hourly_data['avg_stance'].min()

        print(f"\n立场极值:")
        print(f"  最支持时间: {max_stance_hour} (值: {max_stance_val:.3f})")
        print(f"  最批判时间: {min_stance_hour} (值: {min_stance_val:.3f})")

    def load_simulation_data(self, simulation_file_path):
        """加载时间序列模拟数据"""
        print("📊 正在加载模拟数据...")

        try:
            sim_df = pd.read_csv(simulation_file_path, encoding='utf-8-sig')
            sim_df['datetime'] = pd.to_datetime(sim_df['datetime'])
            print(f"✅ 加载模拟数据: {len(sim_df)} 个时间点")

            # 显示模拟数据的时间范围
            print(f"📅 模拟数据时间范围: {sim_df['datetime'].min()} 至 {sim_df['datetime'].max()}")

            return sim_df
        except Exception as e:
            print(f"❌ 加载模拟数据失败: {e}")
            return None

    def plot_comparison_analysis(self, simulation_file_path, photo_name="comparison_analysis.png"):
        """绘制真实数据与模拟数据的立场/情感对比分析 - 模仿原图风格"""
        print("🎨 绘制真实数据与模拟数据对比图...")

        # 加载模拟数据
        sim_data = self.load_simulation_data(simulation_file_path)
        if sim_data is None:
            return None

        # 获取真实数据的小时级聚合
        real_hourly_data = self.aggregate_hourly_data()

        # 创建对比图 - 模仿原图的风格
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('时间序列用户立场情感变化分析 - 真实数据 vs 模拟数据对比',
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
        ax1.text(-0.08, 0, '批判', transform=ax1.transAxes,
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
        import matplotlib.dates as mdates
        from datetime import datetime

        for ax in [ax1, ax2]:
            ax.set_xlabel('时间间隔时间(小时)', fontsize=12)
            # 设置x轴时间格式
            if hasattr(real_hourly_data['hour'].iloc[0], 'strftime'):
                # 如果是datetime对象，设置时间格式
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
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

    def print_comparison_statistics(self, real_data, sim_data):
        """打印真实数据与模拟数据的对比统计"""
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
        sim_avg_count = sim_data['participant_count'].mean()

        print(f"\n参与度对比:")
        print(f"  真实数据平均数量: {real_avg_count:.1f} 条/时间段")
        print(f"  模拟数据平均参与: {sim_avg_count:.1f} 人/时间段")

        # 相关性分析（如果时间点匹配的话）
        try:
            # 尝试对齐时间点进行相关性分析
            # 简化时间对齐：按小时取整
            real_data_aligned = real_data.copy()
            real_data_aligned['hour_rounded'] = real_data_aligned['hour'].dt.floor('h')

            sim_data_aligned = sim_data.copy()
            sim_data_aligned['hour_rounded'] = sim_data_aligned['datetime'].dt.floor('h')

            # 合并数据
            merged = pd.merge(real_data_aligned, sim_data_aligned,
                              left_on='hour_rounded', right_on='hour_rounded',
                              how='inner')

            if len(merged) > 5:  # 至少需要5个匹配点才进行相关性分析
                stance_corr = np.corrcoef(merged['avg_stance'], merged['user_stance'])[0, 1]
                sentiment_corr = np.corrcoef(merged['avg_sentiment'], merged['user_sentiment'])[0, 1]

                print(f"\n相关性分析 (匹配的{len(merged)}个时间点):")
                print(f"  立场相关性: r={stance_corr:.3f}")
                print(f"  情感相关性: r={sentiment_corr:.3f}")
            else:
                print(f"\n相关性分析: 匹配时间点太少({len(merged)}个)，无法进行相关性分析")
        except Exception as e:
            print(f"\n相关性分析: 无法计算 ({e})")

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
        """生成完整分析报告"""
        print("📋 生成情感态度分析报告...")

        # 主要折线图
        fig1, hourly_data = self.plot_sentiment_timeline()

        # 详细分解图
        fig2 = self.plot_detailed_breakdown()

        print("✅ 情感态度分析报告生成完成!")

        return hourly_data

    def generate_comparison_report(self, simulation_file_path, photo_name="comparison_analysis.png"):
        """生成对比分析报告 - 只生成对比图，不重复生成基础图"""
        print("📋 生成对比分析报告...")

        # 直接生成对比分析，不重复生成基础图表
        comparison_result = self.plot_comparison_analysis(simulation_file_path, photo_name)

        if comparison_result is None:
            print("❌ 对比分析失败")
            return None, None, None

        comparison_fig, real_data, sim_data = comparison_result

        print("✅ 对比分析报告生成完成!")

        return None, real_data, sim_data  # 不返回基础报告数据


def main(hour_interval=1, start_date=None, end_date=None, simulation_file=None, photo_name="comparison_analysis.png"):
    """主函数"""
    # 创建分析工具
    analyzer = Analyzer(hour_interval=hour_interval, start_date=start_date, end_date=end_date)

    if simulation_file:
        # 生成对比分析报告
        print("📊 生成包含模拟数据对比的报告...")
        hourly_data, real_data, sim_data = analyzer.generate_comparison_report(simulation_file, photo_name)
        result_data = {'hourly_data': hourly_data, 'real_data': real_data, 'sim_data': sim_data}
    else:
        # 生成常规报告
        print("📊 生成常规分析报告...")
        hourly_data = analyzer.generate_report()
        result_data = hourly_data

    # 关闭图表以释放内存
    plt.close('all')

    return result_data


if __name__ == "__main__":
    # 可以通过修改这个参数来设置时间间隔
    # 例如: hour_interval=3 表示每3小时统计一次
    hour_interval = 6  # 默认每小时统计
    start_date = '2025-04-01'
    end_date = '2025-04-20'

    # 模拟数据文件路径（如果需要对比分析）
    simulation_file = "Output/timeseries/timeseries_sim_20250907_100915/plot_data_points.csv"

    print(f"🚀 开始分析，时间间隔: 每{hour_interval}小时")

    # 如果存在模拟数据文件则进行对比分析
    if Path(simulation_file).exists():
        print("📊 发现模拟数据文件，将进行对比分析")
        result_data = main(hour_interval=hour_interval, start_date=start_date,
                           end_date=end_date, simulation_file=simulation_file, photo_name="comparison_analysis.png")
    else:
        print("📊 未发现模拟数据文件，进行常规分析")
        result_data = main(hour_interval=hour_interval, start_date=start_date, end_date=end_date)
