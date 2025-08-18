"""
数据排序工具

提供多种排序选项来重新整理已整合的数据
"""

import pandas as pd
from pathlib import Path
import argparse
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataSorter:
    """数据排序器"""

    def __init__(self, data_path: str):
        """
        初始化数据排序器

        Args:
            data_path: 整合数据的路径
        """
        self.data_path = Path(data_path)

    def load_data(self, file_prefix: str):
        """加载数据"""
        articles_file = self.data_path / f"{file_prefix}_integrated_articles.csv"
        comments_file = self.data_path / f"{file_prefix}_integrated_comments.csv"

        articles_df = None
        comments_df = None

        if articles_file.exists():
            articles_df = pd.read_csv(articles_file, encoding='utf-8')
            logger.info(f"已加载文章数据: {len(articles_df)} 条")

        if comments_file.exists():
            comments_df = pd.read_csv(comments_file, encoding='utf-8')
            logger.info(f"已加载评论数据: {len(comments_df)} 条")

        return articles_df, comments_df

    def sort_by_time(self, df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
        """按时间排序"""
        try:
            df_copy = df.copy()
            df_copy['created_time_numeric'] = pd.to_numeric(df_copy['created_time'], errors='coerce')

            # 分离有效和无效时间的数据
            valid_mask = ~df_copy['created_time_numeric'].isna()
            df_valid = df_copy[valid_mask].sort_values('created_time_numeric', ascending=ascending)
            df_invalid = df_copy[~valid_mask]

            # 合并数据
            if ascending:
                result = pd.concat([df_valid, df_invalid], ignore_index=True)
            else:
                result = pd.concat([df_valid, df_invalid], ignore_index=True)

            result = result.drop(columns=['created_time_numeric'])

            order_text = "升序（最早的在前）" if ascending else "降序（最新的在前）"
            logger.info(f"按时间{order_text}排序完成")
            return result

        except Exception as e:
            logger.error(f"时间排序失败: {e}")
            return df

    def sort_by_engagement(self, df: pd.DataFrame, metric: str = 'like_count', ascending: bool = False) -> pd.DataFrame:
        """按互动指标排序"""
        try:
            df_copy = df.copy()
            df_copy[f'{metric}_numeric'] = pd.to_numeric(df_copy[metric], errors='coerce').fillna(0)
            result = df_copy.sort_values(f'{metric}_numeric', ascending=ascending)
            result = result.drop(columns=[f'{metric}_numeric'])

            order_text = "升序" if ascending else "降序"
            logger.info(f"按{metric}{order_text}排序完成")
            return result

        except Exception as e:
            logger.error(f"按{metric}排序失败: {e}")
            return df

    def sort_by_platform(self, df: pd.DataFrame) -> pd.DataFrame:
        """按平台排序"""
        try:
            # 定义平台优先级
            platform_order = ['douyin', 'weibo', 'weixin', 'xhs', 'zhihu']
            df_copy = df.copy()

            # 创建平台排序键
            df_copy['platform_order'] = df_copy['platform'].map(
                {platform: i for i, platform in enumerate(platform_order)}
            ).fillna(999)

            result = df_copy.sort_values('platform_order')
            result = result.drop(columns=['platform_order'])

            logger.info("按平台排序完成")
            return result

        except Exception as e:
            logger.error(f"平台排序失败: {e}")
            return df

    def sort_by_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """按情感倾向排序"""
        try:
            # 定义情感排序优先级：积极 -> 中立 -> 消极
            sentiment_order = {'积极': 0, '中立': 1, '消极': 2}
            df_copy = df.copy()

            df_copy['sentiment_order'] = df_copy['sentiment'].map(sentiment_order).fillna(999)
            result = df_copy.sort_values('sentiment_order')
            result = result.drop(columns=['sentiment_order'])

            logger.info("按情感倾向排序完成")
            return result

        except Exception as e:
            logger.error(f"情感排序失败: {e}")
            return df

    def sort_by_stance(self, df: pd.DataFrame) -> pd.DataFrame:
        """按立场排序"""
        try:
            # 定义立场排序优先级：支持 -> 中立 -> 批判
            stance_order = {'支持': 0, '中立': 1, '批判': 2}
            df_copy = df.copy()

            df_copy['stance_order'] = df_copy['stance'].map(stance_order).fillna(999)
            result = df_copy.sort_values('stance_order')
            result = result.drop(columns=['stance_order'])

            logger.info("按立场排序完成")
            return result

        except Exception as e:
            logger.error(f"立场排序失败: {e}")
            return df

    def save_sorted_data(self, articles_df: pd.DataFrame, comments_df: pd.DataFrame,
                         output_prefix: str, sort_method: str):
        """保存排序后的数据"""
        if articles_df is not None and not articles_df.empty:
            articles_file = self.data_path / f"{output_prefix}_articles_sorted_by_{sort_method}.csv"
            articles_df.to_csv(articles_file, index=False, encoding='utf-8-sig')
            logger.info(f"排序后的文章数据已保存到: {articles_file}")

        if comments_df is not None and not comments_df.empty:
            comments_file = self.data_path / f"{output_prefix}_comments_sorted_by_{sort_method}.csv"
            comments_df.to_csv(comments_file, index=False, encoding='utf-8-sig')
            logger.info(f"排序后的评论数据已保存到: {comments_file}")

    def sort_data(self, file_prefix: str, sort_method: str, **kwargs):
        """执行数据排序"""
        logger.info(f"开始对 {file_prefix} 数据执行 {sort_method} 排序")

        # 加载数据
        articles_df, comments_df = self.load_data(file_prefix)

        # 根据排序方法执行排序
        if sort_method == 'time_desc':
            if articles_df is not None:
                articles_df = self.sort_by_time(articles_df, ascending=False)
            if comments_df is not None:
                comments_df = self.sort_by_time(comments_df, ascending=False)

        elif sort_method == 'time_asc':
            if articles_df is not None:
                articles_df = self.sort_by_time(articles_df, ascending=True)
            if comments_df is not None:
                comments_df = self.sort_by_time(comments_df, ascending=True)

        elif sort_method == 'like_count':
            metric = kwargs.get('metric', 'like_count')
            ascending = kwargs.get('ascending', False)
            if articles_df is not None:
                articles_df = self.sort_by_engagement(articles_df, metric, ascending)
            if comments_df is not None:
                comments_df = self.sort_by_engagement(comments_df, metric, ascending)

        elif sort_method == 'platform':
            if articles_df is not None:
                articles_df = self.sort_by_platform(articles_df)
            if comments_df is not None:
                comments_df = self.sort_by_platform(comments_df)

        elif sort_method == 'sentiment':
            if articles_df is not None:
                articles_df = self.sort_by_sentiment(articles_df)
            if comments_df is not None:
                comments_df = self.sort_by_sentiment(comments_df)

        elif sort_method == 'stance':
            if articles_df is not None:
                articles_df = self.sort_by_stance(articles_df)
            if comments_df is not None:
                comments_df = self.sort_by_stance(comments_df)

        else:
            logger.error(f"不支持的排序方法: {sort_method}")
            return

        # 保存排序后的数据
        self.save_sorted_data(articles_df, comments_df, file_prefix, sort_method)
        logger.info(f"{sort_method} 排序完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据排序工具')
    parser.add_argument('--data_path', default='Data/integrated_data', help='数据路径')
    parser.add_argument('--file_prefix', default='XMSU7D', help='文件前缀')
    parser.add_argument('--sort_method', required=True,
                        choices=['time_desc', 'time_asc', 'like_count', 'platform', 'sentiment', 'stance'],
                        help='排序方法')
    parser.add_argument('--metric', default='like_count', help='排序指标（用于engagement排序）')
    parser.add_argument('--ascending', action='store_true', help='升序排序')

    args = parser.parse_args()

    # 创建排序器
    sorter = DataSorter(args.data_path)

    # 执行排序
    kwargs = {}
    if args.sort_method == 'like_count':
        kwargs['metric'] = args.metric
        kwargs['ascending'] = args.ascending

    sorter.sort_data(args.file_prefix, args.sort_method, **kwargs)


def demo_all_sorts():
    """演示所有排序方法"""
    logger.info("开始演示所有排序方法")

    sorter = DataSorter("Data/integrated_data")

    # 时间排序（最新在前）
    sorter.sort_data("XMSU7D", "time_desc")

    # 时间排序（最早在前）
    sorter.sort_data("XMSU7D", "time_asc")

    # 按点赞数排序（最多在前）
    sorter.sort_data("XMSU7D", "like_count", metric="like_count", ascending=False)

    # 按评论数排序（最多在前）
    sorter.sort_data("XMSU7D", "like_count", metric="comment_count", ascending=False)

    # 按平台排序
    sorter.sort_data("XMSU7D", "platform")

    # 按情感倾向排序
    sorter.sort_data("XMSU7D", "sentiment")

    # 按立场排序
    sorter.sort_data("XMSU7D", "stance")

    logger.info("所有排序演示完成")


if __name__ == "__main__":
    # 如果没有命令行参数，运行演示
    import sys
    if len(sys.argv) == 1:
        demo_all_sorts()
    else:
        main()
