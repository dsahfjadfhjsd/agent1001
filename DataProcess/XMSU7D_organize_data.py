"""
多平台数据整合脚本

将不同平台（抖音、微博、小红书、知乎、微信）的原始数据整合成统一格式
支持文章数据和评论数据的分别处理
"""

import pandas as pd
import os
from pathlib import Path
import json
from typing import List, Dict, Optional
import logging
from datetime import datetime
import hashlib

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataOrganizer:
    """数据整合器"""

    def __init__(self, data_root_path: str):
        """
        初始化数据整合器

        Args:
            data_root_path: 数据根目录路径
        """
        self.data_root_path = Path(data_root_path)
        self.platforms = ['douyin', 'weibo', 'weixin', 'xhs', 'zhihu']

        # 用于存储article_id到post_id的映射
        self.article_id_mapping = {}

        # 定义需要保留的重要字段
        self.article_fields = [
            'post_id', 'content', 'created_time', 'created_date', 'like_count',
            'comment_count', 'share_count', 'uid', 'username',
            'location', 'stance', 'sentiment', 'intent',
            'stance_content', 'sentiment_content', 'intent_content',
            'video_urls', 'img_urls', 'platform', 'original_article_id'
        ]

        self.comment_fields = [
            'comment_id', 'content', 'post_id', 'created_time', 'created_date',
            'like_count', 'parent_comment_id', 'reply_comment',
            'location', 'stance', 'sentiment', 'intent',
            'stance_content', 'sentiment_content', 'intent_content',
            'video_urls', 'img_urls', 'platform', 'original_article_id'
        ]

    def find_data_folders(self) -> List[Path]:
        """查找所有数据文件夹"""
        data_folders = []
        for item in self.data_root_path.iterdir():
            if item.is_dir():
                data_folders.append(item)
        return data_folders

    def generate_post_id(self, platform: str, article_id: str) -> str:
        """生成统一的post_id格式"""
        try:
            # 使用平台和原始ID生成6位哈希
            combined = f"{platform}_{article_id}"
            hash_object = hashlib.md5(combined.encode())
            hex_dig = hash_object.hexdigest()
            # 取前6位作为post_id
            post_id = f"post_{hex_dig[:6]}"
            return post_id
        except Exception as e:
            logger.warning(f"生成post_id失败: {e}")
            # 备用方案：使用平台前缀 + 哈希
            backup_hash = str(hash(combined))[-6:]
            return f"post_{backup_hash}"

    def safe_get_field(self, row: pd.Series, field: str, default='') -> str:
        """安全获取字段值"""
        try:
            if field in row.index:
                value = row[field]
                if pd.isna(value):
                    return default
                # 处理列表格式的字段
                if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                    return value
                return str(value)
            return default
        except Exception as e:
            logger.warning(f"获取字段 {field} 时出错: {e}")
            return default

    def convert_timestamp_to_date(self, timestamp_str: str) -> str:
        """将时间戳转换为日期格式(YYYY-MM-DD HH:mm)"""
        try:
            if pd.isna(timestamp_str) or timestamp_str == '' or timestamp_str == '0':
                return ''

            # 转换为整数时间戳
            timestamp = int(float(timestamp_str))

            # 转换为datetime对象
            dt = datetime.fromtimestamp(timestamp)

            # 返回YYYY-MM-DD HH:mm格式（精确到小时和分钟）
            return dt.strftime('%Y-%m-%d %H:%M')
        except Exception as e:
            logger.warning(f"时间戳转换失败: {timestamp_str}, 错误: {e}")
            return ''

    def sort_by_time(self, df: pd.DataFrame, time_column: str) -> pd.DataFrame:
        """按时间排序数据框"""
        try:
            # 转换时间列为数值类型进行排序
            df_copy = df.copy()
            df_copy[f'{time_column}_numeric'] = pd.to_numeric(df_copy[time_column], errors='coerce')

            # 按时间降序排序（最新的在前）
            df_sorted = df_copy.sort_values(f'{time_column}_numeric')

            # 将无效时间的行放到最后
            valid_time_mask = ~df_sorted[f'{time_column}_numeric'].isna()
            df_valid = df_sorted[valid_time_mask]
            df_invalid = df_sorted[~valid_time_mask]
            df_sorted = pd.concat([df_valid, df_invalid], ignore_index=True)

            # 删除临时排序列
            df_sorted = df_sorted.drop(columns=[f'{time_column}_numeric'])

            logger.info(f"数据已按 {time_column} 排序完成")
            return df_sorted

        except Exception as e:
            logger.warning(f"时间排序失败: {e}，返回原始数据")
            return df

    def process_article_data(self, platform: str, df: pd.DataFrame) -> pd.DataFrame:
        """处理单个平台的文章数据"""
        logger.info(f"处理 {platform} 平台的文章数据，共 {len(df)} 条记录")

        # 创建标准化的数据结构
        processed_data = []

        for _, row in df.iterrows():
            original_article_id = self.safe_get_field(row, 'article_id')
            created_time = self.safe_get_field(row, 'created_time')

            # 生成统一的post_id
            post_id = self.generate_post_id(platform, original_article_id)

            # 存储映射关系
            self.article_id_mapping[original_article_id] = post_id

            article_data = {
                'platform': platform,
                'post_id': post_id,
                'original_article_id': original_article_id,
                'content': self.safe_get_field(row, 'content'),
                'created_time': created_time,
                'created_date': self.convert_timestamp_to_date(created_time),
                'like_count': self.safe_get_field(row, 'like_count', '0'),
                'comment_count': self.safe_get_field(row, 'comment_count', '0'),
                'share_count': self.safe_get_field(row, 'share_count', '0'),
                'uid': self.safe_get_field(row, 'uid'),
                'username': self.safe_get_field(row, 'username'),
                'location': self.safe_get_field(row, 'location'),
                'stance': self.safe_get_field(row, 'stance'),
                'sentiment': self.safe_get_field(row, 'sentiment'),
                'intent': self.safe_get_field(row, 'intent'),
                'stance_content': self.safe_get_field(row, 'stance_content'),
                'sentiment_content': self.safe_get_field(row, 'sentiment_content'),
                'intent_content': self.safe_get_field(row, 'intent_content'),
                'video_urls': self.safe_get_field(row, 'video_urls'),
                'img_urls': self.safe_get_field(row, 'img_urls')
            }

            # 平台特殊字段处理
            # if platform == 'douyin':
            # article_data['reposts_count'] = self.safe_get_field(row, 'share_count', '0')
            # article_data['collect_count'] = self.safe_get_field(row, 'collect_count', '0')
            # elif platform == 'weibo':
            #     article_data['reposts_count'] = self.safe_get_field(row, 'reposts_count', '0')
            # elif platform == 'xhs':
            if platform == 'xhs':
                # article_data['collected_count'] = self.safe_get_field(row, 'collected_count', '0')
                article_data['title'] = self.safe_get_field(row, 'title')
            elif platform == 'zhihu':
                article_data['title'] = self.safe_get_field(row, 'title')
                article_data['question_id'] = self.safe_get_field(row, 'question_id')
                article_data['answer_id'] = self.safe_get_field(row, 'answer_id')

            processed_data.append(article_data)

        return pd.DataFrame(processed_data)

    def process_comment_data(self, platform: str, df: pd.DataFrame) -> pd.DataFrame:
        """处理单个平台的评论数据"""
        logger.info(f"处理 {platform} 平台的评论数据，共 {len(df)} 条记录")

        processed_data = []

        for _, row in df.iterrows():
            original_article_id = self.safe_get_field(row, 'article_id')
            created_time = self.safe_get_field(row, 'created_time')

            # 查找对应的post_id，如果没有找到则生成一个
            post_id = self.article_id_mapping.get(original_article_id)
            if not post_id:
                post_id = self.generate_post_id(platform, original_article_id)
                self.article_id_mapping[original_article_id] = post_id
                logger.warning(f"评论引用了未处理的文章ID: {original_article_id}, 生成post_id: {post_id}")

            comment_data = {
                'platform': platform,
                'comment_id': self.safe_get_field(row, 'comment_id'),
                'content': self.safe_get_field(row, 'content'),
                'post_id': post_id,
                'original_article_id': original_article_id,
                'created_time': created_time,
                'created_date': self.convert_timestamp_to_date(created_time),
                'like_count': self.safe_get_field(row, 'like_count', '0'),
                'parent_comment_id': self.safe_get_field(row, 'parent_comment_id', '0'),
                'reply_comment': self.safe_get_field(row, 'reply_comment'),
                'location': self.safe_get_field(row, 'location'),
                'stance': self.safe_get_field(row, 'stance'),
                'sentiment': self.safe_get_field(row, 'sentiment'),
                'intent': self.safe_get_field(row, 'intent'),
                'stance_content': self.safe_get_field(row, 'stance_content'),
                'sentiment_content': self.safe_get_field(row, 'sentiment_content'),
                'intent_content': self.safe_get_field(row, 'intent_content'),
                'video_urls': '',  # 评论通常不包含视频
                'img_urls': self.safe_get_field(row, 'img_urls')
            }

            # 平台特殊字段处理
            if platform == 'douyin':
                comment_data['uid'] = self.safe_get_field(row, 'uid')
                comment_data['username'] = self.safe_get_field(row, 'username')
                comment_data['level'] = self.safe_get_field(row, 'level', '1')
            elif platform == 'weibo':
                comment_data['uid'] = self.safe_get_field(row, 'uid')
                comment_data['gender'] = self.safe_get_field(row, 'gender')
                comment_data['comment_count'] = self.safe_get_field(row, 'comment_count', '0')

            processed_data.append(comment_data)

        return pd.DataFrame(processed_data)

    def integrate_platform_data(self, folder_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """整合单个文件夹内所有平台的数据"""
        all_articles = []
        all_comments = []

        logger.info(f"开始处理文件夹: {folder_path}")

        # 第一步：先处理所有平台的文章数据，建立完整的ID映射
        logger.info("第一步：处理所有平台的文章数据")
        for platform in self.platforms:
            platform_path = folder_path / platform

            if not platform_path.exists():
                logger.warning(f"平台目录不存在: {platform_path}")
                continue

            # 处理文章数据
            article_file = platform_path / f"{platform}_article.csv"
            if article_file.exists():
                try:
                    df_article = pd.read_csv(article_file, encoding='utf-8', dtype=str)
                    processed_articles = self.process_article_data(platform, df_article)
                    all_articles.append(processed_articles)
                    logger.info(f"成功处理 {platform} 的 {len(processed_articles)} 条文章数据")
                except Exception as e:
                    logger.error(f"处理 {platform} 文章数据时出错: {e}")

        # 第二步：处理所有平台的评论数据，使用已建立的ID映射
        logger.info("第二步：处理所有平台的评论数据")
        for platform in self.platforms:
            platform_path = folder_path / platform

            if not platform_path.exists():
                continue

            # 处理评论数据
            comment_file = platform_path / f"{platform}_comments.csv"
            if comment_file.exists():
                try:
                    df_comment = pd.read_csv(comment_file, encoding='utf-8', dtype=str)
                    processed_comments = self.process_comment_data(platform, df_comment)
                    all_comments.append(processed_comments)
                    logger.info(f"成功处理 {platform} 的 {len(processed_comments)} 条评论数据")
                except Exception as e:
                    logger.error(f"处理 {platform} 评论数据时出错: {e}")

        # 合并所有平台数据
        final_articles = pd.concat(all_articles, ignore_index=True) if all_articles else pd.DataFrame()
        final_comments = pd.concat(all_comments, ignore_index=True) if all_comments else pd.DataFrame()

        return final_articles, final_comments

    def sort_data_by_time(self, df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
        """按时间排序数据"""
        if df.empty:
            return df

        try:
            # 优先使用created_date排序，如果不存在则使用created_time
            if 'created_date' in df.columns:
                # 先按日期排序，然后按时间戳排序
                df['created_time_num'] = pd.to_numeric(df['created_time'], errors='coerce')
                df_sorted = df.sort_values(['created_date', 'created_time_num'],
                                           ascending=[ascending, ascending])
                df_sorted = df_sorted.drop('created_time_num', axis=1)
                return df_sorted
            elif 'created_time' in df.columns:
                # 转换为数字后排序
                df['created_time_num'] = pd.to_numeric(df['created_time'], errors='coerce')
                df_sorted = df.sort_values('created_time_num', ascending=ascending)
                df_sorted = df_sorted.drop('created_time_num', axis=1)
                return df_sorted
            else:
                logger.warning("数据中没有找到时间字段，无法排序")
                return df
        except Exception as e:
            logger.error(f"排序时出错: {e}")
            return df

    def save_integrated_data(self, articles_df: pd.DataFrame, comments_df: pd.DataFrame,
                             output_folder: Path, folder_name: str):
        """保存整合后的数据"""
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)

        # 保存文章数据
        if not articles_df.empty:
            # 按时间排序文章数据
            # articles_df = self.sort_data_by_time(articles_df, ascending=False)
            articles_df = self.sort_data_by_time(articles_df, ascending=True)
            article_file = output_folder / f"{folder_name}_integrated_articles.csv"
            articles_df.to_csv(article_file, index=False, encoding='utf-8-sig')
            logger.info(f"文章数据已保存到: {article_file}")
            logger.info(f"文章数据统计: 共 {len(articles_df)} 条，来自 {articles_df['platform'].nunique()} 个平台")
            # logger.info(f"文章数据已按时间排序（最新在前）")

        # 保存评论数据
        if not comments_df.empty:
            # 按时间排序评论数据
            # comments_df = self.sort_data_by_time(comments_df, ascending=False)
            comments_df = self.sort_data_by_time(comments_df, ascending=True)
            comment_file = output_folder / f"{folder_name}_integrated_comments.csv"
            comments_df.to_csv(comment_file, index=False, encoding='utf-8-sig')
            logger.info(f"评论数据已保存到: {comment_file}")
            logger.info(f"评论数据统计: 共 {len(comments_df)} 条，来自 {comments_df['platform'].nunique()} 个平台")
            # logger.info(f"评论数据已按时间排序（最新在前）")

        # 保存ID映射关系
        self.save_id_mapping(output_folder, folder_name)

        # 生成数据统计报告
        self.generate_data_report(articles_df, comments_df, output_folder, folder_name)

    def save_id_mapping(self, output_folder: Path, folder_name: str):
        """保存article_id到post_id的映射关系"""
        if self.article_id_mapping:
            mapping_file = output_folder / f"{folder_name}_id_mapping.json"
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.article_id_mapping, f, ensure_ascii=False, indent=2)
            logger.info(f"ID映射关系已保存到: {mapping_file}")
            logger.info(f"映射关系统计: 共 {len(self.article_id_mapping)} 条映射")

    def generate_data_report(self, articles_df: pd.DataFrame, comments_df: pd.DataFrame,
                             output_folder: Path, folder_name: str):
        """生成数据统计报告"""
        report = {
            "数据整合报告": {
                "处理时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "数据源": folder_name,
                "ID转换": {
                    "转换说明": "原始article_id已转换为统一的post_id格式",
                    "映射数量": len(self.article_id_mapping),
                    "示例映射": dict(list(self.article_id_mapping.items())[:5]) if self.article_id_mapping else {}
                },
                "文章数据": {
                    "总数": len(articles_df) if not articles_df.empty else 0,
                    "平台分布": articles_df['platform'].value_counts().to_dict() if not articles_df.empty else {},
                    "情感分析": articles_df['sentiment'].value_counts().to_dict() if not articles_df.empty else {},
                    "立场分析": articles_df['stance'].value_counts().to_dict() if not articles_df.empty else {}
                },
                "评论数据": {
                    "总数": len(comments_df) if not comments_df.empty else 0,
                    "平台分布": comments_df['platform'].value_counts().to_dict() if not comments_df.empty else {},
                    "情感分析": comments_df['sentiment'].value_counts().to_dict() if not comments_df.empty else {},
                    "立场分析": comments_df['stance'].value_counts().to_dict() if not comments_df.empty else {}
                }
            }
        }

        report_file = output_folder / f"{folder_name}_data_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"数据报告已保存到: {report_file}")

    def run_integration(self, output_path: str = None):
        """运行数据整合流程"""
        if output_path is None:
            output_path = self.data_root_path / "XMSU7D" / "integrated_data"

        output_path = Path(output_path)

        # 查找所有数据文件夹
        data_folders = self.find_data_folders()
        logger.info(f"发现 {len(data_folders)} 个数据文件夹")

        for folder in data_folders:
            logger.info(f"\n{'='*50}")
            logger.info(f"开始处理数据文件夹: {folder.name}")

            # 重置ID映射（每个文件夹独立）
            self.article_id_mapping = {}

            # 整合该文件夹的数据
            articles_df, comments_df = self.integrate_platform_data(folder)

            # 保存整合后的数据
            self.save_integrated_data(articles_df, comments_df, output_path, folder.name)

            logger.info(f"完成文件夹 {folder.name} 的数据整合")
            logger.info(f"生成了 {len(self.article_id_mapping)} 个post_id映射")

        logger.info(f"\n{'='*50}")
        logger.info("所有数据整合完成！")
        logger.info(f"输出目录: {output_path}")


def main():
    """主函数"""
    # 设置数据路径
    data_root = "Data"

    # 创建数据整合器
    organizer = DataOrganizer(data_root)

    # 运行数据整合
    organizer.run_integration()


if __name__ == "__main__":
    main()
