"""
通用帖子生成模块

支持:
- 灵活的配置系统
- 并发API调用
- 增量保存
- 多维度分类生成
"""

import pandas as pd
from pathlib import Path
import asyncio
import random
import uuid
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 禁用 httpx 和 openai 的日志输出，避免干扰进度条
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ============== 配置示例 ==============
Article_Config = {
    "description": "小米SU7车祸事件相关讨论",
    "Options": {
        "stance": {
            "subclass": ["支持小米", "中立观望", "质疑批评"],
            "probability": [0.3, 0.4, 0.3]
        },
        "sentiment": {
            "subclass": ["积极", "中立", "消极"],
            "probability": [0.2, 0.3, 0.5]
        },
        "intent": {
            "subclass": ["信息验证", "情感表达", "利益实践"],
            "probability": [0.4, 0.4, 0.2]
        }
    }
}


@dataclass
class GenerationConfig:
    """生成配置"""
    # API配置
    model_name: str = "qwen-max"
    max_concurrent_requests: int = 5
    request_timeout: int = 60

    # 生成配置
    total_articles: int = 100  # 总共生成的帖子数量
    batch_size: int = 10  # 每批生成的数量（增量保存间隔）

    # 输出配置
    output_file: str = "generated_articles.csv"
    created_date: str = None  # 统一的创建时间，None则使用当前时间
    platform: str = "生成"

    # 参考配置
    reference_csv: str = None  # 参考帖子CSV路径
    reference_sample_size: int = 3  # 每次从参考帖子中抽样的数量


class ArticleGenerator:
    """通用帖子生成器"""

    def __init__(self, article_config: Dict, generation_config: GenerationConfig = None):
        """
        初始化生成器

        Args:
            article_config: 帖子配置（包含description和Options）
            generation_config: 生成配置
        """
        # 加载环境变量
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Config', '.env'))

        self.article_config = article_config
        self.config = generation_config or GenerationConfig()

        # 初始化OpenAI客户端
        self.client = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL')
        )

        # 创建信号量控制并发
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        # 加载参考帖子
        self.reference_articles = self._load_reference_articles()

        # 设置创建时间
        if self.config.created_date is None:
            self.config.created_date = datetime.now().strftime('%Y-%m-%d %H:%M')

        # 提取所有分类维度
        self.class_names = list(self.article_config['Options'].keys())

        logger.info(f"初始化完成: 分类维度={self.class_names}, 参考帖子数={len(self.reference_articles)}")

    def _load_reference_articles(self) -> List[str]:
        """加载参考帖子"""
        if not self.config.reference_csv or not Path(self.config.reference_csv).exists():
            logger.warning("未提供参考帖子CSV或文件不存在")
            return []

        try:
            df = pd.read_csv(self.config.reference_csv)
            # title
            titles = df['title'].dropna().tolist()
            # 假设参考帖子在'content'列
            articles = df['content'].dropna().tolist()
            # 合并标题和内容
            articles = [f"{title}\n{content}" for title, content in zip(titles, articles)]
            random.shuffle(articles)  # 打乱顺序
            logger.info(f"成功加载 {len(articles)} 条参考帖子")
            return articles
        except Exception as e:
            logger.error(f"加载参考帖子失败: {e}")
            return []

    def _sample_options(self) -> Dict[str, str]:
        """
        根据概率从各个分类中抽样选项

        Returns:
            选项字典 {class_name: selected_option}
        """
        selected_options = {}

        for class_name, class_config in self.article_config['Options'].items():
            subclasses = class_config['subclass']
            probabilities = class_config['probability']

            # 根据概率选择一个选项
            selected = random.choices(subclasses, weights=probabilities, k=1)[0]
            selected_options[class_name] = selected

        return selected_options

    def _get_reference_samples(self) -> List[str]:
        """获取参考帖子样本"""
        if not self.reference_articles:
            return []

        sample_size = min(self.config.reference_sample_size, len(self.reference_articles))
        return random.sample(self.reference_articles, sample_size)

    def _build_generation_prompt(self, selected_options: Dict[str, str],
                                 reference_samples: List[str]) -> str:
        """
        构建生成提示词

        Args:
            selected_options: 选中的选项
            reference_samples: 参考帖子样本

        Returns:
            提示词字符串
        """
        prompt = f"""你是一个社交媒体内容生成助手。请根据以下要求生成一篇帖子内容。

事件描述：
{self.article_config['description']}

"""

        # 添加参考帖子
        if reference_samples:
            prompt += "参考帖子示例（作为风格和内容参考）：\n"
            for i, sample in enumerate(reference_samples, 1):
                prompt += f"\n示例{i}:\n{sample}\n"
            prompt += "\n"

        # 添加生成要求
        prompt += "生成要求：\n"
        for class_name, option in selected_options.items():
            prompt += f"- {class_name}: {option}\n"

        prompt += """
请生成一篇新的帖子内容，要求：
1. 内容要符合上述所有要求和倾向
2. 风格自然，像真实用户发布的内容
3. 长度适中（100-500字）
4. 不要完全照搬参考示例，要有创新，要符合当前的网络语言风格
5. 使用中文

请直接输出帖子内容，不要包含任何额外说明或格式标记。
"""

        return prompt

    async def _generate_single_article(self, selected_options: Dict[str, str]) -> Optional[str]:
        """
        生成单篇帖子

        Args:
            selected_options: 选中的选项

        Returns:
            生成的帖子内容
        """
        reference_samples = self._get_reference_samples()
        prompt = self._build_generation_prompt(selected_options, reference_samples)

        async with self.semaphore:
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.config.model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": "你是一个专业的社交媒体内容生成助手，擅长根据要求生成真实自然的帖子内容。"
                            },
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.9,  # 提高随机性以获得更多样化的内容
                    ),
                    timeout=self.config.request_timeout
                )

                content = response.choices[0].message.content.strip()
                return content

            except asyncio.TimeoutError:
                logger.error("API请求超时")
                return None
            except Exception as e:
                logger.error(f"生成帖子失败: {e}")
                return None

    def _create_article_record(self, content: str, selected_options: Dict[str, str]) -> Dict[str, Any]:
        """
        创建帖子记录

        Args:
            content: 帖子内容
            selected_options: 选中的选项

        Returns:
            帖子记录字典
        """
        record = {
            'post_id': f"post_{uuid.uuid4().hex[:6]}",
            'title': '',
            'content': content,
            'created_date': self.config.created_date,
            'platform': self.config.platform,
            'like_count': 0,
            'comment_count': 0,
            'share_count': 0,
            'uid': '',
            'video_urls': '',
            'img_urls': '',
            'username': self.config.model_name,

        }

        # 添加所有分类维度的选项
        for class_name in self.class_names:
            record[class_name] = selected_options.get(class_name, '')

        return record

    async def _generate_batch(self, batch_num: int, batch_size: int) -> List[Dict[str, Any]]:
        """
        生成一批帖子

        Args:
            batch_num: 批次号
            batch_size: 批次大小

        Returns:
            帖子记录列表
        """
        logger.info(f"开始生成第 {batch_num} 批，数量: {batch_size}")

        tasks = []
        options_list = []

        # 为每篇帖子抽样选项并创建任务
        for _ in range(batch_size):
            selected_options = self._sample_options()
            options_list.append(selected_options)
            task = self._generate_single_article(selected_options)
            tasks.append(task)

        # 并发执行所有任务
        results = await tqdm.gather(*tasks, desc=f"批次 {batch_num}")

        # 创建记录
        records = []
        for content, options in zip(results, options_list):
            if content:
                record = self._create_article_record(content, options)
                records.append(record)
            else:
                logger.warning("生成失败，跳过该帖子")

        logger.info(f"第 {batch_num} 批完成，成功生成 {len(records)}/{batch_size} 篇")
        return records

    def _save_to_csv(self, records: List[Dict[str, Any]], mode: str = 'a'):
        """
        保存记录到CSV

        Args:
            records: 帖子记录列表
            mode: 写入模式 ('w' 或 'a')
        """
        if not records:
            return

        df = pd.DataFrame(records)

        # 确保列顺序：固定列 + 动态分类列
        fixed_columns = ['post_id', 'title', 'content', 'created_date', 'platform',
                         'like_count', 'comment_count', 'share_count',
                         'uid', 'username', 'video_urls', 'img_urls']
        all_columns = fixed_columns + self.class_names
        df = df[all_columns]

        # 保存
        file_exists = Path(self.config.output_file).exists()

        if mode == 'w' or not file_exists:
            df.to_csv(self.config.output_file, index=False, encoding='utf-8-sig')
            logger.info(f"创建新文件: {self.config.output_file}")
        else:
            df.to_csv(self.config.output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
            logger.info(f"追加到文件: {self.config.output_file}")

    async def generate(self):
        """执行生成任务"""
        logger.info("="*60)
        logger.info("开始生成帖子")
        logger.info(f"总数量: {self.config.total_articles}")
        logger.info(f"批次大小: {self.config.batch_size}")
        logger.info(f"输出文件: {self.config.output_file}")
        logger.info("="*60)

        # 删除已存在的输出文件（重新开始）
        if Path(self.config.output_file).exists():
            Path(self.config.output_file).unlink()
            logger.info("删除旧文件，重新开始生成")

        total_generated = 0
        batch_num = 1

        while total_generated < self.config.total_articles:
            # 计算本批次大小
            remaining = self.config.total_articles - total_generated
            batch_size = min(self.config.batch_size, remaining)

            # 生成一批
            records = await self._generate_batch(batch_num, batch_size)

            # 增量保存
            if records:
                self._save_to_csv(records, mode='a')
                total_generated += len(records)
                logger.info(f"进度: {total_generated}/{self.config.total_articles}")

            batch_num += 1

        logger.info("="*60)
        logger.info(f"生成完成！总共生成 {total_generated} 篇帖子")
        logger.info(f"保存位置: {self.config.output_file}")
        logger.info("="*60)

    async def close(self):
        """关闭客户端连接"""
        try:
            if hasattr(self.client, 'close'):
                await self.client.close()
        except:
            pass


def combine_csv_files(
    input_files: List[str],
    output_file: str,
    column_mode: str = 'union',
    fixed_columns: List[str] = None,
    keep_column_order: bool = False
):
    """
    合并多个CSV文件，支持多种列选择模式

    Args:
        input_files: 输入CSV文件路径列表
        output_file: 输出CSV文件路径
        column_mode: 列选择模式
            - 'union': 所有文件的列的并集（默认）
            - 'intersection': 所有文件共有的列（交集）
            - 'fixed': 只保留指定的固定列
        fixed_columns: 当column_mode='fixed'时，指定要保留的列名列表
        keep_column_order: 是否保持原始列顺序（False则按字母排序）

    Example:
        # 模式1: 保留所有列（默认）
        combine_csv_files(
            ['file1.csv', 'file2.csv'],
            'merged.csv'
        )

        # 模式2: 只保留共有列
        combine_csv_files(
            ['file1.csv', 'file2.csv'],
            'merged.csv',
            column_mode='intersection'
        )

        # 模式3: 只保留指定列
        combine_csv_files(
            ['file1.csv', 'file2.csv'],
            'merged.csv',
            column_mode='fixed',
            fixed_columns=['post_id', 'content', 'created_date']
        )
    """
    logger.info("="*60)
    logger.info("开始合并CSV文件")
    logger.info(f"输入文件数量: {len(input_files)}")
    logger.info(f"列选择模式: {column_mode}")
    if column_mode == 'fixed' and fixed_columns:
        logger.info(f"指定列: {fixed_columns}")
    logger.info("="*60)

    if not input_files:
        logger.error("未提供输入文件")
        return

    # 验证参数
    if column_mode not in ['union', 'intersection', 'fixed']:
        logger.error(f"无效的column_mode: {column_mode}，必须是 'union', 'intersection' 或 'fixed'")
        return

    if column_mode == 'fixed' and not fixed_columns:
        logger.error("column_mode='fixed' 时必须提供 fixed_columns 参数")
        return

    all_dataframes = []
    all_columns_sets = []  # 存储每个文件的列集合

    # 读取所有CSV文件并收集所有列名
    for i, file_path in enumerate(input_files, 1):
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            logger.warning(f"文件 {i}/{len(input_files)} 不存在，跳过: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path, dtype=str, encoding='utf-8-sig')  # 统一读取为字符串类型
            logger.info(f"读取文件 {i}/{len(input_files)}: {file_path} ({len(df)} 行, {len(df.columns)} 列)")

            # 收集列名集合
            all_columns_sets.append(set(df.columns.tolist()))
            all_dataframes.append(df)

        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            continue

    if not all_dataframes:
        logger.error("没有成功读取任何文件")
        return

    # 根据模式确定最终要保留的列
    if column_mode == 'union':
        # 并集：所有文件的列合并
        final_columns = set()
        for cols in all_columns_sets:
            final_columns.update(cols)
        logger.info(f"并集模式: 总共 {len(final_columns)} 个不同的列")

    elif column_mode == 'intersection':
        # 交集：所有文件共有的列
        final_columns = all_columns_sets[0]
        for cols in all_columns_sets[1:]:
            final_columns = final_columns.intersection(cols)
        logger.info(f"交集模式: 找到 {len(final_columns)} 个共有列")

        if not final_columns:
            logger.warning("警告: 没有找到任何共有列！")
            return

    elif column_mode == 'fixed':
        # 固定列：只保留指定的列
        final_columns = set(fixed_columns)
        logger.info(f"固定列模式: 保留 {len(final_columns)} 个指定列")

        # 检查哪些列在所有文件中都不存在
        all_available_columns = set()
        for cols in all_columns_sets:
            all_available_columns.update(cols)

        missing_columns = final_columns - all_available_columns
        if missing_columns:
            logger.warning(f"以下指定列在所有文件中都不存在: {missing_columns}")

    # 转换为列表
    if keep_column_order and column_mode == 'fixed':
        # 保持fixed_columns指定的顺序
        final_columns_list = [col for col in fixed_columns if col in final_columns]
    elif keep_column_order:
        # 保持第一个文件的列顺序，后续文件中不存在的列放在最后
        first_file_columns = list(all_dataframes[0].columns)
        final_columns_list = [col for col in first_file_columns if col in final_columns]
        # 添加剩余的列
        final_columns_list += [col for col in final_columns if col not in final_columns_list]
    else:
        # 按字母排序
        final_columns_list = sorted(list(final_columns))

    logger.info(f"最终列列表: {final_columns_list[:10]}{'...' if len(final_columns_list) > 10 else ''}")

    # 为每个dataframe处理列
    normalized_dataframes = []
    for i, df in enumerate(all_dataframes, 1):
        df_copy = df.copy()

        # 找出需要的列中，当前dataframe缺失的列
        missing_columns = set(final_columns_list) - set(df_copy.columns)

        # 为缺失的列添加空值
        for col in missing_columns:
            df_copy[col] = ''

        # 只保留需要的列
        available_columns = [col for col in final_columns_list if col in df_copy.columns or col in missing_columns]
        df_copy = df_copy[available_columns]

        logger.debug(f"文件 {i}: 保留 {len(available_columns)} 列")
        normalized_dataframes.append(df_copy)

    # 合并所有dataframe
    merged_df = pd.concat(normalized_dataframes, ignore_index=True)

    logger.info(f"合并完成: 总共 {len(merged_df)} 行, {len(merged_df.columns)} 列")

    # 保存到输出文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"已保存到: {output_file}")
    logger.info("="*60)

    return merged_df


def change_column_values(input_file: str, output_file: str, column: str, value_map: Dict[str, str]):
    """
    修改CSV文件中指定列的值

    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        column: 要修改的列名
        value_map: 值映射字典 {old_value: new_value}

    Example:
        change_column_values(
            'input.csv',
            'output.csv',
            'stance',
            {
                '支持小米': '支持',
                '中立观望': '中立',
                '质疑批评': '批判'
            }
        )
    """
    logger.info("="*60)
    logger.info(f"开始修改列值: {column}")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"值映射: {value_map}")
    logger.info("="*60)

    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_file}")
        return

    try:
        df = pd.read_csv(input_file, dtype=str, encoding='utf-8-sig')  # 统一读取为字符串类型
        logger.info(f"读取文件: {input_file} ({len(df)} 行, {len(df.columns)} 列)")

        if column not in df.columns:
            logger.error(f"列 '{column}' 在文件中不存在")
            return

        # 替换值
        df[column] = df[column].map(value_map).fillna(df[column])

        # 保存到输出文件
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"已保存修改后的文件到: {output_file}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"处理文件失败: {e}")


async def main():
    """主函数示例"""
    # 配置示例
    article_config = {
        "description": "小米SU7高速车祸事件，涉及智能驾驶、电池安全、车门设计等多个争议点",
        "Options": {
            "stance": {
                "subclass": ["支持小米", "中立观望", "质疑批评"],
                "probability": [0.45, 0.3, 0.25]
            },
            "sentiment": {
                "subclass": ["积极", "中立", "消极"],
                "probability": [0.35, 0.3, 0.35]
            },
            "intent": {
                "subclass": ["信息验证", "情感表达", "利益实践"],
                "probability": [0.3, 0.4, 0.3]
            }
        }
    }

    generation_config = GenerationConfig(
        model_name="qwen3-max",
        max_concurrent_requests=10,
        total_articles=100,  # 生成100篇作为测试
        batch_size=20,  # 每20篇保存一次
        output_file="Data/XMSU7D/generated/generated_articles_100.csv",
        created_date="2025-03-29 20:00",
        platform="生成",
        reference_csv="Data/XMSU7D/integrated_data/XMSU7D_integrated_articles.csv",
        reference_sample_size=2
    )

    generator = ArticleGenerator(article_config, generation_config)

    try:
        await generator.generate()
    finally:
        await generator.close()


if __name__ == "__main__":
    # 选择运行模式
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "combine":
        # 合并CSV文件模式
        input_files = [
            "Data/XMSU7D/integrated_data/XMSU7D_integrated_articles.csv",
            "Data/XMSU7D/generated/generated_articles_100_.csv"
        ]
        output_file = "Data/XMSU7D/generated/generated_articles_combined.csv"

        if not all(Path(f).exists() for f in input_files):
            logger.error("某些输入文件不存在，无法合并")

        # 默认使用union模式
        if len(sys.argv) < 3 or sys.argv[2] == "union":
            # 并集模式
            combine_csv_files(input_files, output_file, column_mode='union')
        elif sys.argv[2] == "fixed":
            # 固定列模式
            fixed_columns = ['post_id', 'title', 'content', 'created_date', 'platform',
                             'like_count', 'comment_count', 'share_count',
                             'uid', 'username', 'video_urls', 'img_urls',
                             'stance', 'sentiment', 'intent']
            combine_csv_files(input_files, output_file, column_mode='fixed', fixed_columns=fixed_columns)
        elif sys.argv[2] == "intersection":
            # 交集模式
            combine_csv_files(input_files, output_file, column_mode='intersection', keep_column_order=True)


    elif len(sys.argv) > 1 and sys.argv[1] == "change":
        # 修改列值模式
        input_file = "Data/XMSU7D/generated/generated_articles_combined.csv"
        output_file = "Data/XMSU7D/generated/generated_articles_combined_fixed.csv"
        column = "stance"
        value_map = {
            "支持小米": "支持",
            "中立观望": "中立",
            "质疑批评": "批判"
        }
        if not Path(input_file).exists():
            logger.error(f"输入文件不存在，无法修改列值: {input_file}")
        else:
            change_column_values(input_file, output_file, column, value_map)

    else:
        # 默认生成模式
        asyncio.run(main())
