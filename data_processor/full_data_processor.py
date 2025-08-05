import pandas as pd
import json
import re
from typing import Dict, List, Any
import logging
from datetime import datetime
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FullDataProcessor:
    def __init__(self, csv_file_path: str):
        """
        初始化完整数据处理器
        
        Args:
            csv_file_path: CSV文件路径
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.output_dir = "output"
        
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_data(self) -> pd.DataFrame:
        """
        加载CSV数据
        
        Returns:
            加载的数据框
        """
        try:
            logger.info(f"正在加载数据文件: {self.csv_file_path}")
            self.data = pd.read_csv(self.csv_file_path)
            logger.info(f"成功加载数据，共 {len(self.data)} 条记录")
            return self.data
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        清理文本数据
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        if pd.isna(text) or text is None:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', str(text).strip())
        return text
    
    def extract_attributes(self, row: pd.Series) -> Dict[str, Any]:
        """
        从单行数据中提取七个关键属性
        
        Args:
            row: 数据行
            
        Returns:
            包含七个属性的字典
        """
        return {
            "content": self.clean_text(row.get('content', '')),
            "stance": self.clean_text(row.get('stance', '')),
            "sentiment": self.clean_text(row.get('sentiment', '')),
            "intent": self.clean_text(row.get('intent', '')),
            "stance_content": self.clean_text(row.get('stance_content', '')),
            "sentiment_content": self.clean_text(row.get('sentiment_content', '')),
            "intent_content": self.clean_text(row.get('intent_content', ''))
        }
    
    def process_all_data(self, batch_size: int = 1000) -> List[Dict[str, Any]]:
        """
        分批处理所有数据
        
        Args:
            batch_size: 批处理大小
            
        Returns:
            提取的数据列表
        """
        if self.data is None:
            self.load_data()
        
        extracted_data = []
        total_records = len(self.data)
        
        logger.info(f"开始处理全部数据，共 {total_records} 条记录，批处理大小: {batch_size}")
        
        for start_idx in range(0, total_records, batch_size):
            end_idx = min(start_idx + batch_size, total_records)
            batch_data = self.data.iloc[start_idx:end_idx]
            
            logger.info(f"处理批次 {start_idx//batch_size + 1}/{(total_records + batch_size - 1)//batch_size} "
                       f"(记录 {start_idx + 1}-{end_idx})")
            
            for idx, row in batch_data.iterrows():
                try:
                    attributes = self.extract_attributes(row)
                    extracted_data.append(attributes)
                except Exception as e:
                    logger.warning(f"处理第 {idx + 1} 条记录时出错: {str(e)}")
                    continue
        
        logger.info(f"数据处理完成，成功提取 {len(extracted_data)} 条记录")
        return extracted_data
    
    def save_to_json(self, data: List[Dict[str, Any]], output_file: str):
        """
        保存数据为JSON格式
        
        Args:
            data: 要保存的数据
            output_file: 输出文件路径
        """
        try:
            output_path = os.path.join(self.output_dir, output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"数据已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存JSON文件失败: {str(e)}")
            raise
    
    def get_detailed_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取详细的数据统计信息
        
        Args:
            data: 提取的数据
            
        Returns:
            详细统计信息字典
        """
        stats = {
            "total_records": len(data),
            "stance_distribution": {},
            "sentiment_distribution": {},
            "intent_distribution": {},
            "stance_content_distribution": {},
            "sentiment_content_distribution": {},
            "intent_content_distribution": {},
            "empty_content_count": 0,
            "non_empty_content_count": 0,
            "content_length_stats": {
                "min": float('inf'),
                "max": 0,
                "avg": 0
            },
            "processing_time": datetime.now().isoformat()
        }
        
        total_length = 0
        
        for record in data:
            # 统计立场分布
            stance = record.get('stance', '')
            if stance:
                stats['stance_distribution'][stance] = stats['stance_distribution'].get(stance, 0) + 1
            
            # 统计情感分布
            sentiment = record.get('sentiment', '')
            if sentiment:
                stats['sentiment_distribution'][sentiment] = stats['sentiment_distribution'].get(sentiment, 0) + 1
            
            # 统计意图分布
            intent = record.get('intent', '')
            if intent:
                stats['intent_distribution'][intent] = stats['intent_distribution'].get(intent, 0) + 1
            
            # 统计立场内容分布
            stance_content = record.get('stance_content', '')
            if stance_content:
                stats['stance_content_distribution'][stance_content] = stats['stance_content_distribution'].get(stance_content, 0) + 1
            
            # 统计情感内容分布
            sentiment_content = record.get('sentiment_content', '')
            if sentiment_content:
                stats['sentiment_content_distribution'][sentiment_content] = stats['sentiment_content_distribution'].get(sentiment_content, 0) + 1
            
            # 统计意图内容分布
            intent_content = record.get('intent_content', '')
            if intent_content:
                stats['intent_content_distribution'][intent_content] = stats['intent_content_distribution'].get(intent_content, 0) + 1
            
            # 统计内容完整性
            content = record.get('content', '')
            content_length = len(content)
            if content.strip():
                stats['non_empty_content_count'] += 1
                total_length += content_length
                stats['content_length_stats']['min'] = min(stats['content_length_stats']['min'], content_length)
                stats['content_length_stats']['max'] = max(stats['content_length_stats']['max'], content_length)
            else:
                stats['empty_content_count'] += 1
        
        # 计算平均长度
        if stats['non_empty_content_count'] > 0:
            stats['content_length_stats']['avg'] = total_length / stats['non_empty_content_count']
        
        # 如果所有内容都为空，设置最小长度为0
        if stats['content_length_stats']['min'] == float('inf'):
            stats['content_length_stats']['min'] = 0
        
        return stats
    
    def generate_report(self, stats: Dict[str, Any], output_file: str = "detailed_report.json"):
        """
        生成详细报告
        
        Args:
            stats: 统计信息
            output_file: 输出文件名
        """
        report = {
            "报告生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "数据概览": {
                "总记录数": stats['total_records'],
                "有效内容数": stats['non_empty_content_count'],
                "空内容数": stats['empty_content_count'],
                "内容完整率": f"{stats['non_empty_content_count']/stats['total_records']*100:.2f}%"
            },
            "内容长度统计": stats['content_length_stats'],
            "立场分布": stats['stance_distribution'],
            "情感分布": stats['sentiment_distribution'],
            "意图分布": stats['intent_distribution'],
            "热门立场内容": dict(sorted(stats['stance_content_distribution'].items(), 
                                    key=lambda x: x[1], reverse=True)[:10]),
            "热门情感内容": dict(sorted(stats['sentiment_content_distribution'].items(), 
                                    key=lambda x: x[1], reverse=True)[:10]),
            "热门意图内容": dict(sorted(stats['intent_content_distribution'].items(), 
                                    key=lambda x: x[1], reverse=True)[:10])
        }
        
        self.save_to_json(report, output_file)
        return report

def main():
    """
    主函数
    """
    # 初始化数据处理器
    processor = FullDataProcessor('E:\mydown\zfqagent\douyin_comments.csv')
    
    # 处理全部数据
    logger.info("开始处理全部数据...")
    extracted_data = processor.process_all_data(batch_size=1000)
    
    # 保存提取的数据
    processor.save_to_json(extracted_data, "full_extracted_data.json")
    
    # 获取详细统计信息
    stats = processor.get_detailed_statistics(extracted_data)
    
    # 生成详细报告
    report = processor.generate_report(stats, "detailed_report.json")
    
    logger.info("数据处理和报告生成完成！")
    logger.info(f"总记录数: {stats['total_records']}")
    logger.info(f"有效内容数: {stats['non_empty_content_count']}")
    logger.info(f"内容完整率: {stats['non_empty_content_count']/stats['total_records']*100:.2f}%")

if __name__ == "__main__":
    main() 