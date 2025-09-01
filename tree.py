"""
项目结构显示

打印项目结构，只显示到文件夹第二层，支持忽视部分文件或文件夹
"""

import os
from pathlib import Path
from typing import Set, Optional


class DirectoryTreeGenerator:
    """目录树生成器"""

    def __init__(self, max_depth: int = 2, ignore_patterns: Optional[Set[str]] = None):
        """
        初始化目录树生成器

        Args:
            max_depth: 最大显示深度，默认为2层
            ignore_patterns: 要忽略的文件/文件夹名称集合
        """
        self.max_depth = max_depth
        self.ignore_patterns = ignore_patterns or {
            '__pycache__', '.git', '.vscode', 'node_modules',
            '.env', '.DS_Store', '*.pyc', '.pytest_cache'
        }

    def should_ignore(self, path: Path) -> bool:
        """判断路径是否应该被忽略"""
        name = path.name
        # 检查是否在忽略列表中
        if name in self.ignore_patterns:
            return True
        # 检查是否匹配忽略模式
        for pattern in self.ignore_patterns:
            if pattern.startswith('*') and name.endswith(pattern[1:]):
                return True
            if pattern.startswith('.') and name.startswith(pattern):
                return True
        return False

    def generate_tree(self, root_path: str, current_depth: int = 0, prefix: str = "") -> str:
        """
        生成目录树字符串

        Args:
            root_path: 根目录路径
            current_depth: 当前深度
            prefix: 当前行的前缀

        Returns:
            树状结构字符串
        """
        if current_depth > self.max_depth:
            return ""

        tree_str = ""
        root = Path(root_path)

        if not root.exists():
            return f"{prefix}[路径不存在: {root_path}]\n"

        # 获取目录下的所有项目，并排序（文件夹在前，文件在后）
        try:
            items = [item for item in root.iterdir() if not self.should_ignore(item)]
            # 排序：文件夹在前，文件在后，同类型按名称排序
            items.sort(key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            return f"{prefix}[权限拒绝: {root_path}]\n"

        for i, item in enumerate(items):
            is_last = i == len(items) - 1

            # 选择合适的连接符
            if is_last:
                current_prefix = prefix + "└── "
                next_prefix = prefix + "    "
            else:
                current_prefix = prefix + "├── "
                next_prefix = prefix + "│   "

            # 添加当前项目
            if item.is_dir():
                tree_str += f"{current_prefix}{item.name}/\n"
                # 递归处理子目录
                if current_depth < self.max_depth:
                    tree_str += self.generate_tree(str(item), current_depth + 1, next_prefix)
            else:
                tree_str += f"{current_prefix}{item.name}\n"

        return tree_str

    def print_tree(self, root_path: str, title: Optional[str] = None):
        """
        打印目录树

        Args:
            root_path: 根目录路径
            title: 可选的标题
        """
        if title:
            print(f"\n{title}")
            print("=" * len(title))

        root = Path(root_path)
        print(f"{root.name}/")
        print(self.generate_tree(root_path, 0, ""))


def main():
    """主函数 - 演示用法"""
    # 获取当前工作目录
    current_dir = os.getcwd()

    # 创建目录树生成器
    tree_generator = DirectoryTreeGenerator(
        max_depth=2,
        ignore_patterns={
            '__pycache__', '.git', '.vscode', 'node_modules',
            '.DS_Store', '*.pyc', '.pytest_cache',
            'uv.lock', '.venv'
        }  # 项目特定的忽略文件
    )

    # 生成并打印目录树
    tree_generator.print_tree(current_dir, "项目结构")

    # 也可以获取树状字符串用于其他用途
    tree_string = tree_generator.generate_tree(current_dir)

    # 保存到txt
    with open("project_structure.txt", "w", encoding="utf-8") as f:
        f.write("项目结构\n")
        f.write("=" * 10 + "\n")
        f.write("agent1001/\n")
        f.write(tree_string)

    print("树状结构已保存到 project_structure.txt 文件")

    print("\n" + "="*50)
    print("您也可以调用以下方法获取树状字符串：")
    print("tree_string = tree_generator.generate_tree(directory_path)")


if __name__ == "__main__":
    main()
