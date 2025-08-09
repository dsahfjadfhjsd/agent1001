"""
配置管理工具 - 超参数和用户配置的统一管理
"""
from Config.settings import config, config_loader, reload_config
import json
import os
import sys
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ConfigManager:
    """统一配置管理器"""

    def __init__(self):
        self.config_dir = os.path.dirname(__file__)
        self.hyperparams_file = os.path.join(self.config_dir, "hyperparameters.json")
        self.user_profiles_file = os.path.join(self.config_dir, "user_profiles.json")

    def show_current_config(self):
        """显示当前配置概览"""
        print("🔧 当前系统配置")
        print("=" * 50)

        print("📊 超参数配置:")
        print(f"  模型: {config.model.model_name}")
        print(f"  温度: {config.model.temperature}")
        print(f"  最大令牌: {config.model.max_tokens}")
        print(f"  并发数: {config.concurrency.max_concurrent_users}")
        print(f"  并发方法: {config.concurrency.method}")
        print(f"  批处理大小: {config.concurrency.batch_size}")
        print(f"  最大轮次: {config.simulation.max_rounds}")

        print("\n👥 用户配置:")
        from Config.settings import get_user_generation_rules, get_sample_users
        rules = get_user_generation_rules()
        users = get_sample_users()

        if rules:
            age_dist = rules.get('age_distribution', {})
            print(f"  年龄分布: {age_dist}")
            occupations = rules.get('occupation_list', [])
            print(f"  职业类型: {len(occupations)} 种")

        print(f"  示例用户: {len(users)} 个")

        print(f"\n🌍 环境变量:")
        print(f"  调试模式: {config.debug}")
        print(f"  日志级别: {config.log_level}")
        print(f"  数据路径: {config.data_path}")
        print(f"  输出路径: {config.output_path}")

    def update_hyperparams(self, updates: Dict[str, Any]):
        """更新超参数"""
        try:
            with open(self.hyperparams_file, 'r', encoding='utf-8') as f:
                current_config = json.load(f)

            # 深度更新
            def deep_update(d, u):
                for k, v in u.items():
                    if isinstance(v, dict):
                        d[k] = deep_update(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d

            deep_update(current_config, updates)

            # 保存
            with open(self.hyperparams_file, 'w', encoding='utf-8') as f:
                json.dump(current_config, f, ensure_ascii=False, indent=2)

            # 重新加载配置
            reload_config()
            print(f"✅ 超参数更新成功: {updates}")

        except Exception as e:
            print(f"❌ 超参数更新失败: {e}")

    def update_user_profiles(self, updates: Dict[str, Any]):
        """更新用户画像配置"""
        try:
            with open(self.user_profiles_file, 'r', encoding='utf-8') as f:
                current_config = json.load(f)

            # 深度更新
            def deep_update(d, u):
                for k, v in u.items():
                    if isinstance(v, dict):
                        d[k] = deep_update(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d

            deep_update(current_config, updates)

            # 保存
            with open(self.user_profiles_file, 'w', encoding='utf-8') as f:
                json.dump(current_config, f, ensure_ascii=False, indent=2)

            # 重新加载配置
            reload_config()
            print(f"✅ 用户配置更新成功")

        except Exception as e:
            print(f"❌ 用户配置更新失败: {e}")

    def set_concurrency_preset(self, preset: str):
        """设置并发预设"""
        presets = {
            "development": {
                "concurrency": {
                    "max_concurrent_users": 5,
                    "method": "asyncio",
                    "batch_size": 10
                },
                "simulation": {
                    "max_rounds": 3
                }
            },
            "production": {
                "concurrency": {
                    "max_concurrent_users": 20,
                    "method": "semaphore",
                    "batch_size": 100
                },
                "simulation": {
                    "max_rounds": 10
                }
            },
            "performance": {
                "concurrency": {
                    "max_concurrent_users": 50,
                    "method": "asyncio",
                    "batch_size": 200
                },
                "simulation": {
                    "max_rounds": 20
                }
            }
        }

        if preset in presets:
            self.update_hyperparams(presets[preset])
            return True
        else:
            print(f"❌ 未知预设: {preset}")
            print(f"可用预设: {list(presets.keys())}")
            return False

    def add_sample_user(self, user_data: Dict[str, Any]):
        """添加示例用户"""
        try:
            with open(self.user_profiles_file, 'r', encoding='utf-8') as f:
                current_config = json.load(f)

            if "sample_users" not in current_config:
                current_config["sample_users"] = []

            current_config["sample_users"].append(user_data)

            with open(self.user_profiles_file, 'w', encoding='utf-8') as f:
                json.dump(current_config, f, ensure_ascii=False, indent=2)

            reload_config()
            print(f"✅ 用户 {user_data.get('user_id', 'unknown')} 添加成功")

        except Exception as e:
            print(f"❌ 添加用户失败: {e}")

    def export_config(self, output_dir: str = None):
        """导出完整配置"""
        if not output_dir:
            output_dir = f"./config_backup_{int(__import__('time').time())}"

        os.makedirs(output_dir, exist_ok=True)

        try:
            # 复制配置文件
            import shutil
            shutil.copy2(self.hyperparams_file, output_dir)
            shutil.copy2(self.user_profiles_file, output_dir)

            # 导出当前配置状态
            current_state = {
                "model": config.model.__dict__,
                "concurrency": config.concurrency.__dict__,
                "simulation": config.simulation.__dict__,
                "evaluation": config.evaluation.__dict__
            }

            with open(os.path.join(output_dir, "current_state.json"), 'w', encoding='utf-8') as f:
                json.dump(current_state, f, ensure_ascii=False, indent=2)

            print(f"✅ 配置已导出到: {output_dir}")

        except Exception as e:
            print(f"❌ 导出失败: {e}")


# 创建全局管理器实例
config_manager = ConfigManager()


if __name__ == "__main__":
    # 测试配置管理器
    print("🧪 配置管理器测试")
    print("=" * 50)

    # 显示当前配置
    config_manager.show_current_config()

    print("\n🔄 测试配置更新:")

    # 测试更新超参数
    config_manager.update_hyperparams({
        "concurrency": {
            "max_concurrent_users": 15
        }
    })

    print(f"更新后并发数: {config.concurrency.max_concurrent_users}")

    # 测试预设
    print("\n📦 测试预设配置:")
    config_manager.set_concurrency_preset("development")
    print(f"开发预设并发数: {config.concurrency.max_concurrent_users}")
