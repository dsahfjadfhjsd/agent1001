"""
异步运行工具模块

提供更好的 asyncio 运行方式，避免 Windows 上的错误信息
"""

import asyncio
import sys
import signal


def run_async(coro):
    """
    安全运行异步协程，避免 Windows 上的清理错误

    Args:
        coro: 异步协程函数
    """
    if sys.platform == "win32":
        # Windows 特殊处理
        try:
            # 使用 ProactorEventLoop 减少错误
            if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # 设置信号处理
            if hasattr(signal, 'SIGINT'):
                def signal_handler():
                    print("\n收到中断信号，正在关闭...")
                    for task in asyncio.all_tasks(loop):
                        task.cancel()

                loop.add_signal_handler(signal.SIGINT, signal_handler)

            return loop.run_until_complete(coro)

        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"运行出错: {e}")
        finally:
            try:
                # 清理挂起的任务
                pending = asyncio.all_tasks(loop)
                if pending:
                    print("正在清理挂起的任务...")
                    for task in pending:
                        task.cancel()
                    # 等待任务完成或取消
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except:
                pass
            finally:
                try:
                    loop.close()
                except:
                    pass
    else:
        # 其他平台使用标准方式
        return asyncio.run(coro)


def run_async_simple(coro):
    """
    简单的异步运行方式，完全忽略清理错误

    Args:
        coro: 异步协程函数
    """
    import warnings

    # 忽略特定的警告
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(coro)
        return result
    except KeyboardInterrupt:
        print("\n程序被中断")
        return None
    except Exception as e:
        print(f"运行出错: {e}")
        return None
    finally:
        # 完全静默关闭，不做任何清理
        try:
            import gc
            # 强制垃圾回收
            gc.collect()
            # 直接设置为None，让Python自己处理
            loop = None
        except:
            pass

        # 在Windows上，设置不显示错误
        import sys
        if sys.platform == "win32":
            try:
                import os
                # 重定向stderr到null以隐藏错误消息
                with open(os.devnull, 'w') as devnull:
                    old_stderr = sys.stderr
                    sys.stderr = devnull
                    # 短暂延迟后恢复
                    import time
                    time.sleep(0.01)
                    sys.stderr = old_stderr
            except:
                pass


if __name__ == "__main__":
    # 测试代码
    async def test_async():
        await asyncio.sleep(0.1)
        print("异步函数运行成功!")
        return "完成"

    print("测试异步运行工具...")
    result = run_async_simple(test_async())
    print(f"结果: {result}")
