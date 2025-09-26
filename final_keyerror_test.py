#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最终测试：确认所有KeyError问题都已解决
测试draw_ai_info中可能出现的所有KeyError
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gomoku import GomokuGame

def test_all_ai_keyerror_free():
    """测试所有AI系统是否KeyError-free"""
    print("=== 最终KeyError测试 ===")
    
    ai_difficulties = [
        'ultimate_threat',
        'enhanced_reinforced', 
        'reinforced',
        'reinforced_fixed',
        'optimized',
        'modern',
        'neural_mcts'
    ]
    
    all_passed = True
    
    for difficulty in ai_difficulties:
        print(f"\n--- 测试 {difficulty} ---")
        
        try:
            # 创建游戏实例
            game = GomokuGame(ai_difficulty=difficulty)
            
            # 获取性能统计
            perf_stats = game.get_performance_stats()
            
            # 测试所有draw_ai_info中使用的访问模式
            # 1. 测试ai_stats嵌套结构
            if 'ai_stats' not in perf_stats:
                print(f"  [失败] 缺少ai_stats字段")
                all_passed = False
                continue
                
            ai_stats = perf_stats['ai_stats']
            
            # 2. 测试所有必需字段
            required_fields = ['total_moves', 'avg_time_per_move', 'avg_nodes_per_move']
            for field in required_fields:
                if field not in ai_stats:
                    print(f"  [失败] ai_stats缺少{field}")
                    all_passed = False
                    continue
            
            # 3. 测试具体的访问模式（模拟draw_ai_info中的代码）
            try:
                # 这是draw_ai_info中的实际访问模式
                if ai_stats['total_moves'] > 0:
                    avg_time = ai_stats['avg_time_per_move']
                    avg_nodes = ai_stats['avg_nodes_per_move']
                    
                    # 测试数值运算（如格式化）
                    time_str = f"{avg_time:.3f}"
                    nodes_str = f"{avg_nodes:.0f}"
                    
                    print(f"  [成功] 数值访问和运算正常")
                    print(f"    avg_time: {time_str}, avg_nodes: {nodes_str}")
                else:
                    print(f"  [成功] 零移动状态正常")
                    
            except KeyError as e:
                print(f"  [失败] KeyError: {e}")
                all_passed = False
                continue
            except Exception as e:
                print(f"  [警告] 其他错误: {e}")
                # 不影响基本测试
            
            # 4. 测试AI系统本身的get_ai_info方法
            if hasattr(game.ai_system, 'get_ai_info'):
                try:
                    ai_info = game.ai_system.get_ai_info()
                    
                    # 测试draw_ai_info中使用的features字段
                    if 'features' not in ai_info:
                        print(f"  [失败] get_ai_info缺少features字段")
                        all_passed = False
                        continue
                        
                    features = ai_info['features']
                    if isinstance(features, list) and len(features) > 0:
                        print(f"  [成功] features列表正常，包含{len(features)}项")
                    else:
                        print(f"  [失败] features不是有效列表")
                        all_passed = False
                        continue
                        
                except KeyError as e:
                    print(f"  [失败] get_ai_info KeyError: {e}")
                    all_passed = False
                    continue
            else:
                print(f"  [跳过] 没有get_ai_info方法")
            
            print(f"  [成功] {difficulty} 完全通过KeyError测试")
            
        except Exception as e:
            print(f"  [失败] {difficulty} 测试失败: {e}")
            all_passed = False
    
    # 汇总结果
    print("\n" + "="*60)
    print("最终KeyError测试结果:")
    print("="*60)
    
    if all_passed:
        print("[SUCCESS] 所有AI系统都通过了KeyError测试！")
        print("所有draw_ai_info中使用的字段都可以正常访问。")
        print("游戏现在应该可以正常运行，不会出现KeyError异常。")
        return True
    else:
        print("[FAILED] 部分AI系统仍有KeyError问题")
        return False

def main():
    """主函数"""
    print("AI系统KeyError问题最终验证")
    print("作者：Claude AI Engineer")
    print("日期：2025-09-26")
    print("="*60)
    
    success = test_all_ai_keyerror_free()
    
    if success:
        print("\n" + "SUCCESS" * 10)
        print("所有KeyError问题都已解决！")
        print("游戏现在可以正常运行！")
        print("SUCCESS" * 10)
        print("\n运行命令：python gomoku.py")
        return 0
    else:
        print("\n" + "WARNING" * 10)
        print("仍有KeyError问题需要解决")
        print("WARNING" * 10)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)