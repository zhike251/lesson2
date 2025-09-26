#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
验证所有AI系统的性能统计接口修复
确保不再出现KeyError: 'avg_time_per_move'错误
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gomoku import GomokuGame

def test_all_ai_performance_stats():
    """测试所有AI系统的性能统计接口"""
    print("=== 验证所有AI系统的性能统计接口修复 ===")
    
    # 所有需要测试的AI难度
    ai_difficulties = [
        'ultimate_threat',
        'enhanced_reinforced', 
        'reinforced',
        'reinforced_fixed',
        'optimized',
        'modern',
        'neural_mcts'
    ]
    
    results = {}
    
    for difficulty in ai_difficulties:
        print(f"\n--- 测试 {difficulty} ---")
        
        try:
            # 创建游戏实例
            game = GomokuGame(ai_difficulty=difficulty)
            print(f"  AI系统类型: {type(game.ai_system).__name__}")
            
            # 测试get_performance_summary方法
            if hasattr(game.ai_system, 'get_performance_summary'):
                print("  有get_performance_summary方法")
                
                # 获取性能统计
                perf_summary = game.ai_system.get_performance_summary()
                print("  成功调用get_performance_summary")
                
                # 检查必需的嵌套结构
                if 'ai_stats' in perf_summary:
                    ai_stats = perf_summary['ai_stats']
                    
                    # 检查必需字段
                    required_fields = ['total_moves', 'avg_time_per_move', 'avg_nodes_per_move']
                    missing_fields = []
                    
                    for field in required_fields:
                        if field not in ai_stats:
                            missing_fields.append(field)
                    
                    if missing_fields:
                        print(f"  [失败] 缺少字段: {missing_fields}")
                        results[difficulty] = False
                    else:
                        print(f"  [成功] 所有必需字段存在")
                        print(f"     total_moves: {ai_stats['total_moves']}")
                        print(f"     avg_time_per_move: {ai_stats['avg_time_per_move']}")
                        print(f"     avg_nodes_per_move: {ai_stats['avg_nodes_per_move']}")
                        results[difficulty] = True
                else:
                    print("  [失败] 缺少ai_stats嵌套结构")
                    results[difficulty] = False
            else:
                print("  [失败] 没有get_performance_summary方法")
                results[difficulty] = False
            
            # 测试游戏的get_performance_stats方法
            print("  测试游戏性能统计...")
            game_perf_stats = game.get_performance_stats()
            
            if 'ai_stats' in game_perf_stats:
                ai_stats = game_perf_stats['ai_stats']
                
                # 尝试访问draw_ai_info中使用的字段
                try:
                    avg_time = game_perf_stats['ai_stats']['avg_time_per_move']
                    avg_nodes = game_perf_stats['ai_stats']['avg_nodes_per_move']
                    total_moves = game_perf_stats['ai_stats']['total_moves']
                    print(f"  [成功] draw_ai_info访问测试通过")
                    print(f"     avg_time_per_move: {avg_time}")
                    print(f"     avg_nodes_per_move: {avg_nodes}")
                    print(f"     total_moves: {total_moves}")
                except KeyError as e:
                    print(f"  [失败] draw_ai_info访问测试失败: {e}")
                    results[difficulty] = False
            else:
                print("  [失败] 游戏性能统计缺少ai_stats字段")
                results[difficulty] = False
                
        except Exception as e:
            print(f"  [失败] 测试失败: {e}")
            results[difficulty] = False
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总:")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for ai, result in results.items():
        status = "[通过]" if result else "[失败]"
        print(f"{ai:25} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("="*60)
    print(f"总计: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n[成功] 所有AI系统的性能统计接口都已修复！")
        print("不再会出现KeyError: 'avg_time_per_move'错误")
        return True
    else:
        print(f"\n[警告] 还有 {failed} 个AI系统需要修复")
        return False

def test_draw_ai_info_compatibility():
    """测试与draw_ai_info的兼容性"""
    print("\n=== 测试与draw_ai_info的兼容性 ===")
    
    try:
        # 模拟draw_ai_info的调用
        from integrated_ai import draw_ai_info
        import pygame
        
        # 初始化pygame（用于字体）
        pygame.init()
        font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 18)
        
        # 创建测试游戏
        game = GomokuGame(ai_difficulty="ultimate_threat")
        
        # 创建一个虚拟的screen对象
        screen = pygame.Surface((800, 600))
        
        # 尝试调用draw_ai_info
        print("调用draw_ai_info...")
        draw_ai_info(screen, game, font, small_font)
        print("[成功] draw_ai_info调用成功，无KeyError错误")
        
        pygame.quit()
        return True
        
    except Exception as e:
        print(f"[失败] draw_ai_info调用失败: {e}")
        return False

if __name__ == "__main__":
    # 运行所有测试
    success1 = test_all_ai_performance_stats()
    success2 = test_draw_ai_info_compatibility()
    
    if success1 and success2:
        print("\n[成功] 所有测试通过！性能统计接口修复完成！")
        print("现在可以安全运行游戏：python gomoku.py")
    else:
        print("\n[警告] 还有问题需要解决")