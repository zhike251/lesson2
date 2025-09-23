"""
训练管理系统
整合自我对弈、数据管理、分析和可视化功能

作者：Claude AI Engineer
日期：2025-09-22
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from auto_selfplay import AutoSelfPlay
from data_analyzer import DataAnalyzer
from data_visualizer import DataVisualizer
from data_manager import DataManager
from training_data_collector import TrainingDataCollector
from self_play_engine import SelfPlayConfig

class TrainingManager:
    """训练管理系统"""
    
    def __init__(self, workspace_dir: str = "training_workspace"):
        """
        初始化训练管理系统
        
        Args:
            workspace_dir: 工作空间目录
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)
        
        # 子目录
        self.data_dir = self.workspace_dir / "data"
        self.models_dir = self.workspace_dir / "models"
        self.analysis_dir = self.workspace_dir / "analysis"
        self.reports_dir = self.workspace_dir / "reports"
        
        # 创建目录结构
        for dir_path in [self.data_dir, self.models_dir, self.analysis_dir, self.reports_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 初始化组件
        self.data_manager = DataManager(str(self.data_dir))
        self.data_analyzer = DataAnalyzer(str(self.data_dir))
        self.data_visualizer = DataVisualizer()
        
        # 状态跟踪
        self.training_session = {
            'session_id': int(time.time()),
            'start_time': time.time(),
            'phases_completed': [],
            'current_phase': None,
            'status': 'initialized'
        }
        
        print(f"🚀 训练管理系统初始化完成")
        print(f"📁 工作空间: {self.workspace_dir}")
    
    def run_selfplay_training(self, config: Dict) -> Dict:
        """
        运行自我对弈训练
        
        Args:
            config: 自我对弈配置
            
        Returns:
            训练结果
        """
        print("\\n" + "="*60)
        print("🎮 开始自我对弈训练阶段")
        print("="*60)
        
        self.training_session['current_phase'] = 'selfplay'
        
        # 创建自我对弈配置
        selfplay_config = SelfPlayConfig(**config)
        
        # 运行自我对弈
        auto_play = AutoSelfPlay()
        auto_play.config = selfplay_config
        
        results = auto_play.run(
            mode=config.get('mode', 'single'),
            output_dir=str(self.data_dir)
        )
        
        if results:
            self.training_session['phases_completed'].append('selfplay')
            print("✅ 自我对弈训练完成")
        else:
            print("❌ 自我对弈训练失败")
        
        return results
    
    def run_data_analysis(self, file_pattern: str = "*.json") -> Dict:
        """
        运行数据分析
        
        Args:
            file_pattern: 文件模式
            
        Returns:
            分析结果
        """
        print("\\n" + "="*60)
        print("📊 开始数据分析阶段")
        print("="*60)
        
        self.training_session['current_phase'] = 'analysis'
        
        # 运行完整分析
        analysis_output = self.analysis_dir / f"session_{self.training_session['session_id']}"
        results = self.data_analyzer.run_full_analysis(
            file_pattern=file_pattern,
            output_dir=str(analysis_output)
        )
        
        if results:
            self.training_session['phases_completed'].append('analysis')
            print("✅ 数据分析完成")
        else:
            print("❌ 数据分析失败")
        
        return results
    
    def run_data_management(self, operations: List[str]) -> Dict:
        """
        运行数据管理操作
        
        Args:
            operations: 操作列表 ['clean', 'split', 'augment', 'merge']
            
        Returns:
            操作结果
        """
        print("\\n" + "="*60)
        print("🗂️ 开始数据管理阶段")
        print("="*60)
        
        self.training_session['current_phase'] = 'data_management'
        results = {}
        
        # 添加数据文件到管理器
        data_files = list(self.data_dir.glob("*.json"))
        for data_file in data_files:
            self.data_manager.add_data_file(str(data_file))
        
        # 执行操作
        if 'clean' in operations:
            print("🧹 清理数据...")
            clean_results = self.data_manager.clean_data({
                'min_game_length': 10,
                'max_game_length': 400,
                'min_thinking_time': 0.1,
                'remove_duplicates': True
            })
            results['clean'] = clean_results
        
        if 'split' in operations:
            print("✂️ 分割数据集...")
            split_results = self.data_manager.split_dataset(
                train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
            )
            results['split'] = split_results
        
        if 'augment' in operations and 'split' in results:
            print("🔄 数据增强...")
            aug_results = self.data_manager.augment_dataset(
                results['split']['train_file'],
                str(self.data_dir / "augmented_train.json"),
                augmentation_factor=4
            )
            results['augment'] = aug_results
        
        if 'export' in operations:
            print("📤 导出训练样本...")
            export_dir = self.data_dir / "exported_samples"
            export_results = self.data_manager.export_training_samples(
                str(self.data_dir / "augmented_train.json") if 'augment' in results 
                else results.get('split', {}).get('train_file', ''),
                str(export_dir),
                format_type="numpy"
            )
            results['export'] = export_results
        
        if results:
            self.training_session['phases_completed'].append('data_management')
            print("✅ 数据管理完成")
        
        return results
    
    def generate_comprehensive_report(self) -> str:
        """
        生成综合训练报告
        
        Returns:
            报告文件路径
        """
        print("\\n" + "="*60)
        print("📝 生成综合训练报告")
        print("="*60)
        
        timestamp = int(time.time())
        report_file = self.reports_dir / f"training_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 五子棋AI训练综合报告\\n\\n")
            
            # 会话信息
            f.write("## 训练会话信息\\n\\n")
            f.write(f"- **会话ID**: {self.training_session['session_id']}\\n")
            f.write(f"- **开始时间**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.training_session['start_time']))}\\n")
            f.write(f"- **总耗时**: {time.time() - self.training_session['start_time']:.2f} 秒\\n")
            f.write(f"- **完成阶段**: {', '.join(self.training_session['phases_completed'])}\\n")
            f.write(f"- **工作空间**: {self.workspace_dir}\\n\\n")
            
            # 数据统计
            f.write("## 数据统计概览\\n\\n")
            stats = self.data_manager.get_data_statistics()
            f.write(f"- **总游戏数**: {stats.total_games:,}\\n")
            f.write(f"- **总移动数**: {stats.total_moves:,}\\n")
            f.write(f"- **平均游戏长度**: {stats.avg_game_length:.1f}\\n")
            f.write(f"- **数据大小**: {stats.data_size_mb:.2f} MB\\n")
            f.write(f"- **黑棋胜率**: {stats.black_wins / max(1, stats.total_games) * 100:.1f}%\\n")
            f.write(f"- **白棋胜率**: {stats.white_wins / max(1, stats.total_games) * 100:.1f}%\\n")
            f.write(f"- **平局率**: {stats.draws / max(1, stats.total_games) * 100:.1f}%\\n\\n")
            
            # 文件列表
            f.write("## 生成的文件\\n\\n")
            f.write("### 数据文件\\n")
            for data_file in self.data_dir.glob("*"):
                f.write(f"- `{data_file.name}` ({data_file.stat().st_size / 1024:.1f} KB)\\n")
            
            f.write("\\n### 分析结果\\n")
            for analysis_file in self.analysis_dir.rglob("*"):
                if analysis_file.is_file():
                    f.write(f"- `{analysis_file.relative_to(self.workspace_dir)}`\\n")
            
            f.write("\\n## 使用建议\\n\\n")
            f.write("1. **数据质量**: 建议定期清理和验证训练数据\\n")
            f.write("2. **模型训练**: 使用导出的numpy格式数据进行神经网络训练\\n")
            f.write("3. **超参数调优**: 根据分析结果调整MCTS参数\\n")
            f.write("4. **持续改进**: 定期重新运行自我对弈以获取更多数据\\n\\n")
            
            f.write("## 下一步操作\\n\\n")
            f.write("```bash\\n")
            f.write("# 使用训练数据训练神经网络\\n")
            f.write("python train_neural_network.py --data-dir training_workspace/data/exported_samples\\n")
            f.write("\\n")
            f.write("# 评估模型性能\\n")
            f.write("python evaluate_model.py --model-path training_workspace/models/latest.pth\\n")
            f.write("\\n")
            f.write("# 继续自我对弈训练\\n")
            f.write("python auto_selfplay.py --games 1000 --output-dir training_workspace/data\\n")
            f.write("```\\n")
        
        print(f"✅ 综合报告已生成: {report_file}")
        return str(report_file)
    
    def run_full_pipeline(self, pipeline_config: Dict) -> Dict:
        """
        运行完整训练流水线
        
        Args:
            pipeline_config: 流水线配置
            
        Returns:
            执行结果
        """
        print("\\n" + "🌟"*30)
        print("🚀 开始完整训练流水线")
        print("🌟"*30)
        
        results = {}
        
        try:
            # 1. 自我对弈训练
            if pipeline_config.get('run_selfplay', True):
                selfplay_results = self.run_selfplay_training(
                    pipeline_config.get('selfplay_config', {})
                )
                results['selfplay'] = selfplay_results
            
            # 2. 数据管理
            if pipeline_config.get('run_data_management', True):
                management_results = self.run_data_management(
                    pipeline_config.get('management_operations', ['clean', 'split', 'augment', 'export'])
                )
                results['data_management'] = management_results
            
            # 3. 数据分析
            if pipeline_config.get('run_analysis', True):
                analysis_results = self.run_data_analysis()
                results['analysis'] = analysis_results
            
            # 4. 生成报告
            if pipeline_config.get('generate_report', True):
                report_file = self.generate_comprehensive_report()
                results['report'] = report_file
            
            self.training_session['status'] = 'completed'
            print("\\n🎉 完整训练流水线执行完成!")
            
        except Exception as e:
            self.training_session['status'] = 'failed'
            print(f"\\n❌ 训练流水线执行失败: {e}")
            results['error'] = str(e)
        
        return results
    
    def save_session_config(self, config: Dict):
        """保存会话配置"""
        config_file = self.workspace_dir / f"session_{self.training_session['session_id']}_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def load_session_config(self, session_id: int) -> Optional[Dict]:
        """加载会话配置"""
        config_file = self.workspace_dir / f"session_{session_id}_config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

def create_default_config() -> Dict:
    """创建默认配置"""
    return {
        'selfplay_config': {
            'num_games': 100,
            'mcts_simulations': 800,
            'thinking_time_limit': 3.0,
            'temperature': 1.0,
            'temperature_decay': 0.95,
            'min_temperature': 0.1,
            'parallel_games': 1,
            'max_threads': 4,
            'mode': 'single'
        },
        'management_operations': ['clean', 'split', 'augment', 'export'],
        'run_selfplay': True,
        'run_data_management': True,
        'run_analysis': True,
        'generate_report': True
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="五子棋AI训练管理系统")
    
    parser.add_argument("--workspace", type=str, default="training_workspace",
                       help="工作空间目录 (默认: training_workspace)")
    parser.add_argument("--config", type=str,
                       help="配置文件路径")
    parser.add_argument("--games", type=int, default=100,
                       help="自我对弈游戏数 (默认: 100)")
    parser.add_argument("--simulations", type=int, default=800,
                       help="MCTS模拟次数 (默认: 800)")
    parser.add_argument("--mode", choices=["single", "multi", "batch"], 
                       default="single", help="运行模式")
    parser.add_argument("--skip-selfplay", action="store_true",
                       help="跳过自我对弈阶段")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="跳过数据分析阶段")
    parser.add_argument("--only-report", action="store_true",
                       help="仅生成报告")
    
    args = parser.parse_args()
    
    # 创建训练管理器
    manager = TrainingManager(args.workspace)
    
    # 加载或创建配置
    if args.config and Path(args.config).exists():
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = create_default_config()
        
        # 更新配置
        config['selfplay_config'].update({
            'num_games': args.games,
            'mcts_simulations': args.simulations,
            'mode': args.mode
        })
        
        config['run_selfplay'] = not args.skip_selfplay
        config['run_analysis'] = not args.skip_analysis
    
    # 保存配置
    manager.save_session_config(config)
    
    if args.only_report:
        # 仅生成报告
        report_file = manager.generate_comprehensive_report()
        print(f"\\n📝 报告已生成: {report_file}")
    else:
        # 运行完整流水线
        results = manager.run_full_pipeline(config)
        
        if results.get('error'):
            print(f"\\n❌ 执行失败: {results['error']}")
            sys.exit(1)
        else:
            print(f"\\n✅ 训练管理任务完成!")
            print(f"📁 结果保存在: {manager.workspace_dir}")

if __name__ == "__main__":
    main()