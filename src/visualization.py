"""
可视化工具模块
包含各种数据可视化和结果展示功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

class WeatherVisualizer:
    """天气数据可视化工具"""
    
    def __init__(self, figsize=(12, 8), dpi=150):
        """
        初始化可视化工具
        
        Args:
            figsize: 图形大小
            dpi: 分辨率
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'primary': '#3498db',
            'secondary': '#e74c3c',
            'success': '#2ecc71',
            'warning': '#f39c12',
            'info': '#9b59b6',
            'dark': '#34495e',
            'light': '#ecf0f1'
        }
    
    def plot_weather_data_overview(self, df, save_path=None):
        """
        绘制天气数据概览图
        
        Args:
            df: 天气数据DataFrame
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle('天气数据概览', fontsize=16, fontweight='bold')
        
        # 1. 温度时间序列
        axes[0, 0].plot(df['date'], df['avg_temp'], 
                       color=self.colors['primary'], linewidth=1.5, alpha=0.8)
        axes[0, 0].set_title('平均温度变化', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('日期')
        axes[0, 0].set_ylabel('温度 (°C)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 温度分布直方图
        axes[0, 1].hist(df['avg_temp'], bins=50, 
                       color=self.colors['secondary'], alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('温度分布', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('温度 (°C)')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 湿度和气压
        ax3 = axes[1, 0]
        if 'humidity' in df.columns:
            ax3.plot(df['date'], df['humidity'], 
                    color=self.colors['success'], linewidth=1, alpha=0.7, label='湿度 (%)')
        if 'pressure' in df.columns:
            ax4 = ax3.twinx()
            ax4.plot(df['date'], df['pressure'], 
                    color=self.colors['warning'], linewidth=1, alpha=0.7, label='气压 (hPa)')
            ax4.set_ylabel('气压 (hPa)', color=self.colors['warning'])
        
        ax3.set_title('湿度和气压', fontsize=12, fontweight='bold')
        ax3.set_xlabel('日期')
        ax3.set_ylabel('湿度 (%)', color=self.colors['success'])
        ax3.grid(True, alpha=0.3)
        
        # 4. 季节性分析
        if 'season' in df.columns:
            seasonal_temp = df.groupby('season')['avg_temp'].agg(['mean', 'std'])
            seasons = seasonal_temp.index
            means = seasonal_temp['mean']
            stds = seasonal_temp['std']
            
            bars = axes[1, 1].bar(seasons, means, 
                                color=[self.colors['primary'], self.colors['success'], 
                                      self.colors['warning'], self.colors['secondary']],
                                alpha=0.8, capsize=5)
            axes[1, 1].errorbar(seasons, means, yerr=stds, fmt='none', 
                               color='black', capsize=5)
            axes[1, 1].set_title('季节性温度分析', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('季节')
            axes[1, 1].set_ylabel('平均温度 (°C)')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # 如果没有季节数据，绘制月度分析
            df['month'] = pd.to_datetime(df['date']).dt.month
            monthly_temp = df.groupby('month')['avg_temp'].agg(['mean', 'std'])
            months = monthly_temp.index
            means = monthly_temp['mean']
            stds = monthly_temp['std']
            
            bars = axes[1, 1].bar(months, means, 
                                color=self.colors['info'], alpha=0.8)
            axes[1, 1].errorbar(months, means, yerr=stds, fmt='none', 
                               color='black', capsize=5)
            axes[1, 1].set_title('月度温度分析', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('月份')
            axes[1, 1].set_ylabel('平均温度 (°C)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"天气数据概览图已保存到: {save_path}")
        
        plt.show()
        return fig
    
    def plot_training_history(self, history, save_path=None):
        """
        绘制训练历史
        
        Args:
            history: 训练历史（dict或History对象）
            save_path: 保存路径
        """
        if hasattr(history, 'history'):
            history_dict = history.history
        else:
            history_dict = history
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle('模型训练历史', fontsize=16, fontweight='bold')
        
        # 训练轮数
        epochs = range(1, len(history_dict.get('loss', [])) + 1)
        
        # 1. 损失曲线
        if 'loss' in history_dict:
            axes[0, 0].plot(epochs, history_dict['loss'], 
                           color=self.colors['primary'], linewidth=2, 
                           marker='o', markersize=4, label='训练损失')
        if 'val_loss' in history_dict:
            axes[0, 0].plot(epochs, history_dict['val_loss'], 
                           color=self.colors['secondary'], linewidth=2, 
                           marker='s', markersize=4, label='验证损失')
        axes[0, 0].set_title('损失曲线', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('轮数')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. MAE曲线
        if 'mae' in history_dict:
            axes[0, 1].plot(epochs, history_dict['mae'], 
                           color=self.colors['success'], linewidth=2, 
                           marker='o', markersize=4, label='训练MAE')
        if 'val_mae' in history_dict:
            axes[0, 1].plot(epochs, history_dict['val_mae'], 
                           color=self.colors['warning'], linewidth=2, 
                           marker='s', markersize=4, label='验证MAE')
        axes[0, 1].set_title('平均绝对误差', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('轮数')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 学习率曲线（如果有）
        if 'lr' in history_dict:
            axes[1, 0].plot(epochs, history_dict['lr'], 
                           color=self.colors['info'], linewidth=2, 
                           marker='o', markersize=4)
            axes[1, 0].set_title('学习率变化', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('轮数')
            axes[1, 0].set_ylabel('学习率')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            # 如果没有学习率，绘制MAPE曲线
            if 'mape' in history_dict:
                axes[1, 0].plot(epochs, history_dict['mape'], 
                               color=self.colors['info'], linewidth=2, 
                               marker='o', markersize=4, label='训练MAPE')
            if 'val_mape' in history_dict:
                axes[1, 0].plot(epochs, history_dict['val_mape'], 
                               color=self.colors['dark'], linewidth=2, 
                               marker='s', markersize=4, label='验证MAPE')
            axes[1, 0].set_title('平均绝对百分比误差', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('轮数')
            axes[1, 0].set_ylabel('MAPE (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. R²分数（如果有）
        if 'r2' in history_dict:
            axes[1, 1].plot(epochs, history_dict['r2'], 
                           color=self.colors['dark'], linewidth=2, 
                           marker='o', markersize=4, label='训练R²')
            axes[1, 1].set_title('R² 分数', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('轮数')
            axes[1, 1].set_ylabel('R²')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # 如果没有R²，绘制训练时间
            axes[1, 1].text(0.5, 0.5, '训练完成！', 
                           ha='center', va='center', 
                           fontsize=20, fontweight='bold',
                           transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('训练状态', fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"训练历史图已保存到: {save_path}")
        
        plt.show()
        return fig
    
    def plot_predictions(self, y_true, y_pred, dates=None, save_path=None):
        """
        绘制预测结果
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            dates: 日期（可选）
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle('温度预测结果分析', fontsize=16, fontweight='bold')
        
        # 1. 预测 vs 真实 散点图
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=30, 
                          color=self.colors['primary'])
        
        # 添加完美预测线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 
                       'r--', linewidth=2, label='完美预测')
        
        axes[0, 0].set_xlabel('真实温度 (°C)')
        axes[0, 0].set_ylabel('预测温度 (°C)')
        axes[0, 0].set_title('预测值 vs 真实值', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 残差图
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=30, 
                          color=self.colors['secondary'])
        axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('预测温度 (°C)')
        axes[0, 1].set_ylabel('残差 (°C)')
        axes[0, 1].set_title('残差分析', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 时间序列图
        sample_size = min(100, len(y_true))
        if dates is None:
            x_axis = range(sample_size)
            x_label = '样本序号'
        else:
            x_axis = dates[:sample_size]
            x_label = '日期'
        
        axes[1, 0].plot(x_axis, y_true[:sample_size], 
                       color=self.colors['success'], linewidth=2, 
                       marker='o', markersize=3, label='真实值')
        axes[1, 0].plot(x_axis, y_pred[:sample_size], 
                       color=self.colors['warning'], linewidth=2, 
                       marker='s', markersize=3, label='预测值')
        axes[1, 0].set_xlabel(x_label)
        axes[1, 0].set_ylabel('温度 (°C)')
        axes[1, 0].set_title('时间序列预测', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 如果日期太多，只显示部分刻度
        if dates is not None and len(x_axis) > 10:
            step = len(x_axis) // 10
            axes[1, 0].set_xticks(x_axis[::step])
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 误差分布
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, 
                       color=self.colors['info'], edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('残差 (°C)')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].set_title('残差分布', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加统计信息
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        textstr = f'MAE: {mae:.2f}°C\nRMSE: {rmse:.2f}°C'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        axes[1, 1].text(0.05, 0.95, textstr, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"预测结果图已保存到: {save_path}")
        
        plt.show()
        return fig
    
    def plot_feature_importance(self, feature_columns, importance_scores, save_path=None):
        """
        绘制特征重要性
        
        Args:
            feature_columns: 特征列名
            importance_scores: 重要性分数
            save_path: 保存路径
        """
        # 排序
        indices = np.argsort(importance_scores)[::-1][:20]  # 取前20个
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 创建水平条形图
        y_pos = np.arange(len(indices))
        bars = ax.barh(y_pos, importance_scores[indices], 
                      color=self.colors['primary'], alpha=0.8)
        
        # 设置标签
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_columns[i] for i in indices])
        ax.set_xlabel('重要性分数')
        ax.set_title('特征重要性分析', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, importance_scores[indices])):
            ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"特征重要性图已保存到: {save_path}")
        
        plt.show()
        return fig
    
    def plot_prediction_comparison(self, cities_results, save_path=None):
        """
        绘制多城市预测结果对比
        
        Args:
            cities_results: 字典，键为城市名，值为预测结果
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        colors = list(self.colors.values())
        
        for i, (city, results) in enumerate(cities_results.items()):
            predictions = results['predictions']
            dates = [pred['date'] for pred in predictions]
            temps = [pred['mean_temperature'] for pred in predictions]
            
            ax.plot(dates, temps, 
                   color=colors[i % len(colors)], 
                   linewidth=2, marker='o', markersize=6,
                   label=city)
        
        ax.set_xlabel('日期')
        ax.set_ylabel('预测温度 (°C)')
        ax.set_title('多城市温度预测对比', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 旋转日期标签
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"多城市对比图已保存到: {save_path}")
        
        plt.show()
        return fig
    
    def plot_confidence_interval(self, predictions, save_path=None):
        """
        绘制预测置信区间
        
        Args:
            predictions: 预测结果列表
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        dates = [pred['date'] for pred in predictions]
        mean_temps = [pred['mean_temperature'] for pred in predictions]
        lower_bounds = [pred['confidence_interval_95'][0] for pred in predictions]
        upper_bounds = [pred['confidence_interval_95'][1] for pred in predictions]
        
        # 绘制置信区间
        ax.fill_between(dates, lower_bounds, upper_bounds, 
                       alpha=0.3, color=self.colors['light'], 
                       label='95%置信区间')
        
        # 绘制预测线
        ax.plot(dates, mean_temps, 
               color=self.colors['primary'], linewidth=3, 
               marker='o', markersize=8, label='预测温度')
        
        ax.set_xlabel('日期')
        ax.set_ylabel('温度 (°C)')
        ax.set_title('温度预测与置信区间', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 旋转日期标签
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"置信区间图已保存到: {save_path}")
        
        plt.show()
        return fig
    
    def create_prediction_report(self, prediction_results, save_dir='reports'):
        """
        创建预测报告
        
        Args:
            prediction_results: 预测结果字典
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 保存预测数据
        csv_path = os.path.join(save_dir, f'prediction_report_{timestamp}.csv')
        
        predictions = prediction_results['predictions']
        df_pred = pd.DataFrame(predictions)
        df_pred.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"预测报告已保存到: {csv_path}")
        
        # 2. 创建可视化图表
        # 置信区间图
        ci_path = os.path.join(save_dir, f'confidence_interval_{timestamp}.png')
        self.plot_confidence_interval(predictions, save_path=ci_path)
        
        # 3. 创建HTML报告
        html_path = os.path.join(save_dir, f'prediction_report_{timestamp}.html')
        self.create_html_report(prediction_results, html_path)
        print(f"HTML报告已保存到: {html_path}")
    
    def create_html_report(self, prediction_results, html_path):
        """
        创建HTML格式的预测报告
        
        Args:
            prediction_results: 预测结果字典
            html_path: HTML文件路径
        """
        predictions = prediction_results['predictions']
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>温度预测报告 - {prediction_results['city']}</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 800px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                .info-section {{
                    background-color: #ecf0f1;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .prediction-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .prediction-table th,
                .prediction-table td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: center;
                }}
                .prediction-table th {{
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }}
                .prediction-table tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .highlight {{
                    background-color: #2ecc71;
                    color: white;
                    font-weight: bold;
                }}
                .footer {{
                    margin-top: 30px;
                    text-align: center;
                    color: #7f8c8d;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>温度预测报告</h1>
                
                <div class="info-section">
                    <h3>预测信息</h3>
                    <p><strong>城市:</strong> {prediction_results['city']}</p>
                    <p><strong>预测时间:</strong> {prediction_results['prediction_date']}</p>
                    <p><strong>模型类型:</strong> {prediction_results['model_type']}</p>
                    <p><strong>输入序列长度:</strong> {prediction_results['input_sequence_length']} 天</p>
                    <p><strong>预测天数:</strong> {prediction_results['prediction_days']} 天</p>
                </div>
                
                <h3>详细预测结果</h3>
                <table class="prediction-table">
                    <thead>
                        <tr>
                            <th>日期</th>
                            <th>预测温度 (°C)</th>
                            <th>标准差 (°C)</th>
                            <th>95%置信区间 (°C)</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for pred in predictions:
            html_content += f"""
                        <tr>
                            <td>{pred['date']}</td>
                            <td class="highlight">{pred['mean_temperature']:.2f}</td>
                            <td>{pred['std_temperature']:.2f}</td>
                            <td>[{pred['confidence_interval_95'][0]:.2f}, {pred['confidence_interval_95'][1]:.2f}]</td>
                        </tr>
            """
        
        html_content += f"""
                    </tbody>
                </table>
                
                <div class="footer">
                    <p>本报告由温度预测系统自动生成</p>
                    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

def main():
    """主函数 - 测试可视化功能"""
    print("=== 测试可视化功能 ===")
    
    # 创建可视化工具
    visualizer = WeatherVisualizer()
    
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # 模拟温度数据
    day_of_year = dates.dayofyear
    base_temp = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temp_noise = np.random.normal(0, 5, len(dates))
    avg_temp = base_temp + temp_noise
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        'avg_temp': avg_temp,
        'humidity': np.random.normal(60, 15, len(dates)),
        'pressure': np.random.normal(1013, 20, len(dates)),
        'season': pd.to_datetime(dates).month.map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                                  3: 'Spring', 4: 'Spring', 5: 'Spring',
                                                  6: 'Summer', 7: 'Summer', 8: 'Summer',
                                                  9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})
    })
    
    print("1. 绘制天气数据概览...")
    visualizer.plot_weather_data_overview(df, save_path='weather_overview.png')
    
    print("\n2. 绘制训练历史...")
    # 模拟训练历史
    history = {
        'loss': np.random.exponential(0.1, 100)[::-1] + np.random.normal(0, 0.01, 100),
        'val_loss': np.random.exponential(0.12, 100)[::-1] + np.random.normal(0, 0.01, 100),
        'mae': np.random.exponential(0.08, 100)[::-1] + np.random.normal(0, 0.005, 100),
        'val_mae': np.random.exponential(0.09, 100)[::-1] + np.random.normal(0, 0.005, 100)
    }
    visualizer.plot_training_history(history, save_path='training_history.png')
    
    print("\n3. 绘制预测结果...")
    # 模拟预测结果
    y_true = np.random.normal(20, 10, 200)
    y_pred = y_true + np.random.normal(0, 2, 200)
    visualizer.plot_predictions(y_true, y_pred, save_path='predictions.png')
    
    print("\n4. 绘制特征重要性...")
    features = [f'feature_{i}' for i in range(20)]
    importance = np.random.exponential(0.1, 20)
    visualizer.plot_feature_importance(features, importance, save_path='feature_importance.png')
    
    print("\n5. 绘制置信区间...")
    predictions = []
    base_date = datetime.now()
    for i in range(7):
        date = base_date + timedelta(days=i)
        predictions.append({
            'date': date.strftime('%Y-%m-%d'),
            'mean_temperature': 20 + 5 * np.sin(i * np.pi / 7),
            'std_temperature': 2.0,
            'confidence_interval_95': [15 + 5 * np.sin(i * np.pi / 7), 
                                      25 + 5 * np.sin(i * np.pi / 7)]
        })
    
    visualizer.plot_confidence_interval(predictions, save_path='confidence_interval.png')
    
    print("\n6. 创建预测报告...")
    results = {
        'city': '北京',
        'prediction_date': datetime.now().isoformat(),
        'model_type': 'LSTM',
        'input_sequence_length': 30,
        'prediction_days': 7,
        'predictions': predictions
    }
    visualizer.create_prediction_report(results)
    
    print("\n=== 可视化测试完成 ===")

if __name__ == '__main__':
    main()