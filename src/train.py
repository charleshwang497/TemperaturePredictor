"""
训练脚本
用于训练温度预测模型
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection import WeatherDataCollector
from src.data_preprocessing import WeatherDataPreprocessor
from src.neural_network import TemperaturePredictor

def load_or_create_data(data_dir='data', sequence_length=30, prediction_days=1):
    """
    加载或创建训练数据
    
    Args:
        data_dir: 数据目录
        sequence_length: 序列长度
        prediction_days: 预测天数
        
    Returns:
        dict: 包含所有训练数据的字典
    """
    print("=== 准备训练数据 ===")
    
    # 检查是否已有预处理好的数据
    processed_files = [
        'X_train_scaled.npy', 'X_val_scaled.npy', 'X_test_scaled.npy',
        'y_train_scaled.npy', 'y_val_scaled.npy', 'y_test_scaled.npy',
        'preprocessing_info.json'
    ]
    
    processed_dir = os.path.join(data_dir, 'processed')
    all_files_exist = all(
        os.path.exists(os.path.join(processed_dir, f)) 
        for f in processed_files
    )
    
    if all_files_exist:
        print("发现已处理的数据，直接加载...")
        
        # 加载数据
        data = {}
        for filename in processed_files:
            filepath = os.path.join(processed_dir, filename)
            if filename.endswith('.npy'):
                data[filename.replace('.npy', '')] = np.load(filepath)
            elif filename.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data['preprocessing_info'] = json.load(f)
        
        return data
    
    else:
        print("未找到处理好的数据，开始数据处理流程...")
        
        # 1. 收集数据
        collector = WeatherDataCollector(data_dir)
        
        # 检查是否有数据文件
        data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        if not data_files:
            print("未找到数据文件，生成示例数据...")
            df = collector.load_sample_data()
            data_filepath = os.path.join(data_dir, 'sample_weather_data.csv')
        else:
            # 使用最新的数据文件
            data_filepath = os.path.join(data_dir, data_files[0])
            print(f"使用数据文件: {data_filepath}")
        
        # 2. 预处理数据
        preprocessor = WeatherDataPreprocessor(
            sequence_length=sequence_length,
            prediction_days=prediction_days
        )
        
        result = preprocessor.preprocess_pipeline(data_filepath)
        
        if result is None:
            raise ValueError("数据预处理失败")
        
        # 3. 保存预处理数据
        preprocessor.save_preprocessed_data(result, processed_dir)
        
        # 返回需要的数据
        return {
            'X_train_scaled': result['X_train_scaled'],
            'X_val_scaled': result['X_val_scaled'],
            'X_test_scaled': result['X_test_scaled'],
            'y_train_scaled': result['y_train_scaled'],
            'y_val_scaled': result['y_val_scaled'],
            'y_test_scaled': result['y_test_scaled'],
            'scaler_X': result['scaler_X'],
            'scaler_y': result['scaler_y'],
            'preprocessing_info': result['preprocessing_info']
        }

def train_model(data, model_type='lstm', epochs=100, batch_size=32, learning_rate=0.001, 
                model_dir='models', save_model=True):
    """
    训练模型
    
    Args:
        data: 训练数据字典
        model_type: 模型类型
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        model_dir: 模型保存目录
        save_model: 是否保存模型
        
    Returns:
        tuple: (predictor, metrics, history)
    """
    print("=== 开始训练模型 ===")
    
    # 获取数据
    X_train = data['X_train_scaled']
    y_train = data['y_train_scaled']
    X_val = data['X_val_scaled']
    y_val = data['y_val_scaled']
    X_test = data['X_test_scaled']
    y_test = data['y_test_scaled']
    scaler_y = data['scaler_y']
    
    print(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")
    print(f"验证数据形状: X={X_val.shape}, y={y_val.shape}")
    print(f"测试数据形状: X={X_test.shape}, y={y_test.shape}")
    
    # 创建模型
    input_shape = X_train.shape[1:]  # (sequence_length, num_features)
    output_size = y_train.shape[1]   # 输出大小
    
    predictor = TemperaturePredictor(
        input_shape=input_shape,
        output_size=output_size,
        model_type=model_type
    )
    
    # 编译模型
    predictor.compile_model(learning_rate=learning_rate)
    
    # 显示模型架构
    print("\n模型架构:")
    print(predictor.get_model_summary())
    
    # 训练模型
    history = predictor.train_model(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # 评估模型
    metrics, y_test_original, y_pred_original = predictor.evaluate_model(
        X_test, y_test, scaler_y
    )
    
    # 保存模型和训练历史
    if save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"temperature_model_{model_type}_{timestamp}.h5"
        model_filepath = os.path.join(model_dir, model_filename)
        predictor.save_model(model_filepath)
        
        history_filename = f"training_history_{model_type}_{timestamp}.json"
        history_filepath = os.path.join(model_dir, history_filename)
        predictor.save_training_history(history_filepath)
        
        # 保存训练配置
        config = {
            'model_type': model_type,
            'input_shape': input_shape,
            'output_size': output_size,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'training_date': datetime.now().isoformat(),
            'metrics': metrics,
            'preprocessing_info': data.get('preprocessing_info', {})
        }
        
        config_filename = f"model_config_{model_type}_{timestamp}.json"
        config_filepath = os.path.join(model_dir, config_filename)
        
        with open(config_filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"\n模型配置已保存到: {config_filepath}")
    
    return predictor, metrics, history

def plot_training_history(history, save_path=None):
    """
    绘制训练历史
    
    Args:
        history: 训练历史
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    axes[0].plot(history.history['loss'], label='训练损失', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='验证损失', linewidth=2)
    axes[0].set_title('模型损失', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('轮数')
    axes[0].set_ylabel('损失')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE曲线
    axes[1].plot(history.history['mae'], label='训练MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='验证MAE', linewidth=2)
    axes[1].set_title('平均绝对误差', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('轮数')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存到: {save_path}")
    
    plt.show()

def plot_predictions(y_true, y_pred, save_path=None):
    """
    绘制预测结果
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 预测 vs 真实
    axes[0].scatter(y_true, y_pred, alpha=0.6, s=20)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2)
    axes[0].set_xlabel('真实温度 (°C)')
    axes[0].set_ylabel('预测温度 (°C)')
    axes[0].set_title('预测值 vs 真实值', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 时间序列图
    sample_size = min(100, len(y_true))
    sample_indices = np.random.choice(len(y_true), sample_size, replace=False)
    
    axes[1].plot(y_true[sample_indices], label='真实值', linewidth=2, marker='o', markersize=3)
    axes[1].plot(y_pred[sample_indices], label='预测值', linewidth=2, marker='s', markersize=3)
    axes[1].set_xlabel('样本')
    axes[1].set_ylabel('温度 (°C)')
    axes[1].set_title('时间序列预测示例', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果图已保存到: {save_path}")
    
    plt.show()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练温度预测模型')
    parser.add_argument('--data_dir', type=str, default='data', help='数据目录')
    parser.add_argument('--model_dir', type=str, default='models', help='模型保存目录')
    parser.add_argument('--model_type', type=str, default='lstm', 
                       choices=['lstm', 'gru', 'dense', 'cnn_lstm', 'bidirectional_lstm'],
                       help='模型类型')
    parser.add_argument('--sequence_length', type=int, default=30, help='输入序列长度')
    parser.add_argument('--prediction_days', type=int, default=1, help='预测天数')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--no_save', action='store_true', help='不保存模型')
    parser.add_argument('--plot', action='store_true', help='绘制结果')
    
    args = parser.parse_args()
    
    try:
        # 准备数据
        data = load_or_create_data(
            data_dir=args.data_dir,
            sequence_length=args.sequence_length,
            prediction_days=args.prediction_days
        )
        
        if data is None:
            print("错误: 无法准备训练数据")
            return
        
        # 训练模型
        predictor, metrics, history = train_model(
            data=data,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            model_dir=args.model_dir,
            save_model=not args.no_save
        )
        
        # 绘制结果
        if args.plot:
            # 绘制训练历史
            history_plot_path = os.path.join(args.model_dir, f'training_history_{args.model_type}.png')
            plot_training_history(history, save_path=history_plot_path)
            
            # 绘制预测结果
            X_test = data['X_test_scaled']
            y_test = data['y_test_scaled']
            scaler_y = data['scaler_y']
            
            y_pred = predictor.predict(X_test)
            
            # 反标准化
            y_test_original = scaler_y.inverse_transform(y_test)
            y_pred_original = scaler_y.inverse_transform(y_pred)
            
            predictions_plot_path = os.path.join(args.model_dir, f'predictions_{args.model_type}.png')
            plot_predictions(y_test_original, y_pred_original, save_path=predictions_plot_path)
        
        print("\n=== 训练完成 ===")
        print(f"模型类型: {args.model_type}")
        print(f"最终性能指标:")
        for metric, value in metrics.items():
            if metric == 'R2':
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value:.2f}")
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()