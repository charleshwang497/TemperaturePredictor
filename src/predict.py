"""
预测脚本
用于使用训练好的模型进行温度预测
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_network import TemperaturePredictor
from src.data_preprocessing import WeatherDataPreprocessor

def load_model_and_config(model_path):
    """
    加载模型和配置
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        tuple: (predictor, config)
    """
    print(f"正在加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载模型
    predictor = TemperaturePredictor(
        input_shape=(1, 1),  # 临时值，稍后会更新
        output_size=1,
        model_type='lstm'
    )
    predictor.load_model(model_path)
    
    # 加载配置
    config_path = model_path.replace('.h5', '_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"已加载模型配置")
    else:
        print(f"警告: 未找到配置文件 {config_path}")
        config = {}
    
    return predictor, config

def prepare_prediction_data(data_path, sequence_length, prediction_days, city=None):
    """
    准备预测数据
    
    Args:
        data_path: 数据文件路径
        sequence_length: 序列长度
        prediction_days: 预测天数
        city: 城市名称（可选）
        
    Returns:
        tuple: (X, scaler_X, scaler_y, df, feature_columns)
    """
    print("正在准备预测数据...")
    
    # 加载数据
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # 如果指定了城市，过滤数据
    if city and 'city' in df.columns:
        df = df[df['city'] == city].copy()
        if df.empty:
            raise ValueError(f"未找到城市 '{city}' 的数据")
    
    # 按日期排序
    df = df.sort_values('date').reset_index(drop=True)
    
    # 创建预处理器
    preprocessor = WeatherDataPreprocessor(
        sequence_length=sequence_length,
        prediction_days=prediction_days
    )
    
    # 数据清洗和特征创建
    df_clean = preprocessor.clean_data(df)
    if df_clean is None:
        raise ValueError("数据清洗失败")
    
    df_features = preprocessor.create_features(df_clean)
    
    # 选择特征列
    exclude_cols = ['date', 'city'] + preprocessor.target_columns
    feature_columns = [col for col in df_features.columns if col not in exclude_cols]
    
    # 确保所有特征都是数值类型
    df_features = df_features[feature_columns].astype(float)
    
    # 创建scaler并拟合数据
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    scaler_X.fit(df_features)
    
    # 标准化数据
    X_scaled = scaler_X.transform(df_features)
    
    # 创建序列（使用最后sequence_length天的数据）
    if len(X_scaled) < sequence_length:
        raise ValueError(f"数据不足，需要至少 {sequence_length} 天的数据")
    
    X = X_scaled[-sequence_length:].reshape(1, sequence_length, len(feature_columns))
    
    # 创建目标scaler（用于反标准化预测结果）
    scaler_y = StandardScaler()
    if 'avg_temp' in df.columns:
        scaler_y.fit(df[['avg_temp']])
    else:
        # 如果没有温度列，使用默认scaler
        scaler_y.fit([[0], [1]])
    
    return X, scaler_X, scaler_y, df, feature_columns

def predict_temperature(model_path, data_path, city=None, sequence_length=30, 
                       prediction_days=1, num_predictions=1):
    """
    预测温度
    
    Args:
        model_path: 模型路径
        data_path: 数据路径
        city: 城市名称
        sequence_length: 序列长度
        prediction_days: 预测天数
        num_predictions: 预测次数（用于生成多个预测结果）
        
    Returns:
        dict: 预测结果
    """
    print("=== 开始温度预测 ===")
    
    # 加载模型
    predictor, config = load_model_and_config(model_path)
    
    # 准备预测数据
    X, scaler_X, scaler_y, df, feature_columns = prepare_prediction_data(
        data_path, sequence_length, prediction_days, city
    )
    
    print(f"使用数据: {len(df)} 条记录")
    print(f"预测日期: {df['date'].max() + timedelta(days=1)} 到 {df['date'].max() + timedelta(days=prediction_days)}")
    
    # 进行预测
    predictions = []
    for i in range(num_predictions):
        y_pred_scaled = predictor.predict(X)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        predictions.append(y_pred[0])
    
    # 计算统计信息
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    # 创建结果字典
    results = {
        'city': city or 'Unknown',
        'prediction_date': datetime.now().isoformat(),
        'model_type': config.get('model_type', 'Unknown'),
        'input_sequence_length': sequence_length,
        'prediction_days': prediction_days,
        'predictions': []
    }
    
    for day in range(prediction_days):
        pred_info = {
            'date': (df['date'].max() + timedelta(days=day + 1)).strftime('%Y-%m-%d'),
            'mean_temperature': float(mean_pred[day]) if len(mean_pred) > day else float(mean_pred[0]),
            'std_temperature': float(std_pred[day]) if len(std_pred) > day else float(std_pred[0]),
            'confidence_interval_95': [
                float(mean_pred[day] - 1.96 * std_pred[day]) if len(mean_pred) > day else float(mean_pred[0] - 1.96 * std_pred[0]),
                float(mean_pred[day] + 1.96 * std_pred[day]) if len(mean_pred) > day else float(mean_pred[0] + 1.96 * std_pred[0])
            ]
        }
        results['predictions'].append(pred_info)
    
    return results

def predict_with_recent_data(model_path, city='北京', days_back=60, **kwargs):
    """
    使用最近的数据进行预测
    
    Args:
        model_path: 模型路径
        city: 城市名称
        days_back: 使用最近多少天的数据
        **kwargs: 其他参数
        
    Returns:
        dict: 预测结果
    """
    print(f"=== 使用最近数据预测 {city} 的温度 ===")
    
    # 检查是否有可用的数据文件
    data_dir = 'data'
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not data_files:
        print("未找到数据文件，生成示例数据...")
        from src.data_collection import WeatherDataCollector
        collector = WeatherDataCollector()
        df = collector.load_sample_data()
        data_path = os.path.join(data_dir, 'sample_weather_data.csv')
        df.to_csv(data_path, index=False)
    else:
        # 使用最新的数据文件
        data_path = os.path.join(data_dir, data_files[0])
        print(f"使用数据文件: {data_path}")
    
    # 预测
    results = predict_temperature(
        model_path=model_path,
        data_path=data_path,
        city=city,
        **kwargs
    )
    
    return results

def print_prediction_results(results):
    """
    打印预测结果
    
    Args:
        results: 预测结果字典
    """
    print("\n" + "="*60)
    print(f"温度预测结果 - {results['city']}")
    print("="*60)
    print(f"预测时间: {results['prediction_date']}")
    print(f"模型类型: {results['model_type']}")
    print(f"输入序列长度: {results['input_sequence_length']} 天")
    print(f"预测天数: {results['prediction_days']} 天")
    print("\n详细预测:")
    print("-" * 60)
    
    for pred in results['predictions']:
        date = pred['date']
        mean_temp = pred['mean_temperature']
        std_temp = pred['std_temperature']
        ci_lower = pred['confidence_interval_95'][0]
        ci_upper = pred['confidence_interval_95'][1]
        
        print(f"日期: {date}")
        print(f"  预测温度: {mean_temp:.2f}°C")
        print(f"  标准差: {std_temp:.2f}°C")
        print(f"  95%置信区间: [{ci_lower:.2f}°C, {ci_upper:.2f}°C]")
        print()

def save_prediction_results(results, filepath):
    """
    保存预测结果
    
    Args:
        results: 预测结果字典
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"预测结果已保存到: {filepath}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用训练好的模型预测温度')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--data', type=str, help='数据文件路径')
    parser.add_argument('--city', type=str, default='北京', help='城市名称')
    parser.add_argument('--sequence_length', type=int, default=30, help='输入序列长度')
    parser.add_argument('--prediction_days', type=int, default=1, help='预测天数')
    parser.add_argument('--num_predictions', type=int, default=1, help='预测次数（用于生成置信区间）')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--use_recent', action='store_true', help='使用最近数据进行预测')
    
    args = parser.parse_args()
    
    try:
        if args.use_recent:
            # 使用最近数据进行预测
            results = predict_with_recent_data(
                model_path=args.model,
                city=args.city,
                sequence_length=args.sequence_length,
                prediction_days=args.prediction_days,
                num_predictions=args.num_predictions
            )
        else:
            # 使用指定数据进行预测
            if not args.data:
                print("错误: 请指定数据文件路径 (--data)")
                return
            
            results = predict_temperature(
                model_path=args.model,
                data_path=args.data,
                city=args.city,
                sequence_length=args.sequence_length,
                prediction_days=args.prediction_days,
                num_predictions=args.num_predictions
            )
        
        # 打印结果
        print_prediction_results(results)
        
        # 保存结果
        if args.output:
            save_prediction_results(results, args.output)
        else:
            # 自动生成输出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = 'predictions'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'prediction_{args.city}_{timestamp}.json')
            save_prediction_results(results, output_path)
        
    except Exception as e:
        print(f"预测过程中出错: {e}")
        import traceback
        traceback.print_exc()

def interactive_predict():
    """交互式预测函数"""
    print("=== 交互式温度预测 ===")
    
    # 询问模型路径
    model_path = input("请输入模型文件路径（或按Enter使用默认模型）: ").strip()
    if not model_path:
        # 查找最新的模型
        model_dir = 'models'
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
            if model_files:
                model_files.sort()
                model_path = os.path.join(model_dir, model_files[-1])
                print(f"使用最新模型: {model_path}")
            else:
                print("错误: 未找到模型文件")
                return
        else:
            print("错误: 模型目录不存在")
            return
    
    # 询问城市
    city = input("请输入城市名称（默认: 北京）: ").strip()
    if not city:
        city = '北京'
    
    # 询问预测天数
    try:
        days = input("请输入预测天数（默认: 1）: ").strip()
        prediction_days = int(days) if days else 1
    except ValueError:
        prediction_days = 1
    
    # 询问序列长度
    try:
        seq_len = input("请输入输入序列长度（默认: 30）: ").strip()
        sequence_length = int(seq_len) if seq_len else 30
    except ValueError:
        sequence_length = 30
    
    try:
        # 进行预测
        results = predict_with_recent_data(
            model_path=model_path,
            city=city,
            sequence_length=sequence_length,
            prediction_days=prediction_days,
            num_predictions=5  # 生成5次预测用于置信区间
        )
        
        # 显示结果
        print_prediction_results(results)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'predictions/interactive_prediction_{city}_{timestamp}.json'
        save_prediction_results(results, output_path)
        
    except Exception as e:
        print(f"预测失败: {e}")

if __name__ == '__main__':
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        main()
    else:
        # 进入交互模式
        interactive_predict()