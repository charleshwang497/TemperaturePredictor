"""
主程序入口
整合所有模块，提供完整的温度预测功能
"""

import os
import sys
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_welcome():
    """打印欢迎信息"""
    welcome_message = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           明日气温预测器 - Neural Weather Predictor          ║
    ║                                                              ║
    ║     基于神经网络的气温预测系统                                 ║
    ║     支持多种神经网络架构                                      ║
    ║     提供完整的机器学习流程                                    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    功能说明:
    1. 数据采集 - 从历史天气API获取数据
    2. 数据预处理 - 特征工程和数据清洗
    3. 模型训练 - 训练神经网络模型
    4. 温度预测 - 预测未来气温
    5. 结果可视化 - 展示预测结果
    
    支持的模型类型:
    • LSTM - 长短期记忆网络
    • GRU - 门控循环单元
    • Dense - 全连接神经网络
    • CNN-LSTM - 混合模型
    • Bidirectional LSTM - 双向LSTM
    
    """
    print(welcome_message)

def show_menu():
    """显示主菜单"""
    menu = """
    ┌────────────────────────────────────────────────────────────┐
    │                        主菜单                              │
    ├────────────────────────────────────────────────────────────┤
    │  1. 数据采集与预处理                                       │
    │  2. 训练模型                                               │
    │  3. 进行预测                                               │
    │  4. 可视化结果                                             │
    │  5. 完整流程演示                                           │
    │  0. 退出程序                                               │
    └────────────────────────────────────────────────────────────┘
    """
    print(menu)

def option_1_data_processing():
    """选项1: 数据采集与预处理"""
    print("\n=== 数据采集与预处理 ===")
    
    from src.data_collection import WeatherDataCollector
    from src.data_preprocessing import WeatherDataPreprocessor
    
    # 创建数据收集器
    collector = WeatherDataCollector()
    
    print("请选择数据来源:")
    print("1. 使用示例数据（推荐，快速开始）")
    print("2. 获取真实天气数据（需要网络连接）")
    
    choice = input("请选择 (1/2): ").strip()
    
    if choice == '1':
        print("\n正在生成示例数据...")
        df = collector.load_sample_data()
        data_path = 'data/sample_weather_data.csv'
        
    elif choice == '2':
        print("\n获取真实天气数据...")
        city = input("请输入城市名称（如：北京）: ").strip() or '北京'
        start_date = input("请输入开始日期（如：2020-01-01）: ").strip() or '2020-01-01'
        end_date = input("请输入结束日期（如：2023-12-31）: ").strip() or '2023-12-31'
        
        df = collector.collect_weather_data(city, start_date, end_date)
        if df is None:
            print("数据获取失败，使用示例数据...")
            df = collector.load_sample_data()
        data_path = f'data/weather_data_{city}_{start_date}_{end_date}.csv'
    else:
        print("无效选择，使用示例数据...")
        df = collector.load_sample_data()
        data_path = 'data/sample_weather_data.csv'
    
    # 保存数据
    df.to_csv(data_path, index=False)
    print(f"数据已保存到: {data_path}")
    
    # 数据预处理
    print("\n开始数据预处理...")
    sequence_length = int(input("请输入输入序列长度（默认30）: ") or 30)
    prediction_days = int(input("请输入预测天数（默认1）: ") or 1)
    
    preprocessor = WeatherDataPreprocessor(sequence_length, prediction_days)
    
    result = preprocessor.preprocess_pipeline(data_path)
    if result:
        print("\n数据预处理完成!")
        print(f"训练数据形状: {result['X_train_scaled'].shape}")
        print(f"特征数量: {len(result['preprocessing_info']['feature_columns'])}")
        
        # 保存预处理数据
        preprocessor.save_preprocessed_data(result, 'data/processed')
        
        print("\n预处理数据已保存到 data/processed/ 目录")
    else:
        print("数据预处理失败")

def option_2_train_model():
    """选项2: 训练模型"""
    print("\n=== 训练模型 ===")
    
    from src.train import train_model, load_or_create_data, plot_training_history
    
    # 检查是否有预处理数据
    if not os.path.exists('data/processed/X_train_scaled.npy'):
        print("未找到预处理数据，请先执行数据采集与预处理")
        return
    
    # 加载数据
    print("正在加载预处理数据...")
    data = load_or_create_data()
    
    if data is None:
        print("数据加载失败")
        return
    
    # 选择模型类型
    print("\n请选择模型类型:")
    print("1. LSTM (推荐)")
    print("2. GRU")
    print("3. Dense (全连接)")
    print("4. CNN-LSTM")
    print("5. Bidirectional LSTM")
    
    model_choice = input("请选择 (1-5): ").strip()
    model_types = {'1': 'lstm', '2': 'gru', '3': 'dense', '4': 'cnn_lstm', '5': 'bidirectional_lstm'}
    model_type = model_types.get(model_choice, 'lstm')
    
    # 训练参数
    print("\n设置训练参数:")
    epochs = int(input("请输入训练轮数（默认100）: ") or 100)
    batch_size = int(input("请输入批次大小（默认32）: ") or 32)
    learning_rate = float(input("请输入学习率（默认0.001）: ") or 0.001)
    
    # 开始训练
    print(f"\n开始训练 {model_type.upper()} 模型...")
    predictor, metrics, history = train_model(
        data=data,
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    print("\n训练完成!")
    print("模型性能指标:")
    for metric, value in metrics.items():
        if metric == 'R2':
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value:.2f}")
    
    # 绘制训练历史
    print("\n生成训练历史图...")
    plot_training_history(history)
    
    # 绘制预测结果
    print("\n生成预测结果图...")
    from src.train import plot_predictions
    
    X_test = data['X_test_scaled']
    y_test = data['y_test_scaled']
    scaler_y = data['scaler_y']
    
    y_pred = predictor.predict(X_test)
    y_test_original = scaler_y.inverse_transform(y_test)
    y_pred_original = scaler_y.inverse_transform(y_pred)
    
    plot_predictions(y_test_original, y_pred_original)
    
    print(f"\n模型已保存到 models/ 目录")

def option_3_make_prediction():
    """选项3: 进行预测"""
    print("\n=== 进行预测 ===")
    
    from src.predict import predict_with_recent_data, print_prediction_results, save_prediction_results
    
    # 选择模型
    model_dir = 'models'
    if not os.path.exists(model_dir):
        print("未找到模型目录，请先训练模型")
        return
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
    if not model_files:
        print("未找到模型文件，请先训练模型")
        return
    
    print("\n可用模型:")
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")
    
    try:
        model_choice = int(input("请选择模型 (输入序号): ").strip()) - 1
        model_path = os.path.join(model_dir, model_files[model_choice])
    except (ValueError, IndexError):
        print("无效选择，使用最新模型")
        model_path = os.path.join(model_dir, model_files[-1])
    
    # 预测参数
    city = input("请输入城市名称（默认：北京）: ").strip() or '北京'
    prediction_days = int(input("请输入预测天数（默认：1）: ") or 1)
    sequence_length = int(input("请输入输入序列长度（默认：30）: ") or 30)
    
    # 进行预测
    print(f"\n正在预测 {city} 未来 {prediction_days} 天的温度...")
    
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
    output_path = f'predictions/prediction_{city}_{timestamp}.json'
    save_prediction_results(results, output_path)

def option_4_visualization():
    """选项4: 可视化结果"""
    print("\n=== 可视化结果 ===")
    
    from src.visualization import WeatherVisualizer
    
    visualizer = WeatherVisualizer()
    
    print("请选择可视化类型:")
    print("1. 天气数据概览")
    print("2. 训练历史")
    print("3. 预测结果对比")
    print("4. 特征重要性")
    print("5. 置信区间图")
    
    choice = input("请选择 (1-5): ").strip()
    
    if choice == '1':
        # 天气数据概览
        data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
        if not data_files:
            print("未找到数据文件")
            return
        
        data_path = os.path.join('data', data_files[0])
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        visualizer.plot_weather_data_overview(df)
    
    elif choice == '2':
        # 训练历史
        history_files = [f for f in os.listdir('models') if f.endswith('.json') and 'history' in f]
        if not history_files:
            print("未找到训练历史文件")
            return
        
        history_path = os.path.join('models', history_files[-1])
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        visualizer.plot_training_history(history)
    
    elif choice == '3':
        # 预测结果对比
        pred_files = [f for f in os.listdir('predictions') if f.endswith('.json')]
        if not pred_files:
            print("未找到预测结果文件")
            return
        
        print("可用预测结果:")
        for i, f in enumerate(pred_files):
            print(f"{i+1}. {f}")
        
        try:
            choice = int(input("请选择文件 (输入序号): ")) - 1
            pred_path = os.path.join('predictions', pred_files[choice])
        except (ValueError, IndexError):
            pred_path = os.path.join('predictions', pred_files[-1])
        
        with open(pred_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 这里需要真实值和预测值，简化处理
        print("此功能需要真实值和预测值，请先训练模型")
    
    elif choice == '4':
        # 特征重要性
        print("此功能需要在模型训练后使用")
    
    elif choice == '5':
        # 置信区间图
        pred_files = [f for f in os.listdir('predictions') if f.endswith('.json')]
        if not pred_files:
            print("未找到预测结果文件")
            return
        
        pred_path = os.path.join('predictions', pred_files[-1])
        with open(pred_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        visualizer.plot_confidence_interval(results['predictions'])

def option_5_demo():
    """选项5: 完整流程演示"""
    print("\n=== 完整流程演示 ===")
    print("将执行完整的温度预测流程...")
    
    # 1. 数据准备
    print("\n步骤1: 数据准备")
    from src.data_collection import WeatherDataCollector
    collector = WeatherDataCollector()
    df = collector.load_sample_data()
    data_path = 'data/demo_weather_data.csv'
    df.to_csv(data_path, index=False)
    print(f"示例数据已保存到: {data_path}")
    
    # 2. 数据预处理
    print("\n步骤2: 数据预处理")
    from src.data_preprocessing import WeatherDataPreprocessor
    preprocessor = WeatherDataPreprocessor(sequence_length=30, prediction_days=1)
    result = preprocessor.preprocess_pipeline(data_path)
    
    if result:
        print("数据预处理完成")
        preprocessor.save_preprocessed_data(result, 'data/demo_processed')
    else:
        print("数据预处理失败")
        return
    
    # 3. 模型训练
    print("\n步骤3: 模型训练")
    from src.train import train_model
    
    predictor, metrics, history = train_model(
        data=result,
        model_type='lstm',
        epochs=10,  # 演示用较少轮数
        batch_size=32,
        learning_rate=0.001
    )
    
    print("\n模型训练完成，性能指标:")
    for metric, value in metrics.items():
        if metric == 'R2':
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value:.2f}")
    
    # 4. 进行预测
    print("\n步骤4: 进行预测")
    from src.predict import predict_with_recent_data, print_prediction_results
    
    # 保存模型以便预测使用
    model_path = 'models/demo_model.h5'
    predictor.save_model(model_path)
    
    # 进行预测
    pred_results = predict_with_recent_data(
        model_path=model_path,
        city='北京',
        sequence_length=30,
        prediction_days=1,
        num_predictions=5
    )
    
    print_prediction_results(pred_results)
    
    # 5. 可视化
    print("\n步骤5: 可视化结果")
    from src.visualization import WeatherVisualizer
    visualizer = WeatherVisualizer()
    
    # 绘制训练历史
    visualizer.plot_training_history(history, save_path='demo_training_history.png')
    
    # 绘制天气数据概览
    visualizer.plot_weather_data_overview(df, save_path='demo_weather_overview.png')
    
    # 绘制置信区间
    visualizer.plot_confidence_interval(pred_results['predictions'], 
                                       save_path='demo_confidence_interval.png')
    
    print("\n演示完成！")
    print("生成的文件:")
    print("  - demo_weather_data.csv: 示例数据")
    print("  - demo_model.h5: 训练好的模型")
    print("  - demo_training_history.png: 训练历史图")
    print("  - demo_weather_overview.png: 天气数据概览")
    print("  - demo_confidence_interval.png: 置信区间图")

def main():
    """主函数"""
    print_welcome()
    
    while True:
        show_menu()
        
        choice = input("\n请选择操作 (0-5): ").strip()
        
        if choice == '0':
            print("\n感谢使用明日气温预测器！再见！")
            break
        
        elif choice == '1':
            option_1_data_processing()
        
        elif choice == '2':
            option_2_train_model()
        
        elif choice == '3':
            option_3_make_prediction()
        
        elif choice == '4':
            option_4_visualization()
        
        elif choice == '5':
            option_5_demo()
        
        else:
            print("无效选择，请重新输入")
        
        input("\n按Enter键继续...")

if __name__ == '__main__':
    # 检查是否直接运行
    if len(sys.argv) == 1:
        main()
    else:
        # 命令行模式
        parser = argparse.ArgumentParser(description='明日气温预测器')
        parser.add_argument('--mode', type=str, choices=['collect', 'train', 'predict', 'demo'],
                           help='运行模式')
        parser.add_argument('--model', type=str, help='模型类型', default='lstm')
        parser.add_argument('--city', type=str, help='城市名称', default='北京')
        parser.add_argument('--days', type=int, help='预测天数', default=1)
        parser.add_argument('--epochs', type=int, help='训练轮数', default=100)
        
        args = parser.parse_args()
        
        if args.mode == 'collect':
            from src.data_collection import WeatherDataCollector
            collector = WeatherDataCollector()
            df = collector.collect_weather_data(args.city, '2020-01-01', '2023-12-31')
            if df is not None:
                df.to_csv(f'data/{args.city}_weather_data.csv', index=False)
        
        elif args.mode == 'train':
            from src.train import main as train_main
            sys.argv = ['train.py', '--model_type', args.model, '--epochs', str(args.epochs)]
            train_main()
        
        elif args.mode == 'predict':
            from src.predict import main as predict_main
            sys.argv = ['predict.py', '--model', f'models/{args.model}.h5', 
                       '--city', args.city, '--prediction_days', str(args.days)]
            predict_main()
        
        elif args.mode == 'demo':
            option_5_demo()