# 明日气温预测器 (Neural Weather Predictor)

一个基于神经网络的气温预测系统，使用历史天气数据训练模型，实现对第二天气温的精准预测。

## 🌟 项目特色

- **易于学习**: 专为神经网络算法学习设计，代码结构清晰，注释详细
- **多种模型**: 支持LSTM、GRU、全连接网络等多种神经网络架构
- **完整流程**: 包含数据获取、预处理、模型训练、预测和可视化全流程
- **CPU友好**: 可在普通CPU上运行，无需GPU加速
- **实战导向**: 使用真实天气数据，具备实际应用价值

## 📋 目录结构

```
weather_predictor/
├── data/                   # 数据文件夹
│   ├── sample_weather_data.csv    # 示例天气数据
│   └── processed/          # 预处理后的数据
├── models/                 # 模型文件夹
│   └── *.h5               # 训练好的模型文件
├── src/                    # 源代码
│   ├── data_collection.py  # 数据收集模块
│   ├── data_preprocessing.py # 数据预处理模块
│   ├── neural_network.py   # 神经网络模型
│   ├── train.py           # 训练脚本
│   ├── predict.py         # 预测脚本
│   └── visualization.py   # 可视化工具
├── predictions/            # 预测结果
├── requirements.txt        # 依赖包列表
├── main.py                # 主程序入口
├── README.md              # 项目说明
└── 项目说明.md            # 详细设计文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目（或下载项目文件）
git clone <项目地址>
cd weather_predictor

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行主程序

```bash
# 启动交互式程序
python main.py
```

程序将显示主菜单，您可以选择不同的操作：

```
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
```

### 3. 快速演示

```bash
# 运行完整流程演示
python main.py --mode demo
```

## 📖 使用说明

### 数据采集

项目支持两种数据获取方式：

1. **示例数据**: 快速开始，无需网络连接
2. **真实数据**: 从Meteostat API获取历史天气数据

```python
from src.data_collection import WeatherDataCollector

collector = WeatherDataCollector()

# 获取示例数据
df = collector.load_sample_data()

# 获取真实数据
df = collector.collect_weather_data('北京', '2020-01-01', '2023-12-31')
```

### 数据预处理

```python
from src.data_preprocessing import WeatherDataPreprocessor

preprocessor = WeatherDataPreprocessor(sequence_length=30, prediction_days=1)
result = preprocessor.preprocess_pipeline('data/weather_data.csv')

# 获取处理后的数据
X_train = result['X_train_scaled']
y_train = result['y_train_scaled']
```

### 模型训练

```python
from src.neural_network import TemperaturePredictor

# 创建模型
predictor = TemperaturePredictor(
    input_shape=(30, 20),  # (序列长度, 特征数)
    output_size=1,
    model_type='lstm'
)

# 编译和训练
predictor.compile_model(learning_rate=0.001)
history = predictor.train_model(X_train, y_train, X_val, y_val, epochs=100)

# 保存模型
predictor.save_model('models/temperature_model.h5')
```

### 温度预测

```python
from src.predict import predict_with_recent_data

results = predict_with_recent_data(
    model_path='models/temperature_model.h5',
    city='北京',
    sequence_length=30,
    prediction_days=1
)

# 打印预测结果
print_prediction_results(results)
```

## 🧠 支持的模型类型

### 1. LSTM (长短期记忆网络)
- **特点**: 擅长处理时间序列数据，能够捕捉长期依赖关系
- **适用场景**: 温度预测、股票价格预测等时间序列任务
- **推荐程度**: ⭐⭐⭐⭐⭐

### 2. GRU (门控循环单元)
- **特点**: LSTM的简化版本，训练速度更快
- **适用场景**: 与LSTM类似，但计算效率更高
- **推荐程度**: ⭐⭐⭐⭐

### 3. Dense (全连接神经网络)
- **特点**: 结构简单，易于理解和实现
- **适用场景**: 小规模数据集，快速原型验证
- **推荐程度**: ⭐⭐⭐

### 4. CNN-LSTM (混合模型)
- **特点**: 结合CNN的特征提取能力和LSTM的时序建模能力
- **适用场景**: 复杂的时间序列预测任务
- **推荐程度**: ⭐⭐⭐⭐

### 5. Bidirectional LSTM (双向LSTM)
- **特点**: 同时考虑过去和未来的信息
- **适用场景**: 需要双向上下文信息的任务
- **推荐程度**: ⭐⭐⭐⭐

## 📊 项目特点详解

### 1. 数据处理
- **数据清洗**: 自动处理缺失值和异常值
- **特征工程**: 创建时间特征、滞后特征、移动平均等
- **数据标准化**: 使用StandardScaler进行特征标准化
- **序列生成**: 滑动窗口方法生成时间序列样本

### 2. 模型架构
```
输入层 → LSTM层1 → LSTM层2 → LSTM层3 → 全连接层 → 输出层
   ↓         ↓          ↓          ↓           ↓          ↓
多维输入  128节点    64节点     32节点      50节点      1节点
```

### 3. 训练策略
- **优化器**: Adam（自适应学习率）
- **损失函数**: 均方误差（MSE）
- **正则化**: Dropout + BatchNormalization
- **早停**: 防止过拟合
- **学习率调度**: 自动调整学习率

### 4. 评估指标
- **MSE**: 均方误差
- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差
- **MAPE**: 平均绝对百分比误差
- **R²**: 决定系数

## 🎯 学习要点

### 神经网络基础
1. **前向传播**: 数据如何在网络中流动
2. **反向传播**: 如何计算梯度并更新权重
3. **激活函数**: ReLU、Sigmoid、Tanh的作用
4. **损失函数**: 回归问题中的损失函数选择
5. **优化器**: SGD、Adam、RMSprop的区别

### 时间序列处理
1. **序列生成**: 滑动窗口方法
2. **特征工程**: 时间特征的提取
3. **数据标准化**: 为什么需要标准化
4. **序列模型**: RNN、LSTM、GRU的原理

### 机器学习实践
1. **数据预处理**: 清洗、特征选择、标准化
2. **模型训练**: 批次训练、验证、早停
3. **模型评估**: 各种评估指标的含义
4. **模型保存**: 如何保存和加载模型
5. **结果可视化**: 如何展示预测结果

## 📈 示例输出

### 训练结果示例
```
=== 模型训练完成 ===
评估结果:
  MSE: 4.23
  RMSE: 2.06
  MAE: 1.65
  MAPE: 8.45%
  R2: 0.8923
```

### 预测结果示例
```
温度预测结果 - 北京
============================================
预测时间: 2024-01-15 10:30:45
模型类型: LSTM
输入序列长度: 30 天
预测天数: 1 天

详细预测:
日期: 2024-01-16
  预测温度: 12.56°C
  标准差: 1.23°C
  95%置信区间: [10.15°C, 14.97°C]
```

## 🔧 高级功能

### 命令行参数
```bash
# 数据采集模式
python main.py --mode collect --city 北京

# 训练模式
python main.py --mode train --model lstm --epochs 200

# 预测模式
python main.py --mode predict --model models/lstm_model.h5 --city 上海

# 演示模式
python main.py --mode demo
```

### 批量预测
```python
from src.predict import predict_temperature

cities = ['北京', '上海', '广州', '深圳']
results = {}

for city in cities:
    results[city] = predict_temperature(
        model_path='models/temperature_model.h5',
        data_path=f'data/{city}_weather_data.csv'
    )
```

### 模型比较
```python
from src.visualization import WeatherVisualizer

visualizer = WeatherVisualizer()
visualizer.plot_prediction_comparison(results)
```

## 📚 相关文档

- [项目详细说明](项目说明.md) - 包含完整的搭建思路和技术方案
- [代码注释] - 每个模块都有详细的中文注释
- [API文档] - 每个函数都有完整的文档字符串

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 如何贡献
1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [Meteostat](https://meteostat.net/) - 提供历史天气数据
- [TensorFlow](https://www.tensorflow.org/) - 深度学习框架
- [Scikit-learn](https://scikit-learn.org/) - 机器学习工具

## 📞 联系方式

如有问题或建议，欢迎交流讨论！

---

**祝您学习愉快！** 🎉