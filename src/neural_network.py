"""
神经网络模型模块
包含温度预测的神经网络架构定义
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import json
import warnings
warnings.filterwarnings('ignore')

class TemperaturePredictor:
    """温度预测神经网络模型"""
    
    def __init__(self, input_shape, output_size=1, model_type='lstm'):
        """
        初始化温度预测器
        
        Args:
            input_shape: 输入数据形状 (sequence_length, num_features)
            output_size: 输出大小（预测的天数 * 每天的预测值数量）
            model_type: 模型类型 ('lstm', 'gru', 'dense', 'cnn_lstm')
        """
        self.input_shape = input_shape
        self.output_size = output_size
        self.model_type = model_type.lower()
        self.model = None
        self.history = None
        
        print(f"初始化温度预测器:")
        print(f"  输入形状: {input_shape}")
        print(f"  输出大小: {output_size}")
        print(f"  模型类型: {model_type}")
    
    def build_lstm_model(self):
        """构建LSTM模型"""
        print("构建LSTM模型...")
        
        model = keras.Sequential([
            layers.Input(shape=self.input_shape),
            
            # 第一层LSTM
            layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            
            # 第二层LSTM
            layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            
            # 第三层LSTM
            layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            
            # 全连接层
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(25, activation='relu'),
            layers.Dropout(0.2),
            
            # 输出层
            layers.Dense(self.output_size)
        ])
        
        return model
    
    def build_gru_model(self):
        """构建GRU模型"""
        print("构建GRU模型...")
        
        model = keras.Sequential([
            layers.Input(shape=self.input_shape),
            
            # 第一层GRU
            layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            
            # 第二层GRU
            layers.GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            
            # 第三层GRU
            layers.GRU(32, dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            
            # 全连接层
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(25, activation='relu'),
            layers.Dropout(0.2),
            
            # 输出层
            layers.Dense(self.output_size)
        ])
        
        return model
    
    def build_dense_model(self):
        """构建全连接神经网络模型"""
        print("构建全连接模型...")
        
        # 将输入展平
        flattened_input = np.prod(self.input_shape)
        
        model = keras.Sequential([
            layers.Input(shape=(flattened_input,)),
            
            # 第一层
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # 第二层
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # 第三层
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # 第四层
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # 输出层
            layers.Dense(self.output_size)
        ])
        
        return model
    
    def build_cnn_lstm_model(self):
        """构建CNN-LSTM混合模型"""
        print("构建CNN-LSTM混合模型...")
        
        model = keras.Sequential([
            layers.Input(shape=self.input_shape),
            
            # 1D CNN层
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # LSTM层
            layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            
            layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            
            # 全连接层
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(25, activation='relu'),
            layers.Dropout(0.2),
            
            # 输出层
            layers.Dense(self.output_size)
        ])
        
        return model
    
    def build_bidirectional_lstm_model(self):
        """构建双向LSTM模型"""
        print("构建双向LSTM模型...")
        
        model = keras.Sequential([
            layers.Input(shape=self.input_shape),
            
            # 双向LSTM层
            layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
            layers.BatchNormalization(),
            
            layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
            layers.BatchNormalization(),
            
            layers.Bidirectional(layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2)),
            layers.BatchNormalization(),
            
            # 全连接层
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(25, activation='relu'),
            layers.Dropout(0.2),
            
            # 输出层
            layers.Dense(self.output_size)
        ])
        
        return model
    
    def build_model(self):
        """根据模型类型构建相应的模型"""
        if self.model == 'lstm':
            self.model = self.build_lstm_model()
        elif self.model == 'gru':
            self.model = self.build_gru_model()
        elif self.model == 'dense':
            self.model = self.build_dense_model()
        elif self.model == 'cnn_lstm':
            self.model = self.build_cnn_lstm_model()
        elif self.model == 'bidirectional_lstm':
            self.model = self.build_bidirectional_lstm_model()
        else:
            print(f"警告: 未知的模型类型 '{self.model_type}'，使用默认LSTM模型")
            self.model = self.build_lstm_model()
            self.model_type = 'lstm'
        
        return self.model
    
    @property
    def model(self):
        """获取模型（懒加载）"""
        if self._model is None:
            self._model = self.build_model()
        return self._model
    
    @model.setter
    def model(self, value):
        self._model = value
    
    def compile_model(self, learning_rate=0.001):
        """
        编译模型
        
        Args:
            learning_rate: 学习率
        """
        print(f"编译模型（学习率={learning_rate}）...")
        
        # 定义优化器
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # 编译模型
        self.model.compile(
            optimizer=optimizer,
            loss='mse',  # 均方误差
            metrics=['mae', 'mape']  # 平均绝对误差、平均绝对百分比误差
        )
        
        print("模型编译完成")
    
    def get_callbacks(self, model_dir='models'):
        """
        获取训练回调函数
        
        Args:
            model_dir: 模型保存目录
            
        Returns:
            list: 回调函数列表
        """
        os.makedirs(model_dir, exist_ok=True)
        
        callbacks = [
            # 早停
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            
            # 学习率调度
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # 模型检查点
            ModelCheckpoint(
                filepath=os.path.join(model_dir, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard日志
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(model_dir, 'logs'),
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        return callbacks
    
    def train_model(self, X_train, y_train, X_val, y_val, 
                   epochs=100, batch_size=32, verbose=1):
        """
        训练模型
        
        Args:
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 详细程度
            
        Returns:
            History: 训练历史
        """
        print("=== 开始训练模型 ===")
        print(f"训练样本数: {len(X_train)}")
        print(f"验证样本数: {len(X_val)}")
        print(f"批次大小: {batch_size}")
        print(f"训练轮数: {epochs}")
        
        # 获取回调函数
        callbacks = self.get_callbacks()
        
        # 训练模型
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=False  # 时间序列数据不要打乱
        )
        
        print("=== 模型训练完成 ===")
        return self.history
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 输入数据
            
        Returns:
            np.array: 预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未构建或加载")
        
        return self.model.predict(X)
    
    def evaluate_model(self, X_test, y_test, scaler_y=None):
        """
        评估模型性能
        
        Args:
            X_test, y_test: 测试数据
            scaler_y: 目标变量的scaler（用于反标准化）
            
        Returns:
            dict: 评估指标
        """
        print("=== 评估模型性能 ===")
        
        # 预测
        y_pred = self.predict(X_test)
        
        # 如果提供了scaler，反标准化
        if scaler_y is not None:
            y_test_original = scaler_y.inverse_transform(y_test)
            y_pred_original = scaler_y.inverse_transform(y_pred)
        else:
            y_test_original = y_test
            y_pred_original = y_pred
        
        # 计算评估指标
        mse = np.mean((y_test_original - y_pred_original) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test_original - y_pred_original))
        mape = np.mean(np.abs((y_test_original - y_pred_original) / (y_test_original + 1e-8))) * 100
        
        # R²分数
        ss_res = np.sum((y_test_original - y_pred_original) ** 2)
        ss_tot = np.sum((y_test_original - np.mean(y_test_original)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }
        
        print("评估结果:")
        for metric, value in metrics.items():
            if metric == 'R2':
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value:.2f}")
        
        return metrics, y_test_original, y_pred_original
    
    def save_model(self, filepath):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        if self.model is None:
            raise ValueError("模型尚未构建")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """
        加载模型
        
        Args:
            filepath: 模型路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        self.model = keras.models.load_model(filepath)
        print(f"模型已从 {filepath} 加载")
    
    def save_training_history(self, filepath):
        """
        保存训练历史
        
        Args:
            filepath: 保存路径
        """
        if self.history is None:
            print("警告: 没有训练历史可以保存")
            return
        
        import json
        history_dict = self.history.history
        
        # 转换numpy类型为Python原生类型
        history_dict = {k: [float(v) for v in vs] for k, vs in history_dict.items()}
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_dict, f, ensure_ascii=False, indent=2)
        
        print(f"训练历史已保存到: {filepath}")
    
    def get_model_summary(self):
        """获取模型摘要"""
        if self.model is None:
            return "模型尚未构建"
        
        import io
        import sys
        
        # 捕获模型摘要输出
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        self.model.summary()
        
        sys.stdout = old_stdout
        summary = buffer.getvalue()
        
        return summary

def main():
    """主函数 - 测试神经网络模型"""
    print("=== 测试神经网络模型 ===")
    
    # 创建示例数据
    sequence_length = 30
    num_features = 20
    output_size = 1
    
    # 生成随机数据用于测试
    np.random.seed(42)
    X_train = np.random.randn(1000, sequence_length, num_features)
    y_train = np.random.randn(1000, output_size)
    X_val = np.random.randn(200, sequence_length, num_features)
    y_val = np.random.randn(200, output_size)
    X_test = np.random.randn(200, sequence_length, num_features)
    y_test = np.random.randn(200, output_size)
    
    # 测试不同的模型类型
    model_types = ['lstm', 'gru', 'dense', 'cnn_lstm', 'bidirectional_lstm']
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"测试 {model_type.upper()} 模型")
        print(f"{'='*50}")
        
        try:
            # 创建预测器
            if model_type == 'dense':
                # 对于dense模型，需要展平输入
                predictor = TemperaturePredictor(
                    input_shape=(sequence_length * num_features,),
                    output_size=output_size,
                    model_type=model_type
                )
                # 展平数据
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_val_flat = X_val.reshape(X_val.shape[0], -1)
                X_test_flat = X_test.reshape(X_test.shape[0], -1)
            else:
                predictor = TemperaturePredictor(
                    input_shape=(sequence_length, num_features),
                    output_size=output_size,
                    model_type=model_type
                )
                X_train_flat, X_val_flat, X_test_flat = X_train, X_val, X_test
            
            # 编译模型
            predictor.compile_model(learning_rate=0.001)
            
            # 显示模型摘要
            print("\n模型架构:")
            print(predictor.get_model_summary())
            
            # 训练模型（使用较少的轮数进行测试）
            print("\n开始训练...")
            history = predictor.train_model(
                X_train_flat, y_train,
                X_val_flat, y_val,
                epochs=5,  # 测试时只用5轮
                batch_size=32,
                verbose=1
            )
            
            # 评估模型
            metrics, _, _ = predictor.evaluate_model(X_test_flat, y_test)
            
            print(f"\n{model_type.upper()} 模型测试完成!")
            
        except Exception as e:
            print(f"测试 {model_type} 模型时出错: {e}")
            continue

if __name__ == '__main__':
    main()