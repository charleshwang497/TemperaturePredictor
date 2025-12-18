"""
数据预处理模块
包含数据清洗、特征工程、数据标准化等功能
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

class WeatherDataPreprocessor:
    """天气数据预处理器"""
    
    def __init__(self, sequence_length=30, prediction_days=1):
        """
        初始化预处理器
        
        Args:
            sequence_length: 输入序列长度（使用过去多少天的数据）
            prediction_days: 预测天数（预测未来多少天）
        """
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_columns = []
        self.target_columns = ['avg_temp']
        
    def load_data(self, filepath):
        """
        加载天气数据
        
        Args:
            filepath: 数据文件路径
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        print(f"正在加载数据文件: {filepath}")
        
        if not os.path.exists(filepath):
            print(f"错误: 文件 {filepath} 不存在")
            return None
        
        try:
            df = pd.read_csv(filepath)
            print(f"成功加载数据，共 {len(df)} 条记录")
            
            # 转换日期列
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return None
    
    def clean_data(self, df):
        """
        清洗数据
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            pd.DataFrame: 清洗后的数据
        """
        print("正在清洗数据...")
        
        # 创建副本以避免修改原始数据
        df_clean = df.copy()
        
        # 检查必要列
        required_columns = ['date', 'avg_temp']
        missing_columns = [col for col in required_columns if col not in df_clean.columns]
        if missing_columns:
            print(f"警告: 缺少必要列 {missing_columns}")
            return None
        
        # 按日期排序
        df_clean = df_clean.sort_values('date').reset_index(drop=True)
        
        # 处理缺失值
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df_clean[col].isnull().sum() > 0:
                print(f"列 {col} 有 {df_clean[col].isnull().sum()} 个缺失值")
                # 使用线性插值填充时间序列数据
                df_clean[col] = df_clean[col].interpolate(method='linear')
                # 如果还有缺失值，使用前向填充
                df_clean[col] = df_clean[col].fillna(method='ffill')
                df_clean[col] = df_clean[col].fillna(method='bfill')
        
        # 移除温度异常值（超出合理范围的数据）
        if 'avg_temp' in df_clean.columns:
            df_clean = df_clean[(df_clean['avg_temp'] >= -50) & (df_clean['avg_temp'] <= 60)]
        
        if 'min_temp' in df_clean.columns:
            df_clean = df_clean[(df_clean['min_temp'] >= -60) & (df_clean['min_temp'] <= 55)]
        
        if 'max_temp' in df_clean.columns:
            df_clean = df_clean[(df_clean['max_temp'] >= -45) & (df_clean['max_temp'] <= 65)]
        
        # 重置索引
        df_clean = df_clean.reset_index(drop=True)
        
        print(f"数据清洗完成，剩余 {len(df_clean)} 条记录")
        return df_clean
    
    def create_features(self, df):
        """
        创建额外的特征
        
        Args:
            df: 清洗后的数据
            
        Returns:
            pd.DataFrame: 包含新特征的数据
        """
        print("正在创建特征...")
        
        df_features = df.copy()
        
        # 确保日期列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df_features['date']):
            df_features['date'] = pd.to_datetime(df_features['date'])
        
        # 时间特征
        df_features['year'] = df_features['date'].dt.year
        df_features['month'] = df_features['date'].dt.month
        df_features['day'] = df_features['date'].dt.day
        df_features['day_of_year'] = df_features['date'].dt.dayofyear
        df_features['day_of_week'] = df_features['date'].dt.dayofweek
        df_features['quarter'] = df_features['date'].dt.quarter
        
        # 周期性特征（将时间转换为正弦/余弦）
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        df_features['day_of_year_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365)
        df_features['day_of_year_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365)
        df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        
        # 滞后特征（过去几天的数据）
        if 'avg_temp' in df_features.columns:
            for lag in [1, 2, 3, 7, 14]:
                df_features[f'avg_temp_lag_{lag}'] = df_features['avg_temp'].shift(lag)
        
        if 'humidity' in df_features.columns:
            for lag in [1, 2, 3]:
                df_features[f'humidity_lag_{lag}'] = df_features['humidity'].shift(lag)
        
        if 'pressure' in df_features.columns:
            for lag in [1, 2, 3]:
                df_features[f'pressure_lag_{lag}'] = df_features['pressure'].shift(lag)
        
        # 移动平均特征
        if 'avg_temp' in df_features.columns:
            df_features['avg_temp_ma_7'] = df_features['avg_temp'].rolling(window=7).mean()
            df_features['avg_temp_ma_14'] = df_features['avg_temp'].rolling(window=14).mean()
            df_features['avg_temp_ma_30'] = df_features['avg_temp'].rolling(window=30).mean()
        
        # 温度变化特征
        if 'avg_temp' in df_features.columns:
            df_features['temp_change_1d'] = df_features['avg_temp'].diff(1)
            df_features['temp_change_7d'] = df_features['avg_temp'].diff(7)
        
        # 温度范围
        if all(col in df_features.columns for col in ['max_temp', 'min_temp']):
            df_features['temp_range'] = df_features['max_temp'] - df_features['min_temp']
        
        # 天气条件分类（如果有降水数据）
        if 'precipitation' in df_features.columns:
            df_features['is_rainy'] = (df_features['precipitation'] > 0).astype(int)
            df_features['is_heavy_rain'] = (df_features['precipitation'] > 10).astype(int)
        
        print(f"特征创建完成，共有 {len(df_features.columns)} 列")
        return df_features
    
    def create_sequences(self, df):
        """
        创建时间序列样本
        
        Args:
            df: 包含特征的数据
            
        Returns:
            tuple: (X, y) 输入序列和目标值
        """
        print(f"正在创建时间序列样本（序列长度={self.sequence_length}）...")
        
        # 选择特征列（排除非数值列和目标列）
        exclude_cols = ['date', 'city'] + self.target_columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        print(f"使用特征: {len(feature_cols)} 个")
        
        # 确保所有特征都是数值类型
        df_features = df[feature_cols].astype(float)
        
        X, y = [], []
        
        # 创建序列
        for i in range(len(df_features) - self.sequence_length - self.prediction_days + 1):
            # 输入序列
            X_sequence = df_features.iloc[i:(i + self.sequence_length)].values
            X.append(X_sequence)
            
            # 目标值（预测未来几天的平均温度）
            target_idx = i + self.sequence_length + self.prediction_days - 1
            if target_idx < len(df):
                y_value = []
                for target_col in self.target_columns:
                    if target_col in df.columns:
                        y_value.append(df[target_col].iloc[target_idx])
                    else:
                        y_value.append(np.nan)
                y.append(y_value)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"创建完成: X shape = {X.shape}, y shape = {y.shape}")
        return X, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """
        分割数据集
        
        Args:
            X: 输入数据
            y: 目标数据
            test_size: 测试集比例
            val_size: 验证集比例
            random_state: 随机种子
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("正在分割数据集...")
        
        # 第一次分割：分离出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        # 第二次分割：分离出训练集和验证集
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, shuffle=False
        )
        
        print(f"数据集分割完成:")
        print(f"  训练集: {X_train.shape[0]} 样本")
        print(f"  验证集: {X_val.shape[0]} 样本")
        print(f"  测试集: {X_test.shape[0]} 样本")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        标准化数据
        
        Args:
            X_train, X_val, X_test: 输入数据
            y_train, y_val, y_test: 目标数据
            
        Returns:
            tuple: 标准化后的数据和scaler对象
        """
        print("正在标准化数据...")
        
        # 重塑 X 数据以便标准化 (samples * timesteps, features)
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        
        # 标准化 X
        X_train_scaled = self.scaler_X.fit_transform(X_train_reshaped)
        X_val_scaled = self.scaler_X.transform(X_val_reshaped)
        X_test_scaled = self.scaler_X.transform(X_test_reshaped)
        
        # 重塑回原始形状
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_val_scaled = X_val_scaled.reshape(X_val.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        # 标准化 y
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_val_scaled = self.scaler_y.transform(y_val)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        print("数据标准化完成")
        
        return (X_train_scaled, X_val_scaled, X_test_scaled,
                y_train_scaled, y_val_scaled, y_test_scaled)
    
    def inverse_transform_y(self, y_scaled):
        """
        将标准化的目标值转换回原始尺度
        
        Args:
            y_scaled: 标准化的目标值
            
        Returns:
            np.array: 原始尺度的目标值
        """
        return self.scaler_y.inverse_transform(y_scaled)
    
    def inverse_transform_X(self, X_scaled):
        """
        将标准化的输入数据转换回原始尺度
        
        Args:
            X_scaled: 标准化的输入数据
            
        Returns:
            np.array: 原始尺度的输入数据
        """
        # 重塑数据
        X_reshaped = X_scaled.reshape(-1, X_scaled.shape[-1])
        X_inverse = self.scaler_X.inverse_transform(X_reshaped)
        return X_inverse.reshape(X_scaled.shape)
    
    def preprocess_pipeline(self, filepath, test_size=0.2, val_size=0.1):
        """
        完整的预处理流程
        
        Args:
            filepath: 数据文件路径
            test_size: 测试集比例
            val_size: 验证集比例
            
        Returns:
            dict: 包含所有预处理结果的字典
        """
        print("=== 开始数据预处理流程 ===")
        
        # 1. 加载数据
        df = self.load_data(filepath)
        if df is None:
            return None
        
        # 2. 清洗数据
        df_clean = self.clean_data(df)
        if df_clean is None:
            return None
        
        # 3. 创建特征
        df_features = self.create_features(df_clean)
        
        # 4. 创建序列
        X, y = self.create_sequences(df_features)
        if len(X) == 0:
            print("错误: 无法创建序列，数据可能不足")
            return None
        
        # 5. 分割数据
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            X, y, test_size=test_size, val_size=val_size
        )
        
        # 6. 标准化数据
        (X_train_scaled, X_val_scaled, X_test_scaled,
         y_train_scaled, y_val_scaled, y_test_scaled) = self.scale_data(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # 保存预处理信息
        preprocessing_info = {
            'sequence_length': self.sequence_length,
            'prediction_days': self.prediction_days,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'original_shape': df.shape,
            'processed_shape': df_features.shape,
            'X_shape': X.shape,
            'y_shape': y.shape
        }
        
        print("=== 数据预处理完成 ===")
        
        return {
            'original_data': df,
            'processed_data': df_features,
            'X': X, 'y': y,
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'X_train_scaled': X_train_scaled, 'X_val_scaled': X_val_scaled, 'X_test_scaled': X_test_scaled,
            'y_train_scaled': y_train_scaled, 'y_val_scaled': y_val_scaled, 'y_test_scaled': y_test_scaled,
            'scaler_X': self.scaler_X, 'scaler_y': self.scaler_y,
            'preprocessing_info': preprocessing_info
        }
    
    def save_preprocessed_data(self, data_dict, output_dir='data'):
        """
        保存预处理后的数据
        
        Args:
            data_dict: 预处理结果字典
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存各个数据集
        datasets = {
            'X_train': data_dict['X_train'],
            'X_val': data_dict['X_val'],
            'X_test': data_dict['X_test'],
            'y_train': data_dict['y_train'],
            'y_val': data_dict['y_val'],
            'y_test': data_dict['y_test'],
            'X_train_scaled': data_dict['X_train_scaled'],
            'X_val_scaled': data_dict['X_val_scaled'],
            'X_test_scaled': data_dict['X_test_scaled'],
            'y_train_scaled': data_dict['y_train_scaled'],
            'y_val_scaled': data_dict['y_val_scaled'],
            'y_test_scaled': data_dict['y_test_scaled']
        }
        
        for name, data in datasets.items():
            filepath = os.path.join(output_dir, f'{name}.npy')
            np.save(filepath, data)
            print(f"已保存 {name} 到 {filepath}")
        
        # 保存处理信息
        info_filepath = os.path.join(output_dir, 'preprocessing_info.json')
        import json
        with open(info_filepath, 'w', encoding='utf-8') as f:
            json.dump(data_dict['preprocessing_info'], f, ensure_ascii=False, indent=2)
        print(f"已保存预处理信息到 {info_filepath}")

def main():
    """主函数 - 测试数据预处理功能"""
    # 创建预处理器
    preprocessor = WeatherDataPreprocessor(sequence_length=30, prediction_days=1)
    
    # 使用示例数据
    from data_collection import WeatherDataCollector
    
    collector = WeatherDataCollector()
    sample_data = collector.load_sample_data()
    
    # 保存示例数据以便处理
    sample_filepath = 'data/sample_weather_data.csv'
    sample_data.to_csv(sample_filepath, index=False)
    
    # 运行预处理流程
    result = preprocessor.preprocess_pipeline(sample_filepath)
    
    if result:
        print("\n=== 预处理结果 ===")
        print(f"训练集大小: {result['X_train'].shape}")
        print(f"验证集大小: {result['X_val'].shape}")
        print(f"测试集大小: {result['X_test'].shape}")
        
        print(f"\n特征列数量: {len(result['preprocessing_info']['feature_columns'])}")
        print(f"目标列: {result['preprocessing_info']['target_columns']}")
        
        # 保存预处理数据
        preprocessor.save_preprocessed_data(result)
        
        # 显示数据样本
        print("\n=== 数据样本 ===")
        print("原始数据前5行:")
        print(result['original_data'].head())
        
        print("\n处理后数据前5行:")
        print(result['processed_data'].head())

if __name__ == '__main__':
    main()