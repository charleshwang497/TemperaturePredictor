"""
数据收集模块
使用Meteostat库获取历史天气数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from meteostat import Point, Daily, Hourly
import os
import warnings
warnings.filterwarnings('ignore')

class WeatherDataCollector:
    """天气数据收集器"""
    
    def __init__(self, data_dir='data'):
        """
        初始化数据收集器
        
        Args:
            data_dir: 数据保存目录
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def get_city_coordinates(self, city_name):
        """
        获取城市坐标（简化版本，实际使用需要地理编码）
        
        Args:
            city_name: 城市名称
            
        Returns:
            tuple: (纬度, 经度) 或 None
        """
        # 常见城市坐标字典
        city_coords = {
            '北京': (39.9042, 116.4074),
            '上海': (31.2304, 121.4737),
            '广州': (23.1291, 113.2644),
            '深圳': (22.5431, 114.0579),
            '成都': (30.5728, 104.0668),
            '杭州': (30.2741, 120.1551),
            '武汉': (30.5928, 114.3055),
            '西安': (34.3416, 108.9398),
            '南京': (32.0603, 118.7969),
            '重庆': (29.5630, 106.5516),
            '天津': (39.3434, 117.3616),
            '苏州': (31.2990, 120.5853),
            '长沙': (28.2104, 112.9388),
            '沈阳': (41.8057, 123.4315),
            '青岛': (36.0986, 120.3719),
            '郑州': (34.7472, 113.6249),
            '大连': (38.9140, 121.6147),
            '东莞': (23.0489, 113.7447),
            '济南': (36.6512, 117.1201),
            '哈尔滨': (45.8038, 126.5340),
            'Londen': (51.5074, -0.1278),
            'New York': (40.7128, -74.0060),
            'Tokyo': (35.6762, 139.6503),
            'Paris': (48.8566, 2.3522),
            'Sydney': (-33.8688, 151.2093)
        }
        
        return city_coords.get(city_name)
    
    def collect_weather_data(self, city_name, start_date, end_date, save=True):
        """
        收集指定城市的天气数据
        
        Args:
            city_name: 城市名称
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            save: 是否保存到文件
            
        Returns:
            pd.DataFrame: 天气数据
        """
        print(f"正在获取 {city_name} 的天气数据...")
        
        # 获取城市坐标
        coords = self.get_city_coordinates(city_name)
        if coords is None:
            print(f"警告: 未找到城市 '{city_name}' 的坐标")
            print("可用城市:", list(self.get_city_coordinates('').keys()))
            return None
        
        lat, lon = coords
        
        try:
            # 创建地点对象
            location = Point(lat, lon)
            
            # 转换日期格式
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            # 获取日级数据
            data = Daily(location, start, end)
            df = data.fetch()
            
            if df.empty:
                print(f"警告: 未获取到 {city_name} 的数据")
                return None
            
            # 重置索引，将日期作为列
            df = df.reset_index()
            
            # 添加城市信息
            df['city'] = city_name
            df['latitude'] = lat
            df['longitude'] = lon
            
            # 重命名列使其更有意义
            column_mapping = {
                'time': 'date',
                'tavg': 'avg_temp',
                'tmin': 'min_temp',
                'tmax': 'max_temp',
                'prcp': 'precipitation',
                'snow': 'snow_depth',
                'wdir': 'wind_direction',
                'wspd': 'wind_speed',
                'wpgt': 'wind_gust',
                'pres': 'pressure',
                'tsun': 'sunshine',
                'rhum': 'humidity'
            }
            
            # 只保留存在的列
            existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
            df = df.rename(columns=existing_columns)
            
            # 添加派生特征
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['day_of_year'] = df['date'].dt.dayofyear
            df['day_of_week'] = df['date'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # 季节性特征
            df['season'] = df['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                          3: 'Spring', 4: 'Spring', 5: 'Spring',
                                          6: 'Summer', 7: 'Summer', 8: 'Summer',
                                          9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})
            
            # 季节编码
            season_dummies = pd.get_dummies(df['season'], prefix='season')
            df = pd.concat([df, season_dummies], axis=1)
            
            print(f"成功获取 {len(df)} 条天气记录")
            print(f"日期范围: {df['date'].min()} 到 {df['date'].max()}")
            print(f"温度范围: {df['avg_temp'].min():.1f}°C 到 {df['avg_temp'].max():.1f}°C")
            
            if save:
                # 保存原始数据
                filename = f"weather_data_{city_name}_{start_date}_{end_date}.csv"
                filepath = os.path.join(self.data_dir, filename)
                df.to_csv(filepath, index=False, encoding='utf-8')
                print(f"数据已保存到: {filepath}")
            
            return df
            
        except Exception as e:
            print(f"获取数据时出错: {e}")
            return None
    
    def collect_multiple_cities(self, cities, start_date, end_date):
        """
        收集多个城市的天气数据
        
        Args:
            cities: 城市名称列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: 合并的天气数据
        """
        all_data = []
        
        for city in cities:
            city_data = self.collect_weather_data(city, start_date, end_date, save=False)
            if city_data is not None:
                all_data.append(city_data)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # 保存合并数据
            filename = f"weather_data_combined_{start_date}_{end_date}.csv"
            filepath = os.path.join(self.data_dir, filename)
            combined_df.to_csv(filepath, index=False, encoding='utf-8')
            print(f"合并数据已保存到: {filepath}")
            
            return combined_df
        else:
            print("未能获取任何城市的数据")
            return None
    
    def load_sample_data(self):
        """
        加载示例数据用于演示
        
        Returns:
            pd.DataFrame: 示例天气数据
        """
        print("正在生成示例天气数据...")
        
        # 生成示例数据
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        
        # 模拟北京的气候特征
        np.random.seed(42)
        
        # 基础温度曲线（正弦波模拟季节变化）
        day_of_year = dates.dayofyear
        base_temp = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # 添加随机噪声
        temp_noise = np.random.normal(0, 5, len(dates))
        avg_temp = base_temp + temp_noise
        
        # 生成其他特征
        min_temp = avg_temp - np.random.exponential(3, len(dates))
        max_temp = avg_temp + np.random.exponential(3, len(dates))
        
        humidity = np.random.normal(60, 15, len(dates))
        humidity = np.clip(humidity, 20, 100)
        
        pressure = np.random.normal(1013, 20, len(dates))
        
        wind_speed = np.random.exponential(3, len(dates))
        wind_speed = np.clip(wind_speed, 0, 20)
        
        precipitation = np.random.exponential(2, len(dates))
        precipitation = np.where(np.random.random(len(dates)) < 0.7, 0, precipitation)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates,
            'avg_temp': avg_temp,
            'min_temp': min_temp,
            'max_temp': max_temp,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'precipitation': precipitation,
            'city': '北京',
            'latitude': 39.9042,
            'longitude': 116.4074
        })
        
        # 添加派生特征
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # 季节性特征
        df['season'] = df['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                      3: 'Spring', 4: 'Spring', 5: 'Spring',
                                      6: 'Summer', 7: 'Summer', 8: 'Summer',
                                      9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})
        
        # 季节编码
        season_dummies = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season_dummies], axis=1)
        
        # 保存示例数据
        filepath = os.path.join(self.data_dir, 'sample_weather_data.csv')
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"示例数据已保存到: {filepath}")
        
        return df

def main():
    """主函数 - 用于测试数据收集功能"""
    collector = WeatherDataCollector()
    
    # 收集单个城市的数据
    print("=== 收集单个城市天气数据 ===")
    beijing_data = collector.collect_weather_data(
        city_name='北京',
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    if beijing_data is not None:
        print("\n数据预览:")
        print(beijing_data.head())
        print(f"\n数据形状: {beijing_data.shape}")
        print(f"\n列名: {list(beijing_data.columns)}")
    
    # 收集多个城市的数据
    print("\n\n=== 收集多个城市天气数据 ===")
    cities = ['北京', '上海', '广州', '深圳']
    multi_data = collector.collect_multiple_cities(
        cities=cities,
        start_date='2022-01-01',
        end_date='2022-12-31'
    )
    
    if multi_data is not None:
        print(f"\n多城市数据形状: {multi_data.shape}")
        print(f"\n城市分布:")
        print(multi_data['city'].value_counts())
    
    # 生成示例数据
    print("\n\n=== 生成示例数据 ===")
    sample_data = collector.load_sample_data()
    print(f"\n示例数据形状: {sample_data.shape}")
    print("\n示例数据统计:")
    print(sample_data[['avg_temp', 'humidity', 'pressure']].describe())

if __name__ == '__main__':
    main()