import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_eia_data(file_path='data/raw/Weekly_U.S._Ending_Stocks_excluding_SPR_of_Crude_Oil.csv'):
    """
    加载和处理EIA周度库存数据
    """
    # 跳过前4行元数据
    df = pd.read_csv(file_path, skiprows=4)
    
    # 重命名列
    df.columns = ['date', 'inventory']
    
    # 转换日期格式
    df['date'] = pd.to_datetime(df['date'])
    
    # 设置日期为索引
    df = df.set_index('date')
    
    # 按日期升序排序
    df = df.sort_index()
    
    return df

def load_jodi_data(file_path='data/raw/QDL-JODI.csv'):
    """
    加载和处理JODI数据
    """
    df = pd.read_csv(file_path)
    
    # 只保留石油相关数据
    df = df[df['energy'] == 'OIL']
    
    # 将日期转换为datetime格式
    df['date'] = pd.to_datetime(df['date'])
    
    # 按国家和日期分组计算总库存变化
    inventory_change = df.groupby(['date', 'country'])['value'].sum().reset_index()
    
    # 透视表以获得每个国家的单独列
    inventory_wide = inventory_change.pivot(index='date', columns='country', values='value')
    
    # 计算全球总库存变化
    inventory_wide['global_change'] = inventory_wide.sum(axis=1)
    
    return inventory_wide

def calculate_features(eia_data, jodi_data):
    """
    计算用于预测的特征
    """
    # 将EIA周度数据转换为月度数据以匹配JODI
    eia_monthly = eia_data.resample('M').last()
    
    # 计算EIA特征
    eia_features = pd.DataFrame(index=eia_monthly.index)
    
    # 1. 库存水平
    eia_features['us_inventory_level'] = eia_monthly['inventory']
    
    # 2. 库存变化率（环比）
    eia_features['us_inventory_mom_change'] = eia_monthly['inventory'].pct_change()
    
    # 3. 库存水平的5年百分位数
    rolling_window = 260  # 5年的周数 (52周 * 5年)
    eia_features['us_inventory_5y_percentile'] = (
        eia_data['inventory']
        .rolling(rolling_window)
        .apply(lambda x: pd.Series(x).rank().iloc[-1] / len(x))
    )
    
    # 4. 库存变化速度（一阶导数）
    eia_features['us_inventory_velocity'] = eia_data['inventory'].diff()
    
    # 5. 库存加速度（二阶导数）
    eia_features['us_inventory_acceleration'] = eia_data['inventory'].diff().diff()
    
    # 合并JODI特征
    # 确保日期对齐
    jodi_aligned = jodi_data.reindex(eia_features.index)
    
    # 6. 全球库存变化
    eia_features['global_inventory_change'] = jodi_aligned['global_change']
    
    # 7. 主要地区库存变化
    for country in ['CHN', 'EU', 'JPN', 'KOR']:  # 主要消费国
        if country in jodi_data.columns:
            eia_features[f'{country.lower()}_inventory_change'] = jodi_aligned[country]
    
    return eia_features

def main():
    # 加载数据
    print("加载EIA数据...")
    eia_data = load_eia_data()
    print("EIA数据加载完成，时间范围：", eia_data.index.min(), "至", eia_data.index.max())
    
    print("\n加载JODI数据...")
    jodi_data = load_jodi_data()
    print("JODI数据加载完成，时间范围：", jodi_data.index.min(), "至", jodi_data.index.max())
    
    # 计算特征
    print("\n计算特征...")
    features = calculate_features(eia_data, jodi_data)
    
    # 保存处理后的数据
    output_path = os.path.join('data', 'processed', 'inventory_features.csv')
    features.to_csv(output_path)
    print(f"\n特征数据已保存到: {output_path}")
    
    # 打印数据预览
    print("\n特征数据预览:")
    print(features.head())
    print("\n特征统计信息:")
    print(features.describe())
    
    # 打印缺失值信息
    print("\n缺失值统计:")
    print(features.isnull().sum())

if __name__ == "__main__":
    main() 