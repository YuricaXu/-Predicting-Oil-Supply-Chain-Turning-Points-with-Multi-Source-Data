import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def fetch_eia_inventory_data(api_key=None):
    """
    从EIA获取美国原油库存数据
    包括商业原油库存和战略石油储备(SPR)数据
    """
    if api_key is None:
        api_key = os.getenv('EIA_API_KEY')
        if api_key is None:
            raise ValueError("请提供EIA API密钥")
    
    # EIA API端点
    base_url = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
    
    # 设置请求参数
    params = {
        'api_key': api_key,
        'frequency': 'weekly',
        'data[]': ['value'],
        'facets[series][]': [
            'WCESTUS1',  # 商业原油库存
            'WCSSTUS1'   # 战略石油储备
        ],
        'sort[0][column]': 'period',
        'sort[0][direction]': 'desc',
        'offset': 0,
        'length': 5000  # 获取最近的5000条数据
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # 转换为DataFrame
        df = pd.DataFrame(data['response']['data'])
        
        # 重命名列
        df.columns = ['date', 'series_id', 'value']
        
        # 将数据转换为宽格式
        df_wide = df.pivot(index='date', columns='series_id', values='value')
        df_wide.columns = ['commercial_stocks', 'spr']
        
        # 转换日期格式
        df_wide.index = pd.to_datetime(df_wide.index)
        
        # 按日期排序
        df_wide.sort_index(inplace=True)
        
        # 保存数据
        output_path = os.path.join('data', 'raw', 'eia_inventory.csv')
        df_wide.to_csv(output_path)
        print(f"数据已保存到: {output_path}")
        
        return df_wide
        
    except requests.exceptions.RequestException as e:
        print(f"获取数据时发生错误: {e}")
        return None

if __name__ == "__main__":
    inventory_data = fetch_eia_inventory_data()
    if inventory_data is not None:
        print("\n数据预览:")
        print(inventory_data.head())
        print("\n数据统计:")
        print(inventory_data.describe()) 