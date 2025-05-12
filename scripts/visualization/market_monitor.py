import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

class MarketMonitor:
    def __init__(self):
        self.thresholds = None
        self.historical_metrics = None
        
    def calculate_core_metrics(self, data):
        """计算核心监控指标"""
        metrics = {
            # 供应指标
            'supply_change': data['value_supply'].pct_change(),
            'supply_ma3': data['value_supply'].rolling(3).mean(),
            'supply_ma6': data['value_supply'].rolling(6).mean(),
            
            # 库存指标
            'inventory_supply_ratio': data['us_inventory_level'] / data['value_supply'],
            'inventory_velocity': data['us_inventory_level'].diff(),
            
            # 复合指标
            'supply_inventory_momentum': (
                data['supply_pct_change'] * data['inventory_supply_ratio']
            )
        }
        return pd.DataFrame(metrics)
    
    def calculate_advanced_features(self, data):
        """计算高级特征"""
        # 供应趋势强度
        data['supply_trend_strength'] = (
            data['supply_ma_3'] - data['supply_ma_6']
        ) / data['supply_ma_6']
        
        # 库存压力指标
        data['inventory_pressure'] = (
            data['us_inventory_level'] - data['us_inventory_level'].rolling(60).mean()
        ) / data['us_inventory_level'].rolling(60).std()
        
        # 综合动量指标
        data['combined_momentum'] = (
            data['supply_pct_change'] * 0.4 +
            data['inventory_pct_change'] * 0.3 +
            data['supply_trend_strength'] * 0.3
        )
        
        return data
    
    def set_alert_thresholds(self, historical_data):
        """设置预警阈值"""
        self.thresholds = {
            'supply_pct_change': {
                'critical': historical_data['supply_pct_change'].quantile([0.05, 0.95]).values,
                'warning': historical_data['supply_pct_change'].quantile([0.1, 0.9]).values
            },
            'inventory_supply_ratio': {
                'critical': historical_data['inventory_supply_ratio'].quantile([0.05, 0.95]).values,
                'warning': historical_data['inventory_supply_ratio'].quantile([0.1, 0.9]).values
            },
            'inventory_pressure': {
                'critical': historical_data['inventory_pressure'].quantile([0.05, 0.95]).values,
                'warning': historical_data['inventory_pressure'].quantile([0.1, 0.9]).values
            }
        }
        return self.thresholds
    
    def check_alerts(self, current_data):
        """检查是否触发预警"""
        alerts = []
        
        for metric, thresholds in self.thresholds.items():
            current_value = current_data[metric].iloc[-1]
            
            if current_value <= thresholds['critical'][0] or current_value >= thresholds['critical'][1]:
                alerts.append({
                    'metric': metric,
                    'level': 'CRITICAL',
                    'value': current_value,
                    'threshold': thresholds['critical']
                })
            elif current_value <= thresholds['warning'][0] or current_value >= thresholds['warning'][1]:
                alerts.append({
                    'metric': metric,
                    'level': 'WARNING',
                    'value': current_value,
                    'threshold': thresholds['warning']
                })
        
        return alerts
    
    def generate_report(self, data, output_dir='data/reports'):
        """生成监控报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 计算指标
        metrics = self.calculate_core_metrics(data)
        data_with_features = self.calculate_advanced_features(data.copy())
        
        # 检查预警
        alerts = self.check_alerts(data_with_features)
        
        # 生成图表
        plt.figure(figsize=(15, 10))
        
        # 供应变化图
        plt.subplot(2, 2, 1)
        plt.plot(data_with_features.index, data_with_features['supply_pct_change'])
        plt.title('Supply Change')
        plt.axhline(y=0, color='r', linestyle='--')
        
        # 库存压力图
        plt.subplot(2, 2, 2)
        plt.plot(data_with_features.index, data_with_features['inventory_pressure'])
        plt.title('Inventory Pressure')
        
        # 综合动量图
        plt.subplot(2, 2, 3)
        plt.plot(data_with_features.index, data_with_features['combined_momentum'])
        plt.title('Combined Momentum')
        
        # 库存供应比图
        plt.subplot(2, 2, 4)
        plt.plot(data_with_features.index, data_with_features['inventory_supply_ratio'])
        plt.title('Inventory/Supply Ratio')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/market_monitor_{datetime.now().strftime("%Y%m%d")}.png')
        plt.close()
        
        # 生成报告文本
        report = f"Market Monitor Report - {datetime.now().strftime('%Y-%m-%d')}\n\n"
        
        if alerts:
            report += "⚠️ ALERTS:\n"
            for alert in alerts:
                report += f"- {alert['level']}: {alert['metric']} = {alert['value']:.4f}\n"
        else:
            report += "✅ No alerts triggered\n"
        
        report += "\nKey Metrics (Latest):\n"
        for col in metrics.columns:
            report += f"- {col}: {metrics[col].iloc[-1]:.4f}\n"
        
        with open(f'{output_dir}/report_{datetime.now().strftime("%Y%m%d")}.txt', 'w') as f:
            f.write(report)
        
        return report

def main():
    # 加载数据
    print("Loading data...")
    data = pd.read_csv('data/processed/enhanced_data_v3.csv')
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    
    # 初始化监控器
    monitor = MarketMonitor()
    
    # 计算指标和设置阈值
    print("\nProcessing data...")
    metrics = monitor.calculate_core_metrics(data)
    data_with_features = monitor.calculate_advanced_features(data.copy())
    monitor.set_alert_thresholds(data_with_features)
    
    # 生成报告
    print("\nGenerating report...")
    report = monitor.generate_report(data)
    print("\nReport:")
    print(report)
    
    print("\nCheck data/reports directory for detailed report and visualizations.")

if __name__ == "__main__":
    main() 