# 供应链分析模型 v4 更新日志

## 项目发展历程

### 1. 初始阶段
- 建立了基于供应端数据的拐点预测模型
- 解决了文件路径和库依赖相关的技术问题
- 完成了基础模型的搭建和验证

### 2. OPEC事件数据整合
- 分析并整合了OPEC政策事件数据
- 研究OPEC政策对油价和供应的影响
- 发现OPEC政策对模型的直接影响相对有限
- 添加了OPEC相关特征：
  - opec_meeting：OPEC会议事件
  - opec_cut：减产决策
  - opec_maintain：维持产量决策
  - opec_increase：增产决策
  - opec_no_agreement：未达成一致
  - opec_no_event：无相关事件

### 3. 库存数据分析
- 引入了库存数据来增强供需动态分析
- 整合了EIA和JODI的库存数据
- 验证了库存水平对市场趋势和潜在拐点的预测价值
- 添加的库存相关特征：
  - inventory_supply_ratio：库存供应比
  - us_inventory_level：美国库存水平
  - us_inventory_velocity：库存变化速度
  - us_inventory_acceleration：库存变化加速度
  - inventory_price_ratio：库存价格比
  - inventory_trend：库存趋势指标

### 4. 特征工程优化
- 时序特征：
  - 添加了滞后特征（lag_1, lag_2, lag_3）
  - 计算了移动平均（MA3, MA6）
  - 加入了变化率指标（pct_change）
- 标准化处理：
  - 对数值型特征进行标准化
  - 保持目标变量（is_turning_point）的二元分类性质
- 特征选择：
  - 移除了高度相关的冗余特征
  - 保留了对预测最有影响力的特征

## 今日更新内容

### 1. 代码优化
- 删除了未使用的导入语句（xgboost、GridSearchCV等）
- 移除了冗余的`evaluate_model`函数，将评估逻辑整合到`train_model`函数中
- 优化了特征工程过程，确保目标变量`is_turning_point`正确处理为二元分类问题
- 改进了特征标准化流程，排除目标变量不被标准化

### 2. 模型改进
- 确认使用RandomForestClassifier进行拐点预测
- 模型参数设置：
  - n_estimators: 200
  - max_depth: 10
  - min_samples_split: 5
  - min_samples_leaf: 2
  - random_state: 42

### 3. 性能提升
与v3版本相比，v4版本实现了以下改进：

#### 模型性能
- v4版本：
  - 准确率：82%
  - 精确率：82%
  - 召回率：81%
  - F1分数：82%
  - 更均衡的预测性能

- v3版本：
  - 精确率：70.97%
  - 召回率：94.79%
  - F1分数：80.59%
  - 倾向于更高的召回率

#### 特征重要性
前五大影响因素：
1. supply_pct_change (15.66%): 供应量变化百分比
2. opec_no_event_supply_change (15.00%): OPEC无事件时的供应变化
3. inventory_supply_ratio (12.33%): 库存供应比率
4. value_supply (9.17%): 供应量
5. supply_lag_1 (6.26%): 前一期供应量

### 4. 输出改进
- 增加了详细的分类报告
- 添加了混淆矩阵显示
- 保持了特征重要性的可视化
- 模型和特征重要性结果保存到相应文件

## 文件更新
- 模型保存位置：`data/models/supply_chain_model.joblib`
- 特征重要性保存位置：`data/processed/feature_importance.csv`
- 特征重要性可视化：`data/plots/feature_importance.png`

## 结论
v4版本在保持较高预测能力的同时，实现了更均衡的性能表现。模型在预测拐点时既保证了较高的准确率，也维持了合理的召回率，整体F1分数优于v3版本。代码结构更加简洁，移除了冗余部分，使维护和理解更加容易。 