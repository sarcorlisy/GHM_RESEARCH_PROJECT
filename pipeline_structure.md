# Hospital Readmission Prediction Pipeline - 结构说明

## 整体架构

```
Hospital Readmission Pipeline
├── 配置层 (Configuration)
│   └── pipeline_config.py
├── 数据层 (Data Layer)
│   └── data_loader.py
├── 预处理层 (Preprocessing Layer)
│   └── data_preprocessor.py
├── 特征工程层 (Feature Engineering Layer)
│   └── feature_selector.py
├── 模型层 (Model Layer)
│   └── model_trainer.py
├── 控制层 (Control Layer)
│   └── main_pipeline.py
└── 工具层 (Utility Layer)
    ├── run_example.py
    ├── test_pipeline.py
    └── requirements.txt
```

## 模块详细说明

### 1. 配置层 (pipeline_config.py)

**职责**: 集中管理所有配置参数

**主要组件**:
- `DATA_PATHS`: 数据文件路径配置
- `MODEL_CONFIG`: 模型训练参数配置
- `FEATURE_SELECTION_METHODS`: 特征选择方法定义
- `MODELS`: 可用模型类型定义
- `FEATURE_CATEGORIES`: 特征分类定义
- `ICD9_CATEGORIES`: ICD-9代码分类映射
- `PLOT_CONFIG`: 可视化配置

**优势**:
- 集中配置管理
- 易于修改和维护
- 支持不同环境配置

### 2. 数据层 (data_loader.py)

**职责**: 数据加载和合并

**主要功能**:
- `DataLoader.load_diabetic_data()`: 加载糖尿病数据集
- `DataLoader.load_ids_mapping()`: 加载ID映射数据
- `DataLoader.split_ids_mapping()`: 分割ID映射表
- `DataLoader.merge_data()`: 合并所有数据表
- `DataLoader.get_data_info()`: 获取数据摘要信息
- `DataLoader.save_merged_data()`: 保存合并后的数据

**数据流程**:
```
原始数据文件 → 加载 → 分割映射表 → 合并 → 输出合并数据
```

### 3. 预处理层 (data_preprocessor.py)

**职责**: 数据清洗、特征工程和数据转换

**主要功能**:

#### 特征工程
- `create_age_features()`: 创建年龄相关特征
- `create_diagnosis_features()`: 创建诊断分类特征
- `create_comorbidity_feature()`: 创建合并症特征
- `create_encounter_features()`: 创建就诊相关特征

#### 数据清洗
- `handle_missing_values()`: 处理缺失值
- 处理特殊字符 '?'

#### 数据转换
- `encode_categorical_features()`: 分类特征编码
- `scale_numerical_features()`: 数值特征标准化
- `prepare_target_variable()`: 目标变量准备

#### 数据平衡
- `apply_smote()`: 使用SMOTE处理类别不平衡

**数据流程**:
```
合并数据 → 特征工程 → 数据清洗 → 数据转换 → 数据分割 → 数据平衡 → 输出预处理数据
```

### 4. 特征工程层 (feature_selector.py)

**职责**: 特征选择和重要性分析

**主要功能**:
- `select_features_by_l1()`: L1正则化特征选择
- `select_features_by_mi()`: 互信息特征选择
- `select_features_by_tree()`: 树模型特征重要性选择
- `select_all_features()`: 使用所有方法选择特征
- `get_common_features()`: 获取共同选择的特征
- `plot_feature_importance()`: 特征重要性可视化

**特征选择流程**:
```
预处理数据 → 多种特征选择方法 → 特征重要性排序 → 选择Top-N特征 → 输出选择的特征
```

### 5. 模型层 (model_trainer.py)

**职责**: 模型训练、评估和比较

**主要功能**:

#### 模型管理
- `get_models()`: 获取可用模型列表
- `train_single_model()`: 训练单个模型
- `train_all_models()`: 训练所有模型

#### 模型评估
- `evaluate_model_with_cv()`: 交叉验证评估
- `evaluate_on_test_set()`: 测试集评估
- `get_best_model()`: 获取最佳模型

#### 结果管理
- `save_models()`: 保存训练好的模型
- `load_models()`: 加载保存的模型
- `generate_model_report()`: 生成模型报告
- `plot_model_comparison()`: 模型比较可视化

**模型训练流程**:
```
选择的特征 → 模型初始化 → 训练 → 交叉验证 → 验证集评估 → 测试集评估 → 模型保存 → 报告生成
```

### 6. 控制层 (main_pipeline.py)

**职责**: 协调整个pipeline的执行流程

**主要功能**:
- `HospitalReadmissionPipeline.run_data_loading()`: 执行数据加载
- `HospitalReadmissionPipeline.run_data_preprocessing()`: 执行数据预处理
- `HospitalReadmissionPipeline.run_feature_selection()`: 执行特征选择
- `HospitalReadmissionPipeline.run_model_training()`: 执行模型训练
- `HospitalReadmissionPipeline.run_full_pipeline()`: 执行完整pipeline
- `HospitalReadmissionPipeline.generate_final_report()`: 生成最终报告
- `HospitalReadmissionPipeline.load_and_predict()`: 加载模型进行预测

**Pipeline执行流程**:
```
初始化 → 数据加载 → 数据预处理 → 特征选择 → 模型训练 → 结果评估 → 报告生成 → 完成
```

### 7. 工具层

**职责**: 提供辅助功能和测试

**主要组件**:
- `run_example.py`: 使用示例脚本
- `test_pipeline.py`: 测试脚本
- `requirements.txt`: 项目依赖

## 数据流向图

```
原始数据文件
    ↓
[数据加载层] → 合并数据
    ↓
[预处理层] → 特征工程 → 数据清洗 → 数据转换 → 数据分割 → 数据平衡
    ↓
[特征工程层] → 特征选择 → 特征重要性分析
    ↓
[模型层] → 模型训练 → 交叉验证 → 模型评估 → 模型比较
    ↓
[控制层] → 流程协调 → 结果整合 → 报告生成
    ↓
输出文件 (模型、报告、可视化)
```

## 模块间依赖关系

```
main_pipeline.py
├── data_loader.py
├── data_preprocessor.py
├── feature_selector.py
├── model_trainer.py
└── pipeline_config.py

data_preprocessor.py
├── data_loader.py
└── pipeline_config.py

feature_selector.py
├── data_preprocessor.py
└── data_loader.py

model_trainer.py
├── feature_selector.py
├── data_preprocessor.py
└── pipeline_config.py
```

## 扩展性设计

### 1. 添加新的特征选择方法
在 `feature_selector.py` 中添加新方法，并在 `get_feature_selectors()` 中注册。

### 2. 添加新的模型
在 `model_trainer.py` 的 `get_models()` 方法中添加新模型。

### 3. 添加新的特征工程
在 `data_preprocessor.py` 中添加新的特征工程方法。

### 4. 添加新的数据源
在 `data_loader.py` 中添加新的数据加载方法。

## 配置管理

所有配置参数集中在 `pipeline_config.py` 中管理，支持：
- 环境特定配置
- 参数验证
- 默认值设置
- 配置继承

## 错误处理

每个模块都包含：
- 异常捕获和处理
- 详细的错误日志
- 优雅的错误恢复
- 用户友好的错误信息

## 日志系统

统一的日志系统提供：
- 不同级别的日志记录
- 文件和控制台输出
- 时间戳和模块标识
- 详细的执行跟踪

## 测试覆盖

测试脚本覆盖：
- 模块功能测试
- 集成测试
- 错误处理测试
- 性能测试

这种模块化设计使得pipeline具有：
- **高可维护性**: 每个模块职责单一，易于理解和修改
- **高可扩展性**: 易于添加新功能和模型
- **高可重用性**: 模块可以独立使用
- **高可测试性**: 每个模块都可以独立测试
- **高可配置性**: 集中配置管理，易于调整参数 