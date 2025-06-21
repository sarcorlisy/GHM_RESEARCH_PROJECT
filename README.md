# Hospital Readmission Prediction Pipeline

一个完整的医院再入院预测数据科学pipeline，用于预测患者在出院后30天内再次入院的风险。

## 项目概述

本项目将原始的Jupyter notebook重构为一个模块化的、可重用的数据科学pipeline，包含以下主要功能：

- **数据加载和合并**: 自动加载和合并多个数据源
- **数据预处理**: 特征工程、数据清洗、编码和标准化
- **特征选择**: 多种特征选择方法（L1正则化、互信息、树模型重要性）
- **模型训练**: 多种机器学习模型（逻辑回归、随机森林、XGBoost）
- **模型评估**: 交叉验证、测试集评估、性能比较
- **结果可视化**: 特征重要性图、模型比较图
- **报告生成**: 自动生成详细的训练和评估报告

## 项目结构

```
rp0609/
├── pipeline_config.py          # 配置文件
├── data_loader.py              # 数据加载模块
├── data_preprocessor.py        # 数据预处理模块
├── feature_selector.py         # 特征选择模块
├── model_trainer.py            # 模型训练模块
├── main_pipeline.py            # 主pipeline文件
├── requirements.txt            # 项目依赖
├── README.md                   # 项目文档
├── outputs/                    # 输出目录
│   ├── models/                 # 保存的模型
│   ├── *.png                   # 可视化图表
│   ├── *.txt                   # 报告文件
│   └── *.csv                   # 处理后的数据
└── data/                       # 数据文件
    ├── diabetic_data.csv       # 主要数据集
    ├── IDS_mapping.csv         # ID映射数据
    └── ccs_icd9_mapping.csv    # ICD-9映射数据
```

## 安装和设置

1. **克隆项目**
```bash
git clone <repository-url>
cd rp0609
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **准备数据文件**
确保以下数据文件在项目根目录中：
- `diabetic_data.csv`
- `IDS_mapping.csv`
- `ccs_icd9_mapping.csv`

## 使用方法

### 1. 运行完整pipeline

```bash
python main_pipeline.py
```

这将执行完整的数据科学流程：
- 数据加载和合并
- 特征工程和预处理
- 特征选择
- 模型训练和评估
- 生成报告和可视化

### 2. 使用预测功能

```bash
python main_pipeline.py --predict new_data.csv --model RandomForest
```

### 3. 单独运行各个模块

你也可以单独运行各个模块进行测试：

```python
# 数据加载
python data_loader.py

# 数据预处理
python data_preprocessor.py

# 特征选择
python feature_selector.py

# 模型训练
python model_trainer.py
```

## Pipeline流程

### 步骤1: 数据加载 (`data_loader.py`)
- 加载糖尿病数据集
- 加载ID映射数据
- 合并所有数据表
- 生成数据摘要报告

### 步骤2: 数据预处理 (`data_preprocessor.py`)
- **特征工程**:
  - 创建年龄相关特征（年龄中点、年龄组）
  - 创建诊断分类特征（ICD-9代码分类）
  - 创建合并症特征
  - 创建就诊相关特征（就诊索引、滚动平均）
- **数据清洗**:
  - 处理缺失值
  - 处理特殊字符
- **数据转换**:
  - 分类特征编码
  - 数值特征标准化
  - 目标变量准备
- **数据平衡**:
  - 使用SMOTE处理类别不平衡

### 步骤3: 特征选择 (`feature_selector.py`)
- **L1正则化特征选择**: 使用逻辑回归的L1惩罚
- **互信息特征选择**: 基于互信息分数
- **树模型特征重要性**: 使用随机森林的特征重要性
- 生成特征重要性可视化
- 保存选择的特征

### 步骤4: 模型训练 (`model_trainer.py`)
- **模型类型**:
  - 逻辑回归
  - 随机森林
  - XGBoost
- **评估方法**:
  - 5折交叉验证
  - 验证集评估
  - 测试集评估
- **性能指标**:
  - 准确率 (Accuracy)
  - 精确率 (Precision)
  - 召回率 (Recall)
  - F1分数
  - AUC-ROC

## 输出文件

运行pipeline后，会在`outputs/`目录中生成以下文件：

### 数据文件
- `merged_data.csv`: 合并后的原始数据
- `X_train.csv`, `X_val.csv`, `X_test.csv`: 预处理后的特征数据
- `y_train.csv`, `y_val.csv`, `y_test.csv`: 目标变量数据

### 模型文件
- `models/LogisticRegression.joblib`: 逻辑回归模型
- `models/RandomForest.joblib`: 随机森林模型
- `models/XGBoost.joblib`: XGBoost模型

### 特征选择结果
- `selected_features_top15.json`: 选择的特征列表
- `feature_importance.png`: 特征重要性可视化

### 报告和可视化
- `model_report.txt`: 模型训练报告
- `model_comparison.png`: 模型性能比较图
- `final_pipeline_report.txt`: 完整的pipeline报告
- `pipeline.log`: 详细的执行日志

## 配置选项

在`pipeline_config.py`中可以修改以下配置：

```python
MODEL_CONFIG = {
    'test_size': 0.2,           # 测试集比例
    'val_size': 0.2,            # 验证集比例
    'random_state': 42,         # 随机种子
    'cv_folds': 5,              # 交叉验证折数
    'feature_selection_top_n': 15  # 特征选择数量
}
```

## 特征分类

项目中的特征按以下类别组织：

- **Demographic**: 人口统计学特征（年龄、性别、种族等）
- **Administrative**: 管理特征（入院类型、出院处置等）
- **Clinical**: 临床特征（诊断、合并症等）
- **Utilization**: 使用特征（住院时间、检查数量等）
- **Medication**: 药物特征（各种糖尿病药物）

## 模型性能

基于测试集评估，各模型的典型性能表现：

| 模型 | AUC | F1-Score | 准确率 |
|------|-----|----------|--------|
| 逻辑回归 | 0.670 | 0.580 | 0.65 |
| 随机森林 | 0.965 | 0.933 | 0.94 |
| XGBoost | 0.958 | 0.931 | 0.93 |

## 扩展和定制

### 添加新的特征选择方法

在`feature_selector.py`中添加新的方法：

```python
def select_features_by_new_method(self, X, y, top_n=15):
    # 实现新的特征选择逻辑
    pass
```

### 添加新的模型

在`model_trainer.py`的`get_models()`方法中添加：

```python
def get_models(self):
    return {
        # 现有模型...
        'NewModel': NewModelClass(random_state=self.random_state)
    }
```

### 修改特征工程

在`data_preprocessor.py`中添加新的特征工程方法：

```python
def create_new_feature(self, df):
    # 实现新的特征创建逻辑
    return df
```

## 故障排除

### 常见问题

1. **内存不足**: 对于大数据集，可以减少特征选择的数量或使用数据采样
2. **依赖包版本冲突**: 使用虚拟环境并严格按照`requirements.txt`安装
3. **数据文件缺失**: 确保所有必需的数据文件都在正确的位置

### 日志文件

查看`pipeline.log`文件获取详细的执行信息和错误信息。

## 贡献

欢迎提交问题报告和功能请求。如果要贡献代码：

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 这是一个用于教育和研究目的的项目。在实际医疗应用中，请确保遵守相关的医疗数据隐私法规和伦理准则。
