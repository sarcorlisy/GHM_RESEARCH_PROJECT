# Hospital Readmission Prediction Pipeline - 快速启动指南

## 🚀 5分钟快速开始

### 1. 环境准备

确保你的系统已安装Python 3.8+，然后执行：

```bash
# 克隆或下载项目文件
cd rp0609

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据文件检查

确保以下文件在项目根目录：
- ✅ `diabetic_data.csv`
- ✅ `IDS_mapping.csv`
- ✅ `ccs_icd9_mapping.csv`

### 3. 运行完整Pipeline

```bash
# 方法1: 使用主pipeline
python main_pipeline.py

# 方法2: 使用示例脚本（推荐新手）
python run_example.py

# 方法3: 使用测试脚本验证功能
python test_pipeline.py
```

### 4. 查看结果

运行完成后，检查 `outputs/` 目录：

```
outputs/
├── models/                    # 训练好的模型
├── feature_importance.png     # 特征重要性图
├── model_comparison.png       # 模型比较图
├── final_pipeline_report.txt  # 完整报告
└── pipeline.log              # 执行日志
```

## 📊 预期结果

成功运行后，你应该看到类似这样的输出：

```
🏥 Hospital Readmission Prediction Pipeline
==================================================
✅ Pipeline completed successfully!

📊 Results Summary:
------------------------------
Best Model: RandomForest
Best AUC Score: 0.965
Total Features Selected: 15
Training Samples: 3 models trained

📈 Model Performance Comparison:
----------------------------------------
model_name         auc    f1  accuracy
LogisticRegression 0.670 0.580     0.65
RandomForest       0.965 0.933     0.94
XGBoost            0.958 0.931     0.93
```

## 🔧 常见问题解决

### 问题1: 数据文件缺失
```
❌ Error: Data file not found
```
**解决方案**: 确保所有CSV文件都在项目根目录

### 问题2: 依赖包安装失败
```
❌ ModuleNotFoundError: No module named 'xgboost'
```
**解决方案**: 
```bash
pip install xgboost
# 或者重新安装所有依赖
pip install -r requirements.txt
```

### 问题3: 内存不足
```
❌ MemoryError
```
**解决方案**: 在 `pipeline_config.py` 中减少特征选择数量：
```python
MODEL_CONFIG = {
    'feature_selection_top_n': 10  # 从15减少到10
}
```

## 🎯 下一步操作

### 1. 自定义配置

编辑 `pipeline_config.py` 调整参数：

```python
MODEL_CONFIG = {
    'test_size': 0.2,           # 测试集比例
    'val_size': 0.2,            # 验证集比例
    'random_state': 42,         # 随机种子
    'cv_folds': 5,              # 交叉验证折数
    'feature_selection_top_n': 15  # 特征选择数量
}
```

### 2. 单独运行模块

```bash
# 只运行数据加载
python data_loader.py

# 只运行数据预处理
python data_preprocessor.py

# 只运行特征选择
python feature_selector.py

# 只运行模型训练
python model_trainer.py
```

### 3. 使用预测功能

```bash
# 对新数据进行预测
python main_pipeline.py --predict new_data.csv --model RandomForest
```

### 4. 添加新功能

参考 `pipeline_structure.md` 了解如何扩展pipeline。

## 📈 性能优化建议

### 1. 大数据集处理
- 使用数据采样减少内存使用
- 减少特征选择数量
- 使用更简单的模型

### 2. 提高训练速度
- 减少交叉验证折数
- 使用更少的树模型参数
- 并行处理（如果支持）

### 3. 提高模型性能
- 尝试不同的特征选择方法
- 调整模型超参数
- 使用集成方法

## 🔍 调试技巧

### 1. 查看详细日志
```bash
# 查看pipeline.log文件
tail -f pipeline.log
```

### 2. 逐步调试
```python
# 在代码中添加断点
import pdb; pdb.set_trace()
```

### 3. 检查中间结果
```python
# 在pipeline中添加打印语句
print(f"Data shape: {df.shape}")
print(f"Selected features: {selected_features}")
```

## 📚 学习资源

1. **项目文档**:
   - `README.md`: 完整项目说明
   - `pipeline_structure.md`: 架构设计说明
   - `outputs/final_pipeline_report.txt`: 详细结果报告

2. **代码示例**:
   - `run_example.py`: 使用示例
   - `test_pipeline.py`: 测试示例

3. **相关技术**:
   - scikit-learn: 机器学习库
   - pandas: 数据处理
   - matplotlib: 数据可视化

## 🆘 获取帮助

如果遇到问题：

1. 查看 `pipeline.log` 文件中的错误信息
2. 运行 `python test_pipeline.py` 检查各模块功能
3. 检查数据文件格式和内容
4. 确认Python环境和依赖包版本

## 🎉 恭喜！

你已经成功运行了医院再入院预测pipeline！现在你可以：

- 分析生成的报告和可视化结果
- 调整参数优化模型性能
- 扩展pipeline添加新功能
- 将模型应用到实际医疗数据中

记住：这是一个用于教育和研究目的的项目。在实际医疗应用中，请确保遵守相关的医疗数据隐私法规和伦理准则。 