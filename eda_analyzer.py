import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

class EDAAnalyzer:
    def __init__(self, df):
        self.df = df

    def _create_readmission_binary(self, readmission_col='readmitted'):
        """
        辅助方法：将readmission分组为Early Readmission (<30) 和 No Early Readmission (NO + >30)
        返回临时列名，使用后需要清理
        """
        temp_col = 'readmit_bin'
        self.df[temp_col] = self.df[readmission_col].apply(
            lambda x: 'Early Readmission' if x == '<30' else 'No Early Readmission'
        )
        return temp_col

    def plot_readmission_distribution(self):
        """Plot the distribution of readmission categories."""
        plt.figure(figsize=(6,4))
        sns.countplot(x='readmitted', data=self.df, order=['NO', '>30', '<30'])
        plt.title('Readmission Distribution', fontsize=11)
        plt.xlabel('Readmission Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        ax = plt.gca()
        plt.show()
        print(self.df['readmitted'].value_counts())

    def plot_missing_values(self):
        """Visualize missing values as a bar plot."""
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            plt.figure(figsize=(10,5))
            missing.plot(kind='bar')
            plt.title('Missing Values per Feature', fontsize=11)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            ax = plt.gca()
            plt.show()
        else:
            print("No missing values found.")

    def plot_feature_distributions(self, features):
        """Plot histograms for selected numerical features."""
        for col in features:
            if col in self.df.columns:
                plt.figure(figsize=(6,4))
                sns.histplot(self.df[col].dropna(), kde=True)
                plt.title(f'Distribution of {col}', fontsize=11)
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                ax = plt.gca()
                plt.show()

    def plot_categorical_distributions(self, features):
        """Plot bar charts for selected categorical features."""
        for col in features:
            if col in self.df.columns:
                plt.figure(figsize=(6,4))
                self.df[col].value_counts().plot(kind='bar')
                plt.title(f'Distribution of {col}', fontsize=11)
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                ax = plt.gca()
                plt.show()

    def plot_correlation_heatmap(self, save_path: str = None):
        """
        Plot a heatmap of feature correlations, removing constant, all-NaN columns, 以及ID类特征（encounter_id, patient_nbr）和rolling_avg。
        """
        # 剔除ID类特征和rolling_avg
        drop_cols = [col for col in ['encounter_id', 'patient_nbr', 'rolling_avg'] if col in self.df.columns]
        df_num = self.df.drop(columns=drop_cols).select_dtypes(include=[np.number])
        # Remove constant columns
        nunique = df_num.nunique()
        df_num = df_num.loc[:, nunique > 1]
        # Compute correlation
        corr = df_num.corr()
        # Remove all-NaN rows/cols
        corr = corr.dropna(axis=0, how='all').dropna(axis=1, how='all')
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Feature Correlation Heatmap', fontsize=11)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        ax = plt.gca()
        plt.show()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def plot_top_diagnoses(self, top_n=10, icd9_mapping_path=None, show_category=True):
        """
        统计并可视化最常见的诊断编码（diag_1, diag_2, diag_3），可选显示疾病类型。
        Args:
            top_n: 显示前N个诊断
            icd9_mapping_path: ICD9映射文件路径（csv），用于显示疾病类型
            show_category: 是否显示疾病类型（否则只显示编码）
        """
        # ====== ICD9常见重要编码人工注释 ======
        # 250.x — Diabetes mellitus | 糖尿病
        # 428.x — Congestive heart failure (CHF) | 充血性心力衰竭
        # 401.x — Essential hypertension | 原发性高血压
        # 414.x — Other chronic ischemic heart disease | 慢性缺血性心脏病
        # 276.x — Disorders of fluid, electrolyte, and acid-base balance | 水电解质代谢紊乱
        # 427.x — Cardiac dysrhythmias | 心律不齐
        # 599.x — Other disorders of urinary system | 泌尿系统其他疾病（如尿路感染）
        # 786.x — Symptoms involving respiratory system and other chest symptoms | 呼吸系统症状（如呼吸困难、胸痛）
        # 496 — Chronic airway obstruction, not elsewhere classified | 慢性阻塞性肺病（COPD）
        # 486 — Pneumonia, organism unspecified | 肺炎（病原体不明）
        # =====================================
        plt.figure(figsize=(10, 6))
        diag_cols = [col for col in ['diag_1', 'diag_2', 'diag_3'] if col in self.df.columns]
        diagnosis_combined = pd.concat([self.df[col] for col in diag_cols])
        diagnosis_combined = diagnosis_combined.dropna().astype(str)

        def normalize_code(code):
            code = code.strip().replace("'", "")
            code = re.sub(r'[^0-9.]', '', code)
            if '.' in code:
                parts = code.split('.')
                parts[0] = parts[0].zfill(3)
                return parts[0] + '.' + parts[1]
            if code == '':
                return ''
            return code.zfill(5)

        diagnosis_combined = diagnosis_combined.apply(normalize_code)
        diagnosis_counts = diagnosis_combined.value_counts().head(top_n)
        labels = diagnosis_counts.index.astype(str)

        if show_category:
            # 人工兜底常见重要编码（含主码和补零形式）
            manual_map = {
                # Diabetes mellitus
                '250': 'Diabetes mellitus\n糖尿病',
                '00250': 'Diabetes mellitus\n糖尿病',
                # Congestive heart failure
                '428': 'Congestive heart failure\n充血性心力衰竭',
                '00428': 'Congestive heart failure\n充血性心力衰竭',
                # Essential hypertension
                '401': 'Essential hypertension\n原发性高血压',
                '00401': 'Essential hypertension\n原发性高血压',
                # Other chronic ischemic heart disease
                '414': 'Other chronic ischemic heart disease\n慢性缺血性心脏病',
                '00414': 'Other chronic ischemic heart disease\n慢性缺血性心脏病',
                # Disorders of fluid, electrolyte, and acid-base balance
                '276': 'Fluid/electrolyte/acid-base disorder\n水电解质代谢紊乱',
                '00276': 'Fluid/electrolyte/acid-base disorder\n水电解质代谢紊乱',
                # Cardiac dysrhythmias
                '427': 'Cardiac dysrhythmias\n心律不齐',
                '00427': 'Cardiac dysrhythmias\n心律不齐',
                # Other disorders of urinary system
                '599': 'Urinary system disorder\n泌尿系统其他疾病',
                '00599': 'Urinary system disorder\n泌尿系统其他疾病',
                # Symptoms involving respiratory system and other chest symptoms
                '786': 'Respiratory/chest symptoms\n呼吸系统症状',
                '00786': 'Respiratory/chest symptoms\n呼吸系统症状',
                # COPD
                '496': 'COPD\n慢性阻塞性肺病',
                '00496': 'COPD\n慢性阻塞性肺病',
                # Pneumonia
                '486': 'Pneumonia\n肺炎',
                '00486': 'Pneumonia\n肺炎',
            }
            def get_category(code):
                if code in manual_map:
                    return manual_map[code]
                main_code = code.split('.')[0] if '.' in code else code
                if main_code in manual_map:
                    return manual_map[main_code]
                return '未知'
            labels = [
                f"{code}\n{get_category(code)}" for code in diagnosis_counts.index.astype(str)
            ]

        sns.barplot(x=labels, y=diagnosis_counts.values, palette="viridis")
        plt.title(f"Top {top_n} Most Common Diagnoses Among Patients", fontsize=11)
        plt.xlabel("Diagnosis Code" if not show_category else "Diagnosis Code & Category", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        ax = plt.gca()
        plt.show()
        print(diagnosis_counts)

    
    def plot_readmission_heatmap_by_age_gender(self, readmission_col='readmitted', age_col='age_group', gender_col='gender'):
        """
        绘制不同年龄组和性别下的再入院率热力图。
        如果age_group列不存在，自动尝试用age_midpoint或age分箱生成。
        """
        df = self.df.copy()
        # 自动生成age_group
        if age_col not in df.columns:
            if 'age_midpoint' in df.columns:
                bins = [0, 30, 40, 50, 60, 70, 80, 100]
                labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
                df[age_col] = pd.cut(df['age_midpoint'], bins=bins, labels=labels, right=False)
            elif 'age' in df.columns:
                bins = [0, 30, 40, 50, 60, 70, 80, 100]
                labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
                df[age_col] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
            else:
                raise ValueError(f'Column {age_col} not found and cannot be auto-generated.')
        # 只保留<30和>30
        df = df[df[readmission_col] != 'NO']
        # 计算百分比
        heatmap_data = (
            df.groupby([age_col, gender_col])[readmission_col]
            .value_counts(normalize=True)
            .rename('percentage')
            .mul(100)
            .reset_index()
            .pivot_table(index=age_col, columns=gender_col, values='percentage', aggfunc='sum')
        )
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu', cbar_kws={'label': 'Readmission Rate (%)'}, linewidths=0.8, linecolor='black')
        plt.title('Readmission Rate by Age Group and Gender', fontsize=11)
        plt.ylabel('Age Group', fontsize=12)
        plt.xlabel('Gender', fontsize=12)
        plt.tight_layout()
        ax = plt.gca()
        plt.show()

    def plot_readmission_percentage_by_age_gender(self, readmission_col='readmitted', age_col='age_group', gender_col='gender'):
        """
        参考10Yi_Hospital_Readmission_Analysis.ipynb，绘制不同年龄组和性别下各readmitted类别的百分比分布柱状图。
        """
        df = self.df.copy()
        # 自动生成age_group
        if age_col not in df.columns:
            if 'age_midpoint' in df.columns:
                bins = [0, 30, 40, 50, 60, 70, 80, 100]
                labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
                df[age_col] = pd.cut(df['age_midpoint'], bins=bins, labels=labels, right=False)
            elif 'age' in df.columns:
                bins = [0, 30, 40, 50, 60, 70, 80, 100]
                labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
                df[age_col] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
            else:
                raise ValueError(f'Column {age_col} not found and cannot be auto-generated.')
        # 只保留非NO
        df_readmitted = df[df[readmission_col] != 'NO']
        # 分组计数
        readmission_distribution = df_readmitted.groupby([age_col, gender_col, readmission_col]).size().unstack(fill_value=0)
        # 百分比
        readmission_percentage = readmission_distribution.div(readmission_distribution.sum(axis=1), axis=0) * 100
        # 变形用于barplot
        readmission_percentage_melted = readmission_percentage.reset_index().melt(id_vars=[age_col, gender_col], var_name=readmission_col, value_name='percentage')
        plt.figure(figsize=(14, 8))
        sns.barplot(x=age_col, y='percentage', hue=readmission_col, data=readmission_percentage_melted, palette='Set2')
        plt.title('Readmission Rate Distribution by Age Group and Gender', fontsize=11)
        plt.xlabel('Age Group', fontsize=12)
        plt.ylabel('Readmission Rate (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        ax = plt.gca()
        plt.show()

    def plot_readmission_heatmap_by_age_gender_advanced(self, readmission_col='readmitted', age_col='age_group', gender_col='gender'):
        """
        参考10Yi_Hospital_Readmission_Analysis.ipynb，绘制不同年龄组和性别下<30和>30再入院率的热力图。
        """
        df = self.df.copy()
        # 自动生成age_group
        if age_col not in df.columns:
            if 'age_midpoint' in df.columns:
                bins = [0, 30, 40, 50, 60, 70, 80, 100]
                labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
                df[age_col] = pd.cut(df['age_midpoint'], bins=bins, labels=labels, right=False)
            elif 'age' in df.columns:
                bins = [0, 30, 40, 50, 60, 70, 80, 100]
                labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
                df[age_col] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
            else:
                raise ValueError(f'Column {age_col} not found and cannot be auto-generated.')
        # 只保留非NO
        df_readmitted = df[df[readmission_col] != 'NO']
        # 分组计数
        readmission_distribution = df_readmitted.groupby([age_col, gender_col, readmission_col]).size().unstack(fill_value=0)
        # 百分比
        readmission_percentage = readmission_distribution.div(readmission_distribution.sum(axis=1), axis=0) * 100
        # 只保留<30和>30
        for col in ['<30', '>30']:
            if col not in readmission_percentage.columns:
                readmission_percentage[col] = 0.0
        heatmap_data = readmission_percentage[['<30', '>30']].reset_index().pivot(index=age_col, columns=gender_col, values=['<30', '>30'])
        sns.set(style="whitegrid", palette="muted")
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu', cbar_kws={'label': 'Readmission Rate (%)'}, linewidths=0.8, linecolor='black')
        # 手动加百分号
        for text in ax.texts:
            val = text.get_text()
            try:
                text.set_text(f'{float(val):.2f}%')
            except:
                pass
        plt.title('Readmission Rate by Age Group and Gender', fontsize=11)
        plt.ylabel('Age Group', fontsize=12)
        plt.xlabel('Gender', fontsize=12)
        plt.grid(visible=True, linestyle='--', linewidth=0.5, color='black')
        plt.tight_layout()
        ax = plt.gca()
        plt.show()

    # ========== 新增：年龄区间中点函数 ==========
    @staticmethod
    def get_age_midpoint(age_range):
        lower, upper = age_range.strip('[]').split('-')
        return (int(lower) + int(upper.replace(')', ''))) / 2

    def plot_average_age(self, age_col='age'):
        """
        计算并打印平均年龄，风格与10Yi一致。
        """
        if 'age_midpoint' not in self.df.columns:
            self.df['age_midpoint'] = self.df[age_col].apply(self.get_age_midpoint)
        average_age = self.df['age_midpoint'].mean()
        print(f"The average age of patients is: {average_age:.2f} years")

    def plot_avg_stay_by_age_group(self, age_col='age', stay_col='time_in_hospital'):
        """
        按age_group分组画住院时长均值柱状图，风格与10Yi一致。
        """
        # 生成age_group
        self.df['age_group'] = self.df[age_col].str.extract(r'(\d+-\d+)')
        avg_stay_by_age_group = self.df.groupby('age_group')[stay_col].mean()
        print("Average stay at hospital by Age Group In Days")
        print(avg_stay_by_age_group)
        plt.figure(figsize=(10,6))
        sns.barplot(x=avg_stay_by_age_group.index, y=avg_stay_by_age_group.values, palette="viridis")
        plt.title("Average Hospital Stay Time by Age Group", fontsize=11)
        plt.xlabel("Age Group")
        plt.ylabel("Average Time in Hospital (days)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        ax = plt.gca()
        plt.show()

    def plot_avg_stay_by_age_gender_box(self, age_col='age', stay_col='time_in_hospital', gender_col='gender'):
        """
        按age_group和gender画住院时长箱线图，风格与10Yi一致。
        """
        self.df['age_group'] = self.df[age_col].str.extract(r'(\d+-\d+)')
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='age_group', y=stay_col, hue=gender_col, data=self.df, palette='Set2')
        plt.title('Hospital Stay Time vs Gender and Age Group', fontsize=11)
        plt.xticks(rotation=90)
        plt.tight_layout()
        ax = plt.gca()
        plt.show()

    def plot_top_diagnoses_simple(self, top_n=10):
        """
        统计并画前10诊断柱状图（不做复杂映射），风格与10Yi一致。
        """
        diagnosis_combined = pd.concat([self.df['diag_1'], self.df['diag_2'], self.df['diag_3']])
        diagnosis_counts = diagnosis_combined.value_counts()
        top_10_diagnoses = diagnosis_counts.head(top_n)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_10_diagnoses.index, y=top_10_diagnoses.values, palette="viridis")
        plt.title("Top 10 Most Common Diagnoses Among Patients", fontsize=11)
        plt.xlabel("Diagnosis Code", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.xticks(rotation=90)
        ax = plt.gca()
        plt.show()

    def plot_readmission_rate_by_age_gender(self, age_col='age', gender_col='gender', readmission_col='readmitted'):
        """
        绘制不同年龄组和性别下的再入院率分组柱状图，横轴为age_group，每组内按gender分组显示Early/No Early Readmission。
        如果age_group列不存在，自动尝试用age分箱生成。
        """
        df = self.df.copy()
        # 自动生成age_group
        if 'age_group' not in df.columns:
            bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
            df['age_group'] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)
        else:
            df['age_group'] = self.df['age_group']
        # 创建分组
        temp_col = 'readmit_bin'
        df[temp_col] = df[readmission_col].apply(lambda x: 'Early Readmission Rate (%)' if x == '<30' else 'No Early Readmission Rate (%)')
        # 计算百分比
        summary = df.groupby(['age_group', gender_col])[temp_col].value_counts(normalize=True).unstack(fill_value=0) * 100
        summary = summary.reset_index()
        melted = summary.melt(id_vars=['age_group', gender_col], var_name='readmitted', value_name='percentage')
        # 分组柱状图
        plt.figure(figsize=(14, 7))
        sns.barplot(
            x='age_group',
            y='percentage',
            hue=gender_col,
            data=melted[melted['readmitted'] == 'Early Readmission Rate (%)'],
            palette='Set2',
            dodge=True
        )
        plt.title('Early Readmission Rate by Age Group and Gender', fontsize=11)
        plt.xlabel('Age Group', fontsize=12)
        plt.ylabel('Early Readmission Rate (%)', fontsize=12)
        plt.legend(title='Gender')
        plt.tight_layout()
        ax = plt.gca()
        plt.show()
        # 清理临时列
        if temp_col in df.columns:
            df.drop(columns=[temp_col], inplace=True)
        return summary

    def plot_readmission_rate_heatmap_by_age_gender(self, age_col='age', gender_col='gender', readmission_col='readmitted'):
        """
        画age_group为y轴，gender为x轴，x轴每个gender下有Early/No Early Readmission（No Early包含NO和>30），单元格为百分比。
        """
        self.df['age_group'] = self.df[age_col].str.extract(r'(\d+-\d+)')
        self.df['readmit_bin'] = self.df[readmission_col].apply(lambda x: 'Early Readmission' if x == '<30' else 'No Early Readmission')
        self.df['gender_readmit'] = self.df['readmit_bin'] + '-' + self.df[gender_col].astype(str)
        summary = self.df.groupby(['age_group', 'gender_readmit']).size().unstack(fill_value=0)
        summary_pct = summary.div(summary.sum(axis=1), axis=0) * 100
        # 只保留Early/No Early + Female/Male
        valid_cols = [col for col in summary_pct.columns if (('Early Readmission' in col or 'No Early Readmission' in col) and ('Female' in col or 'Male' in col))]
        summary_pct = summary_pct[valid_cols]
        print("Readmission Rate Heatmap Table (age_group x gender_readmit):")
        print(summary_pct)
        plt.figure(figsize=(14, 8))
        ax = sns.heatmap(summary_pct, annot=True, fmt='.2f', cmap='YlGnBu', cbar_kws={'label': 'Percentage (%)'}, linewidths=0.8, linecolor='black')
        # 手动加百分号
        for text in ax.texts:
            val = text.get_text()
            try:
                text.set_text(f'{float(val):.2f}%')
            except:
                pass
        plt.title('Early vs No Early Readmission Rate by Age Group and Gender', fontsize=11)
        plt.ylabel('Age Group', fontsize=12)
        plt.xlabel('Gender')
        plt.tight_layout()
        ax = plt.gca()
        plt.show()
        if 'readmit_bin' in self.df.columns:
            self.df.drop(columns=['readmit_bin'], inplace=True)
        if 'gender_readmit' in self.df.columns:
            self.df.drop(columns=['gender_readmit'], inplace=True)
        return summary_pct

    def plot_comorbidity_vs_readmission(self, categorize_disease_func, readmission_col='readmitted'):
        """
        统计并可视化不同comorbidity下Early Readmission和No Early Readmission（NO+>30）再入院率的分布，分母为所有患者。
        """
        self.df['comorbidity_1'] = self.df['diag_1'].apply(categorize_disease_func)
        self.df['comorbidity_2'] = self.df['diag_2'].apply(categorize_disease_func)
        self.df['comorbidity_3'] = self.df['diag_3'].apply(categorize_disease_func)
        self.df['comorbidity'] = self.df[['comorbidity_1', 'comorbidity_2', 'comorbidity_3']].mode(axis=1)[0]
        # 合并NO和>30为No Early Readmission
        self.df['readmit_bin'] = self.df[readmission_col].apply(lambda x: 'Early Readmission' if x == '<30' else 'No Early Readmission')
        summary = self.df.groupby('comorbidity')['readmit_bin'].value_counts().unstack(fill_value=0)
        summary['Total'] = summary.sum(axis=1)
        summary['Early Readmission Rate (%)'] = summary.get('Early Readmission', 0) / summary['Total'] * 100
        summary['No Early Readmission Rate (%)'] = summary.get('No Early Readmission', 0) / summary['Total'] * 100
        print("Readmission rate by comorbidity (%):")
        print(summary[['Early Readmission Rate (%)', 'No Early Readmission Rate (%)']])
        summary = summary.reset_index()
        melted = summary.melt(id_vars=['comorbidity'], value_vars=['Early Readmission Rate (%)', 'No Early Readmission Rate (%)'], var_name='readmitted', value_name='percentage')
        plt.figure(figsize=(14,6))
        sns.barplot(x='comorbidity', y='percentage', hue='readmitted', data=melted, palette='Set2')
        plt.title("Early vs No Early Readmission Rate for Different Comorbidities", fontsize=11)
        plt.xlabel("Comorbidities", fontsize=12)
        plt.ylabel("Readmission Rate (%)", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        ax = plt.gca()
        plt.show()
        self.df.drop(columns=['comorbidity_1', 'comorbidity_2', 'comorbidity_3', 'readmit_bin'], inplace=True)
        return summary[['comorbidity', 'Early Readmission Rate (%)', 'No Early Readmission Rate (%)']]

    def plot_readmission_by_medication_and_dose(self, save_path="Readmission Rate (<30 Days) by Medication & Dose Change.png"):
        """
        严格参考10Yi notebook，NO不参与分母，分母为<30和>30，y轴为<30的比例，图例只显示剂量分组。
        """
        medication_cols = [
            'metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
            'rosiglitazone', 'insulin', 'glyburide-metformin', 'glipizide-metformin'
        ]
        med_df = self.df[['readmitted'] + [col for col in medication_cols if col in self.df.columns]].copy()
        med_df = med_df[med_df['readmitted'] != 'NO']
        dose_mapping = {'Up': 'Increased', 'Down': 'Decreased', 'Steady': 'No Change', 'No': 'Not Given'}
        for col in medication_cols:
            if col in med_df.columns:
                med_df[col] = med_df[col].map(dose_mapping).fillna("Not Given")
        med_long = med_df.melt(id_vars=['readmitted'], var_name='Medication', value_name='Dose Change')
        medication_summary = med_long.groupby(['Medication', 'Dose Change'])['readmitted'].value_counts(normalize=True).unstack() * 100
        # 处理100%异常
        if '<30' in medication_summary.columns:
            medication_summary.loc[medication_summary["<30"] == 100, "<30"] = np.nan
        if '>30' in medication_summary.columns:
            medication_summary.loc[medication_summary[">30"] == 100, ">30"] = np.nan
        medication_summary_reset = medication_summary.reset_index()
        print("Readmission Rate (<30 Days) by Medication & Dose Change:")
        print(medication_summary_reset[["Medication", "Dose Change", "<30"]])
        plt.figure(figsize=(12, 6))
        sns.barplot(data=medication_summary_reset, x="Medication", y="<30", hue="Dose Change", palette="coolwarm")
        plt.xticks(rotation=45)
        plt.xlabel("Medication", fontsize=12)
        plt.ylabel("Readmission Rate (<30 Days)", fontsize=12)
        plt.title("Readmission Rate (<30 Days) by Medication & Dose Change", fontsize=11)
        plt.legend(title="Dose Change")
        plt.tight_layout()
        ax = plt.gca()
        plt.show()
        return medication_summary_reset[["Medication", "Dose Change", "<30"]]

    def plot_readmission_by_medication_and_dose_heatmap(self, save_path="📊 Readmission Rates (<30 Days) by Medication & Dose Change.png"):
        """
        严格参考10Yi notebook，NO不参与分母，分母为<30和>30，y轴为<30的比例，输出热力图。
        """
        medication_summary = self.plot_readmission_by_medication_and_dose(save_path=None)
        if medication_summary is None or medication_summary.empty:
            print('No data for heatmap.')
            return
        pivot_table = medication_summary.pivot(index="Medication", columns="Dose Change", values="<30")
        print("Readmission Rate (<30 Days) Heatmap Table:")
        print(pivot_table)
        plt.figure(figsize=(12, 7))
        sns.heatmap(
            pivot_table,
            annot=True,
            cmap="magma",
            fmt=".2f",
            linewidths=0.7,
            cbar_kws={'label': 'Readmission Rate (<30 Days)'}
        )
        plt.title("📊 Readmission Rates (<30 Days) by Medication & Dose Change", fontsize=11)
        plt.ylabel("Medication", fontsize=12)
        plt.xlabel("Dose Change", fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        ax = plt.gca()
        plt.show()
        return pivot_table 