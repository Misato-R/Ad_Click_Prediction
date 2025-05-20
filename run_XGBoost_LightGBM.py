import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, Booster

# 加载训练时保存的特征列顺序
with open('feature_columns.json', 'r', encoding='utf-8') as f:
    feature_columns = json.load(f)

# 读取测试集
df_test = pd.read_csv('test_data.csv')
# 标准化
mean_age = df_test['age'].mean()
std_age = df_test['age'].std()
df_test['age'] = (df_test['age'] - mean_age) / std_age

# 与训练时保持完全一致的预处理
# 数值特征填补
df_test['age'] = df_test['age'].fillna(df_test['age'].median())

# 少量缺失的类别特征填补
for col in ['device_type', 'ad_position', 'time_of_day']:
    df_test[col] = df_test[col].fillna(df_test[col].mode()[0])

# 高缺失类别特征gender与browsing_history
# 添加缺失指示器
df_test['gender_missing'] = df_test['gender'].isnull().astype(int)
df_test['browsing_history_missing'] = df_test['browsing_history'].isnull().astype(int)
# 用常数 'Missing' 填补原列
df_test['gender'] = df_test['gender'].fillna('Missing')
df_test['browsing_history'] = df_test['browsing_history'].fillna('Missing')

# One-Hot 编码
X_test = pd.get_dummies(
    df_test.drop(columns=['id', 'full_name']),  # 如果 test_data.csv 没有 full_name，可删去
    columns=['gender','device_type','ad_position','browsing_history','time_of_day'],
    prefix_sep='_',
    drop_first=False
)

# 对齐训练时的特征列顺序，多余列被丢弃，不存在的列补0
X_test = X_test.reindex(columns=feature_columns, fill_value=0)
# 转为numpy
X_test_np = X_test.to_numpy(dtype=np.float32)

# 加载 XGBoost 并预测
xgb2 = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=48)
xgb2.load_model('xgb_model.json')
probs_xgb = xgb2.predict_proba(X_test_np)[:,1]
preds_xgb = (probs_xgb >= 0.5).astype(int)

# 直接用 Booster 加载 LightGBM 模型并预测
lgb_bst = Booster(model_file='lgb_model.txt')
# 对于二分类，predict 返回的是对“正类”的概率
probs_lgb = lgb_bst.predict(X_test_np)
preds_lgb = (probs_lgb >= 0.5).astype(int)

final_prob = 0.5 * probs_xgb + 0.5 * probs_lgb
final_pred = (final_prob >= 0.5).astype(int)

# 输出结果到csv
df_test['xgb_probability'] = probs_xgb    # 连续概率
df_test['xgb_click'] = preds_xgb    # 0/1 预测
df_test['lightgbm_probability'] = probs_lgb    # 连续概率
df_test['lightgbm_click'] = preds_lgb    # 0/1 预测
df_test['blending_probability'] = final_prob    # 连续概率
df_test['blending_probability_click'] = final_pred    # 0/1 预测
print("xgb_probability:", probs_xgb)
print("xgb_click:", preds_xgb)
print("lightgbm_probability:", probs_lgb)
print("lightgbm_click :", preds_lgb)
print("blending_probability:", final_prob)
print("blending_probability_click :", final_pred)
df_test.to_csv('test_predictions_xgboost.csv', index=False)
print("Save to test_predictions_xgboost.csv")