import json
import pandas as pd
import numpy as np
import torch
from torch import nn

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

# 高缺失类别特征 gender 与 browsing_history
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

# 转为 numpy 和 PyTorch 张量
X_test_np = X_test.to_numpy(dtype=np.float32)
X_test_tensor = torch.from_numpy(X_test_np)

# 重建模型结构并加载 .pth 参数
input_dim = len(feature_columns)
model = nn.Sequential(
    nn.Linear(input_dim, 1),
    nn.Sigmoid()
)
model.load_state_dict(torch.load('logistic_regression.pth'))
model.eval()

# 推断并保存结果
with torch.no_grad():
    probs = model(X_test_tensor).numpy().ravel()

preds = (probs > 0.5141).astype(int)

# 8. 保存到 DataFrame
df_test['predicted_click_probability'] = probs    # 连续概率
df_test['predicted_click'] = preds    # 0/1 预测
print("Probabilities:", probs)
print("Predictions  :", preds)
df_test.to_csv('test_predictions.csv', index=False)
print("Save to test_predictions.csv")