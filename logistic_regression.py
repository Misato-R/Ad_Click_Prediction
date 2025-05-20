import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import json

# 加载数据
df = pd.read_csv('ad_click_dataset.csv')

# 数值特征 age 用中位数填补
df['age'] = df['age'].fillna(df['age'].median())
# 标准化
mean_age = df['age'].mean()
std_age = df['age'].std()
df['age'] = (df['age'] - mean_age) / std_age

# 类别特征 device_type、ad_position、time_of_day 用众数填补
for col in ['device_type', 'ad_position', 'time_of_day']:
    mode_value = df[col].mode()[0]
    df[col] = df[col].fillna(mode_value)

# 高缺失类别特征 gender 与 browsing_history
# 添加缺失指示器
df['gender_missing'] = df['gender'].isnull().astype(int)
df['browsing_history_missing'] = df['browsing_history'].isnull().astype(int)
# 用常数 'Missing' 填补原列
df['gender'] = df['gender'].fillna('Missing')
df['browsing_history'] = df['browsing_history'].fillna('Missing')

# 数据分离
pd.set_option('display.max_rows', 100)
X = df.drop(columns=['id', 'full_name', 'click'])
y = df['click']
print(X.head(10).to_string(), y)

# 查看处理后示例
df[['age', 'device_type', 'ad_position', 'time_of_day',
    'gender', 'gender_missing',
    'browsing_history', 'browsing_history_missing']].head()

# 类别特征数字化
X_encoded = pd.get_dummies(
    X,
    columns=['gender', 'device_type', 'ad_position', 'browsing_history', 'time_of_day'],
    prefix_sep='_',
    drop_first=False
)
print(X_encoded.head(10).to_string())

# 按 80% / 20% 比例划分，并保持正负样本比例一致（stratify=y）
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=48,    # 固定随机种子以保证可复现
    stratify=y          # 按 y 的分布做分层抽样
)

# 查看划分后各集大小
print(f"训练集样本数：{X_train.shape[0]}, 测试集样本数：{X_test.shape[0]}")
print(f"训练集点击率：{y_train.mean():%}, 测试集点击率：{y_test.mean():%}")

# 保存特征列顺序到json文件
feature_columns = X_encoded.columns.tolist()
with open('feature_columns.json', 'w', encoding='utf-8') as f:
    json.dump(feature_columns, f, ensure_ascii=False, indent=2)
print("Saved feature_columns.json")

# 显式转numpy并指定dtype
X_train_np = X_train.to_numpy(dtype=np.float32)
y_train_np = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
X_test_np = X_test.to_numpy(dtype=np.float32)
y_test_np = y_test.to_numpy(dtype=np.float32).reshape(-1, 1)

# 转PyTorch张量
X_tr = torch.from_numpy(X_train_np)
y_tr = torch.from_numpy(y_train_np)
X_te = torch.from_numpy(X_test_np)
y_te = torch.from_numpy(y_test_np)

# 构建DataLoader mini batch
train_ds = TensorDataset(X_tr, y_tr)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

# 定义模型、损失和优化器
model = nn.Sequential(
    nn.Linear(X_tr.shape[1], 1),
    nn.Sigmoid()
)
criterion = nn.BCELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# 训练循环
n_epochs = 50
for epoch in range(n_epochs):
    model.train()
    for xb, yb in train_loader:
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 测试集评估
model.eval()
with torch.no_grad():
    probs = model(X_te)
    y_pred = (probs.numpy() > 0.5121).astype(int)

# 把y_train_np拉平成一维
y_true = y_test_np.ravel()
probs_true = probs.ravel()

print("y_test_np shape:", y_test_np.shape)
print("probs shape:", probs_true.shape)

# 构造一个 DataFrame
df_out = pd.DataFrame({
    'true_click':        y_true,   # 真实标签 0/1
    'predicted_prob':    probs_true     # 模型输出的概率
})

# 保存到 CSV
df_out.to_csv('predictions_with_probs.csv', index=False)
print("Save to predictions_with_probs.csv")

accuracy = np.mean(y_pred == y_test_np)
tp = np.sum((y_pred == 1) & (y_test_np == 1))
fp = np.sum((y_pred == 1) & (y_test_np == 0))
fn = np.sum((y_pred == 0) & (y_test_np == 1))
precision = tp / (tp + fp)
recall = tp / (tp + fn)

print("PyTorch Logistic Regression:\n")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

torch.save(model.state_dict(), 'logistic_regression.pth')
print("Saved to logistic_regression.pth")