import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score
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
# 设置最大显示行数为100
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

# 初始化模型
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
lgb_clf = LGBMClassifier()

# 取出搜索到的最优参数
params_for_xgb = {
    'colsample_bytree': float(0.7918271674141664),
    'learning_rate':    float(0.15193870279314878),
    'max_depth':        int(9),
    'n_estimators':     int(319),
    'reg_alpha':        float(0.11231202539270979),
    'reg_lambda':       float(2.277074393634674),
    'subsample':        float(0.777983630684874)
}
print("Using best_params:", params_for_xgb)

param_for_lgb = {
    'n_estimators':        int(428),    # 树数量
    'num_leaves':          int(116),     # 叶子节点数
    'max_depth':           int(7),       # 树深度
    'learning_rate':       float(0.11634183869089182),  # 学习率
    'subsample':           float(0.779622352998791),    # 行采样比例
    'colsample_bytree':    float(0.910050632137683),    # 列采样比例
    'min_child_samples':   int(10),       # 最小叶子样本数
    'reg_alpha':           float(0.27324955993679545),        # L1 正则
    'reg_lambda':          float(3.1049300184913697)         # L2 正则
}

# 训练,传入一维标签 y_train_np.ravel()
xgb_clf = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=48,
    **params_for_xgb
)
lgb_clf = LGBMClassifier(
    random_state=48,
    **param_for_lgb
)

xgb_clf.fit(X_train_np, y_train_np.ravel())
lgb_clf.fit(X_train_np, y_train_np.ravel())

# 在测试集上预测概率
probs_xgb = xgb_clf.predict_proba(X_test_np)[:, 1]
preds_xgb = (probs_xgb >= 0.5).astype(int)

probs_lgb = lgb_clf.predict_proba(X_test_np)[:, 1]
preds_lgb = (probs_lgb >= 0.5).astype(int)

# 混合模型测试，计算并打印指标
final_prob = 0.2 * probs_xgb + 0.8 * probs_lgb
final_pred = (final_prob >= 0.5).astype(int)
# print("Ensemble precision:", precision_score(y_test_np, final_pred))
def report(name, y_true, preds, probs):
    accuracy = accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)
    average_precision = average_precision_score(y_true, probs)
    print(f"{name}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, AP={average_precision:.4f}")

y_test_flat = y_test_np.ravel()
report("XGBoost", y_test_flat, preds_xgb, probs_xgb)
report("LightGBM", y_test_flat, preds_lgb, probs_lgb)
report("Blending", y_test_flat, final_pred, final_prob)

# 保存模型
xgb_clf.save_model('xgb_model.json')
print("XGBoost model saved to xgb_model.json")
lgb_clf.booster_.save_model('lgb_model.txt')
print("LightGBM model saved to lgb_model.txt")
