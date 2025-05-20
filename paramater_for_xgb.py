import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import precision_score, make_scorer
from scipy.stats import uniform, randint
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

# 显式转numpy并指定dtype
X_train_np = X_train.to_numpy(dtype=np.float32)
y_train_np = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
X_test_np = X_test.to_numpy(dtype=np.float32)
y_test_np = y_test.to_numpy(dtype=np.float32).reshape(-1, 1)

# 初始化模型
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# 训练,传入一维标签 y_train_np.ravel()
xgb_clf = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=48
)

# 指定超参数分布
param_dist = {
    'n_estimators': randint(100, 500),        # 树的数量
    'max_depth': randint(3, 10),              # 树的最大深度
    'learning_rate': uniform(0.01, 0.19),     # 学习率（0.01~0.2）
    'subsample': uniform(0.6, 0.4),           # 样本采样比例（0.6~1.0）
    'colsample_bytree': uniform(0.6, 0.4),    # 列采样比例（0.6~1.0）
    'reg_alpha': uniform(0, 1),               # L1 正则（0~1）
    'reg_lambda': uniform(1, 9)               # L2 正则（1~10）
}

# 精准率 (Precision) 作为评分指标
precision_scorer = make_scorer(precision_score)

# 交叉验证策略
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=48)

# 随机搜索
rs = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    n_iter=30,               # 随机测试 30 组参数
    scoring=precision_scorer,
    cv=cv,
    verbose=2,
    n_jobs=-1,
    random_state=48
)

# 拟合
rs.fit(X_train_np, y_train_np)

# 输出最优参数和得分
print("Best parameters found:")
print(rs.best_params_)
print(f"Best precision score: {rs.best_score_:.4f}")
