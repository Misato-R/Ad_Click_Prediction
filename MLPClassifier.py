import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score

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

X = df.drop(columns=['id','full_name','click'])
y = df['click']
X_encoded = pd.get_dummies(
    X,
    columns=['gender','device_type','ad_position','browsing_history','time_of_day'],
    drop_first=False
)

# 按 80% / 20% 比例划分，并保持正负样本比例一致
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=48,    # 固定随机种子以保证可复现
    stratify=y          # 按 y 的分布做分层抽样
)

# 定义基础模型及其调好的超参数
best_params_xgb = {
    'colsample_bytree': 0.7918271674141664,
    'learning_rate':    0.15193870279314878,
    'max_depth':        9,
    'n_estimators':     319,
    'reg_alpha':        0.11231202539270979,
    'reg_lambda':       2.277074393634674,
    'subsample':        0.777983630684874
}
best_params_lgb = {
    'colsample_bytree': 0.910050632137683,
    'learning_rate':    0.11634183869089182,
    'max_depth':        7,
    'min_child_samples':10,
    'n_estimators':     428,
    'num_leaves':       116,
    'reg_alpha':        0.27324955993679545,
    'reg_lambda':       3.1049300184913697,
    'subsample':        0.779622352998791
}

# 一个基础学习器（base learner）的列表，每个元素是 (名称, 模型实例)
estimators = [
    ('lr',  LogisticRegression(solver='liblinear', max_iter=1000)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                          random_state=48, **best_params_xgb)),
    ('lgb', LGBMClassifier(random_state=48, **best_params_lgb))
]

# 构建 StackingClassifier，元学习器用 MLPClassifier（小型神经网）
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=MLPClassifier(
        hidden_layer_sizes=(16,),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=48
    ),
    cv=7,
    n_jobs=-1,
    passthrough=False
)

# 训练并预测
stack.fit(X_train, y_train)
probs = stack.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)

# 评估
print("Stacked Ensemble with MLP Meta-Learner:")
print(f" Accuracy : {accuracy_score(y_test, preds):.4f}")
print(f" Precision: {precision_score(y_test, preds):.4f}")
print(f" Recall   : {recall_score(y_test, preds):.4f}")
print(f" AP       : {average_precision_score(y_test, probs):.4f}")
