import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 加载数据 load data
data = pd.read_csv('ad_click_dataset.csv')

# 数据规模 Data size
print("shape", data.shape)

# 缺失值统计 Missing value statistics
missing_data = pd.DataFrame({
    'Non-Missing Count': data.count(),
    'Missing Count': data.isnull().sum(),
    'Missing %': (data.isnull().mean() * 100)
})
print("Missing Count：\n")
print(missing_data)

# 点击分布统计 Click distribution statistics
click_counts = data['click'].value_counts().sort_index()
click_dist = pd.DataFrame({
    'Click': click_counts.index,
    'Count': click_counts.values,
    'Percentage %': (click_counts.values / len(data) * 100)
})
print("Click distribution statistics: \n")
print(click_dist)

# 绘制条形图 bar chart
plt.figure()
click_counts.plot(kind='bar')
plt.xlabel('Click')
plt.ylabel('Count')
plt.title('Click Distribution')
plt.xticks([0, 1], ['No Click', 'Click'], rotation=0)
plt.tight_layout()
plt.show()

# Calculate impressions, clicks and CTR by gender
gender = data.groupby('gender')['click'].agg(
    impressions='count',
    clicks='sum'
)
gender['ctr_pct'] = (gender['clicks'] / gender['impressions'] * 100)
print("gender CTR：")
print(gender)

# 绘制CTR条形图
plt.figure()
ax = gender['ctr_pct'].plot(kind='bar')
plt.xlabel('Gender')
plt.ylabel('CTR (%)')
plt.title('CTR by Gender')
plt.xticks(rotation=0)  # 横坐标不旋转
# 在每个柱子的顶部加上数值标签
for bar in ax.patches:
    height = bar.get_height()
    ax.annotate(
        f'{height:.2f}%',                       # 文本内容，保留两位小数并加“%”
        xy=(bar.get_x() + bar.get_width() / 2, height),  # 文本位置：柱顶中心
        xytext=(0, 3),                          # 在柱顶上方偏移 3 点
        textcoords="offset points",
        ha='center', va='bottom'                # 水平居中、垂直底部对齐
    )
plt.tight_layout()
plt.show()

# Calculate impressions, clicks and CTR by device_type
device = data.groupby('device_type')['click'].agg(
    impressions='count',
    clicks='sum'
)
device['ctr_pct'] = (device['clicks'] / device['impressions'] * 100)
print("device type CTR：")
print(device)

# 绘制CTR条形图
plt.figure()
ax = device['ctr_pct'].plot(kind='bar')
plt.xlabel('device')
plt.ylabel('CTR (%)')
plt.title('CTR by device')
plt.xticks(rotation=0)  # 横坐标不旋转
# 在每个柱子的顶部加上数值标签
for bar in ax.patches:
    height = bar.get_height()
    ax.annotate(
        f'{height:.2f}%',                       # 文本内容，保留两位小数并加“%”
        xy=(bar.get_x() + bar.get_width() / 2, height),  # 文本位置：柱顶中心
        xytext=(0, 3),                          # 在柱顶上方偏移 3 点
        textcoords="offset points",
        ha='center', va='bottom'                # 水平居中、垂直底部对齐
    )
plt.tight_layout()
plt.show()

# Calculate impressions, clicks and CTR by ad_position
ad_position = data.groupby('ad_position')['click'].agg(
    impressions='count',
    clicks='sum'
)
ad_position['ctr_pct'] = (ad_position['clicks'] / ad_position['impressions'] * 100)
print("Ad position CTR：")
print(ad_position)

# 绘制CTR条形图
plt.figure()
ax = ad_position['ctr_pct'].plot(kind='bar')
plt.xlabel('ad_position')
plt.ylabel('CTR (%)')
plt.title('CTR by ad_position')
plt.xticks(rotation=0)  # 横坐标不旋转
# 在每个柱子的顶部加上数值标签
for bar in ax.patches:
    height = bar.get_height()
    ax.annotate(
        f'{height:.2f}%',                       # 文本内容，保留两位小数并加“%”
        xy=(bar.get_x() + bar.get_width() / 2, height),  # 文本位置：柱顶中心
        xytext=(0, 3),                          # 在柱顶上方偏移 3 点
        textcoords="offset points",
        ha='center', va='bottom'                # 水平居中、垂直底部对齐
    )
plt.tight_layout()
plt.show()

# Calculate impressions, clicks and CTR by browsing_history
browsing_history = data.groupby('browsing_history')['click'].agg(
    impressions='count',
    clicks='sum'
)
browsing_history['ctr_pct'] = (browsing_history['clicks'] / browsing_history['impressions'] * 100)
print("browsing history CTR：")
print(browsing_history)

# 绘制CTR条形图
plt.figure()
ax = browsing_history['ctr_pct'].plot(kind='bar')
plt.xlabel('browsing_history')
plt.ylabel('CTR (%)')
plt.title('CTR by browsing_history')
plt.xticks(rotation=0)  # 横坐标不旋转
# 在每个柱子的顶部加上数值标签
for bar in ax.patches:
    height = bar.get_height()
    ax.annotate(
        f'{height:.2f}%',                       # 文本内容，保留两位小数并加“%”
        xy=(bar.get_x() + bar.get_width() / 2, height),  # 文本位置：柱顶中心
        xytext=(0, 3),                          # 在柱顶上方偏移 3 点
        textcoords="offset points",
        ha='center', va='bottom'                # 水平居中、垂直底部对齐
    )
plt.tight_layout()
plt.show()

# Calculate impressions, clicks and CTR by time_of_day
time = data.groupby('time_of_day')['click'].agg(
    impressions='count',
    clicks='sum'
)
time['ctr_pct'] = (time['clicks'] / time['impressions'] * 100)
print("time CTR：")
print(time)

# 绘制CTR条形图
plt.figure()
ax = time['ctr_pct'].plot(kind='bar')
plt.xlabel('time_of_day')
plt.ylabel('CTR (%)')
plt.title('CTR by time_of_day')
plt.xticks(rotation=0)  # 横坐标不旋转
# 在每个柱子的顶部加上数值标签
for bar in ax.patches:
    height = bar.get_height()
    ax.annotate(
        f'{height:.2f}%',                       # 文本内容，保留两位小数并加“%”
        xy=(bar.get_x() + bar.get_width() / 2, height),  # 文本位置：柱顶中心
        xytext=(0, 3),                          # 在柱顶上方偏移 3 点
        textcoords="offset points",
        ha='center', va='bottom'                # 水平居中、垂直底部对齐
    )
plt.tight_layout()
plt.show()
