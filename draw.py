import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# % matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

## DataFrame格式
# diabetesDF = pd.read_csv('diabetes.csv')
# cleanedDF = pd.read_csv('cleaned.csv')

# 热力图5.2
def corr_heat(df):
    dfData = abs(df.corr())
    plt.subplots(figsize=(9, 9)) # 设置画面大小
    sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")
   # plt.savefig('./BluesStateRelation.png')
    plt.show()

# corr_heat(diabetesDF)
# corr_heat(cleanedDF)


# 图4.3
def autolabel(rects, ax, xpos='center'):
	xpos = xpos.lower()  # normalize the case of the parameter
	ha = {'center': 'center', 'right': 'left', 'left': 'right'}
	offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off
	for rect in rects:
		height = rect.get_height()
		ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')
def bar_exam():

	x = ('50', '100', '150', '200')	#任务数
	#实验一
	y1 = (55094, 104914, 158476, 203867)	#类型A时间 ms
	y2 = (54932, 99941, 156932, 201743)		#类型B时间 ms
	#实验二
	y3 = (56726, 117421, 173741, 236907)	#类型A时间 ms 未优化
	y4 = (55412, 114597, 163948, 220110)	#类型B时间 ms 优化

	n_groups = 4
	fig, ax = plt.subplots()
	index = np.arange(n_groups)
	bar_width = 0.30
	opacity = 0.4
	rects1 = ax.bar(index, y3, bar_width,
                alpha=opacity, color='b',
                label='class A')

	rects2 = ax.bar(index + bar_width, y4, bar_width,
                alpha=opacity, color='r',
                label='class B')
	autolabel(rects1, ax, 'center')
	autolabel(rects2, ax, 'center')
	ax.set_xlabel('tasks Number')
	ax.set_ylabel('time/ms')
	ax.set_title('')
	ax.set_xticks(index + bar_width / 2)
	ax.set_xticklabels(x)
	ax.legend()

	fig.tight_layout()
	plt.show()

# bar_exam()

# 折线图 图4.6
def line_chart():
	x = ('50', '100', '150', '200')	#任务数
	#实验一
	y1 = (55094, 104914, 158476, 203867)	#类型A时间 ms
	y2 = (54932, 99941, 156932, 201743)		#类型B时间 ms
	#实验二
	y3 = (56726, 117421, 173741, 236907)	#类型A时间 ms 未优化
	y4 = (55412, 114597, 163948, 220110)	#类型B时间 ms 优化

	fig, ax = plt.subplots()
	ax.plot(x, y1, label="experiment 1 A")
	ax.plot(x, y2, label="experiment 1 B")
	ax.plot(x, y3, label="experiment 2 A")
	ax.plot(x, y4, label="experiment 2 B")
	ax.legend()
	plt.show()
line_chart()