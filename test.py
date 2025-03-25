import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

import pickle 
from os import path

from sklearn.preprocessing import MinMaxScaler


from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report

from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# 读取数据集，尝试从指定路径'datasets/UNSW_NB15.csv'读取数据
data = pd.read_csv('datasets/UNSW_NB15.csv')
# 查看数据集的前5行数据，方便快速了解数据的大致样子，比如各列的数据类型、数据内容等情况
data.head(n=5)

# 获取数据集的详细信息，包括每列的数据类型、非空值数量等信息，有助于进一步分析数据集的结构特点
data.info()

# 筛选出数据集中'service'列值为'-'的所有行，查看是否存在这样的特殊值以及其分布情况等
data[data['service']=='-']

# 将数据集中'service'列值为'-'的地方替换为缺失值（numpy中的NaN），方便后续统一处理缺失值相关问题
data['service'].replace('-',np.nan,inplace=True)

# 统计数据集中每列的缺失值数量，以便清楚知道哪些列存在缺失值以及缺失的程度如何
data.isnull().sum()

# 查看数据集的形状，也就是行数和列数，了解数据集的规模大小
data.shape

# 删除含有缺失值的行，这是一种处理缺失值的方式，但要注意可能会丢失一些有用信息，尤其是数据量不大时
# 可以考虑采用其他更合适的缺失值处理方法，比如填充等方式来替代直接删除
data.dropna(inplace=True)

# 再次查看数据集的形状，确认删除缺失值行后数据集的规模变化情况
data.shape

# 统计数据集中'attack_cat'列不同值出现的频次，了解不同攻击类别在数据集中的分布情况
data['attack_cat'].value_counts()

# 统计数据集中'state'列不同值出现的频次，查看'state'这一属性在数据集中的分布情况
data['state'].value_counts()

# 查看整个数据集当前的数据情况，可能用于在前面一系列处理操作后，再次整体查看数据的样子
data





# 从指定路径'datasets/UNSW_NB15_features.csv'读取特征数据集，
# 这个数据集可能包含了关于数据集中各特征的一些额外描述信息，比如数据类型等，用于后续对主数据集的处理操作
features = pd.read_csv('datasets/UNSW_NB15_features.csv')

# 查看读取到的特征数据集的前几行数据，方便快速了解这个数据集的大致样子，
# 例如各列的名称、数据格式等情况，有助于后续基于其内容进行相关操作
features.head()

# 将特征数据集中 'Type ' 列（注意列名后面有个空格哦）的所有字符串值转换为小写形式，
# 这样做可能是为了后续进行字符串比较等操作时更方便、统一，避免因大小写不一致导致的匹配问题
features['Type '] = features['Type '].str.lower()

# 以下开始选择不同数据类型对应的列名，也就是从特征数据集中筛选出数据类型为 'nominal'（名义型，通常指分类数据）的列名，
# 通过布尔索引的方式，选择 'Type ' 列值为 'nominal' 的对应 'Name' 列的值，最终得到一个包含所有名义型列名的序列对象
nominal_names = features['Name'][features['Type '] == 'nominal']

# 同样的道理，筛选出数据类型为 'integer'（整数型）的列名，得到一个包含所有整数型列名的序列对象，
# 用于后续识别主数据集中哪些列是整数类型，以便进行相应的处理操作
integer_names = features['Name'][features['Type '] == 'integer']

# 筛选出数据类型为 'binary'（二进制，这里可能指只有两种取值的类别型数据等情况）的列名，
# 获取所有二进制类型的列名序列，方便后续针对这类列进行处理，比如统一转换数据类型等操作
binary_names = features['Name'][features['Type '] == 'binary']

# 筛选出数据类型为 'float'（浮点型）的列名，将所有浮点型列名整理出来，便于后续在主数据集中对这些列进行数据类型相关的操作
float_names = features['Name'][features['Type '] == 'float']

# 以下是从主数据集（前面代码中读取的 'data' 数据集）和特征数据集共同筛选出列名的操作，
# 首先获取主数据集的所有列名，存储在 'cols' 变量中，用于后续与不同类型的特征列名进行交集运算
cols = data.columns

# 通过求交集的方式，从主数据集的列名和前面获取的名义型列名中，找出同时存在于两者之中的列名，
# 也就是确定主数据集中哪些列属于名义型数据，更新 'nominal_names' 变量，使其只包含主数据集中对应的名义型列名
nominal_names = cols.intersection(nominal_names)

# 类似地，找出主数据集中属于整数型的列名，通过与特征数据集中整数型列名求交集，更新 'integer_names' 变量，
# 使其准确指向主数据集中的整数型列，方便后续针对这些列进行处理
integer_names = cols.intersection(integer_names)

# 找出主数据集中属于二进制类型的列名，将主数据集和特征数据集中二进制类型列名取交集，得到准确的对应列名集合，
# 用于后续对这些二进制列进行相关的数据处理操作
binary_names = cols.intersection(binary_names)

# 找出主数据集中属于浮点型的列名，通过求交集操作确定主数据集中的浮点型列，便于后续统一对这些列的数据类型进行规范等处理
float_names = cols.intersection(float_names)

# 以下开始对主数据集中的整数列进行数据类型转换操作，遍历前面筛选出的所有整数型列名（'integer_names' 中包含的列名），
# 对于每一个列名对应的列，使用 'pd.to_numeric' 函数将其转换为数值类型（如果原本不是数值类型的话会进行转换，若是已经是数值类型则无影响），
# 这样可以确保这些列的数据类型符合后续数据分析、模型训练等操作的要求
for c in integer_names:
    pd.to_numeric(data[c])

# 对主数据集中的二进制列进行数据类型转换操作，遍历二进制列名列表 'binary_names'，
# 对每一个对应的列使用 'pd.to_numeric' 函数将其转换为数值类型，使其数据类型统一、规范，便于后续处理
for c in binary_names:
    pd.to_numeric(data[c])

# 对主数据集中的浮点型列进行数据类型转换操作，同样遍历浮点型列名列表 'float_names'，
# 利用 'pd.to_numeric' 函数将每一个对应列转换为数值类型，确保数据格式符合预期，方便后续的数据处理和模型使用
for c in float_names:
    pd.to_numeric(data[c])

# 查看主数据集当前的详细信息，包括每列的数据类型、非空值数量等内容，
# 通过查看这些信息可以确认前面的数据类型转换操作是否生效，以及整体数据集的结构特点等情况
data.info()

# 查看主数据集当前的数据情况，也就是查看经过上述一系列列名筛选、数据类型转换等操作后的数据样子，
# 可以直观了解数据的内容、各列的大致取值情况等信息，便于后续继续进行其他的数据处理步骤或模型相关操作
data






# 绘制二元分类的可视化图表，即展示正常和异常标签的分布情况的饼图，用于直观了解数据的类别分布是否平衡等情况
# 这里假设数据集中'label'列用于表示类别，比如0可能表示正常，1表示异常等情况
# 以下是具体的绘图操作

# 创建一个大小为8x8英寸的图形对象，作为后续绘制饼图的画布
plt.figure(figsize=(8,8))

# 使用数据集的'label'列值的频次统计结果来绘制饼图，设置饼图各部分对应的标签为['normal', 'abnormal']，
# 并在每个扇形区域内以保留两位小数的百分数形式显示占比信息，比如30.00%这样的格式
plt.pie(data.label.value_counts(),labels=['normal','abnormal'],autopct='%0.2f%%')

# 给饼图添加标题，明确图表展示的内容是正常和异常标签的饼图分布情况，同时设置标题字体大小为16号字，使其更清晰醒目
plt.title("饼图展示正常和异常标签的分布情况",fontsize=16)

# 显示图例，通过前面设置的标签['normal', 'abnormal']，能清楚地知道饼图中不同颜色扇形区域对应的类别
plt.legend()

# 将绘制好的饼图保存为指定路径下的图片文件，方便后续查看或者用于文档、报告等展示，如果不需要保存可省略这行
plt.savefig('plots/Pie_chart_binary.png')

# 显示绘制好的饼图，让其在运行代码的环境中展示出来，便于直观查看可视化效果
plt.show()



# 绘制多类别分类的可视化图表，也就是展示数据集中不同攻击类别（attack_cat列的不同取值）分布情况的饼图
# 用于直观了解多类别标签在数据集中的分布是否均匀等情况，方便后续针对不同类别数据量等特点进行分析和处理
# 以下是具体的绘图操作

# 创建一个大小为8x8英寸的图形对象，作为后续绘制饼图的画布
plt.figure(figsize=(8,8))

# 使用数据集'attack_cat'列不同值的频次统计结果来绘制饼图，
# 设置饼图各部分对应的标签为'attack_cat'列的所有不同取值（也就是各个攻击类别名称），
# 并在每个扇形区域内以保留两位小数的百分数形式显示占比信息，比如10.00%这样的格式
plt.pie(data.attack_cat.value_counts(),labels=data.attack_cat.unique(),autopct='%0.2f%%')

# 给饼图添加标题，明确图表展示的内容是多类别标签的饼图分布情况
plt.title('饼图展示多类别标签的分布情况')

# 设置图例的显示位置为best（自动根据图表布局等情况选择合适的放置位置），让图例显示更合理美观
plt.legend(loc='best')

# 将绘制好的饼图保存为指定路径下的图片文件，方便后续查看或者用于文档、报告等展示，如果不需要保存可省略这行
plt.savefig('plots/Pie_chart_multi.png')

# 显示绘制好的饼图，让其在运行代码的环境中展示出来，便于直观查看可视化效果
plt.show()







# 选择数据集中所有数据类型为数值型（整数、浮点数等）的列名，用于后续区分数值列和类别列等操作
num_col = data.select_dtypes(include='number').columns

# 选择数据集中除了数值列之外的所有列名，也就是类别列名，但这里取了差集后还去除了索引为0的列（具体要根据实际数据结构判断是否合理）
# 目的是获取所有的类别属性列，以便后续进行独热编码等处理
cat_col = data.columns.difference(num_col)
cat_col = cat_col[1:]
cat_col

# 根据前面获取的类别列名，创建一个仅包含这些类别属性的新数据集副本，方便单独对类别数据进行处理
# 比如后续进行独热编码等操作，不会影响到原数据集中的其他数值列数据
data_cat = data[cat_col].copy()
data_cat.head()

# 使用pandas库的get_dummies函数对类别属性进行独热编码，将每个类别变量转换为多个二进制的哑变量
# 例如，原本一个类别列有3个不同取值，经过独热编码后会变成3个新的列，每个列对应一个取值，用0或1表示是否属于该取值
# 这样就把类别信息以数值形式更方便地表示出来，便于后续模型处理
data_cat = pd.get_dummies(data_cat,columns=cat_col)

# 查看经过独热编码后的类别数据的前几行，了解编码后的数据集样子，比如新生成的列名、数据值等情况
data_cat.head()

# 查看原始数据集的形状，方便对比后续操作后数据集形状的变化情况
data.shape

# 将独热编码后的类别数据与原数据集按列方向进行合并，也就是在原数据集基础上增加了独热编码生成的新列，
# 使得整个数据集既包含了数值列又包含了编码后的类别列信息，为后续模型训练准备好完整的数据格式
data = pd.concat([data, data_cat],axis=1)

# 再次查看合并后的数据集形状，确认添加独热编码列后数据集的规模变化情况
data.shape

# 删除原始的类别列，因为已经通过独热编码将类别信息转换为新的列了，原类别列不再需要，
# 这样最终数据集只保留了数值列和独热编码后的类别列，数据结构更加规整，便于后续模型使用
data.drop(columns=cat_col,inplace=True)

# 再次查看数据集形状，确认删除操作后数据集的最终规模情况
data.shape






# 重新选择数据集中所有数据类型为数值型（整数、浮点数等）的列名，用于后续的数据归一化操作
# 这里要注意去除了'id'和'label'列，因为通常这两列不需要进行归一化处理，具体要根据实际情况和数据含义判断
num_col = list(data.select_dtypes(include='number').columns)
num_col.remove('id')
num_col.remove('label')
print(num_col)

# 创建一个MinMaxScaler对象，用于将数据归一化到指定的范围，这里指定的范围是[0, 1]区间，
# 也就是会把数据集中选定的数值列数据都缩放到这个区间内，这样做有助于提高模型训练效果和收敛速度等
minmax_scale = MinMaxScaler(feature_range=(0, 1))

# 定义一个函数用于对数据集进行归一化操作，该函数接受一个数据集（DataFrame类型）和一个列名列表作为参数
def normalization(df,col):
    # 遍历要归一化的列名列表中的每一个列名
    for i in col:
        # 获取当前列的数据，将其转换为numpy数组形式，方便后续使用MinMaxScaler进行处理
        arr = df[i]
        arr = np.array(arr)
        # 使用MinMaxScaler的fit_transform方法对当前列数据进行归一化处理，
        # 它会根据当前列数据自动计算最小值和最大值，并将数据进行相应的缩放变换到[0, 1]区间
        # 注意这里需要将一维数组重塑为二维数组形式（列向量形式），因为fit_transform方法要求输入的是二维数组
        df[i] = minmax_scale.fit_transform(arr.reshape(len(arr), 1))
    # 返回归一化后的数据集
    return df

# 查看数据在归一化之前的样子，方便对比归一化操作前后数据的变化情况，这里查看前几行数据即可大致了解
data.head()

# 调用前面定义的归一化函数，对数据集的指定列（由num_col列表指定）进行归一化处理，
# 使用数据集的副本进行操作，避免直接修改原始数据集（如果不希望改变原始数据的话），
# 经过这一步，数据集中选定的数值列就完成了归一化，变为[0, 1]区间内的值了
data = normalization(data.copy(),num_col)

# 查看数据在归一化之后的样子，通过与前面归一化前的数据对比，可直观看到归一化的效果，比如数据值的范围变化等
data.head()




# 对数据集中的攻击标签（假设'label'列表示攻击相关的标签情况）进行处理，
# 将其转换为只有两个类别（'normal'和'abnormal'）的形式，方便后续进行二元分类任务
# 这里通过一个lambda函数来实现转换，将标签值为0的转换为'normal'，其他值转换为'abnormal'，
# 最终得到一个新的只包含这两种分类情况的DataFrame对象
bin_label = pd.DataFrame(data.label.map(lambda x:'normal' if x==0 else 'abnormal'))

# 创建一个数据集的副本，用于后续存放带有二元标签（经过转换后的'normal'和'abnormal'标签）的数据，
# 以便进行与二元分类相关的操作，比如训练二元分类模型等，保持原始数据集不变（如果不希望修改原始数据的话）
bin_data = data.copy()
# 将新创建的数据集副本中的'label'列替换为前面转换好的二元标签数据，这样就完成了标签的初步转换
bin_data['label'] = bin_label

# 创建一个LabelEncoder对象，用于将文本形式的二元标签（'normal'和'abnormal'）编码为数值形式（通常是0和1），
# 方便后续模型进行处理，因为很多模型要求输入的标签是数值类型的
le1 = preprocessing.LabelEncoder()
# 使用LabelEncoder的fit_transform方法对二元标签数据进行编码，将文本标签转换为对应的数值标签，
# 比如可能把'normal'编码为0，'abnormal'编码为1，返回编码后的标签数据
enc_label = bin_label.apply(le1.fit_transform)
# 将数据集副本中的'label'列替换为编码后的数值标签，这样整个数据集就具备了适合二元分类模型处理的数值型标签了
bin_data['label'] = enc_label

# 查看LabelEncoder对象中编码对应的类别信息，也就是可以知道数值标签0和1分别对应原来的哪个文本标签，
# 方便后续根据数值标签还原出实际的类别含义等操作
le1.classes_

# 将LabelEncoder对象中编码对应的类别信息保存为一个npy文件，方便后续在需要的时候加载使用，
# 例如在模型预测后还原标签的实际含义等场景下可以用到，允许保存为可pickle的格式
np.save("le1_classes.npy",le1.classes_,allow_pickle=True)


# 创建一个数据集的副本，用于后续存放带有多类别标签（原始的攻击类别等多类别情况）的数据，
# 以便进行与多类别分类相关的操作，比如训练多类别分类模型等，保持原始数据集不变（如果不希望修改原始数据的话）
multi_data = data.copy()
# 获取数据集副本中'attack_cat'列的数据，也就是原始的多类别攻击标签数据，准备进行后续的编码等处理
multi_label = pd.DataFrame(multi_data.attack_cat)

# 使用pandas库的get_dummies函数对'attack_cat'列的多类别标签进行独热编码，
# 将每个不同的攻击类别转换为多个二进制的哑变量，与前面处理类别列的方式类似，
# 这样就把多类别标签信息以数值形式更方便地表示出来，便于后续模型处理
multi_data = pd.get_dummies(multi_data,columns=['attack_cat'])

# 创建一个LabelEncoder对象，用于将独热编码后的多类别标签（虽然已经是数值形式，但可能是多列的哑变量形式等）
# 进一步编码为连续的整数形式（比如0、1、2等），方便后续模型进行处理，很多模型更适合接收这样简单的整数标签形式
le2 = preprocessing.LabelEncoder()
# 使用LabelEncoder的fit_transform方法对多类别标签数据进行编码，将其转换为连续的整数标签，
# 返回编码后的标签数据，例如可能把不同的攻击类别依次编码为0到某个整数的数值
enc_label = multi_label.apply(le2.fit_transform)
# 将数据集副本中的'label'列替换为编码后的整数标签，这样整个数据集就具备了适合多类别分类模型处理的数值型标签了
multi_data['label'] = enc_label

# 查看LabelEncoder对象中编码对应的类别信息，也就是可以知道整数标签分别对应原来的哪个攻击类别，
# 方便后续根据数值标签还原出实际的类别含义等操作
le2.classes_

# 将LabelEncoder对象中编码对应的类别信息保存为一个npy文件，方便后续在需要的时候加载使用，
# 例如在模型预测后还原标签的实际含义等场景下可以用到，允许保存为可pickle的格式
np.save("le2_classes.npy",le2.classes_,allow_pickle=True)

# 将'label'列的列名添加到前面获取的数值列名列表中，这样后续计算相关性矩阵等操作会包含标签列，
# 便于找出与标签相关性较高的特征，用于特征选择，这里假设是针对二元分类的标签情况（前面经过编码后的）
num_col.append('label')

# 计算二元标签数据集（bin_data）中各特征与编码后的攻击标签（也就是'label'列）的皮尔逊相关系数矩阵，
# 皮尔逊相关系数可以衡量两个变量之间的线性相关程度，取值范围在-1到1之间，绝对值越接近1表示相关性越强
# 这里计算得到的corr_bin是一个DataFrame对象，行列索引对应各特征列名，每个元素表示对应两列特征之间的相关系数
plt.figure(figsize=(20,8))
corr_bin = bin_data[num_col].corr()

# 使用seaborn库的heatmap函数绘制相关系数矩阵的热力图，用于直观展示各特征之间以及特征与标签之间的相关性大小情况，
# 这里设置vmax=1.0表示相关系数矩阵中的最大值以1.0的颜色强度来显示（可根据实际情况调整），annot=False表示不在热力图上显示具体的数值，
# 只是通过颜色深浅来直观体现相关性大小，当然如果想显示具体数值可以设置annot=True并根据需要调整其他显示格式参数
sns.heatmap(corr_bin,vmax=1.0,annot=False)

# 给绘制的相关系数矩阵热力图添加标题，明确图表展示的是二元标签的相关系数矩阵情况，同时设置标题字体大小为16号字，使其更清晰醒目
plt.title('二元标签的相关系数矩阵',fontsize=16)
# 选择多类别标签数据（multi_data）中所有数据类型为数值型（整数、浮点数等）的列名，
# 用于后续计算相关性矩阵等操作，获取的这些列名列表将决定参与相关性分析的具体列
num_col = list(multi_data.select_dtypes(include='number').columns)

# 以下开始计算多类别标签的相关性矩阵，首先创建一个大小为20x8英寸的图形对象，作为绘制相关性矩阵热力图的画布
plt.figure(figsize=(20, 8))

# 计算多类别标签数据（multi_data）中由前面选取的数值列（num_col）之间的皮尔逊相关系数矩阵，
# 皮尔逊相关系数可以衡量两个变量之间的线性相关程度，取值范围在 -1 到 1 之间，
# 这里计算得到的 corr_multi 是一个 DataFrame 对象，行列索引对应各特征列名，每个元素表示对应两列特征之间的相关系数
corr_multi = multi_data[num_col].corr()

# 使用 seaborn 库的 heatmap 函数绘制相关系数矩阵的热力图，用于直观展示各特征之间以及特征与标签之间的相关性大小情况，
# 设置 vmax=1.0 表示相关系数矩阵中的最大值以 1.0 的颜色强度来显示（可根据实际情况调整），annot=False 表示不在热力图上显示具体的数值，
# 只是通过颜色深浅来直观体现相关性大小，当然如果想显示具体数值可以设置 annot=True 并根据需要调整其他显示格式参数
sns.heatmap(corr_multi, vmax=1.0, annot=False)

# 给绘制的相关系数矩阵热力图添加标题，明确图表展示的是多类别标签的相关系数矩阵情况，同时设置标题字体大小为 16 号字，使其更清晰醒目
plt.title('多类别标签的相关系数矩阵', fontsize=16)

# 将绘制好的相关性矩阵热力图保存为指定路径下的图片文件，方便后续查看或者用于文档、报告等展示，如果不需要保存可省略这行
plt.savefig('plots/correlation_matrix_multi.png')

# 显示绘制好的热力图，让其在运行代码的环境中展示出来，便于直观查看可视化效果
plt.show()




# 计算二元标签数据集中各特征与编码后的攻击标签（也就是 'label' 列）的皮尔逊相关系数的绝对值，
# 得到一个以各特征列为索引，对应相关系数绝对值为值的 Series 对象，用于后续筛选与标签相关性较强的特征
corr_ybin = abs(corr_bin['label'])

# 从前面计算得到的相关系数绝对值 Series 对象中，筛选出相关系数绝对值大于 0.3 的特征，
# 这些特征与标签的相关性相对较强，更有可能对分类任务有较大的影响，筛选后得到一个新的 Series 对象
highest_corr_bin = corr_ybin[corr_ybin > 0.3]

# 对筛选出的相关系数较高的特征 Series 对象按照值进行升序排序，方便查看哪些特征与标签的相关性大小顺序情况，
# 排序后的索引就是对应的特征列名，值是相应的相关系数绝对值
highest_corr_bin.sort_values(ascending=True)

# 选择通过上述皮尔逊相关系数筛选出来的特征对应的列名，也就是获取那些与标签相关性较强的特征列名集合，
# 这些列名将用于后续构建新的二元标签数据集，只保留相关性高的特征，达到特征选择、降维的目的
bin_cols = highest_corr_bin.index
bin_cols

# 根据前面筛选出的相关性较高的特征列名（bin_cols），从原始的二元标签数据集（bin_data）中选取这些列对应的数据，
# 创建一个新的数据集副本，只包含与标签相关性较强的特征，完成二元标签数据集的特征选择操作，减少数据维度的同时保留对分类有价值的特征
bin_data = bin_data[bin_cols].copy()
bin_data

# 将经过特征选择后的二元标签数据集保存为 CSV 文件，存储到指定路径 './datasets/bin_data.csv' 下，
# 方便后续直接加载使用这个处理好的数据集，例如在其他分析或模型训练场景中可以直接读取该文件获取处理好的数据
bin_data.to_csv('./datasets/bin_data.csv')


# 类似二元标签特征选择的操作，计算多类别标签数据集中各特征与编码后的攻击标签（也就是 'label' 列）的皮尔逊相关系数的绝对值，
# 得到一个以各特征列为索引，对应相关系数绝对值为值的 Series 对象，用于后续筛选与多类别标签相关性较强的特征
corr_ymulti = abs(corr_multi['label'])

# 从前面计算得到的多类别相关系数绝对值 Series 对象中，筛选出相关系数绝对值大于 0.3 的特征，
# 这些特征与多类别标签的相关性相对较强，对多类别分类任务可能有较大帮助，筛选后得到一个新的 Series 对象
highest_corr_multi = corr_ymulti[corr_ymulti > 0.3]

# 对筛选出的相关系数较高的多类别特征 Series 对象按照值进行升序排序，方便查看各特征与多类别标签相关性大小顺序情况，
# 排序后的索引就是对应的特征列名，值是相应的相关系数绝对值
highest_corr_multi.sort_values(ascending=True)

# 选择通过上述皮尔逊相关系数筛选出来的多类别特征对应的列名，即获取那些与多类别标签相关性较强的特征列名集合，
# 这些列名将用于后续构建新的多类别标签数据集，实现特征选择，保留对多类别分类有价值的特征，减少不必要的特征维度
multi_cols = highest_corr_multi.index
multi_cols

# 根据前面筛选出的相关性较高的多类别特征列名（multi_cols），从原始的多类别标签数据集（multi_data）中选取这些列对应的数据，
# 创建一个新的数据集副本，只包含与多类别标签相关性较强的特征，完成多类别标签数据集的特征选择操作，便于后续基于更精简有效的数据进行模型训练等
multi_data = multi_data[multi_cols].copy()

# 将经过特征选择后的多类别标签数据集保存为 CSV 文件，存储到指定路径 './datasets/multi_data.csv' 下，
# 方便后续直接加载使用这个处理好的数据集，例如在进一步的多类别分类分析或模型训练中可以直接读取该文件获取处理好的数据
multi_data.to_csv('./datasets/multi_data.csv')


# 对于二元分类任务，从经过特征选择后的二元标签数据集（bin_data）中，分离出特征列和标签列，
# 这里通过 drop 方法删除 'label' 列，得到只包含特征的数据集 X，用于后续作为模型的输入特征
X = bin_data.drop(columns=['label'], axis=1)

# 从二元标签数据集（bin_data）中获取 'label' 列作为标签数据，也就是分类任务的目标值，存储在变量 Y 中，后续用于模型训练时的监督信息
Y = bin_data['label']

# 使用 sklearn 库的 train_test_split 函数，按照测试集占比 20% 的比例，将特征数据集 X 和标签数据集 Y 划分为训练集和测试集，
# 随机种子设置为 50，这样每次划分的结果在相同随机种子下是可复现的，方便对比不同模型在相同数据划分下的性能表现
# 划分后得到训练集特征 X_train、测试集特征 X_test、训练集标签 y_train 和测试集标签 y_test，用于后续模型的训练和评估
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=50)
# 创建一个线性回归模型对象，这里虽然线性回归一般用于解决回归问题，但代码后续做了一些处理使其可以尝试用于二元分类任务（不过这种用法不太常规）
lr_bin = LinearRegression()

# 使用训练集特征 X_train 和训练集标签 y_train 对线性回归模型进行训练，拟合数据，使得模型学习到特征与标签之间的关系
lr_bin.fit(X_train, y_train)

# 使用训练好的线性回归模型对测试集特征 X_test 进行预测，得到预测结果，预测结果是连续的数值，后续需要进一步处理转换为类别值用于分类评估
y_pred = lr_bin.predict(X_test)

# 定义一个简单的 lambda 函数，用于将预测得到的连续数值转换为类别值（这里以 0.6 为阈值，大于 0.6 转换为 1，否则为 0），
# 这种将回归结果转换为类别值的方式是一种简单粗暴的处理手段，不太符合常规的分类模型使用逻辑，但代码中用于尝试实现二元分类效果
round = lambda x: 1 if x > 0.6 else 0

# 使用 np.vectorize 函数将前面定义的 lambda 函数向量化，使其可以对数组中的每个元素进行操作，
# 也就是将预测结果数组（y_pred）中的每个元素按照阈值规则转换为类别值，得到最终用于分类评估的预测类别结果
vfunc = np.vectorize(round)
y_pred = vfunc(y_pred)

# 计算并打印预测结果与真实测试集标签之间的平均绝对误差（MAE），它衡量了预测值与真实值之间的平均绝对差异大小，数值越小表示预测越准确
print("平均绝对误差 - ", metrics.mean_absolute_error(y_test, y_pred))

# 计算并打印预测结果与真实测试集标签之间的均方误差（MSE），它是预测误差的平方的平均值，对较大误差的惩罚更明显，数值越小越好
print("均方误差 - ", metrics.mean_squared_error(y_test, y_pred))

# 计算并打印预测结果与真实测试集标签之间的均方根误差（RMSE），它是均方误差的平方根，与原始数据的单位一致，便于直观理解误差大小，数值越小越好
print("均方根误差 - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# 计算并打印决定系数（R2 Score），它衡量了模型对数据的拟合程度，取值范围在 0 到 1 之间，越接近 1 表示模型拟合效果越好
print("R2 得分 - ", metrics.explained_variance_score(y_test, y_pred) * 100)

# 计算并打印准确率（Accuracy），即预测正确的样本数占总样本数的比例，用于直观评估模型在二元分类任务中的分类准确性，以百分数形式展示
print("准确率 - ", accuracy_score(y_test, y_pred) * 100)

# 使用 sklearn 库的 classification_report 函数生成详细的分类报告，包括精确率（Precision）、召回率（Recall）、F1 值（F1-score）等指标，
# 针对不同的类别（通过 target_names 参数指定类别名称，这里使用 le1.classes_ 对应的类别，也就是二元分类的具体类别名）展示各项指标情况，更全面地评估模型性能
cls_report = classification_report(y_true=y_test, y_pred=y_pred, target_names=le1.classes_)
print(cls_report)

# 创建一个 DataFrame 对象，将真实的测试集标签（y_test）作为 'Actual' 列，预测的类别结果（y_pred）作为 'Predicted' 列，
# 方便将真实值和预测值整理在一起查看对比情况，并且可以将这个数据框保存为 CSV 文件，用于后续进一步分析等
lr_bin_df = pd.DataFrame({'实际值': y_test, '预测值': y_pred})
lr_bin_df.to_csv('./predictions/lr_real_pred_bin.csv')
lr_bin_df

# 绘制真实值和预测值的对比图，创建一个大小为 20x8 英寸的图形对象作为画布，
# 先绘制预测值的曲线（取前 200 个数据点，方便展示部分数据情况），设置标签为 "预测值"，线条宽度为 2.0，颜色为蓝色
plt.figure(figsize=(20, 8))
plt.plot(y_pred[:200], label="预测值", linewidth=2.0, color='blue')

# 接着绘制真实值的曲线（取前 200 个测试集标签数据点），设置标签为 "真实值"，线条宽度为 2.0，颜色为浅珊瑚色（lightcoral）
plt.plot(y_test[:200].values, label="真实值", linewidth=2.0, color='lightcoral')

# 显示图例，通过前面设置的标签可以清楚区分图中两条曲线分别代表的是预测值还是真实值，并且图例位置会自动选择best显示
plt.legend(loc="best")

# 给图表添加标题，明确展示的是线性回归用于二元分类任务时真实值和预测值的对比情况，设置标题字体大小为合适字号使其更清晰
plt.title("线性回归二元分类任务中真实值与预测值对比")

# 将绘制好的对比图保存为指定路径下的图片文件，方便后续查看或者用于文档、报告等展示，如果不需要保存可省略这行
plt.savefig('plots/lr_real_pred_bin.png')

# 显示绘制好的对比图，让其在运行代码的环境中展示出来，便于直观查看真实值和预测值的分布及对比情况
plt.show()

# 以下是将训练好的线性回归模型保存到磁盘的操作，首先指定模型保存的文件名和路径（"./models/linear_regressor_binary.pkl"），
# 判断该文件是否已经存在，如果不存在，则使用 pickle 模块将训练好的模型对象（lr_bin）保存到磁盘上，保存为二进制文件格式（.pkl），
# 并打印提示信息表示模型已保存成功；如果文件已存在，则打印提示信息告知用户磁盘上已有该模型，可根据需要先删除旧模型再保存新的
pkl_filename = "./models/linear_regressor_binary.pkl"
if (not path.isfile(pkl_filename)):
    # 将训练好的模型保存到磁盘
    with open(pkl_filename, 'wb') as file:
        pickle.dump(lr_bin, file)
    print("已将模型保存到磁盘")
else:
    print("模型已保存")






# **逻辑回归模型**
# 创建一个逻辑回归模型对象，设置随机种子为123，最大迭代次数为5000。
# 随机种子的设置可以保证每次运行代码时模型训练的初始化情况相同（在相同数据集和参数下），便于结果复现。
# 最大迭代次数限制了模型训练时优化算法的迭代轮数，避免训练时间过长或陷入无限循环等情况。
logr_bin = LogisticRegression(random_state=123, max_iter=5000)
logr_bin

# 使用训练集特征X_train和训练集标签y_train对逻辑回归模型进行训练，
# 这个过程中模型会根据输入的特征和对应的标签学习两者之间的关系，调整模型内部的参数，以达到对数据的良好拟合。
logr_bin.fit(X_train, y_train)

# 使用训练好的逻辑回归模型对测试集特征X_test进行预测，得到预测的类别结果，
# 这些结果是模型基于学习到的规律，对测试集中每个样本所属类别做出的判断，通常是离散的类别值（例如在二元分类中就是两个类别对应的标签值）。
y_pred = logr_bin.predict(X_test)

# 计算并打印预测结果与真实测试集标签之间的平均绝对误差（Mean Absolute Error，MAE）。
# MAE衡量的是预测值与真实值之间的平均绝对差异大小，其值越小，表示模型预测的准确性越高，越接近真实情况。
print("平均绝对误差 - ", metrics.mean_absolute_error(y_test, y_pred))

# 计算并打印预测结果与真实测试集标签之间的均方误差（Mean Squared Error，MSE）。
# MSE是预测误差的平方的平均值，相比于MAE，它对较大误差的惩罚更明显，更关注较大偏差的情况，同样数值越小越好，说明模型预测效果好。
print("均方误差 - ", metrics.mean_squared_error(y_test, y_pred))

# 计算并打印预测结果与真实测试集标签之间的均方根误差（Root Mean Squared Error，RMSE）。
# RMSE是均方误差的平方根，它的好处是与原始数据的单位一致，这样在实际理解误差大小时更加直观，也是数值越小代表模型预测越准确。
print("均方根误差 - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# 计算并打印决定系数（R2 Score），也称为可决系数。
# 它衡量了模型对数据的拟合程度，取值范围在0到1之间，越接近1表示模型对数据的变化解释能力越强，拟合效果越好；越接近0则表示模型拟合效果较差。
print("R2得分 - ", metrics.explained_variance_score(y_test, y_pred) * 100)

# 计算并打印准确率（Accuracy），即预测正确的样本数占总样本数的比例，
# 这是一个直观评估模型在二元分类任务中分类准确性的指标，以百分数形式展示，准确率越高说明模型在该任务上的表现越好。
print("准确率 - ", accuracy_score(y_test, y_pred) * 100)

# 使用sklearn库的classification_report函数生成详细的分类报告，
# 其中包含了精确率（Precision）、召回率（Recall）、F1值（F1-score）等多个评估指标，针对不同的类别（通过target_names参数指定类别名称，这里使用le1.classes_对应的类别，也就是二元分类的具体类别名）展示各项指标情况，
# 这些指标从不同角度反映了模型在分类任务中的性能表现，能更全面地评估模型好坏，帮助分析模型在不同类别上的优势和不足。
cls_report = classification_report(y_true=y_test, y_pred=y_pred, target_names=le1.classes_)
print(cls_report)

# **真实值和预测值数据整理及保存**
# 创建一个DataFrame对象，将真实的测试集标签（y_test）作为'Actual'列，预测的类别结果（y_pred）作为'Predicted'列，
# 这样把真实值和预测值整理在一起，方便查看对比情况，并且可以后续对数据进行进一步的分析、可视化等操作。
logr_bin_df = pd.DataFrame({'实际值': y_test, '预测值': y_pred})
# 将这个包含真实值和预测值的数据框保存为CSV文件，存储到指定路径'./predictions/logr_real_pred_bin.csv'下，
# 方便后续随时读取该文件查看具体的预测和真实数据情况，例如用于在其他环境中进行数据分析或者报告生成等。
logr_bin_df.to_csv('./predictions/logr_real_pred_bin.csv')
logr_bin_df

# **绘制真实值和预测值对比图**
# 创建一个大小为20x8英寸的图形对象作为画布，用于绘制真实值和预测值的对比图，设置合适的图形尺寸可以使图表展示更清晰美观。
plt.figure(figsize=(20, 8))
# 绘制预测值的曲线，取预测结果（y_pred）的前200个数据点进行绘制，方便展示部分数据情况，设置标签为"预测值"，线条宽度为2.0，颜色为蓝色，便于在图中区分不同的曲线。
plt.plot(y_pred[:200], label="预测值", linewidth=2.0, color='blue')
# 接着绘制真实值的曲线，取前200个测试集标签数据点（y_test[:200].values）进行绘制，设置标签为"真实值"，线条宽度同样为2.0，颜色为浅珊瑚色（lightcoral），与预测值曲线颜色区分开，方便对比查看。
plt.plot(y_test[:200].values, label="真实值", linewidth=2.0, color='lightcoral')
# 显示图例，通过前面设置的标签可以清楚区分图中两条曲线分别代表的是预测值还是真实值，并且图例位置会自动选择best显示，使图表更易读。
plt.legend(loc="best")
# 给图表添加标题，明确展示的是逻辑回归用于二元分类任务时真实值和预测值的对比情况，设置标题字体大小为合适字号使其更清晰醒目，便于一眼了解图表内容。
plt.title("逻辑回归二元分类任务中真实值与预测值对比")
# 将绘制好的对比图保存为指定路径下的图片文件，方便后续查看或者用于文档、报告等展示，如果不需要保存可省略这行。
plt.savefig('plots/logr_real_pred_bin.png')
# 显示绘制好的对比图，让其在运行代码的环境中展示出来，便于直观查看真实值和预测值的分布及对比情况，帮助分析模型预测效果。
plt.show()

# **保存训练好的模型到磁盘**
# 指定模型保存的文件名和路径（"./models/logistic_regressor_binary.pkl"），这个文件将以二进制格式（.pkl）存储训练好的逻辑回归模型对象。
pkl_filename = "./models/logistic_regressor_binary.pkl"
# 判断该文件是否已经存在，如果不存在，就执行以下保存模型的操作。
if (not path.isfile(pkl_filename)):
    # 使用pickle模块将训练好的模型对象（logr_bin）保存到磁盘上，
    # 通过'wb'模式（write binary，二进制写入模式）打开文件，然后将模型对象序列化并写入文件中，实现模型的保存，方便后续直接加载使用该模型进行预测等操作。
    with open(pkl_filename, 'wb') as file:
        pickle.dump(logr_bin, file)
    print("已将模型保存到磁盘")
else:
    print("模型已保存")









# **线性支持向量机模型**
# 创建一个线性支持向量机（SVM）模型对象，指定核函数为'linear'（线性核），gamma参数设置为'auto'。
# 线性核函数适用于数据在原始特征空间中可能是线性可分的情况，gamma参数在一定程度上影响了模型对数据的拟合程度和泛化能力，'auto'表示自动选择合适的值。
lsvm_bin = SVC(kernel='linear', gamma='auto')
# 使用训练集特征X_train和训练集标签y_train对线性支持向量机模型进行训练，使模型学习如何区分不同类别，找到一个最优的决策边界来对数据进行分类。
lsvm_bin.fit(X_train, y_train)

# 使用训练好的线性支持向量机模型对测试集特征X_test进行预测，得到预测的类别结果，即判断测试集中每个样本属于哪一个类别。
y_pred = lsvm_bin.predict(X_test)

# 以下计算并打印和前面逻辑回归模型部分类似的一系列评估指标，方便对比不同模型在相同数据集上的性能表现。

# 计算并打印预测结果与真实测试集标签之间的平均绝对误差（MAE），衡量预测准确性。
print("平均绝对误差 - ", metrics.mean_absolute_error(y_test, y_pred))

# 计算并打印预测结果与真实测试集标签之间的均方误差（MSE），对较大误差更敏感的评估指标。
print("均方误差 - ", metrics.mean_squared_error(y_test, y_pred))

# 计算并打印预测结果与真实测试集标签之间的均方根误差（RMSE），与原始数据单位一致的误差指标。
print("均方根误差 - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# 计算并打印决定系数（R2 Score），反映模型对数据的拟合程度。
print("R2得分 - ", metrics.explained_variance_score(y_test, y_pred) * 100)

# 计算并打印准确率（Accuracy），直观体现分类正确的比例情况。
print("准确率 - ", accuracy_score(y_test, y_pred) * 100)

# 生成详细的分类报告，包含精确率、召回率、F1值等多方面评估指标，针对二元分类的具体类别展示模型性能。
cls_report = classification_report(y_true=y_test, y_pred=y_pred, target_names=le1.classes_)
print(cls_report)

# **真实值和预测值数据整理及保存**
# 创建包含真实值和预测值的DataFrame对象，列名分别为'Actual'（实际值）和'Predicted'（预测值），方便对比查看。
lsvm_bin_df = pd.DataFrame({'实际值': y_test, '预测值': y_pred})
# 将该数据框保存为CSV文件，存储到指定路径下，便于后续分析查看具体的预测和真实数据情况。
lsvm_bin_df.to_csv('./predictions/lsvm_real_pred_bin.csv')
lsvm_bin_df

# **绘制真实值和预测值对比图**
# 创建一个大小为20x8英寸的图形画布，用于绘制对比图。
plt.figure(figsize=(20, 8))
# 绘制预测值曲线，取前200个预测结果数据点，设置相应的标签、线条宽度和颜色，便于在图中识别。
plt.plot(y_pred[:200], label="预测值", linewidth=2.0, color='blue')
# 绘制真实值曲线，取前200个测试集标签数据点，同样设置好标签、线条宽度和颜色，与预测值曲线区分开以方便对比。
plt.plot(y_test[:200].values, label="真实值", linewidth=2.0, color='lightcoral')
# 显示图例，使其自动选择best显示，清晰展示两条曲线对应的内容。
plt.legend(loc="best")
# 添加图表标题，明确展示的是线性SVM二元分类任务中真实值和预测值的对比情况。
plt.title("线性SVM二元分类任务中真实值与预测值对比")
# 保存绘制好的对比图为指定路径下的图片文件，可根据需要选择是否保存。
plt.savefig('plots/lsvm_real_pred_bin.png')
# 显示绘制好的对比图，便于直观查看模型预测效果和真实值对比情况。
plt.show()

# **保存训练好的模型到磁盘**
# 指定模型保存的文件名和路径，用于后续判断模型是否已存在以及保存操作。
pkl_filename = "./models/lsvm_binary.pkl"
# 如果模型文件不存在，则将训练好的线性支持向量机模型保存到磁盘，保存为二进制文件格式。
if (not path.isfile(pkl_filename)):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(lsvm_bin, file)
    print("已将模型保存到磁盘")
else:
    print("模型已保存")







# **K近邻分类器模型**
# 创建一个K近邻分类器模型对象，设置邻居数量（n_neighbors）为5。
# K值决定了在进行分类预测时，参考多少个最近的邻居样本的类别来确定当前样本的类别，这里选择5个邻居作为参考依据。
knn_bin = KNeighborsClassifier(n_neighbors=5)
# 使用训练集特征X_train和训练集标签y_train对K近邻分类器模型进行训练，
# 实际上K近邻模型在训练阶段主要是记录训练数据的特征和标签信息，后续预测时根据新样本与训练样本的距离来判断类别。
knn_bin.fit(X_train, y_train)

# 使用训练好的K近邻分类器模型对测试集特征X_test进行预测，得到每个测试样本的预测类别结果。
y_pred = knn_bin.predict(X_test)

# 同样计算并打印一系列评估指标，用于衡量模型性能，与前面其他模型部分的指标含义相同，便于对比分析。

# 计算并打印平均绝对误差（MAE）。
print("平均绝对误差 - ", metrics.mean_absolute_error(y_test, y_pred))

# 计算并打印均方误差（MSE）。
print("均方误差 - ", metrics.mean_squared_error(y_test, y_pred))

# 计算并打印均方根误差（RMSE）。
print("均方根误差 - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# 计算并打印决定系数（R2 Score）。
print("R2得分 - ", metrics.explained_variance_score(y_test, y_pred) * 100)

# 计算并打印准确率（Accuracy）。
print("准确率 - ", accuracy_score(y_test, y_pred) * 100)

# 生成详细的分类报告，展示不同类别上的精确率、召回率、F1值等指标情况。
cls_report = classification_report(y_true=y_test, y_pred=y_pred, target_names=le1.classes_)
print(cls_report)

# **真实值和预测值数据整理及保存**
# 创建包含真实值和预测值的DataFrame对象，方便后续查看对比数据情况。
knn_bin_df = pd.DataFrame({'实际值': y_test, '预测值': y_pred})
# 将该数据框保存为CSV文件，存储到指定路径下，方便后续进一步分析使用。
knn_bin_df.to_csv('./predictions/knn_real_pred_bin.csv')
knn_bin_df

# **绘制真实值和预测值对比图**
# 创建一个大小为20x8英寸的图形画布，用于绘制对比图。
plt.figure(figsize=(20, 8))
# 绘制预测值曲线，取前200个预测结果数据点，设置相关绘制参数。
plt.plot(y_pred[:200], label="预测值", linewidth=2.0, color='blue')
# 绘制真实值曲线，取前200个测试集标签数据点，设置对应绘制参数以区分两条曲线。
plt.plot(y_test[:200].values, label="真实值", linewidth=2.0, color='lightcoral')
# 显示图例，让其自动选择最佳显示位置，便于查看图表内容。
plt.legend(loc="best")
# 添加图表标题，明确展示的是KNN二元分类任务中真实值和预测值的对比情况。
plt.title("KNN二元分类任务中真实值与预测值对比")
# 保存绘制好的对比图为指定路径下的图片文件，可按需选择是否保存。
plt.savefig('plots/knn_real_pred_bin.png')
# 显示绘制好的对比图，直观查看模型预测效果与真实值对比情况。
plt.show()

# **保存训练好的模型到磁盘**
# 指定模型保存的文件名和路径，用于后续判断模型是否已存在以及保存操作。
pkl_filename = "./models/knn_binary.pkl"
# 如果模型文件不存在，则将训练好的K近邻分类器模型保存到磁盘，保存为二进制文件格式。
if (not path.isfile(pkl_filename)):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(knn_bin, file)
    print("已将模型保存到磁盘")
else:
    print("模型已保存")







# **随机森林分类器模型用于二元分类任务**

# 创建一个随机森林分类器对象，设置随机种子为123。随机种子的作用是确保每次运行代码时，
# 模型内部随机初始化等涉及随机性的操作结果是可复现的，方便对比不同次运行或者不同参数调整下模型的表现。
rf_bin = RandomForestClassifier(random_state=123)

# 使用训练集特征数据 X_train 和对应的训练集标签数据 y_train 对随机森林分类器进行训练。
# 在训练过程中，随机森林会构建多个决策树（具体数量由模型默认参数或手动设置决定），
# 每个决策树基于训练数据的不同随机子集和特征子集进行学习，以捕捉数据中的不同模式和规律，最终综合这些决策树的预测结果来进行分类决策。
rf_bin.fit(X_train, y_train)

# 使用训练好的随机森林分类器对测试集特征数据 X_test 进行预测，得到每个测试样本对应的预测类别结果，
# 这些结果将用于后续与真实标签对比，评估模型在测试集上的性能表现。
y_pred = rf_bin.predict(X_test)

# 以下是一系列评估模型性能的指标计算和打印输出，用于从不同角度衡量模型预测结果与真实标签之间的差异和拟合程度。

# 计算并打印平均绝对误差（Mean Absolute Error，MAE）。MAE 衡量的是预测值与真实值之间绝对差值的平均值，
# 它直观地反映了预测结果平均偏离真实值的程度，数值越小表示模型预测的准确性越高，越接近真实标签。
print("平均绝对误差 - ", metrics.mean_absolute_error(y_test, y_pred))

# 计算并打印均方误差（Mean Squared Error，MSE）。MSE 是预测值与真实值误差的平方的平均值，
# 相比于 MAE，它对较大的误差惩罚更重，更关注预测偏差较大的情况，同样数值越小代表模型预测效果越好。
print("均方误差 - ", metrics.mean_squared_error(y_test, y_pred))

# 计算并打印均方根误差（Root Mean Squared Error，RMSE）。RMSE 是 MSE 的平方根，
# 其优点是与原始数据的单位相同，在实际理解误差大小时更加直观，也是数值越小说明模型预测越准确，越能贴合真实数据情况。
print("均方根误差 - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# 计算并打印决定系数（R2 Score），也称为可决系数。它表示模型对数据中因变量变异的解释程度，
# 取值范围在 0 到 1 之间，越接近 1 说明模型能够解释的因变量变异越多，即模型对数据的拟合效果越好；越接近 0 则表示模型拟合能力较差。
print("R2 得分 - ", metrics.explained_variance_score(y_test, y_pred) * 100)

# 计算并打印准确率（Accuracy），它是指预测正确的样本数量占总测试样本数量的比例，
# 以百分数形式展示，能够直观地反映模型在二元分类任务中整体分类正确的情况，准确率越高说明模型在该任务上的性能越好。
print("准确率 - ", accuracy_score(y_test, y_pred) * 100)

# 使用 `classification_report` 函数生成详细的分类报告，该报告包含了精确率（Precision）、召回率（Recall）、F1 值（F1-score）等多个重要的分类评估指标，
# 并且针对不同的类别（通过 `target_names` 参数指定类别名称，这里使用 `le1.classes_` 对应的二元分类的具体类别名）分别展示各项指标情况，
# 这些指标可以更全面、细致地评估模型在不同类别上的分类性能，帮助分析模型的优势和可能存在的不足，例如是否存在某一类别的预测偏差较大等情况。
cls_report = classification_report(y_true=y_test, y_pred=y_pred, target_names=le1.classes_)
print(cls_report)

# **整理真实值和预测值数据并保存为 CSV 文件**
# 创建一个 `DataFrame` 对象，将真实的测试集标签数据 `y_test` 作为 'Actual'（实际值）列，预测得到的类别结果 `y_pred` 作为 'Predicted'（预测值）列，
# 通过这种方式把真实值和预测值整理在一起，方便后续查看、对比两者之间的差异情况，也便于进行进一步的数据处理或分析，例如在其他数据分析工具中导入查看。
rf_bin_df = pd.DataFrame({'实际值': y_test, '预测值': y_pred})
# 将这个包含了真实值和预测值的数据框保存为 CSV 文件，存储到指定路径 './predictions/rf_real_pred_bin.csv' 下，
# 这样可以长期保存预测结果数据，方便后续随时回顾查看具体的预测和真实数据情况，例如在模型优化过程中对比不同版本模型的预测结果变化等。
rf_bin_df.to_csv('./predictions/rf_real_pred_bin.csv')
rf_bin_df

# **绘制真实值和预测值对比图**
# 创建一个大小为 20x8 英寸的图形对象作为画布，用于绘制真实值和预测值的对比可视化图表，合适的图形尺寸有助于清晰展示数据曲线和细节内容。
plt.figure(figsize=(20, 8))
# 在画布上绘制预测值的曲线，这里选取预测结果 `y_pred` 中索引从 200 到 400 的数据点进行绘制，设置标签为 "prediction"（预测值），
# 线条宽度为 2.0，颜色为蓝色，通过这些设置可以使预测值曲线在图表中清晰可见且易于区分。
plt.plot(y_pred[200:400], label="预测值", linewidth=2.0, color='blue')
# 接着绘制真实值的曲线，选取对应的测试集标签数据 `y_test` 中索引从 200 到 400 的数据点（通过 `.values` 获取其数值形式）进行绘制，
# 设置标签为 "真实值"（真实值），线条宽度同样为 2.0，颜色为浅珊瑚色（'lightcoral'），与预测值曲线颜色不同以便于对比查看。
plt.plot(y_test[200:400].values, label="真实值", linewidth=2.0, color='lightcoral')
# 显示图例，让图例自动选择在图表中最合适的位置显示，通过前面设置的标签，能够清晰地表明两条曲线分别代表的是预测值还是真实值，方便查看对比。
plt.legend(loc="best")
# 给图表添加标题，明确展示的是随机森林在二元分类任务中真实值和预测值的对比情况，设置合适的标题字体大小使其更清晰醒目，让人一眼就能了解图表的核心内容。
plt.title("随机森林二元分类任务中真实值与预测值对比")
# 将绘制好的对比图保存为指定路径 'plots/rf_real_pred_bin.png' 下的图片文件，方便后续查看或者将其用于文档、报告等展示，如果不需要保存图片可省略这行代码。
plt.savefig('plots/rf_real_pred_bin.png')
# 显示绘制好的对比图，让其在运行代码的环境中展示出来，便于直观地查看真实值和预测值的分布情况以及它们之间的对比关系，进而帮助分析模型的预测效果是否合理等。
plt.show()

# **保存训练好的模型到磁盘**
# 指定要保存的模型文件的文件名和路径为 "./models/random_forest_binary.pkl"，这里使用 `.pkl` 文件格式，
# 它是 Python 中常用的用于序列化和保存对象的二进制文件格式，方便后续直接加载使用该训练好的模型进行预测等操作。
pkl_filename = "./models/random_forest_binary.pkl"
# 通过判断该文件是否已经存在来决定是否执行保存模型的操作，如果文件不存在，就进入以下保存模型的代码块。
if (not path.isfile(pkl_filename)):
    # 使用 `pickle` 模块将训练好的随机森林分类器模型对象 `rf_bin` 保存到磁盘上，
    # 通过 'wb'（write binary，二进制写入模式）打开指定的文件，然后将模型对象进行序列化并写入到文件中，实现模型的持久化保存。
    with open(pkl_filename, 'wb') as file:
        pickle.dump(rf_bin, file)
    print("已将模型保存到磁盘")
else:
    print("模型已保存")





# **决策树分类器模型用于二元分类任务**

# 创建一个决策树分类器对象，设置随机种子为 123。随机种子同样是为了保证模型训练过程中的随机性是可复现的，
# 例如在决策树构建过程中节点分裂时选择特征等操作涉及随机因素，设置种子可以使每次运行代码生成相同结构的决策树（在相同数据和参数下），便于对比分析。
dt_bin = DecisionTreeClassifier(random_state=123)

# 使用训练集特征数据 X_train 和对应的训练集标签数据 y_train 对决策树分类器进行训练，
# 决策树会根据训练数据学习特征与标签之间的关系，通过不断地选择最优特征进行节点分裂，构建出一棵能够对数据进行分类的树形结构模型，
# 最终每个叶子节点对应一个类别预测结果。
dt_bin.fit(X_train, y_train)

# 使用训练好的决策树分类器对测试集特征数据 X_test 进行预测，得到每个测试样本对应的预测类别结果，用于后续与真实标签对比评估模型性能。
y_pred = dt_bin.predict(X_test)

# 以下计算并打印与前面随机森林分类器模型部分相同的一系列评估指标，方便对比不同模型在相同数据集上的性能表现差异。

# 计算并打印平均绝对误差（MAE），反映预测值与真实值的平均绝对偏差情况，数值越小越好。
print("平均绝对误差 - ", metrics.mean_absolute_error(y_test, y_pred))

# 计算并打印均方误差（MSE），对较大误差更敏感的误差衡量指标，数值越小表示模型预测越准确。
print("均方误差 - ", metrics.mean_squared_error(y_test, y_pred))

# 计算并打印均方根误差（RMSE），与原始数据单位一致的误差指标，便于直观理解误差大小，越小代表模型性能越好。
print("均方根误差 - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# 计算并打印决定系数（R2 Score），衡量模型对数据的拟合程度，取值范围 0 到 1，越接近 1 拟合越好。
print("R2 得分 - ", metrics.explained_variance_score(y_test, y_pred) * 100)

# 计算并打印准确率（Accuracy），直观体现模型在二元分类任务中分类正确的样本比例，以百分数展示，越高越好。
print("准确率 - ", accuracy_score(y_test, y_pred) * 100)

# 生成详细的分类报告，包含精确率、召回率、F1 值等多方面指标，针对二元分类的具体类别展示模型性能情况，帮助全面评估模型。
cls_report = classification_report(y_true=y_test, y_pred=y_pred, target_names=le1.classes_)
print(cls_report)

# **整理真实值和预测值数据并保存为 CSV 文件**
# 创建包含真实值和预测值的 `DataFrame` 对象，列名分别为 'Actual'（实际值）和 'Predicted'（预测值），方便后续查看对比两者的数据情况。
dt_bin_df = pd.DataFrame({'实际值': y_test, '预测值': y_pred})
# 将该数据框保存为 CSV 文件，存储到指定路径 './predictions/dt_real_pred_bin.csv' 下，便于后续随时查看具体的预测和真实数据细节，进行进一步分析等。
dt_bin_df.to_csv('./predictions/dt_real_pred_bin.csv')
dt_bin_df

# **绘制真实值和预测值对比图**
# 创建一个大小为 20x8 英寸的图形画布，用于绘制真实值和预测值对比的可视化图表，设置合适尺寸方便展示清晰。
plt.figure(figsize=(20, 8))
# 绘制预测值曲线，选取预测结果 `y_pred` 中索引从 300 到 500 的数据点进行绘制，设置相应的标签、线条宽度和颜色等属性，便于图表中识别区分。
plt.plot(y_pred[300:500], label="预测值", linewidth=2.0, color='blue')
# 绘制真实值曲线，选取对应的测试集标签数据 `y_test` 中索引从 300 到 500 的数据点进行绘制，同样设置好标签、线条宽度和颜色等，使其与预测值曲线区分开便于对比查看。
plt.plot(y_test[300:500].values, label="真实值", linewidth=2.0, color='lightcoral')
# 显示图例，使其自动选择best显示，通过前面设置的标签清晰展示两条曲线分别对应的内容，方便查看图表信息。
plt.legend(loc="best")
# 添加图表标题，明确展示的是决策树在二元分类任务中真实值和预测值的对比情况，设置合适字体大小使标题更醒目。
plt.title("决策树二元分类任务中真实值与预测值对比")
# 保存绘制好的对比图为指定路径 'plots/dt_real_pred_bin.png' 下的图片文件，可根据实际需求选择是否保存。
plt.savefig('plots/dt_real_pred_bin.png')
# 显示绘制好的对比图，便于直观查看模型预测效果与真实值之间的对比关系，辅助分析模型性能。
plt.show()

# **保存训练好的模型到磁盘**
# 指定模型保存的文件名和路径为 "./models/decision_tree_binary.pkl"，用于后续判断模型是否已存在以及执行保存操作。
pkl_filename = "./models/decision_tree_binary.pkl"
# 如果该文件不存在，就将训练好的决策树分类器模型保存到磁盘，保存为二进制文件格式（`.pkl`），方便后续加载使用。
if (not path.isfile(pkl_filename)):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(dt_bin, file)
    print("已将模型保存到磁盘")
else:
    print("模型已保存")







# **多层感知机（MLP）用于二元分类任务**

# 创建一个多层感知机分类器（MLPClassifier）对象，用于构建神经网络模型进行二元分类。
# 设置了以下参数：
# - `random_state=123`：指定随机种子为123。这是为了保证模型训练过程中的随机性是可复现的，
#   例如在神经网络权重初始化、训练过程中数据的随机打乱等操作涉及随机因素，设置固定的随机种子可以使得每次运行代码时这些随机操作的结果保持一致，方便对比不同次运行或者不同参数调整下模型的表现情况。
# - `solver='adam'`：选择 'adam' 作为优化器。'adam' 是一种自适应学习率的优化算法，它结合了动量法和自适应学习率调整的优点，在训练神经网络时通常能够更快地收敛，并且在处理复杂的非线性问题、寻找最优权重参数方面表现较好，有助于模型更高效地学习数据中的模式和规律。
# - `max_iter=8000`：设定最大迭代次数为8000。在神经网络的训练过程中，一次迭代指的是对整个训练数据集进行一次完整的前向传播（根据输入特征计算预测结果）和反向传播（根据预测结果与真实标签的差异来更新网络权重）操作。
#   通过限制最大迭代次数，可以避免模型训练时间过长，也能防止出现过拟合现象（即模型在训练数据上表现很好，但在未见过的测试数据上表现不佳），当达到设定的迭代次数后，训练过程就会停止。
mlp_bin = MLPClassifier(random_state=123, solver='adam', max_iter=8000)

# 使用训练集特征数据 `X_train` 和对应的训练集标签数据 `y_train` 对多层感知机分类器进行训练。
# 在训练过程中，神经网络按照其内部默认的网络结构（一般会根据输入特征维度等因素自动确定合适的层数和神经元数量，也可以手动配置更复杂的结构）进行操作：
# 首先进行前向传播，将输入的特征数据通过各层神经元的计算逐步转化为预测结果；然后根据预测结果与真实标签 `y_train` 的差异，通过定义的损失函数（例如交叉熵损失等，用于衡量预测的好坏程度）计算损失值；
# 接着利用反向传播算法，根据损失函数对各层权重参数的梯度信息，从输出层往输入层反向依次更新权重参数，不断调整网络的权重，使得损失函数的值逐渐减小，从而让模型对训练数据的拟合效果越来越好，能够更准确地进行分类预测。
mlp_bin.fit(X_train, y_train)

# 使用训练好的多层感知机分类器对测试集特征数据 `X_test` 进行预测，得到每个测试样本对应的预测类别结果，
# 这些预测结果后续将与真实的测试集标签 `y_test` 进行对比，以此来评估模型在新的、未参与训练的数据上的性能表现，判断模型是否具有良好的泛化能力。
y_pred = mlp_bin.predict(X_test)

# 以下是一系列用于评估模型性能的指标计算和打印输出部分，通过不同的指标从多个角度衡量模型预测结果与真实标签之间的差异以及模型对数据的拟合程度。

# 计算并打印平均绝对误差（Mean Absolute Error，MAE）。MAE 的计算方式是先求出预测值与真实值之间的绝对差值，然后取这些差值的平均值。
# 它直观地反映了预测结果平均偏离真实值的程度，单位与数据本身的单位一致，数值越小表示模型预测的准确性越高，意味着预测值越接近真实标签对应的数值，模型在平均意义上的预测偏差越小。
print("平均绝对误差 - ", metrics.mean_absolute_error(y_test, y_pred))

# 计算并打印均方误差（Mean Squared Error，MSE）。MSE 是先计算预测值与真实值误差的平方，再取这些平方值的平均值。
# 相比于 MAE，MSE 对较大的误差惩罚更重，因为误差经过平方后，较大的误差会被放大，所以更关注预测偏差较大的情况。同样，MSE 的数值越小代表模型预测效果越好，说明模型能够更准确地拟合数据，使得预测值与真实值之间的误差尽可能小。
print("均方误差 - ", metrics.mean_squared_error(y_test, y_pred))

# 计算并打印均方根误差（Root Mean Squared Error，RMSE）。RMSE 是 MSE 的平方根，
# 它的好处是与原始数据的单位相同，这样在实际理解误差大小时更加直观，例如如果预测的数据是价格、长度等具有实际物理意义的量，RMSE 的单位就和这些量的单位一致，方便我们直接根据其数值大小直观判断模型预测误差的大小。和前面的误差指标一样，RMSE 的数值越小说明模型预测越准确，越能贴合真实数据情况。
print("均方根误差 - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# 计算并打印决定系数（R2 Score），也称为可决系数。它衡量了模型对数据中因变量变异的解释程度，其取值范围在 0 到 1 之间。
# 计算公式涉及到模型预测结果的方差以及真实数据的方差等信息，具体来说，它表示模型所解释的因变量变异占总变异的比例。越接近 1 说明模型能够解释的因变量变异越多，
# 即模型对数据的拟合效果越好，意味着模型可以很好地捕捉到数据中的规律；越接近 0 则表示模型拟合能力较差，对数据的变化解释能力不足，模型可能还需要进一步改进或者数据本身可能存在一些难以用该模型拟合的复杂特性。
print("R2 得分 - ", metrics.explained_variance_score(y_test, y_pred) * 100)

# 计算并打印准确率（Accuracy），它是指预测正确的样本数量占总测试样本数量的比例。计算过程是简单地对比预测结果 `y_pred` 和真实测试集标签 `y_test` 中相同位置的元素是否相等，统计出正确预测的样本数，
# 然后除以总样本数并乘以 100 将其转换为百分数形式展示。准确率能够直观地反映模型在二元分类任务中整体分类正确的情况，准确率越高说明模型在该任务上的性能越好，能够准确地将测试样本划分到正确的类别中。
print("准确率 - ", accuracy_score(y_test, y_pred) * 100)

# 使用 `classification_report` 函数生成详细的分类报告，该报告包含了精确率（Precision）、召回率（Recall）、F1 值（F1-score）等多个重要的分类评估指标，
# 并且针对不同的类别（通过 `target_names` 参数指定类别名称，这里使用 `le1.classes_` 对应的二元分类的具体类别名）分别展示各项指标情况。
# 精确率表示在预测为某一类别的样本中，真正属于该类别的样本比例，它侧重于衡量模型预测结果的准确性；召回率表示在真实属于某一类别的样本中，被模型正确预测出来的比例，更关注模型对某一类别的查全能力；
# F1 值是精确率和召回率的调和平均数，综合考虑了两者的情况，能够更全面地评价模型在某个类别上的分类性能。这些指标可以更全面、细致地评估模型在不同类别上的分类性能，帮助分析模型的优势和可能存在的不足，
# 例如是否存在某一类别的预测偏差较大，对某一类别的识别能力较弱等情况，以便针对性地对模型进行优化改进。
cls_report = classification_report(y_true=y_test, y_pred=y_pred, target_names=le1.classes_)
print(cls_report)

# **整理真实值和预测值数据并保存为 CSV 文件**
# 创建一个 `DataFrame` 对象，将真实的测试集标签数据 `y_test` 作为 'Actual'（实际值）列，预测得到的类别结果 `y_pred` 作为 'Predicted'（预测值）列，
# 通过这种方式把真实值和预测值整理在一起，方便后续查看、对比两者之间的差异情况，也便于进行进一步的数据处理或分析，例如可以在其他数据分析工具（如 Excel 等）中导入查看，或者基于这些数据进行更复杂的数据可视化等操作。
mlp_bin_df = pd.DataFrame({'实际值': y_test, '预测值': y_pred})
# 将这个包含了真实值和预测值的数据框保存为 CSV 文件，存储到指定路径 './predictions/mlp_real_pred_bin.csv' 下，
# 这样可以长期保存预测结果数据，方便后续随时回顾查看具体的预测和真实数据情况，例如在模型优化过程中对比不同版本模型的预测结果变化，或者在撰写项目报告时引用具体的数据等情况。
mlp_bin_df.to_csv('./predictions/mlp_real_pred_bin.csv')
mlp_bin_df

# **绘制真实值和预测值对比图**
# 创建一个大小为 20x8 英寸的图形对象作为画布，用于绘制真实值和预测值的对比可视化图表，合适的图形尺寸有助于清晰展示数据曲线和细节内容，使我们能够更直观地观察预测值与真实值之间的变化趋势和差异情况。
plt.figure(figsize=(20, 8))
# 在画布上绘制预测值的曲线，这里选取预测结果 `y_pred` 中索引从 100 到 200 的数据点进行绘制，设置标签为 "prediction"（预测值），
# 线条宽度为 2.0，颜色为蓝色，通过这些设置可以使预测值曲线在图表中清晰可见且易于区分，方便与后续绘制的真实值曲线对比观察。
plt.plot(y_pred[100:200], label="预测值", linewidth=2.0, color='blue')
# 接着绘制真实值的曲线，选取对应的测试集标签数据 `y_test` 中索引从 100 到 200 的数据点（通过 `.values` 获取其数值形式）进行绘制，
# 设置标签为 "真实值"（真实值），线条宽度同样为 2.0，颜色为浅珊瑚色（'lightcoral'），与预测值曲线颜色不同以便于对比查看，更清晰地展示两者之间的差异和变化趋势。
plt.plot(y_test[100:200].values, label="真实值", linewidth=2.0, color='lightcoral')
# 显示图例，让图例自动选择在图表中最合适的位置显示，通过前面设置的标签，能够清晰地表明两条曲线分别代表的是预测值还是真实值，方便查看对比，使图表更易读易懂，便于直观分析模型的预测效果。
plt.legend(loc="best")
# 给图表添加标题，明确展示的是多层感知机在二元分类任务中真实值和预测值的对比情况，设置合适的标题字体大小使其更清晰醒目，让人一眼就能了解图表的核心内容，知道图表所展示的具体模型和任务信息。
plt.title("MLP 二元分类任务中真实值与预测值对比")
# 将绘制好的对比图保存为指定路径 'plots/mlp_real_pred_bin.png' 下的图片文件，方便后续查看或者将其用于文档、报告等展示，如果不需要保存图片可省略这行代码。
plt.savefig('plots/mlp_real_pred_bin.png')
# 显示绘制好的对比图，让其在运行代码的环境中展示出来，便于直观地查看真实值和预测值的分布情况以及它们之间的对比关系，进而帮助分析模型的预测效果是否合理等，例如观察预测值是否能够较好地跟随真实值的变化趋势，是否存在明显的偏差等情况。
plt.show()

# **保存训练好的模型到磁盘**
# 指定要保存的模型文件的文件名和路径为 "./models/mlp_binary.pkl"，这里使用 `.pkl` 文件格式，
# 它是 Python 中常用的用于序列化和保存对象的二进制文件格式，通过将训练好的模型对象保存为这种格式的文件，可以方便后续在其他代码中或者不同的运行环境下直接加载使用该模型进行预测等操作，无需重新训练，节省时间和计算资源。
pkl_filename = "./models/mlp_binary.pkl"
# 通过判断该文件是否已经存在来决定是否执行保存模型的操作，如果文件不存在，就进入以下保存模型的代码块。
if (not path.isfile(pkl_filename)):
    # 使用 `pickle` 模块将训练好的多层感知机分类器模型对象 `mlp_bin` 保存到磁盘上，
    # 通过 'wb'（write binary，二进制写入模式）打开指定的文件，然后将模型对象进行序列化并写入到文件中，实现模型的持久化保存，
    # 即将模型当前的状态（包括网络结构、训练好的权重参数等信息）以二进制形式存储在磁盘文件中，以便后续随时读取使用。
    with open(pkl_filename, 'wb') as file:
        pickle.dump(mlp_bin, file)
    print("已将模型保存到磁盘")
else:
    print("模型已保存")




# **多分类任务的数据分割**

# 从 `multi_data` 数据集中提取特征数据，通过 `drop` 方法去掉包含标签信息的 'label' 列（这里假设 `multi_data` 是一个包含特征列和标签列的数据集，比如是一个 `DataFrame` 对象），
# 得到只包含特征的数据集 `X`，用于后续模型训练和测试的输入特征部分。
X = multi_data.drop(columns=['label'], axis=1)
# 从 `multi_data` 数据集中提取标签数据，将 'label' 列的数据单独提取出来作为 `Y`，它将作为模型训练和测试时对应的目标值，也就是要预测的类别信息。
Y = multi_data['label']

# 使用 `train_test_split` 函数将数据集 `X`（特征数据）和 `Y`（标签数据）按照一定比例划分为训练集和测试集。
# 这里设置 `test_size=0.30`，表示将总数据集的 30% 划分为测试集，剩余的 70% 作为训练集；`random_state=100` 同样是设置随机种子，
# 确保每次划分数据集时，在相同的数据和参数设置下，得到的训练集和测试集的样本是固定的，便于结果的复现和不同模型之间的公平比较，
# 最终得到训练集特征 `X_train`、测试集特征 `X_test`、训练集标签 `y_train` 和测试集标签 `y_test` 这四个部分的数据，用于后续不同模型的训练和评估。
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=100)






# 创建逻辑回归模型对象，设置随机种子、最大迭代次数、求解器及多分类方式等参数
logr_multi = LogisticRegression(random_state=123, max_iter=5000, solver='newton-cg', multi_class='multinomial')
# 使用训练集数据训练逻辑回归模型
logr_multi.fit(X_train, y_train)

# 用训练好的模型对测试集进行预测，得到预测的类别结果
y_pred = logr_multi.predict(X_test)

# 输出多个评估指标，查看模型在多分类任务上的性能表现
print("平均绝对误差 - ", metrics.mean_absolute_error(y_test, y_pred))
print("均方误差 - ", metrics.mean_squared_error(y_test, y_pred))
print("均方根误差 - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 得分 - ", metrics.explained_variance_score(y_test, y_pred) * 100)
print("准确率 - ", accuracy_score(y_test, y_pred) * 100)
print(classification_report(y_test, y_pred, target_names=le2.classes_))

# 整理真实值和预测值数据到DataFrame并保存为CSV文件
logr_multi_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
logr_multi_df.to_csv('./predictions/logr_real_pred_multi.csv')
logr_multi_df

# 绘制真实值和预测值对比图（取部分数据展示）
plt.figure(figsize=(20, 8))
plt.plot(y_pred[:200], label="预测值", linewidth=2.0, color='blue')
plt.plot(y_test[:200].values, label="真实值", linewidth=2.0, color='lightcoral')
plt.legend(loc="best")
plt.title("逻辑回归多分类任务中真实值与预测值对比")
plt.savefig('plots/logr_real_pred_multi.png')
plt.show()

# 判断模型文件是否存在，不存在则保存训练好的模型到磁盘
pkl_filename = "./models/logistic_regressor_multi.pkl"
if (not path.isfile(pkl_filename)):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(logr_multi, file)
    print("已将模型保存到磁盘")
else:
    print("模型已保存")


# 创建线性支持向量机模型对象，指定核函数及自动确定gamma参数
lsvm_multi = SVC(kernel='linear', gamma='auto')
# 使用训练集数据训练线性支持向量机模型
lsvm_multi.fit(X_train, y_train)

# 用训练好的模型对测试集进行预测，得到预测的类别结果
y_pred = lsvm_multi.predict(X_test)

# 输出多个评估指标，评估模型在多分类任务中的性能
print("平均绝对误差 - ", metrics.mean_absolute_error(y_test, y_pred))
print("均方误差 - ", metrics.mean_squared_error(y_test, y_pred))
print("均方根误差 - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 得分 - ", metrics.explained_variance_score(y_test, y_pred) * 100)
print("准确率 - ", accuracy_score(y_test, y_pred) * 100)
print(classification_report(y_test, y_pred, target_names=le2.classes_))

# 整理真实值和预测值数据到DataFrame并保存为CSV文件
lsvm_multi_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
lsvm_multi_df.to_csv('./predictions/lsvm_real_pred_multi.csv')
lsvm_multi_df

# 绘制真实值和预测值对比图（取部分数据展示）
plt.figure(figsize=(20, 8))
plt.plot(y_pred[:200], label="预测值", linewidth=2.0, color='blue')
plt.plot(y_test[:200].values, label="真实值", linewidth=2.0, color='lightcoral')
plt.legend(loc="best")
plt.title("线性支持向量机多分类任务中真实值与预测值对比")
plt.savefig('plots/lsvm_real_pred_multi.png')
plt.show()

# 判断模型文件是否存在，不存在则保存训练好的模型到磁盘
pkl_filename = "./models/lsvm_multi.pkl"
if (not path.isfile(pkl_filename)):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(lsvm_multi, file)
    print("已将模型保存到磁盘")
else:
    print("模型已保存")


# 创建K近邻分类器模型对象，指定近邻数量为5
knn_multi = KNeighborsClassifier(n_neighbors=5)
# 使用训练集数据训练K近邻分类器模型
knn_multi.fit(X_train, y_train)

# 用训练好的模型对测试集进行预测，得到预测的类别结果
y_pred = knn_multi.predict(X_test)

# 输出多个评估指标，衡量模型在多分类任务上的性能表现
print("平均绝对误差 - ", metrics.mean_absolute_error(y_test, y_pred))
print("均方误差 - ", metrics.mean_squared_error(y_test, y_pred))
print("均方根误差 - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 得分 - ", metrics.explained_variance_score(y_test, y_pred) * 100)
print("准确率 - ", accuracy_score(y_test, y_pred) * 100)
print(classification_report(y_test, y_pred, target_names=le2.classes_))

# 整理真实值和预测值数据到DataFrame并保存为CSV文件
knn_multi_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
knn_multi_df.to_csv('./predictions/knn_real_pred_multi.csv')
knn_multi_df

# 绘制真实值和预测值对比图（取部分数据展示）
plt.figure(figsize=(20, 8))
plt.plot(y_pred[400:500], label="预测值", linewidth=2.0, color='blue')
plt.plot(y_test[400:500].values, label="真实值", linewidth=2.0, color='lightcoral')
plt.legend(loc="best")
plt.title("K近邻多分类任务中真实值与预测值对比")
plt.savefig('plots/knn_real_pred_multi.png')
plt.show()

# 判断模型文件是否存在，不存在则保存训练好的模型到磁盘
pkl_filename = "./models/knn_multi.pkl"
if (not path.isfile(pkl_filename)):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(knn_multi, file)
    print("已将模型保存到磁盘")
else:
    print("模型已保存")






# 创建随机森林分类器对象，设置随机种子
rf_multi = RandomForestClassifier(random_state=50)
# 使用训练集数据训练随机森林分类器模型
rf_multi.fit(X_train, y_train)

# 用训练好的模型对测试集进行预测，得到预测的类别结果
y_pred = rf_multi.predict(X_test)

# 输出多个评估指标，查看模型在多分类任务上的性能表现
print("平均绝对误差 - ", metrics.mean_absolute_error(y_test, y_pred))
print("均方误差 - ", metrics.mean_squared_error(y_test, y_pred))
print("均方根误差 - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 得分 - ", metrics.explained_variance_score(y_test, y_pred) * 100)
print("准确率 - ", accuracy_score(y_test, y_pred) * 100)
print(classification_report(y_test, y_pred, target_names=le2.classes_))

# 整理真实值和预测值数据到DataFrame并保存为CSV文件
rf_multi_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
rf_multi_df.to_csv('./predictions/rf_real_pred_multi.csv')
rf_multi_df

# 绘制真实值和预测值对比图（取部分数据展示）
plt.figure(figsize=(20, 8))
plt.plot(y_pred[500:600], label="预测值", linewidth=2.0, color='blue')
plt.plot(y_test[500:600].values, label="真实值", linewidth=2.0, color='lightcoral')
plt.legend(loc="best")
plt.title("随机森林多分类任务中真实值与预测值对比")
plt.savefig('plots/rf_real_pred_multi.png')
plt.show()

# 判断模型文件是否存在，不存在则保存训练好的模型到磁盘
pkl_filename = "./models/random_forest_multi.pkl"
if (not path.isfile(pkl_filename)):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(rf_multi, file)
    print("已将模型保存到磁盘")
else:
    print("模型已保存")






# 创建决策树分类器对象，设置随机种子
dt_multi = DecisionTreeClassifier(random_state=123)
# 使用训练集数据训练决策树分类器模型
dt_multi.fit(X_train, y_train)

# 用训练好的模型对测试集进行预测，得到预测的类别结果
y_pred = dt_multi.predict(X_test)

# 输出多个评估指标，评估模型在多分类任务中的性能
print("平均绝对误差 - ", metrics.mean_absolute_error(y_test, y_pred))
print("均方误差 - ", metrics.mean_squared_error(y_test, y_pred))
print("均方根误差 - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 得分 - ", metrics.explained_variance_score(y_test, y_pred) * 100)
print("准确率 - ", accuracy_score(y_test, y_pred) * 100)
print(classification_report(y_test, y_pred, target_names=le2.classes_))

# 整理真实值和预测值数据到DataFrame并保存为CSV文件
dt_multi_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
dt_multi_df.to_csv('./predictions/dt_real_pred_multi.csv')
dt_multi_df

# 绘制真实值和预测值对比图（取部分数据展示）
plt.figure(figsize=(20, 8))
plt.plot(y_pred[400:700], label="预测值", linewidth=2.0, color='blue')
plt.plot(y_test[400:700].values, label="真实值", linewidth=2.0, color='lightcoral')
plt.legend(loc="best")
plt.title("决策树多分类任务中真实值与预测值对比")
plt.savefig('plots/dt_real_pred_multi.png')
plt.show()

# 判断模型文件是否存在，不存在则保存训练好的模型到磁盘
pkl_filename = "./models/decision_tree_multi.pkl"
if (not path.isfile(pkl_filename)):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(dt_multi, file)
    print("已将模型保存到磁盘")
else:
    print("模型已保存")






# 创建多层感知机分类器对象，设置随机种子、优化器及最大迭代次数
mlp_multi = MLPClassifier(random_state=123, solver='adam', max_iter=8000)
# 使用训练集数据训练多层感知机分类器模型
mlp_multi.fit(X_train, y_train)

# 用训练好的模型对测试集进行预测，得到预测的类别结果
y_pred = mlp_multi.predict(X_test)

# 输出多个评估指标，衡量模型在多分类任务上的性能表现
print("平均绝对误差 - ", metrics.mean_absolute_error(y_test, y_pred))
print("均方误差 - ", metrics.mean_squared_error(y_test, y_pred))
print("均方根误差 - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 得分 - ", metrics.explained_variance_score(y_test, y_pred) * 100)
print("准确率 - ", accuracy_score(y_test, y_pred) * 100)
print(classification_report(y_test, y_pred, target_names=le2.classes_))

# 整理真实值和预测值数据到DataFrame并保存为CSV文件
mlp_multi_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
mlp_multi_df.to_csv('./predictions/mlp_real_pred_multi.csv')
mlp_multi_df

# 绘制真实值和预测值对比图（取部分数据展示）
plt.figure(figsize=(20, 8))
plt.plot(y_pred[100:300], label="预测值", linewidth=2.0, color='blue')
plt.plot(y_test[100:300].values, label="真实值", linewidth=2.0, color='lightcoral')
plt.legend(loc="best")
plt.title("多层感知机多分类任务中真实值与预测值对比")
plt.savefig('plots/mlp_real_pred_multi.png')
plt.show()

# 判断模型文件是否存在，不存在则保存训练好的模型到磁盘
pkl_filename = "./models/mlp_multi.pkl"
if (not path.isfile(pkl_filename)):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(mlp_multi, file)
    print("已将模型保存到磁盘")
else:
    print("模型已保存")

