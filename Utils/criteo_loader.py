import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def DataPreprocess():
    # import data
    train_df = pd.read_csv('../Data/criteo/origin_data/train.csv')
    test_df = pd.read_csv('../Data/criteo/origin_data/test.csv')

    print(train_df.shape, test_df.shape)
    # 先将label这一列保存起来，再从train中删除
    label = train_df['Label']
    del train_df['Label']

    # 进行数据合并，为了同时对train和test数据进行预处理
    data_df = pd.concat((train_df, test_df))

    del data_df['Id']

    print(data_df.columns)

    # 特征分开类别
    sparse_feas = [col for col in data_df.columns if col[0] == 'C']
    dense_feas = [col for col in data_df.columns if col[0] == 'I']

    # 填充缺失值
    data_df[sparse_feas] = data_df[sparse_feas].fillna('-1')
    data_df[dense_feas] = data_df[dense_feas].fillna(0)

    # 进行编码  类别特征编码
    for feat in sparse_feas:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat])

    # 数值特征归一化
    mms = MinMaxScaler()
    data_df[dense_feas] = mms.fit_transform(data_df[dense_feas])

    # 分开测试集和训练集
    train = data_df[:train_df.shape[0]]
    test = data_df[train_df.shape[0]:]

    train['Label'] = label

    train_set, val_set = train_test_split(train, test_size = 0.2, random_state=2020)

    print(train_set['Label'].value_counts())
    print(val_set['Label'].value_counts())

    # 保存文件
    train_set.reset_index(drop=True, inplace=True)
    val_set.reset_index(drop=True, inplace=True)

    train_set.to_csv('../Data/criteo/processed_data/train_set.csv', index=0)
    val_set.to_csv('../Data/criteo/processed_data/val_set.csv', index=0)
    test.to_csv('../Data/criteo/processed_data/test_set.csv', index=0)

def getTrainData(filename, feafile):
    df = pd.read_csv(filename)#[1279 rows x 40 columns]
    print(df.columns)

    """Index(['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
       'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
       'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19',
       'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'Label'],
      dtype='object')
      """
    # C开头的列代表稀疏特征，I开头的列代表的是稠密特征 13列是稠密矩阵
    dense_features_col = [col for col in df.columns if col[0] == 'I']

    # 这个文件里面存储了稀疏特征的最大范围，用于设置Embedding的输入维度
    fea_col = np.load(feafile, allow_pickle=True)
    sparse_features_col = []#存储系数特征一共有多少个值
    for f in fea_col[1]:
        sparse_features_col.append(f['feat_num'])

    data, labels = df.drop(columns='Label').values, df['Label'].values

    return data, labels, dense_features_col, sparse_features_col

def getTestData(filename):
    df = pd.read_csv(filename)
    return df.to_numpy()
