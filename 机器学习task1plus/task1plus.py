import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def make_data(data_path):
    train = pd.read_csv(data_path)
    
    # 删除没用的列
    train.drop(columns=['sample_id'], inplace=True)
    # 采用每列的均值，填充到每列的空值中
    train = train.fillna(train.mean())
    # 区分特征和标签r
    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
    
    data_dir = os.path.dirname(data_path)
    test_data_path = os.path.join(data_dir, 'test_data.csv')
    
    test = pd.read_csv(test_data_path)
    test.drop(columns=['sample_id'], inplace=True)
    test = test.fillna(test.mean())
    
    test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1]
    return train_x, train_y, test_x, test_y


def classifier_fit(train_x, train_y, test_x, test_y):
    model = RandomForestClassifier(n_estimators=100)        #更决策树数量尝试：150-出现0.3的低质量，200-下降，不如100
    model.fit(train_x, train_y)

    # 在测试集上进行预测
    predictions = model.predict(test_x)
    result = classification_report(test_y, predictions)
    print(result)

if __name__ == '__main__':
    code_dir = os.path.dirname(__file__)
    data_dir = os.path.join(code_dir, 'datas')
    train_data_path = os.path.join(data_dir, 'train_data.csv')
    
    # 构造训练和测试数据
    train_x, train_y, test_x, test_y = make_data(train_data_path)
    classifier_fit(train_x, train_y, test_x, test_y)

#macro avg 宏平均（算数平均）   weighted avg权重平均