import math
import abc


class LossFunction(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def initialize_f_0(self, data):
        """初始化 F_0 """

    @abc.abstractmethod
    def calculate_residual(self, data, iter):
        """计算负梯度"""

    @abc.abstractmethod
    def update_f_m(self, data, trees, iter, learning_rate, logger):
        """计算 F_m """

    @abc.abstractmethod
    def update_leaf_values(self, targets, y):
        """更新叶子节点的预测值"""

    @abc.abstractmethod
    def get_train_loss(self, y, f, iter, logger):
        """计算训练损失"""


class BinomialDeviance(LossFunction):

    def initialize_f_0(self, data):
        pos = data['label'].sum()
        neg = data.shape[0] - pos
        # 此处log是以e为底，也就是ln
        f_0 = math.log(pos / neg)
        data['f_0'] = f_0
        return f_0

    def calculate_residual(self, data, iter):
        # calculate negative gradient
        res_name = 'res_' + str(iter)
        f_prev_name = 'f_' + str(iter - 1)
        data[res_name] = data['label'] - 1 / (1 + data[f_prev_name].apply(lambda x: math.exp(-x)))

    def update_f_m(self, data, trees, iter, learning_rate, logger):
        f_prev_name = 'f_' + str(iter - 1)
        f_m_name = 'f_' + str(iter)
        data[f_m_name] = data[f_prev_name]
        for leaf_node in trees[iter].leaf_nodes:
            data.loc[leaf_node.data_index, f_m_name] += learning_rate * leaf_node.predict_value
        # 打印每棵树的 train loss
        self.get_train_loss(data['label'], data[f_m_name], iter, logger)

    def update_leaf_values(self, targets, y):
        numerator = targets.sum()
        if numerator == 0:
            return 0.0
        denominator = ((y - targets) * (1 - y + targets)).sum()
        if abs(denominator) < 1e-150:
            return 0.0
        else:
            return numerator / denominator

    def get_train_loss(self, y, f, iter, logger):
        loss = -2.0 * ((y * f) - f.apply(lambda x: math.exp(1 + x))).mean()
        logger.info(('第%d棵树: log-likelihood:%.4f' % (iter, loss)))
