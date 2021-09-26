import math
import abc


class MultinomialDeviance:

    def init_classes(self, classes):
        self.classes = classes

    @abc.abstractmethod
    def initialize_f_0(self, data, class_name):
        label_name = 'label_' + class_name
        f_name = 'f_' + class_name + '_0'
        class_counts = data[label_name].sum()
        f_0 = class_counts / len(data)
        data[f_name] = f_0
        return f_0

    def calculate_residual(self, data, iter):
        # calculate negative gradient
        data['sum_exp'] = data.apply(lambda x:
                                     sum([math.exp(x['f_' + i + '_' + str(iter - 1)]) for i in self.classes]),
                                     axis=1)
        for class_name in self.classes:
            label_name = 'label_' + class_name
            res_name = 'res_' + class_name + '_' + str(iter)
            f_prev_name = 'f_' + class_name + '_' + str(iter - 1)
            data[res_name] = data[label_name] - math.e ** data[f_prev_name] / data['sum_exp']

    def update_f_m(self, data, trees, iter, class_name, learning_rate, logger):
        f_prev_name = 'f_' + class_name + '_' + str(iter - 1)
        f_m_name = 'f_' + class_name + '_' + str(iter)
        data[f_m_name] = data[f_prev_name]
        for leaf_node in trees[iter][class_name].leaf_nodes:
            data.loc[leaf_node.data_index, f_m_name] += learning_rate * leaf_node.predict_value
        # 打印每棵树的 train loss
        self.get_train_loss(data['label'], data[f_m_name], iter, logger)

    def update_leaf_values(self, targets, y):
        numerator = targets.sum()
        if numerator == 0:
            return 0.0
        numerator *= (self.classes.size - 1) / self.classes.size
        denominator = ((y - targets) * (1 - y + targets)).sum()
        if abs(denominator) < 1e-150:
            return 0.0
        else:
            return numerator / denominator

    def get_train_loss(self, y, f, iter, logger):
        loss = -2.0 * ((y * f) - f.apply(lambda x: math.exp(1 + x))).mean()
        logger.info(('第%d棵树: log-likelihood:%.4f' % (iter, loss)))
