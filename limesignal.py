import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn import linear_model

class LimeSingal(object):
    def __init__(self, x, ground_true, model, device, x_partition = None, cumulative_criteria = 2, drop_ratio = 0.2, iteration = 100, lasso_alpha=0.000001):
        self.x = x
        self.ground_true = ground_true
        self.model = model
        self.device = device
        assert self._get_predict() == ground_true, "This is a bad sample which is misclassified."
        if x_partition:
            self.partition = np.array(x_partition)
        else:
            self.cumulative_criteria = cumulative_criteria
            self.partition = self.cumulative_partition()
        self.number_of_partition = len(np.unique(self.partition))
        self.drop_size = int(round(self.number_of_partition * drop_ratio,0))
        self.iteration = iteration
        self.lasso_alpha = lasso_alpha
        
        self.exp()
        self.lasso()
        self.assign_weight_to_partition()
        
        
    def _get_predict(self):
        prob = self.model(torch.Tensor([self.x]).to(self.device))
        prob = self.softmax(prob.cpu().numpy().flatten())
        return np.argmax(prob)
        
    def softmax(self, x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        soft_max_x = exp_x / np.sum(exp_x)
        return soft_max_x
    
    def normalize_coef(self, coef):
        return (coef - coef.mean())/coef.std()
    
    def get_prob(self, x):
        prob = self.model(torch.Tensor([x]).to(self.device))
        prob = self.softmax(prob.cpu().numpy().flatten())
        return prob[self.ground_true]
    
    def cumulative_partition(self):
        """
        data should be D*N 
        """
        data = np.array(self.x)
        dim_singal_part = []
        part = 0
        for one_dim in data:
            one_dim_single_part = []
            cum = 0
            for signal in one_dim:
                signal = np.abs(signal)
                if cum < self.cumulative_criteria:
                    cum += signal
                    one_dim_single_part.append(part)
                else:
                    cum = signal
                    part += 1
                    one_dim_single_part.append(part)
            part += 1
            dim_singal_part.append(one_dim_single_part)
        return np.array(dim_singal_part)
    
    def one_exp(self):
        random_select = np.random.randint(self.number_of_partition, size=self.drop_size)
        
        not_drop_out_label = np.ones(self.number_of_partition, dtype=int)
        not_drop_out_label[random_select] = 0
        x_copy = self.x.copy()
        for i in random_select:
            x_copy[self.partition == i] = 0
        drop_out_prob = self.get_prob(x_copy)
        return not_drop_out_label, drop_out_prob
    
    def exp(self):
        lasso_x = []
        lasso_y = []
        for i in range(self.iteration):
            print(f"Process: {i+1} / {self.iteration}", end="\r")
            not_drop_out_label, drop_out_prob = self.one_exp()
            lasso_x.append(not_drop_out_label)
            lasso_y.append(drop_out_prob)
        self.lasso_x = np.array(lasso_x)
        self.lasso_y = np.array(lasso_y)
    
    def lasso(self):
        clf = linear_model.Lasso(alpha=self.lasso_alpha)
        clf.fit(self.lasso_x, self.lasso_y)
        self.weight = self.normalize_coef(clf.coef_)
    
    def assign_weight_to_partition(self):
        self.weight_partition = self.partition.copy().astype(float)
        for i in range(self.number_of_partition):
            self.weight_partition[self.partition == i] = self.weight[i]
    
    def output_weight(self):
        return self.weight_partition
    
    def show_line_plot(self, channel_name_list, fig_x_size=20, fig_y_size = 3):
        for i in range(len(self.x)):
            plt.figure(figsize=(fig_x_size,fig_y_size))
            plt.plot(self.x[i,:], label = channel_name_list[i])
            plt.plot(self.weight_partition[i,:], label="Importance")
            plt.title("Signal importance")
            plt.legend()
            plt.show()
    def show_scatter_plot(self, channel_name_list, method = "seaborn", fig_x_size=20, fig_y_size = 3):
        if method == "seaborn":
            for i in range(len(self.x)):
                plt.figure(figsize=(fig_x_size,fig_y_size))
                sns.scatterplot(np.arange(len(self.x[i,:])),self.x[i,:], hue=self.weight_partition[i,:], s=15)
                plt.title(f"Importance in {channel_name_list[i]}")
                plt.show()
        elif method == "matplotlib":
            for i in range(len(self.x)):
                plt.figure(figsize=(fig_x_size,fig_y_size))
                plt.scatter(np.arange(len(self.x[i,:])),self.x[i,:], alpha = 1, c = self.weight_partition[i,:], s=3,cmap = 'Purples')
                plt.title(f"Importance in {channel_name_list[i]}")
                plt.colorbar()
                plt.show()
