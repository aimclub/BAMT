
import itertools
from sklearn.metrics import f1_score



class Comparison():

    def child_dict(self, net: list):
        res_dict = dict()
        for e0, e1 in net:
            if e1 in res_dict:
                res_dict[e1].append(e0)
            else:
                res_dict[e1] = [e0]
        return res_dict

    def precision_recall(self, pred_net: list, true_net: list, decimal = 2):
        pred_dict = self.child_dict(pred_net)
        true_dict = self.child_dict(true_net)
        corr_undir = 0
        corr_dir = 0
        for e0, e1 in pred_net:
            flag = True
            if e1 in true_dict:
                if e0 in true_dict[e1]:
                    corr_undir += 1
                    corr_dir += 1
                    flag = False
            if (e0 in true_dict) and flag:
                if e1 in true_dict[e0]:
                    corr_undir += 1
        pred_len = len(pred_net)
        true_len = len(true_net)
        shd = pred_len + true_len - corr_undir - corr_dir
        return {
            # 'AP': round(corr_undir/pred_len, decimal),
            #     'AR': round(corr_undir/true_len, decimal),
            #     'F1_undir':round(2*(corr_undir/pred_len)*(corr_undir/true_len)/(corr_undir/pred_len+corr_undir/true_len), decimal),
            #     'AHP': round(corr_dir/pred_len, decimal),
            #     'AHR': round(corr_dir/true_len, decimal),
            #    'F1_dir': round(2*(corr_dir/pred_len)*(corr_dir/true_len)/(corr_dir/pred_len+corr_dir/true_len), decimal),
                'SHD': shd}


    def func(self, strt, vector, full_set_edges):
        for i in range(len(full_set_edges)):
            if full_set_edges[i] in strt:
                vector[i] = 1
        return vector

    def F1(self, ga, true):
        flatten_edges = list(itertools.chain(*true))
        nodes = list(set(flatten_edges))
        full_set_edges = list(itertools.permutations(nodes,2))
        len_edges = len(full_set_edges)
        true_vector = [0]*len_edges
        ga_vector = [0]*len_edges  
        true_vector = self.func(true, true_vector, full_set_edges)
        ga_vector = self.func(ga, ga_vector, full_set_edges)
        return f1_score(true_vector, ga_vector)