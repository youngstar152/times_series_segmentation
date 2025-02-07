import pandas as pd
import sys
import os
import time
import torch
sys.path.append('./')
from TSpy.TSpy.eval import *
from TSpy.TSpy.label import *
from TSpy.TSpy.utils import *
from TSpy.TSpy.view import *
from TSpy.TSpy.dataset import *
import numpy as np

#from Time2State.time2stateUrl import Time2State
from Time2State.time2state import Time2State
from ts2vec_kato.ts2vec import TS2Vec
from ts2vec.TimesURL.src import timesurl
from ts2vec.TimesURL.src.models.encoder import TSEncoder
from Time2State.adapers import *
from Time2State.clustering import *
from Time2State.default_params import *
from ts2vec_kato.ts2vec import *
import matplotlib.colors as mcolors
from TSpy.TSpy.label import reorder_label
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


device = torch.device("cuda" if cuda_available else "cpu")
import torch

# モデルの中間処理後にメモリを解放
torch.cuda.empty_cache()

# 確実に未使用メモリを解放するには、以下のコードを適宜追加
#del some_tensor  # 不要なテンソルを削除

script_path = os.path.dirname(__file__)
data_path = "/opt/home/tsubasa.kato/E2Usd/data/"
output_path = os.path.join(script_path, '../results/output_Time2State')

class TS2Vec_Adaper(BasicEncoderClass):
    def _set_parmas(self, param):
        input_dim = param['in_channels']
        output_dim = param['out_channels']
        self.win_size =param["win_size"]
        self.encoder = TS2Vec(input_dim, output_dims=output_dim)

    def fit(self, X,win_size,step):
        data = X
        print(data.shape)

        #学習時は分割しない
        # length, dim = data.shape
        # train_seg_len = self.win_size

        # train_seg_num = length/train_seg_len if length%train_seg_len==0 else length//train_seg_len+1

        # pad_length = train_seg_len-(length%train_seg_len)
        # data = np.pad(data,((0,pad_length),(0,0)),'constant')

        # train_seg_list = []

        # train_seg_num = int(train_seg_num)
        # for i in range(train_seg_num):
        #     train_seg = data[i*train_seg_len:(i+1)*train_seg_len]
        #     train_seg_list.append(train_seg)

        # data = np.array(train_seg_list)
        data=data[np.newaxis,:,:]

        self.encoder.fit(data, n_iters=50,verbose=True)

    def encode(self, X, win_size, step):
        length = X.shape[0]
        # print("X.shape")
        # print(X.shape)
        #print("length",length)
        num_window = int((length-win_size)/step)+1
        #print("num_window",num_window)

        windowed_data = []
        i=0
        for k in range(num_window):
            windowed_data.append(X[i:i+win_size])
            i+=step

        windowed_data = np.stack(windowed_data)
        print(windowed_data.shape)
        out = self.encoder.encode(windowed_data, encoding_window='full_series')
        #print(out.shape)
        out = np.vstack(out)[:length]
        return out
    
        # windowed_data = X
        # windowed_data=windowed_data[np.newaxis, :, :]
        # print(windowed_data.shape)
        # #print(windowed_data.shape)
        # out = self.encoder.encode(windowed_data, encoding_window='full_series')
        # #print(out.shape)
        # return out

    def encode_normal(self, X, win_size, step):
        windowed_data = X
        windowed_data=windowed_data[np.newaxis, :, :]
        #print(windowed_data.shape)
        out = self.encoder.encode(windowed_data, encoding_window='full_series')
        #print(out.shape)
        return out
    
    def print_pa(self):
        print()
        
params_TS2Vec = {
    'input_dim' : 10,
    'output_dim' : 4,
    'win_size' : 256,
}
import argparse
class TimesUrl_Adaper(BasicEncoderClass):
    def _set_parmas(self, params):
        input_dim = params['input_dim']
        output_dim = params['output_dim']
        self.win_size = params['win_size']
        self.encoder = timesurl.TimesURL(input_dim, output_dims=output_dim,max_train_length=50000)
        #self.encoder = TimesURL(input_dim, output_dims=output_dim)

    def print_pa(self):
        print("start")
        # for key, value in self.hyperparameters.items():
        #     print(f"{key}: {value}")

    def fit(self, X,win_size,step):
        data = X
        #print(data.shape)
        # 次元を追加して (1, 200, 4) に変換
        #data = np.expand_dims(data, axis=0)
        

        # length,dim = data.shape
        # train_seg_len = self.win_size

        # train_seg_num = length/train_seg_len if length%train_seg_len==0 else length//train_seg_len+1

        # pad_length = train_seg_len-(length%train_seg_len)
        # data = np.pad(data,((0,pad_length),(0,0)),'constant')

        # train_seg_list = []

        # train_seg_num = int(train_seg_num)
        # for i in range(train_seg_num):
        #     train_seg = data[i*train_seg_len:(i+1)*train_seg_len]
        #     train_seg_list.append(train_seg)

        # data = np.array(train_seg_list)

        # _, dim = X["x"].shape
        # X = np.transpose(np.array(X["x"][:,:], dtype=float)).reshape(1, dim, -1)
        # X = all_normalize(X)
        # self.encoder.fit(X,win_size,step, n_iters=5,save_memory=True, verbose=True)
        self.encoder.fit(data, win_size=75,n_iters=100,verbose=True)

    def encode(self, X, win_size, step):
        ts2vec=False
        if ts2vec:
            length = X.shape[0]
            print(X.shape)
            #print("length",length)
            num_window = int((length-win_size)/step)+1
            #print("num_window",num_window)

            windowed_data = []
            i=0
            for k in range(num_window):
                windowed_data.append(X[i:i+win_size])
                i+=step

            windowed_data = np.stack(windowed_data)
            print(windowed_data.shape)
            out = self.encoder.encode(windowed_data, encoding_window='full_series')
            print(out.shape)
            out = np.vstack(out)[:length]
        else:
            #print(X["x"].shape)
            X["x"]=np.squeeze(X["x"], axis=0)
            length = X["x"].shape[0]
            #print(X["x"].shape)
            #print("length",length)
            num_window = int((length-win_size)/step)+1
            #print("num_window",num_window)

            windowed_data = []
            i=0
            for k in range(num_window):
                windowed_data.append(X["x"][i:i+win_size])
                i+=step

            windowed_data = np.stack(windowed_data)
            
            X["mask"]=np.squeeze(X["mask"], axis=0)
            lengthm = X["mask"].shape[0]
            #print(X["mask"].shape)
            #print("length",length)
            num_windowm = int((lengthm-win_size)/step)+1
            #print("num_window",num_window)

            windowed_datam = []
            i=0
            for k in range(num_windowm):
                windowed_datam.append(X["mask"][i:i+win_size])
                i+=step

            windowed_datam = np.stack(windowed_datam)

            encode_data={"x":windowed_data,"mask":windowed_datam}
            #print(windowed_data.shape)
            out = self.encoder.encode(encode_data, encoding_window='full_series')
            out = np.vstack(out)[:length]
            return out
            #return embeddings

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_color(data):
    # regime_listを生成するためのコード
    regime_list = [0]
    for i in range(1, len(data)):
        if data[i] != data[i-1]:
            regime_list.append(i)
    regime_list.append(len(data))

    # colorsを生成するためのコード
    num = [int(data[i]) for i in regime_list[:-1]]
    print(num)
    colors = []
    for value in num:
        if value == 0.0:
            colors.append("r")
        elif value == 1.0:
            colors.append("pink")
        elif value == 2.0:
            colors.append("g")
        elif value == 3.0:
            colors.append("b")
        elif value == 4.0:
            colors.append("c")
        elif value == 5.0:
            colors.append("m")
        elif value == 6.0:
            colors.append("y")
        else:
            colors.append("k")
    return regime_list, colors

dataset_info = {'amc_86_01.4d':{'n_segs':4, 'label':{588:0,1200:1,2006:0,2530:2,3282:0,4048:3,4579:2}},
        'amc_86_02.4d':{'n_segs':8, 'label':{1009:0,1882:1,2677:2,3158:3,4688:4,5963:0,7327:5,8887:6,9632:7,10617:0}},
        'amc_86_03.4d':{'n_segs':7, 'label':{872:0, 1938:1, 2448:2, 3470:0, 4632:3, 5372:4, 6182:5, 7089:6, 8401:0}},
        'amc_86_07.4d':{'n_segs':6, 'label':{1060:0,1897:1,2564:2,3665:1,4405:2,5169:3,5804:4,6962:0,7806:5,8702:0}},
        'amc_86_08.4d':{'n_segs':9, 'label':{1062:0,1904:1,2661:2,3282:3,3963:4,4754:5,5673:6,6362:4,7144:7,8139:8,9206:0}},
        'amc_86_09.4d':{'n_segs':5, 'label':{921:0,1275:1,2139:2,2887:3,3667:4,4794:0}},
        'amc_86_10.4d':{'n_segs':4, 'label':{2003:0,3720:1,4981:0,5646:2,6641:3,7583:0}},
        'amc_86_11.4d':{'n_segs':4, 'label':{1231:0,1693:1,2332:2,2762:1,3386:3,4015:2,4665:1,5674:0}},
        'amc_86_14.4d':{'n_segs':3, 'label':{671:0,1913:1,2931:0,4134:2,5051:0,5628:1,6055:2}},
}

def normalize_with_mask_kato(train, mask_tr,  scaler):
    train[mask_tr == 0] = np.nan
    scaler = scaler.fit(train.reshape(-1, train.shape[-1]))
    train = scaler.transform(train.reshape(-1, train.shape[-1])).reshape(train.shape)
    #test = scaler.transform(test.reshape(-1, test.shape[-1])).reshape(test.shape)
    train[mask_tr == 0]= 0 
    return train

import random
# def generate_mask_kato(data, p = 0.5, remain = 0):
    
#     B, T, C = data.shape
#     mask = np.empty_like(data)# 初期化を 1 (マスクしない状態) にする

#     for b in range(B):
#         ts = data[b, :, 0]
#         et_num = ts[~np.isnan(ts)].size - remain
#         total, num = et_num * C, round(et_num * C * p)

#         while True:
#             i_mask = np.zeros(total)
#             i_mask[random.sample(range(total), num)] = 1
#             i_mask = i_mask.reshape(et_num, C)
#             if 1 not in i_mask.sum(axis = 0) and 0 not in i_mask.sum(axis = 0):
#                 break
#             break

#         i_mask = np.concatenate((i_mask, np.ones((remain, C))), axis = 0)
#         mask[b, ~np.isnan(ts), :] = i_mask
#         mask[b, np.isnan(ts), :] = np.nan

#     # mask = np.concatenate([random.sample(range(total), num) for _ in range(B)])
#     # matrix = np.zeros((B, total))
#     # matrix[(np.arange(B).repeat(num), mask)] = 1.0
#     # matrix = matrix.reshape(B, T, C)
#     # return matrix
#     return mask

def generate_mask_kato(data, p=0.1, remain=0):
    B, T, C = data.shape
    mask = np.ones_like(data)  # 初期化を 1 (マスクしない状態) にする

    for b in range(B):
        ts = data[b, :, 0]
        et_num = ts[~np.isnan(ts)].size - remain
        total, num = et_num * C, round(et_num * C * p)

        if num > 0:
            i_mask = np.zeros(total)
            i_mask[random.sample(range(total), num)] = 1
            i_mask = i_mask.reshape(et_num, C)
            i_mask = np.concatenate((i_mask, np.ones((remain, C))), axis=0)
            mask[b, ~np.isnan(ts), :] = i_mask
        mask[b, np.isnan(ts), :] = np.nan  # NaNの場所はそのまま保持

    return mask

from sklearn.preprocessing import StandardScaler, MinMaxScaler
def load_kato(dataset, load_tp: bool = False):
    
    train_data_df =  pd.read_csv(dataset, sep=' ',usecols=range(0,4))
    train_data_np = train_data_df.to_numpy()
    train_data = train_data_np.reshape(1, train_data_np.shape[0], train_data_np.shape[1])
    print("ここ")
    print(train_data.shape)
       
    #データをマスクする割合
    p = 0.7
    mask_tr = generate_mask_kato(train_data, p)
    print(mask_tr)
    # scaler = MinMaxScaler()
    scaler = StandardScaler()

    train_X = normalize_with_mask_kato(train_data, mask_tr, scaler)
    print(train_X.shape)

    if load_tp:
        tp = np.linspace(0, 1, train_X.shape[1], endpoint=True).reshape(1, -1, 1)
        train_X = np.concatenate((train_X, np.repeat(tp, train_X.shape[0], axis=0)), axis=-1)
        
        print("t_A",train_X[..., -1:])
        #test_X = np.concatenate((test_X, np.repeat(tp, test_X.shape[0], axis=0)), axis=-1)

    # labels = np.unique(train_y)
    # transform = {k: i for i, k in enumerate(labels)}
    # train_y = np.vectorize(transform.get)(train_y)
    # test_y = np.vectorize(transform.get)(test_y)
    return {'x': train_X, 'mask': mask_tr},train_data_df,train_data_np

#from ts2vec.adaper import *
def exp_on_UCR_SEG(win_size, step, verbose=False):
    score_list = []
    score_list_km1 = []
    score_list_km2 = []
    score_list_km3 = []
    score_list_km4 = []
    score_list_gm1 = []
    score_list_gm2 = []
    score_list_gm3 = []
    score_list_gm4 = []
    score_list_dpgm1 = []
    score_list_dpgm2 = []
    score_list_dpgm3 = []
    score_list_dpgm4 = []
    score_list_shift = []
    score_list_dbscan = []
    score_list_hdbscan = []
    out_path = os.path.join(output_path,'UCRSEG3')
    create_path(out_path)
    params_LSE['in_channels'] = 1
    params_LSE['M'] = 10
    params_LSE['N'] = 4
    params_LSE['out_channels'] = 2
    params_LSE['nb_steps'] = 50
    params_LSE['win_size'] = win_size
    params_LSE['kernel_size'] = 3
    params_LSE['cuda'] = True
    #params_LSE['cuda'] = True
    params_LSE['gpu'] = 0
    # params_TS2Vec['input_dim'] = 1
    # params_TS2Vec['output_dim'] = 2
    print("data_path",data_path)
    params_LSE['seed']=42

    dataset_path = os.path.join(data_path, 'UCRSEG/')
    print("dataset_path",dataset_path)
    addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Ucr/0122_UCRSEG_STL"
    create_path(addpath)
    file_path=addpath+"/result.txt"
    for fname in os.listdir(dataset_path):
        info_list = fname[:-4].split('_')
        # f = info_list[0]
        window_size = int(info_list[1])
        seg_info = {}
        i = 0
        for seg in info_list[2:]:
            seg_info[int(seg)]=i
            i+=1
        seg_info[len_of_file(dataset_path+fname)]=i
        num_state=len(seg_info)
        df = pd.read_csv(dataset_path+fname)
        data = df.to_numpy()
        data = normalize(data)
        saveplt=addpath+"/"+fname+str(win_size)+"plt.png"
        groundtruth = seg_to_label(seg_info)[:-1]
        #t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size,step,groundtruth,saveplt)
        
        #t2s = Time2State(win_size, step, TS2Vec_Adaper(params_LSE), DPGMM(None)).fit(data, win_size,step,groundtruth,saveplt)
        t2s = Time2State(win_size, step, TS2Vec_Adaper(params_LSE), DPGMM(None)).fit(data, win_size,step,groundtruth,saveplt)
        #t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step,groundtruth,saveplt)
        #t2s = Time2State(win_size, step, TS2Vec_Adaper(params_LSE), DPGMM(None)).fit(data, win_size,step,groundtruth,saveplt)
        # t2s = Time2State_backup(win_size, step, TS2Vec_Adaper(params_TS2Vec), DPGMM(None, alpha=None)).fit(data, win_size, step)
        
        prediction = t2s.state_seq
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        #np.save(os.path.join(out_path,fname[:-4]), result)
        score_list.append(np.array([ari, anmi, nmi]))

        import matplotlib.cm as cm
        from matplotlib.colors import ListedColormap
        cmap = cm.Set1
        # ラベルの一覧を取得します
        num_clusters = len(set(prediction))  # クラスタの数
        #cmap = cm.get_cmap('tab20', num_clusters) 
        # 10の要素を持つ色リストを定義
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink']
        colors=colors[:num_clusters]
        cmap = mcolors.ListedColormap(colors)
        plt.figure(figsize=(10, 6))
        plt.plot(data, alpha=0.5)  # cパラメータにクラスタラベルを渡す
        data_index=df.index.tolist()
        
        for i in range(len(data)-1):
        #for i in range(1000):
            # print(data[i])
            # print(prediction[i])
            plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
        plt.legend() 
        plt.savefig(addpath+"/"+fname+'.png')
        plt.show()

        plt.figure(figsize=(10, 6))
        regime_list, colors=make_color(groundtruth)
        #print(regime_list)
        plt.plot(data) 
        num_clusters = len(set(regime_list))  # クラスタの数
        cmap = cm.get_cmap('tab20', num_clusters) 
        for i in range(len(regime_list)-1):
            plt.axvspan(regime_list[i], regime_list[i+1], color=colors[i], alpha=0.2)
        plt.legend() 
        plt.savefig(addpath+"/"+fname+'_answer.png')
        plt.show()
        # plot_mulvariate_time_series_and_label_v2(data,groundtruth,prediction)
        # plt.savefig('1.png')
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi))

        def plot_color(num_clusters,name,score_list,data,win_size,step,colors,groundtruth):
            #print(len(num_clusters))
            __embedding_label=reorder_label(num_clusters)
            hight = len(set(__embedding_label))
            __length = data.shape[0]
            #__length = data["x"].shape[0]
            __win_size = win_size
            __step = step
            __offset = int(win_size/2)
            weight_vector = np.ones(shape=(2*__offset)).flatten()
            __state_seq = __embedding_label
            vote_matrix = np.zeros((__length,hight))
            i = 0
            for l in __embedding_label:
                vote_matrix[i:i+__win_size,l]+= weight_vector
                i+=__step
            __state_seq = np.array([np.argmax(row) for row in vote_matrix])
            prediction = np.array(__state_seq, dtype=int)
            min_size = min(len(groundtruth), len(prediction))

            groundtruth = groundtruth[:min_size]
            prediction = prediction[:min_size]
            num_clusters = len(set(prediction)) 

            colors=colors[:num_clusters]
            cmap = mcolors.ListedColormap(colors)
            plt.figure(figsize=(10, 6))
            #ts2vec
            plt.plot(data)  # cパラメータにクラスタラベルを渡す
            data_index=df.index.tolist()
            #timesurl
            # plt.plot(train_data_np)  # cパラメータにクラスタラベルを渡す
            # data_index=train_data_df.index.tolist()
            # addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/0722_max0.3_test"
            # create_path(addpath)
            for i in range(len(data)-1):
                plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
            plt.legend() 
            plt.savefig(addpath+"/"+str(win_size)+str(step)+name+'.png')
            plt.show()

            #np.save(os.path.join(out_path,fname), result)
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            score_list.append(np.array([ari, anmi, nmi]))
            ans='ID:%s, %s, ARI: %f, ANMI: %f, NMI: %f' %(name,fname, ari, anmi, nmi)
            print(ans)
            with open(file_path, 'a') as file:
                file.write(ans+"\n")

        answer_num=len(set(groundtruth))
        prediction_s=t2s.embeddings
        from sklearn.cluster import DBSCAN
        import hdbscan
        
        meanshift1=MeanShift().fit(X=prediction_s).labels_
        dbscan1=DBSCAN().fit(prediction_s).labels_
        hdbscan1=hdbscan.HDBSCAN().fit(prediction_s).labels_

        km_1=KMeans(n_clusters=answer_num+1, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        km_1_data = km_1.fit_predict(prediction_s)# クラスタの数
        #

        km_2=KMeans(n_clusters=answer_num, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        km_2_data = km_2.fit_predict(prediction_s)
        #

        km_3=KMeans(n_clusters=answer_num-1, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        km_3_data = km_3.fit_predict(prediction_s)
        #

        km_4=KMeans(n_clusters=answer_num+2, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        km_4_data = km_4.fit_predict(prediction_s)

        gm_1=GaussianMixture(n_components=answer_num+1,reg_covar=5e-3, init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        gm_1_ = gm_1.fit(prediction_s)
        gm_1_data=gm_1_.predict(prediction_s)

        gm_2=GaussianMixture(n_components=answer_num, reg_covar=5e-3,init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        gm_2_ = gm_2.fit(prediction_s)
        gm_2_data=gm_2_.predict(prediction_s)

        gm_3=GaussianMixture(n_components=answer_num-1,reg_covar=5e-3, init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        gm_3_ = gm_3.fit(prediction_s)
        gm_3_data=gm_3_.predict(prediction_s)

        gm_4=GaussianMixture(n_components=answer_num+2, reg_covar=5e-3, init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        gm_4_ = gm_4.fit(prediction_s)
        gm_4_data=gm_4_.predict(prediction_s)

        dpgmm_1 = mixture.BayesianGaussianMixture(init_params='k-means++',
                                                n_components=answer_num+1,
                                                covariance_type="full",
                                                weight_concentration_prior_type='dirichlet_process',
                                                max_iter=1000).fit(prediction_s)
        dpg_1=dpgmm_1.predict(prediction_s)

        dpgmm_2 = mixture.BayesianGaussianMixture(init_params='k-means++',
                                                n_components=answer_num,
                                                covariance_type="full",
                                                weight_concentration_prior_type='dirichlet_process',
                                                max_iter=1000).fit(prediction_s)
        dpg_2=dpgmm_2.predict(prediction_s) 

        dpgmm_3 = mixture.BayesianGaussianMixture(init_params='k-means++',
                                                n_components=answer_num-1,
                                                covariance_type="full",
                                                weight_concentration_prior_type='dirichlet_process',
                                                max_iter=1000).fit(prediction_s)
        dpg_3=dpgmm_3.predict(prediction_s)

        dpgmm_4 = mixture.BayesianGaussianMixture(init_params='k-means++',
                                                n_components=answer_num+2,
                                                covariance_type="full",
                                                weight_concentration_prior_type='dirichlet_process',
                                                max_iter=1000).fit(prediction_s)
        dpg_4=dpgmm_4.predict(prediction_s)

        plot_color(meanshift1,"Meanshift",score_list_shift,data,win_size,step,colors,groundtruth)
        plot_color(dbscan1,"dbscan",score_list_dbscan,data,win_size,step,colors,groundtruth)
        plot_color(hdbscan1,"hdscan",score_list_hdbscan,data,win_size,step,colors,groundtruth)

        plot_color(dpg_4,"DPGMM+2",score_list_dpgm4,data,win_size,step,colors,groundtruth)
        plot_color(dpg_3,"DPGMM-1",score_list_dpgm3,data,win_size,step,colors,groundtruth)
        plot_color(dpg_2,"DPGMM0",score_list_dpgm2,data,win_size,step,colors,groundtruth)
        plot_color(dpg_1,"DPGMM+1",score_list_dpgm1,data,win_size,step,colors,groundtruth)
        plot_color(gm_4_data,"GMM+2",score_list_gm4,data,win_size,step,colors,groundtruth)
        plot_color(gm_3_data,"GMM-1",score_list_gm3,data,win_size,step,colors,groundtruth)
        plot_color(gm_2_data,"GMM0",score_list_gm2,data,win_size,step,colors,groundtruth)
        plot_color(gm_1_data,"GMM1",score_list_gm1,data,win_size,step,colors,groundtruth)
        plot_color(km_4_data,"kmeans+2",score_list_km4,data,win_size,step,colors,groundtruth)
        plot_color(km_3_data,"kmeans-1",score_list_km3,data,win_size,step,colors,groundtruth)
        plot_color(km_2_data,"kmeans0",score_list_km2,data,win_size,step,colors,groundtruth)
        plot_color(km_1_data,"kmeans1",score_list_km1,data,win_size,step,colors,groundtruth)

        
        
    score_list = np.vstack(score_list)
    ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list_shift = np.vstack(score_list_shift)
    ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list_shift[:,0])\
        ,np.mean(score_list_shift[:,1])
        ,np.mean(score_list_shift[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list_dbscan = np.vstack(score_list_dbscan)
    ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list_dbscan[:,0])\
        ,np.mean(score_list_dbscan[:,1])
        ,np.mean(score_list_dbscan[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list_hdbscan = np.vstack(score_list_hdbscan)
    ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list_hdbscan[:,0])\
        ,np.mean(score_list_hdbscan[:,1])
        ,np.mean(score_list_hdbscan[:,2]))
    print(ans)
    
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_km1)
    ans='KM1 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_km2)
    ans='KM2 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_km3)
    ans='KM3 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_km4)
    ans='KM4 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_gm1)
    ans='GM1 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_gm2)
    ans='GM2 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_gm3)
    ans='GM3 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_gm4)
    ans='GM4 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_dpgm1)
    ans='DPGM1 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_dpgm2)
    ans='DPGM2 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_dpgm3)
    ans='DPGM3 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_dpgm4)
    ans='DPGM4 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")        


    

def exp_on_MoCap(win_size, step,seed=42, verbose=False):
    base_path = os.path.join(data_path,'MoCap/4d/')
    base_path="/opt/home/tsubasa.kato/E2Usd/data/MoCap/4d/"
    out_path = os.path.join(output_path,'MoCap')
    out_path="/opt/home/tsubasa.kato/E2Usd/data/MoCap/"
    create_path(out_path)
    score_list = []
    score_list_shift = []
    score_list_dbscan = []
    score_list_hdbscan = []

    score_list_km1 = []
    score_list_km2 = []
    score_list_km3 = []
    score_list_km4 = []
    score_list_gm1 = []
    score_list_gm2 = []
    score_list_gm3 = []
    score_list_gm4 = []
    score_list_dpgm1 = []
    score_list_dpgm2 = []
    score_list_dpgm3 = []
    score_list_dpgm4 = []

    params_LSE['in_channels'] = 4
    params_LSE['win_size'] = win_size
    params_LSE['M'] = 10
    params_LSE['N'] = 4
    params_LSE['cuda'] = True
    params_LSE['gpu'] = 0
    params_LSE['out_channels'] = 4
    params_LSE['kernel_size'] = 5
    params_LSE['nb_steps']=40
    params_LSE['seed']=seed

    params_TS2Vec['win_size'] = win_size
    params_TS2Vec['input_dim'] = 4
    params_TS2Vec['output_dim'] = 4
    params_TS2Vec['cuda'] = True
    params_TS2Vec['gpu'] = 1
    params_TS2Vec['seed']=42
    # params_ts2['input_dim'] = 4
    # params_ts2['output_dim'] = 4
    # params_ts2['win_size'] = win_size
    print("step",step)
    addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/0121_Mocap_nokyokusyo"
    create_path(addpath)
    # ファイルに追記
    file_path=addpath+"/result.txt"
    with open(file_path, 'a') as file:
        file.write("win_size"+str(win_size)+"\n")
        file.write("step"+str(step)+"\n")
    # time2seg = Time2Seg(win_size, step, CausalConvEncoder(hyperparameters), DPGMM(None))
    f_list = os.listdir(base_path)
    f_list.sort()
    for idx, fname in enumerate(f_list):
        dataset_path = base_path+fname
        #ts2vec
        df = pd.read_csv(dataset_path, sep=' ',usecols=range(0,4))
        data = df.to_numpy()
        #timesurl
        #data,train_data_df,train_data_np=load_kato(dataset_path)
        plt.figure(figsize=(20, 4))

        #ts2vec
        plt.plot(data)  # cパラメータにクラスタラベルを渡す
        data_index=df.index.tolist()
        #timesurl
        # plt.plot(train_data_np)  # cパラメータにクラスタラベルを渡す
        # data_index=train_data_df.index.tolist()

        # addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/0722_max0.3_test"
        # # create_path(addpath)
        # for i in range(len(data)-1):
        # #for i in range(1000):
        #     # print(data[i])
        #     # print(prediction[i])
        #     plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
        plt.legend() 
        plt.savefig(addpath+"/"+fname+str(win_size)+str(step)+'normal.png')
        
        # create_path(addpath)
        saveplt=addpath+"/"+fname+str(win_size)+str(step)+"plt.png"
        n_state=dataset_info[fname]['n_segs']
        groundtruth = seg_to_label(dataset_info[fname]['label'])[:-1]
        # print(data.shape)
        # t2s = Time2State_backup(win_size, step, CausalConv_Triplet_Adaper(params_Triplet), DPGMM(None)).fit(data, win_size, step)
        # t2s = Time2State_backup(win_size, step, CausalConv_CPC_Adaper(params_CPC), DPGMM(None)).fit(data, win_size, step)
       
        ##通常はこれ！！！
        t2s = Time2State(win_size, step, TS2Vec_Adaper(params_LSE), DPGMM(None)).fit(data, win_size,step,groundtruth,saveplt)
        #t2s = Time2State(win_size, step, TimesUrl_Adaper(params_TS2Vec), DPGMM(None)).fit(data, win_size, step,groundtruth,saveplt)
        #t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step,groundtruth,saveplt)
        #t2s = Time2State(win_size, step, TS2Vec_Adaper(win_size), DPGMM(None)).fit(data, win_size,step)
        
        # t2s = Time2State_backup(win_size, step, LSTM_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
        # t2s = Time2State_backup(win_size, step, CausalConv_LSE_Adaper(params_LSE), KMeansClustering(n_state)).fit(data, win_size, step)
        # t2s = Time2State_backup(win_size, step, CausalConv_TNC_Adaper(params_TNC), DPGMM(None)).fit(data, win_size, step)
        prediction = t2s.state_seq
        prediction = np.array(prediction, dtype=int)
        # print(groundtruth.shape)
        # print(prediction.shape)
        min_size = min(len(groundtruth), len(prediction))

        groundtruth = groundtruth[:min_size]
        prediction = prediction[:min_size]
        #print("prediction")
        result = np.vstack([groundtruth, prediction])
        #np.save(os.path.join(out_path,fname), result)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)

        score_list.append(np.array([ari, anmi, nmi]))


        import matplotlib.cm as cm
        from matplotlib.colors import ListedColormap
        cmap = cm.Set1
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink']
        # ラベルの一覧を取得します
        num_clusters = len(set(prediction))  # クラスタの数
        #vmax1=len(np.unique(num_clusters))
        colors=colors[:num_clusters]
        cmap = mcolors.ListedColormap(colors)
        #cmap = cm.get_cmap('tab20', num_clusters) 
        plt.figure(figsize=(10, 6))
        #ts2vec
        plt.plot(data)  # cパラメータにクラスタラベルを渡す
        data_index=df.index.tolist()
        #timesurl
        # plt.plot(train_data_np)  # cパラメータにクラスタラベルを渡す
        # data_index=train_data_df.index.tolist()
        # addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/0722_max0.3_test"
        # create_path(addpath)
        for i in range(len(data)-1):
        #for i in range(1000):
            # print(data[i])
            # print(prediction[i])
            plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
        plt.legend() 
        plt.savefig(addpath+"/"+fname+str(win_size)+str(step)+'DPGMM.png')
        plt.show()
        if verbose:
            ans='ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi)
            print(ans)
            with open(file_path, 'a') as file:
                file.write(ans+"\n")

        def plot_color(num_clusters,name,score_list,data,win_size,step,colors,groundtruth):
            #numm=num_clusters.labels_
            print(len(num_clusters))
            __embedding_label=reorder_label(num_clusters)
            hight = len(set(__embedding_label))
            __length = data.shape[0]
            #__length = data["x"].shape[0]
            __win_size = win_size
            __step = step
            __offset = int(win_size/2)
            # weight_vector = np.concatenate([np.linspace(0,1,self.__offset),np.linspace(1,0,self.__offset)])
            weight_vector = np.ones(shape=(2*__offset)).flatten()
            #print(weight_vector.shape)
            __state_seq = __embedding_label
            vote_matrix = np.zeros((__length,hight))
            #print(vote_matrix.shape)
            i = 0
            for l in __embedding_label:
                vote_matrix[i:i+__win_size,l]+= weight_vector
                i+=__step
            __state_seq = np.array([np.argmax(row) for row in vote_matrix])
            prediction = np.array(__state_seq, dtype=int)
            min_size = min(len(groundtruth), len(prediction))

            groundtruth = groundtruth[:min_size]
            prediction = prediction[:min_size]
            num_clusters = len(set(prediction)) 

            colors=colors[:num_clusters]
            cmap = mcolors.ListedColormap(colors)
            #cmap = cm.get_cmap('tab20', num_clusters) 
            plt.figure(figsize=(10, 6))
            #ts2vec
            plt.plot(data)  # cパラメータにクラスタラベルを渡す
            data_index=df.index.tolist()
            #timesurl
            # plt.plot(train_data_np)  # cパラメータにクラスタラベルを渡す
            # data_index=train_data_df.index.tolist()
            # addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/0722_max0.3_test"
            # create_path(addpath)
            for i in range(len(data)-1):
            #for i in range(1000):
                # print(data[i])
                # print(prediction[i])
                plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
            plt.legend() 
            plt.savefig(addpath+"/"+fname+str(win_size)+str(step)+name+'.png')
            plt.show()

            #np.save(os.path.join(out_path,fname), result)
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            score_list.append(np.array([ari, anmi, nmi]))
            ans='%s ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(name,fname, ari, anmi, nmi)
            print(ans)
            with open(file_path, 'a') as file:
                file.write(ans+"\n")

        def plotfull_color(num_clusters,name,score_list,data,win_size,step,colors,groundtruth):
            
            prediction = np.array(num_clusters, dtype=int)
            num_clusters = len(set(prediction)) 

            colors=colors[:num_clusters]
            cmap = mcolors.ListedColormap(colors)
            #cmap = cm.get_cmap('tab20', num_clusters) 
            plt.figure(figsize=(10, 6))
            #ts2vec
            plt.plot(data)  # cパラメータにクラスタラベルを渡す
            data_index=df.index.tolist()
            #timesurl
            # plt.plot(train_data_np)  # cパラメータにクラスタラベルを渡す
            # data_index=train_data_df.index.tolist()
            # addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/0722_max0.3_test"
            # create_path(addpath)
            for i in range(len(data)-1):
            #for i in range(1000):
                # print(data[i])
                # print(prediction[i])
                plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
            plt.legend() 
            plt.savefig(addpath+"/"+fname+str(win_size)+str(step)+name+'.png')
            plt.show()

            #np.save(os.path.join(out_path,fname), result)
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            min_size = min(len(groundtruth), len(prediction))

            groundtruth = groundtruth[:min_size]
            prediction = prediction[:min_size]
            score_list.append(np.array([ari, anmi, nmi]))
            ans='%s ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(name,fname, ari, anmi, nmi)
            print(ans)
            with open(file_path, 'a') as file:
                file.write(ans+"\n")


        
        answer_num=len(set(groundtruth))
        prediction_s=t2s.embeddings

        km_1=KMeans(n_clusters=answer_num+1, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        km_1_data = km_1.fit_predict(prediction_s)# クラスタの数
        #

        km_2=KMeans(n_clusters=answer_num, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        km_2_data = km_2.fit_predict(prediction_s)
        #

        km_3=KMeans(n_clusters=answer_num-1, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        km_3_data = km_3.fit_predict(prediction_s)
        #

        km_4=KMeans(n_clusters=answer_num+2, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        km_4_data = km_4.fit_predict(prediction_s)
        #

        gm_1=GaussianMixture(n_components=answer_num+1,reg_covar=5e-3, init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        gm_1_ = gm_1.fit(prediction_s)
        gm_1_data=gm_1_.predict(prediction_s)
        #

        gm_2=GaussianMixture(n_components=answer_num, reg_covar=5e-3,init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        gm_2_ = gm_2.fit(prediction_s)
        gm_2_data=gm_2_.predict(prediction_s)
        #

        gm_3=GaussianMixture(n_components=answer_num-1,reg_covar=5e-3, init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        gm_3_ = gm_3.fit(prediction_s)
        gm_3_data=gm_3_.predict(prediction_s)
        #

        gm_4=GaussianMixture(n_components=answer_num+2, reg_covar=5e-3, init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        gm_4_ = gm_4.fit(prediction_s)
        gm_4_data=gm_4_.predict(prediction_s)
        #

        dpgmm_1 = mixture.BayesianGaussianMixture(init_params='k-means++',
                                                n_components=answer_num+1,
                                                covariance_type="full",
                                                weight_concentration_prior_type='dirichlet_process',
                                                max_iter=1000).fit(prediction_s)
        dpg_1=dpgmm_1.predict(prediction_s)
        #

        dpgmm_2 = mixture.BayesianGaussianMixture(init_params='k-means++',
                                                n_components=answer_num,
                                                covariance_type="full",
                                                weight_concentration_prior_type='dirichlet_process',
                                                max_iter=1000).fit(prediction_s)
        dpg_2=dpgmm_2.predict(prediction_s)
        

        dpgmm_3 = mixture.BayesianGaussianMixture(init_params='k-means++',
                                                n_components=answer_num-1,
                                                covariance_type="full",
                                                weight_concentration_prior_type='dirichlet_process',
                                                max_iter=1000).fit(prediction_s)
        dpg_3=dpgmm_3.predict(prediction_s)

        dpgmm_4 = mixture.BayesianGaussianMixture(init_params='k-means++',
                                                n_components=answer_num+2,
                                                covariance_type="full",
                                                weight_concentration_prior_type='dirichlet_process',
                                                max_iter=1000).fit(prediction_s)
        dpg_4=dpgmm_4.predict(prediction_s)

        slide_w=True
        #slide_w=False
        if slide_w:
            plot_color(dpg_4,"DPGMM+2",score_list_dpgm4,data,win_size,step,colors,groundtruth)
            plot_color(dpg_3,"DPGMM-1",score_list_dpgm3,data,win_size,step,colors,groundtruth)
            plot_color(dpg_2,"DPGMM0",score_list_dpgm2,data,win_size,step,colors,groundtruth)
            plot_color(dpg_1,"DPGMM+1",score_list_dpgm1,data,win_size,step,colors,groundtruth)
            plot_color(gm_4_data,"GMM+2",score_list_gm4,data,win_size,step,colors,groundtruth)
            plot_color(gm_3_data,"GMM-1",score_list_gm3,data,win_size,step,colors,groundtruth)
            plot_color(gm_2_data,"GMM0",score_list_gm2,data,win_size,step,colors,groundtruth)
            plot_color(gm_1_data,"GMM1",score_list_gm1,data,win_size,step,colors,groundtruth)
            plot_color(km_4_data,"kmeans+2",score_list_km4,data,win_size,step,colors,groundtruth)
            plot_color(km_3_data,"kmeans-1",score_list_km3,data,win_size,step,colors,groundtruth)
            plot_color(km_2_data,"kmeans0",score_list_km2,data,win_size,step,colors,groundtruth)
            plot_color(km_1_data,"kmeans1",score_list_km1,data,win_size,step,colors,groundtruth)
        
        else:
            plotfull_color(km_1_data,"kmeans1",score_list_km1,data,win_size,step,colors,groundtruth)
            plotfull_color(km_2_data,"kmeans0",score_list_km2,data,win_size,step,colors,groundtruth)
            plotfull_color(km_3_data,"kmeans-1",score_list_km3,data,win_size,step,colors,groundtruth)
            plotfull_color(km_4_data,"kmeans+2",score_list_km4,data,win_size,step,colors,groundtruth)
            plotfull_color(gm_1_data,"GMM1",score_list_gm1,data,win_size,step,colors,groundtruth)
            plotfull_color(gm_2_data,"GMM0",score_list_gm2,data,win_size,step,colors,groundtruth)
            plotfull_color(gm_3_data,"GMM-1",score_list_gm3,data,win_size,step,colors,groundtruth)
            plotfull_color(gm_4_data,"GMM+2",score_list_gm4,data,win_size,step,colors,groundtruth)
            plotfull_color(dpg_1,"DPGMM+1",score_list_dpgm1,data,win_size,step,colors,groundtruth)
            plotfull_color(dpg_2,"DPGMM0",score_list_dpgm2,data,win_size,step,colors,groundtruth)
            plotfull_color(dpg_4,"DPGMM+2",score_list_dpgm4,data,win_size,step,colors,groundtruth)
            plotfull_color(dpg_3,"DPGMM-1",score_list_dpgm3,data,win_size,step,colors,groundtruth)


        # prediction_s=t2s.embeddings
        # #print(prediction)
        # #clust=MeanShift()
        # numm=MeanShift().fit(X=prediction_s)
        # numm=numm.labels_
        # print(len(numm))
        # __embedding_label=reorder_label(numm)
        # hight = len(set(__embedding_label))
        # #__length = data.shape[0]
        # __length = data["x"].shape[0]
        # __win_size = win_size
        # __step = step
        # __offset = int(win_size/2)
        # # weight_vector = np.concatenate([np.linspace(0,1,self.__offset),np.linspace(1,0,self.__offset)])
        # weight_vector = np.ones(shape=(2*__offset)).flatten()
        # print(weight_vector.shape)
        # __state_seq = __embedding_label
        # vote_matrix = np.zeros((__length,hight))
        # print(vote_matrix.shape)
        # i = 0
        # for l in __embedding_label:
        #     vote_matrix[i:i+__win_size,l]+= weight_vector
        #     i+=__step
        # __state_seq = np.array([np.argmax(row) for row in vote_matrix])
        # prediction = np.array(__state_seq, dtype=int)
        # #np.save(os.path.join(out_path,fname), result)
        # ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        # score_list_shift.append(np.array([ari, anmi, nmi]))

        # if verbose:
        #     ans='MeanShift ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi)
        #     print(ans)
        #     with open(file_path, 'a') as file:
        #         file.write(ans+"\n")
        

        # cmap = cm.Set1
        # # ラベルの一覧を取得します
        # num_clusters = len(set(prediction))  # クラスタの数
        # cmap = cm.get_cmap('tab20', num_clusters) 
        # plt.figure(figsize=(10, 6))
        # #ts2vec
        # # plt.plot(data)  # cパラメータにクラスタラベルを渡す
        # # data_index=df.index.tolist()
        # #timesurl
        # plt.plot(train_data_np)  # cパラメータにクラスタラベルを渡す
        # data_index=train_data_df.index.tolist()
        # # addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/0722_max0.3_test"
        # # create_path(addpath)
        # for i in range(len(data)-1):
        # #for i in range(1000):
        #     # print(data[i])
        #     # print(prediction[i])
        #     plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
        # plt.legend() 
        # plt.savefig(addpath+"/"+fname+str(win_size)+'MeanShift.png')
        # plt.show()
    
        # #clust=DBSCAN()
        # numm=DBSCAN().fit(prediction_s).labels_
        # print(len(numm))
        # __embedding_label=reorder_label(numm)
        # #__embedding_label=reorder_label(DBSCAN(None).fit(prediction))
        # hight = len(set(__embedding_label))
        # #__length = data.shape[0]
        # __length = data["x"].shape[0]
        # __win_size = win_size
        # __step = step
        # __offset = int(win_size/2)
        # # weight_vector = np.concatenate([np.linspace(0,1,self.__offset),np.linspace(1,0,self.__offset)])
        # weight_vector = np.ones(shape=(2*__offset)).flatten()
        # __state_seq = __embedding_label
        # vote_matrix = np.zeros((__length,hight))
        # i = 0
        # for l in __embedding_label:
        #     vote_matrix[i:i+__win_size,l]+= weight_vector
        #     i+=__step
        # __state_seq = np.array([np.argmax(row) for row in vote_matrix])
        # prediction = np.array(__state_seq, dtype=int)
        # #np.save(os.path.join(out_path,fname), result)
        # ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        
        # score_list_dbscan.append(np.array([ari, anmi, nmi]))

        # cmap = cm.Set1
        # # ラベルの一覧を取得します
        # num_clusters = len(set(prediction))  # クラスタの数
        # cmap = cm.get_cmap('tab20', num_clusters) 
        # plt.figure(figsize=(10, 6))
        # #ts2vec
        # # plt.plot(data)  # cパラメータにクラスタラベルを渡す
        # # data_index=df.index.tolist()
        # #timesurl
        # plt.plot(train_data_np)  # cパラメータにクラスタラベルを渡す
        # data_index=train_data_df.index.tolist()
        # # addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/0722_max0.3_test"
        # # create_path(addpath)
        # for i in range(len(data)-1):
        # #for i in range(1000):
        #     # print(data[i])
        #     # print(prediction[i])
        #     plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
        # plt.legend() 
        # plt.savefig(addpath+"/"+fname+str(win_size)+'DBSCAN.png')
        # plt.show()
        # if verbose:
        #     ans='DBSCAN ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi)
        #     print(ans)
        #     with open(file_path, 'a') as file:
        #         file.write(ans+"\n")

        # #clust=hdbscan.HDBSCAN()
        # __embedding_label=reorder_label(hdbscan.HDBSCAN().fit(prediction_s).labels_)
        # #__embedding_label=reorder_label(HDBSCAN(None).fit(prediction))
        # hight = len(set(__embedding_label))
        # #__length = data.shape[0]
        # __length = data["x"].shape[0]
        # __win_size = win_size
        # __step = step
        # __offset = int(win_size/2)
        # # weight_vector = np.concatenate([np.linspace(0,1,self.__offset),np.linspace(1,0,self.__offset)])
        # weight_vector = np.ones(shape=(2*__offset)).flatten()
        # __state_seq = __embedding_label
        # vote_matrix = np.zeros((__length,hight))
        # i = 0
        # for l in __embedding_label:
        #     vote_matrix[i:i+__win_size,l]+= weight_vector
        #     i+=__step
        # __state_seq = np.array([np.argmax(row) for row in vote_matrix])
        # prediction = np.array(__state_seq, dtype=int)
        # #np.save(os.path.join(out_path,fname), result)
        # ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        # score_list_hdbscan.append(np.array([ari, anmi, nmi]))

        # cmap = cm.Set1
        # # ラベルの一覧を取得します
        # num_clusters = len(set(prediction))  # クラスタの数
        # cmap = cm.get_cmap('tab20', num_clusters) 
        # plt.figure(figsize=(10, 6))
        # #ts2vec
        # # plt.plot(data)  # cパラメータにクラスタラベルを渡す
        # # data_index=df.index.tolist()
        # #timesurl
        # plt.plot(train_data_np)  # cパラメータにクラスタラベルを渡す
        # data_index=train_data_df.index.tolist()
        # # addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/0722_max0.3_test"
        # # create_path(addpath)
        # for i in range(len(data)-1):
        # #for i in range(1000):
        #     # print(data[i])
        #     # print(prediction[i])
        #     plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
        # plt.legend() 
        # plt.savefig(addpath+"/"+fname+str(win_size)+'HDBSCAN.png')
        # plt.show()
        # if verbose:
        #     ans='HDBSCAN ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi)
        #     print(ans)
        #     with open(file_path, 'a') as file:
        #         file.write(ans+"\n")
        



        # plt.figure(figsize=(10, 6))
        # regime_list, colors=make_color(groundtruth)
        # #print(regime_list)
        # plt.plot(data) 
        # num_clusters = len(set(regime_list))  # クラスタの数
        # cmap = cm.get_cmap('tab20', num_clusters) 
        # for i in range(len(regime_list)-1):
        #     plt.axvspan(regime_list[i], regime_list[i+1], color=colors[i], alpha=0.2)
        # plt.legend() 
        # plt.savefig(addpath+"/"+fname+'_answer.png')
        # plt.show()

        # print(acc(groundtruth, prediction))
        # v_list = calculate_scalar_velocity_list(t2s.embeddings)
        # fig, ax = plt.subplots(nrows=2)
        # for i in range(4):
        #     ax[0].plot(data[:,i])
        # ax[1].plot(v_list)
        # plt.show()
        # plot_mulvariate_time_series_and_label_v2(data, label=prediction, groundtruth=groundtruth)
        # embedding_space(t2s.embeddings, show=True, s=5, label=t2s.embedding_label)
        
         # plot_mulvariate_time_series_and_label(data[0].T, label=prediction, groundtruth=groundtruth)
        # print('Time2State_backup,%d,%f'%(idx,ari))
        
        
    score_list = np.vstack(score_list)
    ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_km1)
    ans='KM1 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_km2)
    ans='KM2 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_km3)
    ans='KM3 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_km4)
    ans='KM4 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    
    score_list = np.vstack(score_list_gm1)
    ans='GM1 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_gm2)
    ans='GM2 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_gm3)
    ans='GM3 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_gm4)
    ans='GM4 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_dpgm1)
    ans='DPGM1 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_dpgm2)
    ans='DPGM2 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_dpgm3)
    ans='DPGM3 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list = np.vstack(score_list_dpgm4)
    ans='DPGM4 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")


        

    # score_list_shift = np.vstack(score_list_shift)
    # ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list_shift[:,0])\
    #     ,np.mean(score_list_shift[:,1])
    #     ,np.mean(score_list_shift[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list_dbscan = np.vstack(score_list_dbscan)
    # ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list_dbscan[:,0])\
    #     ,np.mean(score_list_dbscan[:,1])
    #     ,np.mean(score_list_dbscan[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list_hdbscan = np.vstack(score_list_hdbscan)
    # ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list_hdbscan[:,0])\
    #     ,np.mean(score_list_hdbscan[:,1])
    #     ,np.mean(score_list_hdbscan[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

def exp_on_synthetic(win_size=512, step=100, verbose=False):
    out_path = os.path.join(output_path,'synthetic')
    create_path(out_path)
    params_LSE['in_channels'] = 4
    params_LSE['win_size'] = win_size
    params_LSE['M'] = 10
    params_LSE['N'] = 4
    params_LSE['nb_steps'] = 40
    params_LSE['out_channels'] = 4
    params_LSE['cuda'] = True
    params_LSE['gpu'] = 1
    params_LSE['seed']=42
    prefix = os.path.join(data_path, 'synthetic/test')

    score_list = []
    score_list2 = []
    score_list_shift = []
    score_list_dbscan = []
    score_list_hdbscan = []
    score_list_km1 = []
    score_list_km2 = []
    score_list_km3 = []
    score_list_km4 = []
    score_list_gm1 = []
    score_list_gm2 = []
    score_list_gm3 = []
    score_list_gm4 = []
    score_list_dpgm1 = []
    score_list_dpgm2 = []
    score_list_dpgm3 = []
    score_list_dpgm4 = []

    addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Syn/0122_syn_STL"
    create_path(addpath)
    for i in range(100):
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=range(4), skiprows=1)
        data = df.to_numpy()
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=[4], skiprows=1)
        saveplt=addpath+"/"+str(i)+str(win_size)+"plt.png"
        file_path=addpath+"/result.txt"
        with open(file_path, 'a') as file:
            file.write("win_size"+str(win_size)+"\n")
            file.write("step"+str(step)+"\n")
        groundtruth = df.to_numpy(dtype=int).flatten()
        #Ts2Vecバージョン
        #t2s = Time2State(win_size, step, TS2Vec_Adaper(params_LSE), DPGMM(None)).fit(data, win_size,step,groundtruth,saveplt)
        #t2s = Time2State(win_size, step, TS2Vec_Adaper(win_size), DPGMM(None)).fit(data, win_size,step,groundtruth,saveplt)
        
        # t2s = Time2State_backup(win_size, step, CausalConv_CPC_Adaper(params_CPC), DPGMM(None)).fit(data, win_size, step)
        #LSEバージョン
        t2s = Time2State(win_size, step, TS2Vec_Adaper(params_LSE), DPGMM(None)).fit(data, win_size,step,groundtruth,saveplt)
        #t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step,groundtruth,saveplt)
        # t2s = Time2State_backup(win_size, step, LSTM_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
        # t2s = Time2State_backup(win_size, step, CausalConv_Triplet_Adaper(params_Triplet), DPGMM(None)).fit(data, win_size, step)
        prediction = t2s.state_seq
        # t2s.set_clustering_component(KMeansClustering(5)).predict_without_encode(data, win_size, step)
        # prediction2 = t2s.state_seq
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        #np.save(os.path.join(out_path,str(i)), result)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        # ari2, anmi2, nmi2 = evaluate_clustering(groundtruth, prediction2)
        score_list.append(np.array([ari, anmi, nmi]))
        ans='DPGMM, ARI: %f, ANMI: %f, NMI: %f' %(ari, anmi, nmi)
        print(ans)
        with open(file_path, 'a') as file:
            file.write(ans+"\n")
        import matplotlib.cm as cm
        from matplotlib.colors import ListedColormap
        cmap = cm.Set1
        # ラベルの一覧を取得します
        num_clusters = len(set(prediction))  # クラスタの数
        cmap = cm.get_cmap('tab20', num_clusters) 
        plt.figure(figsize=(10, 6))
        plt.plot(data)  # cパラメータにクラスタラベルを渡す
        data_index=df.index.tolist()
        # addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Syn/1217_ts2vec"
        # create_path(addpath)
        for i in range(len(data)-1):
        #for i in range(1000):
            # print(data[i])
            # print(prediction[i])
            plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
        plt.legend() 
        plt.savefig(addpath+"/"+str(i)+'.png')
        plt.show()

        plt.figure(figsize=(10, 6))
        regime_list, colors=make_color(groundtruth)
        #print(regime_list)
        plt.plot(data) 
        num_clusters = len(set(regime_list))  # クラスタの数
        cmap = cm.get_cmap('tab20', num_clusters) 
        for i in range(len(regime_list)-1):
            plt.axvspan(regime_list[i], regime_list[i+1], color=colors[i], alpha=0.2)
        plt.legend() 
        plt.savefig(addpath+"/"+str(i)+'_answer.png')
        plt.show()

        def plot_color(num_clusters,name,score_list,data,win_size,step,colors,groundtruth):
            #numm=num_clusters.labels_
            #print(len(num_clusters))
            __embedding_label=reorder_label(num_clusters)
            hight = len(set(__embedding_label))
            __length = data.shape[0]
            #__length = data["x"].shape[0]
            __win_size = win_size
            __step = step
            __offset = int(win_size/2)
            # weight_vector = np.concatenate([np.linspace(0,1,self.__offset),np.linspace(1,0,self.__offset)])
            weight_vector = np.ones(shape=(2*__offset)).flatten()
            #print(weight_vector.shape)
            __state_seq = __embedding_label
            vote_matrix = np.zeros((__length,hight))
            #print(vote_matrix.shape)
            i = 0
            for l in __embedding_label:
                vote_matrix[i:i+__win_size,l]+= weight_vector
                i+=__step
            __state_seq = np.array([np.argmax(row) for row in vote_matrix])
            prediction = np.array(__state_seq, dtype=int)
            min_size = min(len(groundtruth), len(prediction))

            groundtruth = groundtruth[:min_size]
            prediction = prediction[:min_size]
            num_clusters = len(set(prediction)) 

            colors=colors[:num_clusters]
            cmap = mcolors.ListedColormap(colors)
            #cmap = cm.get_cmap('tab20', num_clusters) 
            plt.figure(figsize=(10, 6))
            #ts2vec
            plt.plot(data)  # cパラメータにクラスタラベルを渡す
            data_index=df.index.tolist()
            #timesurl
            # plt.plot(train_data_np)  # cパラメータにクラスタラベルを渡す
            # data_index=train_data_df.index.tolist()
            # addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/0722_max0.3_test"
            # create_path(addpath)
            for i in range(len(data)-1):
            #for i in range(1000):
                # print(data[i])
                # print(prediction[i])
                plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
            plt.legend() 
            plt.savefig(addpath+"/"+str(win_size)+str(step)+name+'.png')
            plt.show()

            #np.save(os.path.join(out_path,fname), result)
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            score_list.append(np.array([ari, anmi, nmi]))
            ans='%s , ARI: %f, ANMI: %f, NMI: %f' %(name, ari, anmi, nmi)
            print(ans)
            with open(file_path, 'a') as file:
                file.write(ans+"\n")

        def plotfull_color(num_clusters,name,score_list,data,win_size,step,colors,groundtruth):
            
            prediction = np.array(num_clusters, dtype=int)
            num_clusters = len(set(prediction)) 

            colors=colors[:num_clusters]
            cmap = mcolors.ListedColormap(colors)
            #cmap = cm.get_cmap('tab20', num_clusters) 
            plt.figure(figsize=(10, 6))
            #ts2vec
            plt.plot(data)  # cパラメータにクラスタラベルを渡す
            data_index=df.index.tolist()
            #timesurl
            # plt.plot(train_data_np)  # cパラメータにクラスタラベルを渡す
            # data_index=train_data_df.index.tolist()
            # addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/0722_max0.3_test"
            # create_path(addpath)
            for i in range(len(data)-1):
            #for i in range(1000):
                # print(data[i])
                # print(prediction[i])
                plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
            plt.legend() 
            plt.savefig(addpath+"/"+str(win_size)+str(step)+name+'.png')
            plt.show()

            #np.save(os.path.join(out_path,fname), result)
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            min_size = min(len(groundtruth), len(prediction))

            groundtruth = groundtruth[:min_size]
            prediction = prediction[:min_size]
            score_list.append(np.array([ari, anmi, nmi]))
            ans='%s , ARI: %f, ANMI: %f, NMI: %f' %(name, ari, anmi, nmi)
            print(ans)
            with open(file_path, 'a') as file:
                file.write(ans+"\n")

        answer_num=len(set(groundtruth))
        prediction_s=t2s.embeddings
        from sklearn.cluster import DBSCAN
        import hdbscan
  
        meanshift1=MeanShift().fit(X=prediction_s).labels_
        #meanshift=meanshift1.predict(X=prediction_s)
        dbscan1=DBSCAN().fit(prediction_s).labels_
        #dbscan=dbscan1.predict(X=prediction_s)
        hdbscan1=hdbscan.HDBSCAN().fit(prediction_s).labels_
        #hdbscan=hdbscan1.predict(X=prediction_s)

        # km_1=KMeans(n_clusters=answer_num+1, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        # km_1_data = km_1.fit_predict(prediction_s)# クラスタの数
        # #

        # km_2=KMeans(n_clusters=answer_num, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        # km_2_data = km_2.fit_predict(prediction_s)
        # #

        # km_3=KMeans(n_clusters=answer_num-1, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        # km_3_data = km_3.fit_predict(prediction_s)
        # #

        # km_4=KMeans(n_clusters=answer_num+2, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        # km_4_data = km_4.fit_predict(prediction_s)

        # gm_1=GaussianMixture(n_components=answer_num+1,reg_covar=5e-3, init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        # gm_1_ = gm_1.fit(prediction_s)
        # gm_1_data=gm_1_.predict(prediction_s)

        # gm_2=GaussianMixture(n_components=answer_num, reg_covar=5e-3,init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        # gm_2_ = gm_2.fit(prediction_s)
        # gm_2_data=gm_2_.predict(prediction_s)

        # gm_3=GaussianMixture(n_components=answer_num-1,reg_covar=5e-3, init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        # gm_3_ = gm_3.fit(prediction_s)
        # gm_3_data=gm_3_.predict(prediction_s)

        # gm_4=GaussianMixture(n_components=answer_num+2, reg_covar=5e-3, init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        # gm_4_ = gm_4.fit(prediction_s)
        # gm_4_data=gm_4_.predict(prediction_s)

        # dpgmm_1 = mixture.BayesianGaussianMixture(init_params='k-means++',
        #                                         n_components=answer_num+1,
        #                                         covariance_type="full",
        #                                         weight_concentration_prior_type='dirichlet_process',
        #                                         max_iter=1000).fit(prediction_s)
        # dpg_1=dpgmm_1.predict(prediction_s)

        # dpgmm_2 = mixture.BayesianGaussianMixture(init_params='k-means++',
        #                                         n_components=answer_num,
        #                                         covariance_type="full",
        #                                         weight_concentration_prior_type='dirichlet_process',
        #                                         max_iter=1000).fit(prediction_s)
        # dpg_2=dpgmm_2.predict(prediction_s) 

        # dpgmm_3 = mixture.BayesianGaussianMixture(init_params='k-means++',
        #                                         n_components=answer_num-1,
        #                                         covariance_type="full",
        #                                         weight_concentration_prior_type='dirichlet_process',
        #                                         max_iter=1000).fit(prediction_s)
        # dpg_3=dpgmm_3.predict(prediction_s)

        # dpgmm_4 = mixture.BayesianGaussianMixture(init_params='k-means++',
        #                                         n_components=answer_num+2,
        #                                         covariance_type="full",
        #                                         weight_concentration_prior_type='dirichlet_process',
        #                                         max_iter=1000).fit(prediction_s)
        # dpg_4=dpgmm_4.predict(prediction_s)

        plot_color(meanshift1,"Meanshift",score_list_shift,data,win_size,step,colors,groundtruth)
        plot_color(dbscan1,"dbscan",score_list_dbscan,data,win_size,step,colors,groundtruth)
        plot_color(hdbscan1,"hdscan",score_list_hdbscan,data,win_size,step,colors,groundtruth)

        # plot_color(dpg_4,"DPGMM+2",score_list_dpgm4,data,win_size,step,colors,groundtruth)
        # plot_color(dpg_3,"DPGMM-1",score_list_dpgm3,data,win_size,step,colors,groundtruth)
        # plot_color(dpg_2,"DPGMM0",score_list_dpgm2,data,win_size,step,colors,groundtruth)
        # plot_color(dpg_1,"DPGMM+1",score_list_dpgm1,data,win_size,step,colors,groundtruth)
        # plot_color(gm_4_data,"GMM+2",score_list_gm4,data,win_size,step,colors,groundtruth)
        # plot_color(gm_3_data,"GMM-1",score_list_gm3,data,win_size,step,colors,groundtruth)
        # plot_color(gm_2_data,"GMM0",score_list_gm2,data,win_size,step,colors,groundtruth)
        # plot_color(gm_1_data,"GMM1",score_list_gm1,data,win_size,step,colors,groundtruth)
        # plot_color(km_4_data,"kmeans+2",score_list_km4,data,win_size,step,colors,groundtruth)
        # plot_color(km_3_data,"kmeans-1",score_list_km3,data,win_size,step,colors,groundtruth)
        # plot_color(km_2_data,"kmeans0",score_list_km2,data,win_size,step,colors,groundtruth)
        # plot_color(km_1_data,"kmeans1",score_list_km1,data,win_size,step,colors,groundtruth)

        
        
    score_list = np.vstack(score_list)
    ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list_shift = np.vstack(score_list_shift)
    ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list_shift[:,0])\
        ,np.mean(score_list_shift[:,1])
        ,np.mean(score_list_shift[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list_dbscan = np.vstack(score_list_dbscan)
    ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list_dbscan[:,0])\
        ,np.mean(score_list_dbscan[:,1])
        ,np.mean(score_list_dbscan[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list_hdbscan = np.vstack(score_list_hdbscan)
    ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list_hdbscan[:,0])\
        ,np.mean(score_list_hdbscan[:,1])
        ,np.mean(score_list_hdbscan[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")
    # score_list = np.vstack(score_list_km1)
    # ans='KM1 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_km2)
    # ans='KM2 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_km3)
    # ans='KM3 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_km4)
    # ans='KM4 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")
    # score_list = np.vstack(score_list_gm1)
    # ans='GM1 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_gm2)
    # ans='GM2 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_gm3)
    # ans='GM3 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_gm4)
    # ans='GM4 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_dpgm1)
    # ans='DPGM1 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_dpgm2)
    # ans='DPGM2 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_dpgm3)
    # ans='DPGM3 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_dpgm4)
    # ans='DPGM4 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

def convert_mat_to_df(mat_data):
    # 空の DataFrame を作成
    df = pd.DataFrame()

    # 辞書型の各キーに対して
    for key in mat_data:
        # キーがメタデータではない場合
        if not key.startswith('__'):
            # MATLAB のデータは通常2次元配列です
            data = mat_data[key].flatten() if mat_data[key].ndim > 1 else mat_data[key]

            # 既存のDataFrameが空の場合はそのまま追加
            if df.empty:
                df[key] = data
            # 既存のDataFrameとデータの長さが一致する場合は列を追加
            elif len(data) == len(df):
                df[key] = data
            else:
                print(f"Warning: Length of data for key '{key}' does not match the length of the DataFrame. Skipping this key.")
    
    return df

def exp_on_ActRecTut(win_size, step, verbose=False):
    out_path = os.path.join(output_path,'ActRecTut')
    create_path(out_path)
    params_LSE['in_channels'] = 10
    params_LSE['win_size'] = win_size
    params_LSE['out_channels'] = 4
    params_LSE['M'] = 20
    params_LSE['N'] = 4
    params_LSE['nb_steps'] = 50
    params_LSE['cuda'] = True
    params_LSE['gpu'] = 1
    params_LSE['seed']=42
    score_list = []
  
    score_list = []
    score_list2 = []
    score_list_shift = []
    score_list_dbscan = []
    score_list_hdbscan = []
    score_list_km1 = []
    score_list_km2 = []
    score_list_km3 = []
    score_list_km4 = []
    score_list_gm1 = []
    score_list_gm2 = []
    score_list_gm3 = []
    score_list_gm4 = []
    score_list_dpgm1 = []
    score_list_dpgm2 = []
    score_list_dpgm3 = []
    score_list_dpgm4 = []

    # train
    # if False:
    #     dataset_path = os.path.join(data_path,'ActRecTut/subject1_walk/data.mat')
    #     data = scipy.io.loadmat(dataset_path)
    #     groundtruth = data['labels'].flatten()
    #     groundtruth = reorder_label(groundtruth)
    #     data = data['data'][:,0:10]
    #     data = normalize(data, mode='channel')
    #     # true state number is 6
    #     t2s = Time2State_backup(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit_encoder(data)
    dir_list = ['subject1_walk', 'subject2_walk']
    for dir_name in dir_list:
        # repeat for 10 times
        for j in range(10):
            dataset_path = os.path.join(data_path,'ActRecTut/'+dir_name+'/data.mat')
            data = scipy.io.loadmat(dataset_path)
            df = convert_mat_to_df(data)
            data_index=df.index.tolist()
            groundtruth = data['labels'].flatten()
            groundtruth = reorder_label(groundtruth)
            data = data['data'][:,0:10]
            data = normalize(data)
            addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Act/0122_Act_STL"
            create_path(addpath)
            file_path=addpath+"/result.txt"
            saveplt=addpath+"/"+str(dir_name)+str(j)+str(win_size)+"plt.png"
            # true state number is 6
            t2s = Time2State(win_size, step, TS2Vec_Adaper(params_LSE), DPGMM(None)).fit(data, win_size,step,groundtruth,saveplt)
            #t2s = Time2State(win_size, step, TS2Vec_Adaper(params_LSE), DPGMM(None)).fit(data, win_size,step,groundtruth,saveplt)
            #t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step,groundtruth,saveplt)
            # t2s.predict(data, win_size, step)
            prediction = t2s.state_seq+1
            prediction = np.array(prediction, dtype=int)
            result = np.vstack([groundtruth, prediction])
            #np.save(os.path.join(out_path,dir_name+str(i)), result)
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            score_list.append(np.array([ari, anmi, nmi]))
            import matplotlib.cm as cm
            from matplotlib.colors import ListedColormap
            cmap = cm.Set1
            # ラベルの一覧を取得します
            num_clusters = len(set(prediction))  # クラスタの数
            cmap = cm.get_cmap('tab20', num_clusters) 
            plt.figure(figsize=(10, 6))
            plt.plot(data)  # cパラメータにクラスタラベルを渡す
            
            
            create_path(addpath)
            for i in range(len(data)-1):
            #for i in range(1000):
                # print(data[i])
                # print(prediction[i])
                plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
            plt.legend() 
            plt.savefig(addpath+"/"+dir_name+'.png')
            plt.show()

            plt.figure(figsize=(10, 6))
            regime_list, colors=make_color(groundtruth)
            #print(regime_list)
            plt.plot(data) 
            num_clusters = len(set(regime_list))  # クラスタの数
            cmap = cm.get_cmap('tab20', num_clusters) 
            for i in range(len(regime_list)-1):
                plt.axvspan(regime_list[i], regime_list[i+1], color=colors[i], alpha=0.2)
            plt.legend() 
            plt.savefig(addpath+"/"+dir_name+'_answer.png')
            plt.show()
            # plot_mulvariate_time_series_and_label_v2(data, label=prediction, groundtruth=groundtruth)
            if verbose:
                print('ID: %s,i:%f, ARI: %f, ANMI: %f, NMI: %f' %(dir_name,j, ari, anmi, nmi))

            def plot_color(num_clusters,name,score_list,data,win_size,step,colors,groundtruth):
                print(len(num_clusters))
                __embedding_label=reorder_label(num_clusters)
                hight = len(set(__embedding_label))
                __length = data.shape[0]
                #__length = data["x"].shape[0]
                __win_size = win_size
                __step = step
                __offset = int(win_size/2)
                weight_vector = np.ones(shape=(2*__offset)).flatten()
                __state_seq = __embedding_label
                vote_matrix = np.zeros((__length,hight))
                i = 0
                for l in __embedding_label:
                    vote_matrix[i:i+__win_size,l]+= weight_vector
                    i+=__step
                __state_seq = np.array([np.argmax(row) for row in vote_matrix])
                prediction = np.array(__state_seq, dtype=int)
                min_size = min(len(groundtruth), len(prediction))

                groundtruth = groundtruth[:min_size]
                prediction = prediction[:min_size]
                num_clusters = len(set(prediction)) 

                colors=colors[:num_clusters]
                cmap = mcolors.ListedColormap(colors)
                plt.figure(figsize=(10, 6))
                #ts2vec
                plt.plot(data)  # cパラメータにクラスタラベルを渡す
                data_index=df.index.tolist()
                #timesurl
                # plt.plot(train_data_np)  # cパラメータにクラスタラベルを渡す
                # data_index=train_data_df.index.tolist()
                # addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/0722_max0.3_test"
                # create_path(addpath)
                for i in range(len(data)-1):
                    plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
                plt.legend() 
                plt.savefig(addpath+"/"+str(win_size)+str(step)+name+'.png')
                plt.show()

                #np.save(os.path.join(out_path,fname), result)
                ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
                score_list.append(np.array([ari, anmi, nmi]))
                ans='ID: %s,%s,i:%f, ARI: %f, ANMI: %f, NMI: %f' %(name,dir_name,j, ari, anmi, nmi)
                print(ans)
                with open(file_path, 'a') as file:
                    file.write(ans+"\n")

            answer_num=len(set(groundtruth))
            prediction_s=t2s.embeddings
            from sklearn.cluster import DBSCAN
            import hdbscan
            
            meanshift1=MeanShift().fit(X=prediction_s).labels_
            dbscan1=DBSCAN().fit(prediction_s).labels_
            hdbscan1=hdbscan.HDBSCAN().fit(prediction_s).labels_

            # km_1=KMeans(n_clusters=answer_num+1, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
            # km_1_data = km_1.fit_predict(prediction_s)# クラスタの数
            # #

            # km_2=KMeans(n_clusters=answer_num, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
            # km_2_data = km_2.fit_predict(prediction_s)
            # #

            # km_3=KMeans(n_clusters=answer_num-1, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
            # km_3_data = km_3.fit_predict(prediction_s)
            # #

            # km_4=KMeans(n_clusters=answer_num+2, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
            # km_4_data = km_4.fit_predict(prediction_s)

            # gm_1=GaussianMixture(n_components=answer_num+1,reg_covar=5e-3, init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
            # gm_1_ = gm_1.fit(prediction_s)
            # gm_1_data=gm_1_.predict(prediction_s)

            # gm_2=GaussianMixture(n_components=answer_num, reg_covar=5e-3,init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
            # gm_2_ = gm_2.fit(prediction_s)
            # gm_2_data=gm_2_.predict(prediction_s)

            # gm_3=GaussianMixture(n_components=answer_num-1,reg_covar=5e-3, init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
            # gm_3_ = gm_3.fit(prediction_s)
            # gm_3_data=gm_3_.predict(prediction_s)

            # gm_4=GaussianMixture(n_components=answer_num+2, reg_covar=5e-3, init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
            # gm_4_ = gm_4.fit(prediction_s)
            # gm_4_data=gm_4_.predict(prediction_s)

            # dpgmm_1 = mixture.BayesianGaussianMixture(init_params='k-means++',
            #                                         n_components=answer_num+1,
            #                                         covariance_type="full",
            #                                         weight_concentration_prior_type='dirichlet_process',
            #                                         max_iter=1000).fit(prediction_s)
            # dpg_1=dpgmm_1.predict(prediction_s)

            # dpgmm_2 = mixture.BayesianGaussianMixture(init_params='k-means++',
            #                                         n_components=answer_num,
            #                                         covariance_type="full",
            #                                         weight_concentration_prior_type='dirichlet_process',
            #                                         max_iter=1000).fit(prediction_s)
            # dpg_2=dpgmm_2.predict(prediction_s) 

            # dpgmm_3 = mixture.BayesianGaussianMixture(init_params='k-means++',
            #                                         n_components=answer_num-1,
            #                                         covariance_type="full",
            #                                         weight_concentration_prior_type='dirichlet_process',
            #                                         max_iter=1000).fit(prediction_s)
            # dpg_3=dpgmm_3.predict(prediction_s)

            # dpgmm_4 = mixture.BayesianGaussianMixture(init_params='k-means++',
            #                                         n_components=answer_num+2,
            #                                         covariance_type="full",
            #                                         weight_concentration_prior_type='dirichlet_process',
            #                                         max_iter=1000).fit(prediction_s)
            # dpg_4=dpgmm_4.predict(prediction_s)

            plot_color(meanshift1,"Meanshift",score_list_shift,data,win_size,step,colors,groundtruth)
            plot_color(dbscan1,"dbscan",score_list_dbscan,data,win_size,step,colors,groundtruth)
            plot_color(hdbscan1,"hdscan",score_list_hdbscan,data,win_size,step,colors,groundtruth)

            # plot_color(dpg_4,"DPGMM+2",score_list_dpgm4,data,win_size,step,colors,groundtruth)
            # plot_color(dpg_3,"DPGMM-1",score_list_dpgm3,data,win_size,step,colors,groundtruth)
            # plot_color(dpg_2,"DPGMM0",score_list_dpgm2,data,win_size,step,colors,groundtruth)
            # plot_color(dpg_1,"DPGMM+1",score_list_dpgm1,data,win_size,step,colors,groundtruth)
            # plot_color(gm_4_data,"GMM+2",score_list_gm4,data,win_size,step,colors,groundtruth)
            # plot_color(gm_3_data,"GMM-1",score_list_gm3,data,win_size,step,colors,groundtruth)
            # plot_color(gm_2_data,"GMM0",score_list_gm2,data,win_size,step,colors,groundtruth)
            # plot_color(gm_1_data,"GMM1",score_list_gm1,data,win_size,step,colors,groundtruth)
            # plot_color(km_4_data,"kmeans+2",score_list_km4,data,win_size,step,colors,groundtruth)
            # plot_color(km_3_data,"kmeans-1",score_list_km3,data,win_size,step,colors,groundtruth)
            # plot_color(km_2_data,"kmeans0",score_list_km2,data,win_size,step,colors,groundtruth)
            # plot_color(km_1_data,"kmeans1",score_list_km1,data,win_size,step,colors,groundtruth)

        
        
    score_list = np.vstack(score_list)
    ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list_shift = np.vstack(score_list_shift)
    ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list_shift[:,0])\
        ,np.mean(score_list_shift[:,1])
        ,np.mean(score_list_shift[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list_dbscan = np.vstack(score_list_dbscan)
    ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list_dbscan[:,0])\
        ,np.mean(score_list_dbscan[:,1])
        ,np.mean(score_list_dbscan[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    score_list_hdbscan = np.vstack(score_list_hdbscan)
    ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list_hdbscan[:,0])\
        ,np.mean(score_list_hdbscan[:,1])
        ,np.mean(score_list_hdbscan[:,2]))
    print(ans)
    
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    # score_list = np.vstack(score_list_km1)
    # ans='KM1 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_km2)
    # ans='KM2 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_km3)
    # ans='KM3 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_km4)
    # ans='KM4 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_gm1)
    # ans='GM1 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_gm2)
    # ans='GM2 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_gm3)
    # ans='GM3 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_gm4)
    # ans='GM4 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_dpgm1)
    # ans='DPGM1 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_dpgm2)
    # ans='DPGM2 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_dpgm3)
    # ans='DPGM3 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_dpgm4)
    # ans='DPGM4 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")          
    
def fill_nan(data):
    x_len, y_len = data.shape
    for x in range(x_len):
        for y in range(y_len):
            if np.isnan(data[x,y]):
                data[x,y]=data[x-1,y]
    return data

def exp_on_PAMAP2(win_size, step, verbose=False):
    out_path = os.path.join(output_path,'PAMAP2')
    create_path(out_path)
    params_LSE['in_channels'] = 9
    params_LSE['win_size'] = win_size
    params_LSE['out_channels'] = 9
    params_LSE['M'] = 20
    params_LSE['N'] = 4
    params_LSE['nb_steps'] = 50
    params_LSE['cuda'] = True
    params_LSE['gpu'] = 1
    params_LSE['seed']=42
    # params_LSE['kernel_size'] = 3
    
    dataset_path = os.path.join(data_path,'PAMAP2/Protocol/subject10'+str(1)+'.dat')
    df = pd.read_csv(dataset_path, sep=' ', header=None)
    data = df.to_numpy()
    groundtruth = np.array(data[:,1],dtype=int)
    hand_acc = data[:,4:7]
    chest_acc = data[:,21:24]
    ankle_acc = data[:,38:41]
    # hand_gy = data[:,10:13]
    # chest_gy = data[:,27:30]
    # ankle_gy = data[:,44:47]
    # data = np.hstack([hand_acc, chest_acc, ankle_acc, hand_gy, chest_gy, ankle_gy])
    data = np.hstack([hand_acc, chest_acc, ankle_acc])
    data = fill_nan(data)
    data = normalize(data)
    addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/PAMAP/0122_PAMAP_STL"
    create_path(addpath)
    
    
    file_path=addpath+"/result.txt"
    score_list = []
    score_list2 = []
    score_list_shift = []
    score_list_dbscan = []
    score_list_hdbscan = []
    score_list_km1 = []
    score_list_km2 = []
    score_list_km3 = []
    score_list_km4 = []
    score_list_gm1 = []
    score_list_gm2 = []
    score_list_gm3 = []
    score_list_gm4 = []
    score_list_dpgm1 = []
    score_list_dpgm2 = []
    score_list_dpgm3 = []
    score_list_dpgm4 = []
    for i in range(1, 9):
        dataset_path = os.path.join(data_path,'PAMAP2/Protocol/subject10'+str(i)+'.dat')
        df = pd.read_csv(dataset_path, sep=' ', header=None)
        data = df.to_numpy()
        groundtruth = np.array(data[:,1],dtype=int)
        hand_acc = data[:,4:7]
        chest_acc = data[:,21:24]
        ankle_acc = data[:,38:41]
        # hand_gy = data[:,10:13]
        # chest_gy = data[:,27:30]
        # ankle_gy = data[:,44:47]
        # data = np.hstack([hand_acc, chest_acc, ankle_acc, hand_gy, chest_gy, ankle_gy])
        data = np.hstack([hand_acc, chest_acc, ankle_acc])
        data = fill_nan(data)
        data = normalize(data)
        saveplt=addpath+"/subject10"+str(i)+str(win_size)+"plt.png"
        #t2s = Time2State(win_size, step, TS2Vec_Adaper(params_LSE), DPGMM(None)).fit(data, win_size,step,groundtruth,saveplt)
        t2s = Time2State(win_size, step, TS2Vec_Adaper(params_LSE), DPGMM(None)).fit(data, win_size,step,groundtruth,saveplt)
        #t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size,step,groundtruth,saveplt)
        #t2s.predict(data, win_size, step)
        prediction = t2s.state_seq
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        # np.save(os.path.join(out_path,'10'+str(i)), result)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        import matplotlib.cm as cm
        from matplotlib.colors import ListedColormap
        cmap = cm.Set1
        # ラベルの一覧を取得します
        num_clusters = len(set(prediction))  # クラスタの数
        cmap = cm.get_cmap('tab20', num_clusters) 
        plt.figure(figsize=(10, 6))
        plt.plot(data)  # cパラメータにクラスタラベルを渡す
        data_index=df.index.tolist()
        
        for i in range(len(data)-1):
        #for i in range(1000):
            # print(data[i])
            # print(prediction[i])
            plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
        plt.legend() 
        plt.savefig(addpath+"/subject10"+str(i)+'.png')
        plt.show()

        plt.figure(figsize=(10, 6))
        regime_list, colors=make_color(groundtruth)
        #print(regime_list)
        plt.plot(data) 
        num_clusters = len(set(regime_list))  # クラスタの数
        cmap = cm.get_cmap('tab20', num_clusters) 
        for i in range(len(regime_list)-1):
            plt.axvspan(regime_list[i], regime_list[i+1], color=colors[i], alpha=0.2)
        plt.legend() 
        plt.savefig(addpath+"/subject10"+str(i)+'_answer.png')
        plt.show()
        # plot_mulvariate_time_series_and_label_v2(data,groundtruth,prediction)
        # plt.savefig('1.png')
        if verbose:
            print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(i, ari, anmi, nmi))

        def plot_color(num_clusters,name,score_list,data,win_size,step,colors,groundtruth):
            #numm=num_clusters.labels_
            #print(len(num_clusters))
            __embedding_label=reorder_label(num_clusters)
            hight = len(set(__embedding_label))
            __length = data.shape[0]
            #__length = data["x"].shape[0]
            __win_size = win_size
            __step = step
            __offset = int(win_size/2)
            # weight_vector = np.concatenate([np.linspace(0,1,self.__offset),np.linspace(1,0,self.__offset)])
            weight_vector = np.ones(shape=(2*__offset)).flatten()
            #print(weight_vector.shape)
            __state_seq = __embedding_label
            vote_matrix = np.zeros((__length,hight))
            #print(vote_matrix.shape)
            i = 0
            for l in __embedding_label:
                vote_matrix[i:i+__win_size,l]+= weight_vector
                i+=__step
            __state_seq = np.array([np.argmax(row) for row in vote_matrix])
            prediction = np.array(__state_seq, dtype=int)
            min_size = min(len(groundtruth), len(prediction))

            groundtruth = groundtruth[:min_size]
            prediction = prediction[:min_size]
            num_clusters = len(set(prediction)) 

            colors=colors[:num_clusters]
            cmap = mcolors.ListedColormap(colors)
            #cmap = cm.get_cmap('tab20', num_clusters) 
            plt.figure(figsize=(10, 6))
            #ts2vec
            plt.plot(data)  # cパラメータにクラスタラベルを渡す
            data_index=df.index.tolist()
            #timesurl
            # plt.plot(train_data_np)  # cパラメータにクラスタラベルを渡す
            # data_index=train_data_df.index.tolist()
            # addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/0722_max0.3_test"
            # create_path(addpath)
            for i in range(len(data)-1):
            #for i in range(1000):
                # print(data[i])
                # print(prediction[i])
                plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
            plt.legend() 
            plt.savefig(addpath+"/"+str(win_size)+str(step)+name+'.png')
            plt.show()

            #np.save(os.path.join(out_path,fname), result)
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            score_list.append(np.array([ari, anmi, nmi]))
            ans='%s ID:%d, ARI: %f, ANMI: %f, NMI: %f' %(name,i, ari, anmi, nmi)
            print(ans)
            with open(file_path, 'a') as file:
                file.write(ans+"\n")

        def plotfull_color(num_clusters,name,score_list,data,win_size,step,colors,groundtruth):
            
            prediction = np.array(num_clusters, dtype=int)
            num_clusters = len(set(prediction)) 

            colors=colors[:num_clusters]
            cmap = mcolors.ListedColormap(colors)
            #cmap = cm.get_cmap('tab20', num_clusters) 
            plt.figure(figsize=(10, 6))
            #ts2vec
            plt.plot(data)  # cパラメータにクラスタラベルを渡す
            data_index=df.index.tolist()
            #timesurl
            # plt.plot(train_data_np)  # cパラメータにクラスタラベルを渡す
            # data_index=train_data_df.index.tolist()
            # addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/0722_max0.3_test"
            # create_path(addpath)
            for i in range(len(data)-1):
            #for i in range(1000):
                # print(data[i])
                # print(prediction[i])
                plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
            plt.legend() 
            plt.savefig(addpath+"/"+str(win_size)+str(step)+name+'.png')
            plt.show()

            #np.save(os.path.join(out_path,fname), result)
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            min_size = min(len(groundtruth), len(prediction))

            groundtruth = groundtruth[:min_size]
            prediction = prediction[:min_size]
            score_list.append(np.array([ari, anmi, nmi]))
            ans='%s , ARI: %f, ANMI: %f, NMI: %f' %(name, ari, anmi, nmi)
            print(ans)
            with open(file_path, 'a') as file:
                file.write(ans+"\n")

        # answer_num=len(set(groundtruth))
        # prediction_s=t2s.embeddings
        # from sklearn.cluster import DBSCAN
        # import hdbscan
  
        # meanshift1=MeanShift().fit(X=prediction_s).labels_
        # #meanshift=meanshift1.predict(X=prediction_s)
        # dbscan1=DBSCAN().fit(prediction_s).labels_
        # #dbscan=dbscan1.predict(X=prediction_s)
        # hdbscan1=hdbscan.HDBSCAN().fit(prediction_s).labels_
        # #hdbscan=hdbscan1.predict(X=prediction_s)

        # km_1=KMeans(n_clusters=answer_num+1, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        # km_1_data = km_1.fit_predict(prediction_s)# クラスタの数
        # #

        # km_2=KMeans(n_clusters=answer_num, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        # km_2_data = km_2.fit_predict(prediction_s)
        # #

        # km_3=KMeans(n_clusters=answer_num-1, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        # km_3_data = km_3.fit_predict(prediction_s)
        # #

        # km_4=KMeans(n_clusters=answer_num+2, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        # km_4_data = km_4.fit_predict(prediction_s)

        # gm_1=GaussianMixture(n_components=answer_num+1,reg_covar=5e-3, init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        # gm_1_ = gm_1.fit(prediction_s)
        # gm_1_data=gm_1_.predict(prediction_s)

        # gm_2=GaussianMixture(n_components=answer_num, reg_covar=5e-3,init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        # gm_2_ = gm_2.fit(prediction_s)
        # gm_2_data=gm_2_.predict(prediction_s)

        # gm_3=GaussianMixture(n_components=answer_num-1,reg_covar=5e-3, init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        # gm_3_ = gm_3.fit(prediction_s)
        # gm_3_data=gm_3_.predict(prediction_s)

        # gm_4=GaussianMixture(n_components=answer_num+2, reg_covar=5e-3, init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
        # gm_4_ = gm_4.fit(prediction_s)
        # gm_4_data=gm_4_.predict(prediction_s)

        # dpgmm_1 = mixture.BayesianGaussianMixture(init_params='k-means++',
        #                                         n_components=answer_num+1,
        #                                         covariance_type="full",
        #                                         weight_concentration_prior_type='dirichlet_process',
        #                                         max_iter=1000).fit(prediction_s)
        # dpg_1=dpgmm_1.predict(prediction_s)

        # dpgmm_2 = mixture.BayesianGaussianMixture(init_params='k-means++',
        #                                         n_components=answer_num,
        #                                         covariance_type="full",
        #                                         weight_concentration_prior_type='dirichlet_process',
        #                                         max_iter=1000).fit(prediction_s)
        # dpg_2=dpgmm_2.predict(prediction_s) 

        # dpgmm_3 = mixture.BayesianGaussianMixture(init_params='k-means++',
        #                                         n_components=answer_num-1,
        #                                         covariance_type="full",
        #                                         weight_concentration_prior_type='dirichlet_process',
        #                                         max_iter=1000).fit(prediction_s)
        # dpg_3=dpgmm_3.predict(prediction_s)

        # dpgmm_4 = mixture.BayesianGaussianMixture(init_params='k-means++',
        #                                         n_components=answer_num+2,
        #                                         covariance_type="full",
        #                                         weight_concentration_prior_type='dirichlet_process',
        #                                         max_iter=1000).fit(prediction_s)
        # dpg_4=dpgmm_4.predict(prediction_s)

        # plot_color(meanshift1,"Meanshift",score_list_shift,data,win_size,step,colors,groundtruth)
        # plot_color(dbscan1,"dbscan",score_list_dbscan,data,win_size,step,colors,groundtruth)
        # plot_color(hdbscan1,"hdscan",score_list_hdbscan,data,win_size,step,colors,groundtruth)

        # plot_color(dpg_4,"DPGMM+2",score_list_dpgm4,data,win_size,step,colors,groundtruth)
        # plot_color(dpg_3,"DPGMM-1",score_list_dpgm3,data,win_size,step,colors,groundtruth)
        # plot_color(dpg_2,"DPGMM0",score_list_dpgm2,data,win_size,step,colors,groundtruth)
        # plot_color(dpg_1,"DPGMM+1",score_list_dpgm1,data,win_size,step,colors,groundtruth)
        # plot_color(gm_4_data,"GMM+2",score_list_gm4,data,win_size,step,colors,groundtruth)
        # plot_color(gm_3_data,"GMM-1",score_list_gm3,data,win_size,step,colors,groundtruth)
        # plot_color(gm_2_data,"GMM0",score_list_gm2,data,win_size,step,colors,groundtruth)
        # plot_color(gm_1_data,"GMM1",score_list_gm1,data,win_size,step,colors,groundtruth)
        # plot_color(km_4_data,"kmeans+2",score_list_km4,data,win_size,step,colors,groundtruth)
        # plot_color(km_3_data,"kmeans-1",score_list_km3,data,win_size,step,colors,groundtruth)
        # plot_color(km_2_data,"kmeans0",score_list_km2,data,win_size,step,colors,groundtruth)
        # plot_color(km_1_data,"kmeans1",score_list_km1,data,win_size,step,colors,groundtruth)

        
        
    score_list = np.vstack(score_list)
    ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2]))
    print(ans)
    with open(file_path, 'a') as file:
        file.write(ans+"\n")

    # score_list_shift = np.vstack(score_list_shift)
    # ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list_shift[:,0])\
    #     ,np.mean(score_list_shift[:,1])
    #     ,np.mean(score_list_shift[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list_dbscan = np.vstack(score_list_dbscan)
    # ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list_dbscan[:,0])\
    #     ,np.mean(score_list_dbscan[:,1])
    #     ,np.mean(score_list_dbscan[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list_hdbscan = np.vstack(score_list_hdbscan)
    # ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list_hdbscan[:,0])\
    #     ,np.mean(score_list_hdbscan[:,1])
    #     ,np.mean(score_list_hdbscan[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")
    # score_list = np.vstack(score_list_km1)
    # ans='KM1 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_km2)
    # ans='KM2 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_km3)
    # ans='KM3 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_km4)
    # ans='KM4 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")
    # score_list = np.vstack(score_list_gm1)
    # ans='GM1 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_gm2)
    # ans='GM2 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_gm3)
    # ans='GM3 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_gm4)
    # ans='GM4 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_dpgm1)
    # ans='DPGM1 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_dpgm2)
    # ans='DPGM2 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_dpgm3)
    # ans='DPGM3 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

    # score_list = np.vstack(score_list_dpgm4)
    # ans='DPGM4 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2]))
    # print(ans)
    # with open(file_path, 'a') as file:
    #     file.write(ans+"\n")

# def exp_on_PAMAP22(win_size, step, verbose=False):
#     out_path = os.path.join(output_path,'PAMAP2')
#     create_path(out_path)
#     params_LSE['in_channels'] = 9
#     params_LSE['win_size'] = win_size
#     params_LSE['out_channels'] = 4
#     params_LSE['M'] = 20
#     params_LSE['N'] = 4
#     params_LSE['nb_steps'] = 20
#     params_LSE['cuda'] = True
#     params_LSE['gpu'] = 0
#     params_LSE['seed']=42
#     dataset_path = os.path.join(data_path,'PAMAP2/Protocol/subject10'+str(1)+'.dat')
#     score_list = []
#     for i in range(1, 9):
#         dataset_path = os.path.join(data_path,'PAMAP2/Protocol/subject10'+str(i)+'.dat')
#         df = pd.read_csv(dataset_path, sep=' ', header=None)
#         data = df.to_numpy()
#         groundtruth = np.array(data[:,1],dtype=int)
#         hand_acc = data[:,4:7]
#         chest_acc = data[:,21:24]
#         ankle_acc = data[:,38:41]
#         data = np.hstack([hand_acc, chest_acc, ankle_acc])
#         data = fill_nan(data)
#         data = normalize(data)
#         t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
#         prediction = t2s.state_seq
#         prediction = np.array(prediction, dtype=int)
#         result = np.vstack([groundtruth, prediction])
#         ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
#         score_list.append(np.array([ari, anmi, nmi]))
#         if verbose:
#             print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(i, ari, anmi, nmi))
#     score_list = np.vstack(score_list)
#     print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
#         ,np.mean(score_list[:,1])
#         ,np.mean(score_list[:,2])))

def exp_on_USC_HAD(win_size, step, verbose=False):
    out_path = os.path.join(output_path,'USC-HAD')
    create_path(out_path)
    score_list = []
    score_list2 = []
    f_list = []
    params_LSE['in_channels'] = 6
    params_LSE['win_size'] = win_size
    params_LSE['M'] = 20
    params_LSE['N'] = 4
    params_LSE['nb_steps'] = 40
    params_LSE['kernel_size'] = 3
    params_LSE['cuda'] = True
    params_LSE['gpu'] = 2
    # params_LSE['depth'] = 12
    # params_LSE['out_channels'] = 2
    params_Triplet['in_channels'] = 6
    params_Triplet['win_size'] = win_size
    params_TNC['in_channels'] = 6
    params_TNC['win_size'] = win_size
    params_CPC['in_channels'] = 6
    params_CPC['win_size'] = win_size
    params_CPC['nb_steps'] = 10
    params_LSE['seed']=42
    train, _ = load_USC_HAD(1, 1, data_path)
    train = normalize(train)
    # t2s = Time2State_backup(win_size, step, CausalConv_CPC_Adaper(params_CPC), DPGMM(None)).fit(train, win_size, step)
    t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(train, win_size, step)
    # t2s = Time2State_backup(win_size, step, CausalConv_LSE_Adaper(params_LSE), GMM_HMM(13)).fit(train, win_size, step)
    # t2s = Time2State_backup(win_size, step, CausalConv_TNC_Adaper(params_TNC), DPGMM(None)).fit_encoder(train)
    # t2s = Time2State_backup(win_size, step, CausalConv_LSE_Adaper(params_LSE), HDP_HSMM(None)).fit(train, win_size, step)
    # t2s = Time2State_backup(win_size, step, CausalConv_Triplet_Adaper(params_Triplet), DPGMM(None)).fit_encoder(train)
    file_path=addpath+"/result.txt"
    for subject in range(1,15):
        for target in range(1,6):
            data, groundtruth = load_USC_HAD(subject, target, data_path)
            data = normalize(data)
            data2 = data
            # the true num_state is 13
            t2s.predict(data, win_size, step)
            # print(data.shape)
            # t2s = Time2State_backup(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
            prediction = t2s.state_seq
            # t2s.set_clustering_component(DPGMM(13)).predict_without_encode(data, win_size, step)
            t2s.set_clustering_component(DPGMM(13)).predict_without_encode(data, win_size, step)
            prediction2 = t2s.state_seq
            prediction = np.array(prediction, dtype=int)
            result = np.vstack([groundtruth, prediction])
            #np.save(os.path.join(out_path,'s%d_t%d'%(subject,target)), result)
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            ari2, anmi2, nmi2 = evaluate_clustering(groundtruth, prediction2)
            f1, p, r = evaluate_cut_point(groundtruth, prediction2, 500)
            score_list.append(np.array([ari, anmi, nmi]))
            score_list2.append(np.array([ari2, anmi2, nmi2]))
            f_list.append(np.array([f1, p, r]))
            # plot_mulvariate_time_series_and_label_v2(data2, groundtruth, prediction)
            # plot_mulvariate_time_series_and_label_v2(data,groundtruth,prediction)
            # plt.savefig('1.png')
            if verbose:
                print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %('s'+str(subject)+'t'+str(target), ari, anmi, nmi))
                print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %('s'+str(subject)+'t'+str(target), ari2, anmi2, nmi2))
                # print('ID: %s, F1: %f, Precision: %f, Recall: %f' %('s'+str(subject)+'t'+str(target), f1, p, r))
    score_list = np.vstack(score_list)
    score_list2 = np.vstack(score_list2)
    f_list = np.vstack(f_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list2[:,0])\
        ,np.mean(score_list2[:,1])
        ,np.mean(score_list2[:,2])))

def exp_on_USC_HAD2(win_size, step, verbose=False):
    score_list = []
    score_list2 = []
    score_list_shift = []
    score_list_dbscan = []
    score_list_hdbscan = []
    score_list_km1 = []
    score_list_km2 = []
    score_list_km3 = []
    score_list_km4 = []
    score_list_gm1 = []
    score_list_gm2 = []
    score_list_gm3 = []
    score_list_gm4 = []
    score_list_dpgm1 = []
    score_list_dpgm2 = []
    score_list_dpgm3 = []
    score_list_dpgm4 = []
    out_path = os.path.join(output_path,'USC-HAD2')
    create_path(out_path)
    params_LSE['in_channels'] = 6
    params_LSE['win_size'] = win_size
    params_LSE['M'] = 20
    params_LSE['N'] = 4
    params_LSE['nb_steps'] = 40
    params_LSE['kernel_size'] = 3
    params_LSE['cuda'] = True
    params_LSE['gpu'] = 1
    params_LSE['seed']=42
    addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/USC/0122_HSCHAD_STL"
    create_path(addpath)
    file_path=addpath+"/result.txt"
    
    for subject in range(1,15):
        for target in range(1,6):
            data, groundtruth = load_USC_HAD(subject, target, data_path)
            data = normalize(data)
            #df = convert_mat_to_df(data)
            data_index=[i for i in range(data.shape[0])]
            #print(data.shape)
            # the true num_state is 13
            saveplt=addpath+"/"+str(subject)+str(target)+str(win_size)+"plt.png"
            #t2s = Time2State(win_size, step, TS2Vec_Adaper(params_LSE), DPGMM(None)).fit(data, win_size,step,groundtruth,saveplt)
            t2s = Time2State(win_size, step, TS2Vec_Adaper(params_LSE), DPGMM(None)).fit(data, win_size,step,groundtruth,saveplt)
            #t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step,groundtruth,saveplt)
            #t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
            prediction = t2s.state_seq
            prediction = np.array(prediction, dtype=int)
            result = np.vstack([groundtruth, prediction])
            #np.save(os.path.join(out_path,'s%d_t%d'%(subject,target)), result)
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            score_list.append(np.array([ari, anmi, nmi]))
            import matplotlib.cm as cm
            from matplotlib.colors import ListedColormap
            cmap = cm.Set1
            # ラベルの一覧を取得します
            num_clusters = len(set(prediction))  # クラスタの数
            cmap = cm.get_cmap('tab20', num_clusters) 
            plt.figure(figsize=(10, 6))
            plt.plot(data)  # cパラメータにクラスタラベルを渡す
            
            
            for i in range(len(data)-1):
            #for i in range(1000):
                # print(data[i])
                # print(prediction[i])
                plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
            plt.legend() 
            plt.savefig(addpath+"/"+str(subject)+"_"+str(target)+'.png')
            plt.show()

            plt.figure(figsize=(10, 6))
            regime_list, colors=make_color(groundtruth)
            #print(regime_list)
            plt.plot(data) 
            num_clusters = len(set(regime_list))  # クラスタの数
            cmap = cm.get_cmap('tab20', num_clusters) 
            for i in range(len(regime_list)-1):
                plt.axvspan(regime_list[i], regime_list[i+1], color=colors[i], alpha=0.2)
            plt.legend() 
            plt.savefig(addpath+"/"+str(subject)+"_"+str(target)+'_answer.png')
            plt.show()
            if verbose:
                print('ID:%s,%s, ARI: %f, ANMI: %f, NMI: %f' %(subject,target, ari, anmi, nmi))
    
            def plot_color(num_clusters,name,score_list,data,win_size,step,colors,groundtruth):
                #numm=num_clusters.labels_
                #print(len(num_clusters))
                __embedding_label=reorder_label(num_clusters)
                hight = len(set(__embedding_label))
                __length = data.shape[0]
                #__length = data["x"].shape[0]
                __win_size = win_size
                __step = step
                __offset = int(win_size/2)
                # weight_vector = np.concatenate([np.linspace(0,1,self.__offset),np.linspace(1,0,self.__offset)])
                weight_vector = np.ones(shape=(2*__offset)).flatten()
                #print(weight_vector.shape)
                __state_seq = __embedding_label
                vote_matrix = np.zeros((__length,hight))
                #print(vote_matrix.shape)
                i = 0
                for l in __embedding_label:
                    vote_matrix[i:i+__win_size,l]+= weight_vector
                    i+=__step
                __state_seq = np.array([np.argmax(row) for row in vote_matrix])
                prediction = np.array(__state_seq, dtype=int)
                min_size = min(len(groundtruth), len(prediction))

                groundtruth = groundtruth[:min_size]
                prediction = prediction[:min_size]
                num_clusters = len(set(prediction)) 

                colors=colors[:num_clusters]
                cmap = mcolors.ListedColormap(colors)
                #cmap = cm.get_cmap('tab20', num_clusters) 
                plt.figure(figsize=(10, 6))
                #ts2vec
                plt.plot(data)  # cパラメータにクラスタラベルを渡す
                #data_index=df.index.tolist()
                #timesurl
                # plt.plot(train_data_np)  # cパラメータにクラスタラベルを渡す
                # data_index=train_data_df.index.tolist()
                # addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/0722_max0.3_test"
                # create_path(addpath)
                for i in range(len(data)-1):
                #for i in range(1000):
                    # print(data[i])
                    # print(prediction[i])
                    plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
                plt.legend() 
                plt.savefig(addpath+"/"+str(win_size)+str(step)+name+'.png')
                plt.show()

                #np.save(os.path.join(out_path,fname), result)
                ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
                score_list.append(np.array([ari, anmi, nmi]))
                ans='ID:%s, %s,%s, ARI: %f, ANMI: %f, NMI: %f' %(name,subject,target, ari, anmi, nmi)
                print(ans)
                with open(file_path, 'a') as file:
                    file.write(ans+"\n")

            def plotfull_color(num_clusters,name,score_list,data,win_size,step,colors,groundtruth):
                
                prediction = np.array(num_clusters, dtype=int)
                num_clusters = len(set(prediction)) 

                colors=colors[:num_clusters]
                cmap = mcolors.ListedColormap(colors)
                #cmap = cm.get_cmap('tab20', num_clusters) 
                plt.figure(figsize=(10, 6))
                #ts2vec
                plt.plot(data)  # cパラメータにクラスタラベルを渡す
                #data_index=df.index.tolist()
                #timesurl
                # plt.plot(train_data_np)  # cパラメータにクラスタラベルを渡す
                # data_index=train_data_df.index.tolist()
                # addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/0722_max0.3_test"
                # create_path(addpath)
                for i in range(len(data)-1):
                #for i in range(1000):
                    # print(data[i])
                    # print(prediction[i])
                    plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
                plt.legend() 
                plt.savefig(addpath+"/"+str(win_size)+str(step)+name+'.png')
                plt.show()

                #np.save(os.path.join(out_path,fname), result)
                ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
                min_size = min(len(groundtruth), len(prediction))

                groundtruth = groundtruth[:min_size]
                prediction = prediction[:min_size]
                score_list.append(np.array([ari, anmi, nmi]))
                ans='%s , ARI: %f, ANMI: %f, NMI: %f' %(name, ari, anmi, nmi)
                print(ans)
                with open(file_path, 'a') as file:
                    file.write(ans+"\n")

            answer_num=len(set(groundtruth))
            prediction_s=t2s.embeddings
            from sklearn.cluster import DBSCAN
            import hdbscan
    
            meanshift1=MeanShift().fit(X=prediction_s).labels_
            #meanshift=meanshift1.predict(X=prediction_s)
            dbscan1=DBSCAN().fit(prediction_s).labels_
            #dbscan=dbscan1.predict(X=prediction_s)
            hdbscan1=hdbscan.HDBSCAN().fit(prediction_s).labels_
            #hdbscan=hdbscan1.predict(X=prediction_s)

            km_1=KMeans(n_clusters=answer_num+1, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
            km_1_data = km_1.fit_predict(prediction_s)# クラスタの数
            #

            km_2=KMeans(n_clusters=answer_num, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
            km_2_data = km_2.fit_predict(prediction_s)
            #

            km_3=KMeans(n_clusters=answer_num-1, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
            km_3_data = km_3.fit_predict(prediction_s)
            #

            km_4=KMeans(n_clusters=answer_num+2, init='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
            km_4_data = km_4.fit_predict(prediction_s)

            gm_1=GaussianMixture(n_components=answer_num+1,reg_covar=5e-3, init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
            gm_1_ = gm_1.fit(prediction_s)
            gm_1_data=gm_1_.predict(prediction_s)

            gm_2=GaussianMixture(n_components=answer_num, reg_covar=5e-3,init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
            gm_2_ = gm_2.fit(prediction_s)
            gm_2_data=gm_2_.predict(prediction_s)

            gm_3=GaussianMixture(n_components=answer_num-1,reg_covar=5e-3, init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
            gm_3_ = gm_3.fit(prediction_s)
            gm_3_data=gm_3_.predict(prediction_s)

            gm_4=GaussianMixture(n_components=answer_num+2, reg_covar=5e-3, init_params='k-means++', n_init=10, max_iter=1000, tol=1e-04, random_state=0)
            gm_4_ = gm_4.fit(prediction_s)
            gm_4_data=gm_4_.predict(prediction_s)

            dpgmm_1 = mixture.BayesianGaussianMixture(init_params='k-means++',
                                                    n_components=answer_num+1,
                                                    covariance_type="full",
                                                    weight_concentration_prior_type='dirichlet_process',
                                                    max_iter=1000).fit(prediction_s)
            dpg_1=dpgmm_1.predict(prediction_s)

            dpgmm_2 = mixture.BayesianGaussianMixture(init_params='k-means++',
                                                    n_components=answer_num,
                                                    covariance_type="full",
                                                    weight_concentration_prior_type='dirichlet_process',
                                                    max_iter=1000).fit(prediction_s)
            dpg_2=dpgmm_2.predict(prediction_s) 

            dpgmm_3 = mixture.BayesianGaussianMixture(init_params='k-means++',
                                                    n_components=answer_num-1,
                                                    covariance_type="full",
                                                    weight_concentration_prior_type='dirichlet_process',
                                                    max_iter=1000).fit(prediction_s)
            dpg_3=dpgmm_3.predict(prediction_s)

            dpgmm_4 = mixture.BayesianGaussianMixture(init_params='k-means++',
                                                    n_components=answer_num+2,
                                                    covariance_type="full",
                                                    weight_concentration_prior_type='dirichlet_process',
                                                    max_iter=1000).fit(prediction_s)
            dpg_4=dpgmm_4.predict(prediction_s)

            plot_color(meanshift1,"Meanshift",score_list_shift,data,win_size,step,colors,groundtruth)
            plot_color(dbscan1,"dbscan",score_list_dbscan,data,win_size,step,colors,groundtruth)
            plot_color(hdbscan1,"hdscan",score_list_hdbscan,data,win_size,step,colors,groundtruth)

            plot_color(dpg_4,"DPGMM+2",score_list_dpgm4,data,win_size,step,colors,groundtruth)
            plot_color(dpg_3,"DPGMM-1",score_list_dpgm3,data,win_size,step,colors,groundtruth)
            plot_color(dpg_2,"DPGMM0",score_list_dpgm2,data,win_size,step,colors,groundtruth)
            plot_color(dpg_1,"DPGMM+1",score_list_dpgm1,data,win_size,step,colors,groundtruth)
            plot_color(gm_4_data,"GMM+2",score_list_gm4,data,win_size,step,colors,groundtruth)
            plot_color(gm_3_data,"GMM-1",score_list_gm3,data,win_size,step,colors,groundtruth)
            plot_color(gm_2_data,"GMM0",score_list_gm2,data,win_size,step,colors,groundtruth)
            plot_color(gm_1_data,"GMM1",score_list_gm1,data,win_size,step,colors,groundtruth)
            plot_color(km_4_data,"kmeans+2",score_list_km4,data,win_size,step,colors,groundtruth)
            plot_color(km_3_data,"kmeans-1",score_list_km3,data,win_size,step,colors,groundtruth)
            plot_color(km_2_data,"kmeans0",score_list_km2,data,win_size,step,colors,groundtruth)
            plot_color(km_1_data,"kmeans1",score_list_km1,data,win_size,step,colors,groundtruth)

            
            
        score_list = np.vstack(score_list)
        ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
            ,np.mean(score_list[:,1])
            ,np.mean(score_list[:,2]))
        print(ans)
        with open(file_path, 'a') as file:
            file.write(ans+"\n")

        score_list_shift = np.vstack(score_list_shift)
        ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list_shift[:,0])\
            ,np.mean(score_list_shift[:,1])
            ,np.mean(score_list_shift[:,2]))
        print(ans)
        with open(file_path, 'a') as file:
            file.write(ans+"\n")

        score_list_dbscan = np.vstack(score_list_dbscan)
        ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list_dbscan[:,0])\
            ,np.mean(score_list_dbscan[:,1])
            ,np.mean(score_list_dbscan[:,2]))
        print(ans)
        with open(file_path, 'a') as file:
            file.write(ans+"\n")

        score_list_hdbscan = np.vstack(score_list_hdbscan)
        ans='AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list_hdbscan[:,0])\
            ,np.mean(score_list_hdbscan[:,1])
            ,np.mean(score_list_hdbscan[:,2]))
        print(ans)
        with open(file_path, 'a') as file:
            file.write(ans+"\n")
        score_list = np.vstack(score_list_km1)
        ans='KM1 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
            ,np.mean(score_list[:,1])
            ,np.mean(score_list[:,2]))
        print(ans)
        with open(file_path, 'a') as file:
            file.write(ans+"\n")

        score_list = np.vstack(score_list_km2)
        ans='KM2 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
            ,np.mean(score_list[:,1])
            ,np.mean(score_list[:,2]))
        print(ans)
        with open(file_path, 'a') as file:
            file.write(ans+"\n")

        score_list = np.vstack(score_list_km3)
        ans='KM3 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
            ,np.mean(score_list[:,1])
            ,np.mean(score_list[:,2]))
        print(ans)
        with open(file_path, 'a') as file:
            file.write(ans+"\n")

        score_list = np.vstack(score_list_km4)
        ans='KM4 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
            ,np.mean(score_list[:,1])
            ,np.mean(score_list[:,2]))
        print(ans)
        with open(file_path, 'a') as file:
            file.write(ans+"\n")
        score_list = np.vstack(score_list_gm1)
        ans='GM1 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
            ,np.mean(score_list[:,1])
            ,np.mean(score_list[:,2]))
        print(ans)
        with open(file_path, 'a') as file:
            file.write(ans+"\n")

        score_list = np.vstack(score_list_gm2)
        ans='GM2 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
            ,np.mean(score_list[:,1])
            ,np.mean(score_list[:,2]))
        print(ans)
        with open(file_path, 'a') as file:
            file.write(ans+"\n")

        score_list = np.vstack(score_list_gm3)
        ans='GM3 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
            ,np.mean(score_list[:,1])
            ,np.mean(score_list[:,2]))
        print(ans)
        with open(file_path, 'a') as file:
            file.write(ans+"\n")

        score_list = np.vstack(score_list_gm4)
        ans='GM4 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
            ,np.mean(score_list[:,1])
            ,np.mean(score_list[:,2]))
        print(ans)
        with open(file_path, 'a') as file:
            file.write(ans+"\n")

        score_list = np.vstack(score_list_dpgm1)
        ans='DPGM1 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
            ,np.mean(score_list[:,1])
            ,np.mean(score_list[:,2]))
        print(ans)
        with open(file_path, 'a') as file:
            file.write(ans+"\n")

        score_list = np.vstack(score_list_dpgm2)
        ans='DPGM2 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
            ,np.mean(score_list[:,1])
            ,np.mean(score_list[:,2]))
        print(ans)
        with open(file_path, 'a') as file:
            file.write(ans+"\n")

        score_list = np.vstack(score_list_dpgm3)
        ans='DPGM3 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
            ,np.mean(score_list[:,1])
            ,np.mean(score_list[:,2]))
        print(ans)
        with open(file_path, 'a') as file:
            file.write(ans+"\n")

        score_list = np.vstack(score_list_dpgm4)
        ans='DPGM4 AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
            ,np.mean(score_list[:,1])
            ,np.mean(score_list[:,2]))
        print(ans)
        with open(file_path, 'a') as file:
            file.write(ans+"\n")

   

# def run_exp():
#     for win_size in [128, 256, 512]:
#         for step in [50, 100]:
#             print('window size: %d, step size: %d' %(win_size, step))
#             time_start=time.time()
#             exp_on_synthetic(win_size, step, verbose=True)
#             # exp_on_MoCap(win_size, step, verbose=True)
#             # exp_on_ActRecTut(win_size, step, verbose=True)
#             # exp_on_PAMAP2(win_size, step, verbose=True)
#             # exp_on_USC_HAD(win_size, step, verbose=True)
#             # exp_on_synthetic(beta, lambda_parameter, threshold, verbose=True)
#             time_end=time.time()
#             print('time',time_end-time_start)

if __name__ == '__main__':
    # run_exp()
    
    # for step in [5, 10, 20, 30, 40, 50, 75 , 100]:
    #for win_size in [256]:
    #         for seed in [42,43,44,45,46]:
    #             time_start=time.time()
    #exp_on_UCR_SEG(256, 50, verbose=True)
    win_size=256

    step=20
    for step in [5, 10, 25, 50 ]:
        for win_size in [128,256, 512]:
            
            #exp_on_MoCap(win_size, step, seed=42,verbose=True)
            #exp_on_synthetic(win_size, step, verbose=True)
            #exp_on_ActRecTut(win_size, step, verbose=True)
            exp_on_PAMAP2(win_size,step, verbose=True)
            #exp_on_UCR_SEG(win_size, step, verbose=True) 
            #exp_on_USC_HAD2(win_size, step, verbose=True)
    
            #time_end=time.time()
    #print('time',time_end-time_start)