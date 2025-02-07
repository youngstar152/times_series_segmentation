# from ts2vec import TS2Vec
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import sys
import os
sys.path.append('./')
# from ts2vec_kato.ts2vec import TS2Vec
#from src.timesurl import TimesURL 
from TimesURL.src import timesurl
print(dir(timesurl))
#print(dir(timesurl))
from Time2State.time2stateUrl import Time2State
from Time2State.clustering import *

from TSpy.TSpy.utils import *

from TSpy.TSpy.label import *
from TSpy.TSpy.eval import *
from TSpy.TSpy.dataset import *
import matplotlib.colors as mcolors
from TSpy.TSpy.label import reorder_label
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
import hdbscan
import matplotlib.pyplot as plt
import TimesURL.src.datautils
from TimesURL.src.models.encoder import TSEncoder

import warnings
warnings.filterwarnings("ignore")

import os

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

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

script_path = os.path.dirname(__file__)
# data_path = os.path.join(script_path, '../data/')
data_path = "/opt/home/tsubasa.kato/E2Usd/data/"
output_path ="/opt/home/tsubasa.kato/E2Usd/data0621/"
#save_path="/opt/home/tsubasa.kato/E2Usd/experi_512_meanstd/"
save_path="/opt/home/tsubasa.kato/E2Usd/result_0708_Mocap_100/"

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
    num = [data[i] for i in regime_list[:-1]]
    colors = []
    for value in num:
        if value == 1.0:
            colors.append("r")
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

params_TS2Vec = {
    'input_dim' : 4,
    'output_dim' : 4,
    'win_size' : 256,
}
import argparse
class TimesUrl_Adaper(BasicEncoderClass):
    def _set_parmas(self, params):
        input_dim = params['input_dim']
        output_dim = params['output_dim']
        self.win_size = params['win_size']
        # parser = argparse.ArgumentParser()
        # parser.add_argument('dataset', help='The dataset name')
        # parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
        # parser.add_argument('--loader', type=str, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
        # parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
        # parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
        # parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate (defaults to 0.001)')
        # parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
        # parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
        # parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
        # parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
        # parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
        # parser.add_argument('--seed', type=int, default=None, help='The random seed')
        # parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
        # parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
        # parser.add_argument('--sgd', action="store_true", help='Whether to perform evaluation after training')
        # parser.add_argument('--load_tp', action="store_true", help='Whether to perform evaluation after training')
        # parser.add_argument('--temp', type=float, default=1.0,)
        # parser.add_argument('--lmd', type=float, default=0.01, )
        # parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
        # parser.add_argument('--segment_num', type=int, default=3,
        #                     help='number of time interval segment to mask, default: 3 time intervals')
        # parser.add_argument('--mask_ratio_per_seg', type=float, default=0.05,
        #                     help='fraction of the sequence length to mask for each time interval, deafult: 0.05 * seq_len to be masked for each of the time interval')
        # args = parser.parse_args()
        # args.load_tp = True
        self.encoder = timesurl.TimesURL(input_dim, output_dims=output_dim,max_train_length=10000)
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
        self.encoder.fit(data, win_size=75,n_iters=5,verbose=True)

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
            # _, dim = X.shape
            # X = np.transpose(np.array(X[:,:], dtype=float)).reshape(1, dim, -1)
            # X = all_normalize(X)
            # print(X.shape)

            #ここが部分列の処理
            # embeddings = self.encoder.encode_window(X["x"], win_size=win_size, step=step)
            # print(embeddings.shape)
            # out = self.encoder.encode(embeddings, encoding_window='full_series')

            out = self.encoder.encode(X, encoding_window='full_series')
            return out
            #return embeddings
        

class TS2Vec_Adaper(BasicEncoderClass):
    def _set_parmas(self, params):
        input_dim = params['input_dim']
        output_dim = params['output_dim']
        self.win_size = params['win_size']
        #self.encoder = TSEncoder(input_dim, output_dims=output_dim,max_train_length=10000)
        self.encoder = TSEncoder(input_dim, output_dims=output_dim)
        
    def print_pa(self):
        print("start")
        # for key, value in self.hyperparameters.items():
        #     print(f"{key}: {value}")

    def fit(self, X,win_size,step):
        data = X
        print(data.shape)
        # 次元を追加して (1, 200, 4) に変換
        data = np.expand_dims(data, axis=0)
        

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
        self.encoder.fit(data, n_iters=20,verbose=True)

    def encode(self, X, win_size, step):
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
        return out

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
def generate_mask_kato(data, p = 0.5, remain = 0):
    
    B, T, C = data.shape
    mask = np.empty_like(data)

    for b in range(B):
        ts = data[b, :, 0]
        et_num = ts[~np.isnan(ts)].size - remain
        total, num = et_num * C, round(et_num * C * p)

        while True:
            i_mask = np.zeros(total)
            i_mask[random.sample(range(total), num)] = 1
            i_mask = i_mask.reshape(et_num, C)
            if 1 not in i_mask.sum(axis = 0) and 0 not in i_mask.sum(axis = 0):
                break
            break

        i_mask = np.concatenate((i_mask, np.ones((remain, C))), axis = 0)
        mask[b, ~np.isnan(ts), :] = i_mask
        mask[b, np.isnan(ts), :] = np.nan

    # mask = np.concatenate([random.sample(range(total), num) for _ in range(B)])
    # matrix = np.zeros((B, total))
    # matrix[(np.arange(B).repeat(num), mask)] = 1.0
    # matrix = matrix.reshape(B, T, C)
    # return matrix
    return mask
from sklearn.preprocessing import StandardScaler, MinMaxScaler
def load_kato(dataset, load_tp: bool = False):
    
    train_data_df =  pd.read_csv(dataset, sep=' ',usecols=range(0,4))
    train_data_np = train_data_df.to_numpy()
    train_data = train_data_np.reshape(1, train_data_np.shape[0], train_data_np.shape[1])
    print("ここ")
    print(train_data.shape)
       

    p = 1
    mask_tr = generate_mask_kato(train_data, p)
    # scaler = MinMaxScaler()
    scaler = StandardScaler()

    train_X = normalize_with_mask_kato(train_data, mask_tr, scaler)

    if load_tp:
        tp = np.linspace(0, 1, train_X.shape[1], endpoint=True).reshape(1, -1, 1)
        train_X = np.concatenate((train_X, np.repeat(tp, train_X.shape[0], axis=0)), axis=-1)
        #test_X = np.concatenate((test_X, np.repeat(tp, test_X.shape[0], axis=0)), axis=-1)

    # labels = np.unique(train_y)
    # transform = {k: i for i, k in enumerate(labels)}
    # train_y = np.vectorize(transform.get)(train_y)
    # test_y = np.vectorize(transform.get)(test_y)
    return {'x': train_X, 'mask': mask_tr},train_data_df,train_data_np


def exp_on_ActRecTut(win_size, step, verbose=False):
    out_path = os.path.join(output_path,'ActRecTut')
    create_path(out_path)
    score_list = []
    params_TS2Vec['win_size'] = win_size
    params_TS2Vec['input_dim'] = 10
    params_TS2Vec['output_dim'] = 4
    dir_list = ['subject1_walk', 'subject2_walk']
    for dir_name in dir_list:
        for i in range(10):
            dataset_path = os.path.join(data_path,'ActRecTut/'+dir_name+'/data.mat')
            data = scipy.io.loadmat(dataset_path)
            groundtruth = data['labels'].flatten()
            groundtruth = reorder_label(groundtruth)
            data = data['data'][:,0:10]
            data = normalize(data)

            t2s = TimesURL(win_size, step, TS2Vec_Adaper(params_TS2Vec), DPGMM(10)).fit(data, win_size, step)
            prediction = t2s.state_seq

            prediction = np.array(prediction, dtype=int)
            result = np.vstack([groundtruth, prediction])
            np.save(os.path.join(out_path,dir_name+str(i)), result)

            regime_list, colors=make_color(prediction)
            # ラベルの一覧を取得します
            labels = prediction
            #print(set(labels))
            
            import matplotlib.cm as cm
            from matplotlib.colors import ListedColormap
            data_test=data
            #data_test.shape
            # date=data_test.index
            date_list=np.arange(len(data))
            from sklearn.cluster import KMeans
            import matplotlib.cm as cm
            from matplotlib.colors import ListedColormap
            import matplotlib.pyplot as plt
            cmap = cm.Set1
            # ラベルの一覧を取得します
            num_clusters = len(set(labels))  # クラスタの数
            cmap = cm.get_cmap('tab20', num_clusters) 

            plt.figure(figsize=(10, 6))
            plt.plot(data_test,alpha=0.8)  # cパラメータにクラスタラベルを渡す
            for i in range(len(date_list)-1):
            #for i in range(1000):
                plt.axvspan(date_list[i], date_list[i+1], color=cmap(labels[i]), alpha=0.2)
            plt.legend() 
            create_path(save_path+"Act")
            plt.savefig(save_path+"Act/"+dir_name+'.png')
            plt.show()

            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            score_list.append(np.array([ari, anmi, nmi]))
            if verbose:
                print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(dir_name, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def clusterings_kato(data,Model):
    from collections import Counter
    window_size=20
    #print(data.shape)
    num_points, num_dims = data.shape
    if num_points < window_size:
        raise ValueError("ウィンドウサイズがデータ長より大きいです。")
    
    # スライディングウィンドウの作成
    windows = np.array([data[i:i + window_size].flatten() for i in range(num_points - window_size + 1)])

    # DPGMMでクラスタリング
    dpgmm = Model
    cluster_labels = dpgmm.fit_predict(windows)

    # 各データポイントのラベルを記録
    point_labels = [[] for _ in range(num_points)]
    for i, label in enumerate(cluster_labels):
        for j in range(window_size):
            point_labels[i + j].append(label)

    # 各地点で最も多いラベルを選択
    final_labels = []
    for labels in point_labels:
        if labels:
            most_common_label = Counter(labels).most_common(1)[0][0]
        else:
            most_common_label = -1  # ラベルが存在しない場合のデフォルト値
        final_labels.append(most_common_label)
    return final_labels

def exp_on_MoCap(win_size, step, verbose=False):
    # base_path = os.path.join(data_path,'MoCap/4d/')
    # base_path="/opt/home/tsubasa.kato/E2Usd/data/MoCap/4d/"
    # out_path = os.path.join(output_path,'MoCap')
    # out_path="/opt/home/tsubasa.kato/E2Usd/data/MoCap/"
    base_path = os.path.join(data_path,'MoCap/4d/')
    base_path="/opt/home/tsubasa.kato/E2Usd/data/MoCap/4d/"
    out_path = os.path.join(output_path,'MoCap')
    out_path="/opt/home/tsubasa.kato/E2Usd/data/MoCap/"
    create_path(out_path)
    score_list = []
    score_list_shift = []
    score_list_dbscan = []
    score_list_hdbscan = []
    params_TS2Vec['win_size'] = win_size
    params_TS2Vec['input_dim'] = 4
    params_TS2Vec['output_dim'] = 4
    params_TS2Vec['cuda'] = True
    params_TS2Vec['gpu'] = 0
    params_TS2Vec['seed']=42


    addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/11209_timesurl_epoch40"
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
        # df = pd.read_csv(dataset_path, sep=' ',usecols=range(0,4))
        # data = df.to_numpy()
        data,train_data_df,train_data_np=load_kato(dataset_path)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 4))
        plt.plot(train_data_np)  # cパラメータにクラスタラベルを渡す
        data_index=train_data_df.index.tolist()
        # addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/0722_max0.3_test"
        # # create_path(addpath)
        # for i in range(len(data)-1):
        # #for i in range(1000):
        #     # print(data[i])
        #     # print(prediction[i])
        #     plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
        plt.legend() 
        plt.savefig(addpath+"/"+fname+str(win_size)+'normal.png')
        
        # create_path(addpath)
        saveplt=addpath+"/"+fname+str(win_size)+"plt.png"
        n_state=dataset_info[fname]['n_segs']
        groundtruth = seg_to_label(dataset_info[fname]['label'])[:-1]
        #train_data, train_labels, test_data, test_labels = datautils.load_others(args.dataset, load_tp = args.load_tp)
        # print(data.shape)
        # t2s = Time2State_backup(win_size, step, CausalConv_Triplet_Adaper(params_Triplet), DPGMM(None)).fit(data, win_size, step)
        # t2s = Time2State_backup(win_size, step, CausalConv_CPC_Adaper(params_CPC), DPGMM(None)).fit(data, win_size, step)
        #t2s = Time2State(win_size, step, TS2Vec_Adaper(params_TS2Vec), DPGMM(None)).fit(data, win_size, step,groundtruth,saveplt)
        t2s = Time2State(win_size, step, TimesUrl_Adaper(params_TS2Vec), DPGMM(None)).fit(data, win_size, step,groundtruth,saveplt)
        # t2s = Time2State_backup(win_size, step, LSTM_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
        # t2s = Time2State_backup(win_size, step, CausalConv_LSE_Adaper(params_LSE), KMeansClustering(n_state)).fit(data, win_size, step)
        # t2s = Time2State_backup(win_size, step, CausalConv_TNC_Adaper(params_TNC), DPGMM(None)).fit(data, win_size, step)
        prediction = t2s.state_seq
        prediction = np.array(prediction, dtype=int)
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
        plt.plot(train_data_np)  # cパラメータにクラスタラベルを渡す
        data_index=train_data_df.index.tolist()
        # addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/0722_max0.3_test"
        # create_path(addpath)
        for i in range(len(train_data_np)-1):
        #for i in range(1000):
            # print(data[i])
            # print(prediction[i])
            plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
        plt.legend() 
        plt.savefig(addpath+"/"+fname+str(win_size)+'DPGMM.png')
        plt.show()
        if verbose:
            ans='ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi)
            print(ans)
            with open(file_path, 'a') as file:
                file.write(ans+"\n")


        prediction=clusterings_kato(t2s.embeddings,MeanShift())
        #print(prediction)
        #clust=MeanShift()
        # numm=MeanShift().fit(X=prediction_s)
        # numm=numm.labels_
        # print(len(numm))
        # __embedding_label=reorder_label(numm)
        # hight = len(set(__embedding_label))
        # __length = data.shape[0]
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
        #np.save(os.path.join(out_path,fname), result)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list_shift.append(np.array([ari, anmi, nmi]))

        if verbose:
            ans='MeanShift ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi)
            print(ans)
            with open(file_path, 'a') as file:
                file.write(ans+"\n")
        

        cmap = cm.Set1
        # ラベルの一覧を取得します
        num_clusters = len(set(prediction))  # クラスタの数
        cmap = cm.get_cmap('tab20', num_clusters) 
        plt.figure(figsize=(10, 6))
        plt.plot(train_data_np)  # cパラメータにクラスタラベルを渡す
        data_index=train_data_df.index.tolist()
        # addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/0722_max0.3_test"
        # create_path(addpath)
        for i in range(len(train_data_np)-1):
        #for i in range(1000):
            # print(data[i])
            # print(prediction[i])
            plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
        plt.legend() 
        plt.savefig(addpath+"/"+fname+str(win_size)+'MeanShift.png')
        plt.show()
        
        #clust=DBSCAN()
        prediction=clusterings_kato(t2s.embeddings,DBSCAN())
        # numm=DBSCAN().fit(prediction_s).labels_
        # print(len(numm))
        # __embedding_label=reorder_label(numm)
        # #__embedding_label=reorder_label(DBSCAN(None).fit(prediction))
        # hight = len(set(__embedding_label))
        # __length = train_data_np.shape[0]
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
        #np.save(os.path.join(out_path,fname), result)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        
        score_list_dbscan.append(np.array([ari, anmi, nmi]))
        #df = pd.read_csv(dataset_path, sep=' ',usecols=range(0,4))
        cmap = cm.Set1
        # ラベルの一覧を取得します
        num_clusters = len(set(prediction))  # クラスタの数
        cmap = cm.get_cmap('tab20', num_clusters) 
        plt.figure(figsize=(10, 6))
        plt.plot(train_data_np)  # cパラメータにクラスタラベルを渡す
        data_index=train_data_df.index.tolist()
        # addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/0722_max0.3_test"
        # create_path(addpath)
        for i in range(len(train_data_np)-1):
        #for i in range(1000):
            # print(data[i])
            # print(prediction[i])
            plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
        plt.legend() 
        plt.savefig(addpath+"/"+fname+str(win_size)+'DBSCAN.png')
        plt.show()
        if verbose:
            ans='DBSCAN ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi)
            print(ans)
            with open(file_path, 'a') as file:
                file.write(ans+"\n")

        #clust=hdbscan.HDBSCAN()
        prediction=clusterings_kato(t2s.embeddings,hdbscan.HDBSCAN())
        # __embedding_label=reorder_label(hdbscan.HDBSCAN().fit(prediction_s).labels_)
        # #__embedding_label=reorder_label(HDBSCAN(None).fit(prediction))
        # hight = len(set(__embedding_label))
        # __length = train_data_np.shape[0]
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
        #np.save(os.path.join(out_path,fname), result)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list_hdbscan.append(np.array([ari, anmi, nmi]))

        cmap = cm.Set1
        # ラベルの一覧を取得します
        num_clusters = len(set(prediction))  # クラスタの数
        cmap = cm.get_cmap('tab20', num_clusters) 
        plt.figure(figsize=(10, 6))
        plt.plot(train_data_np)  # cパラメータにクラスタラベルを渡す
        data_index=train_data_df.index.tolist()
        # addpath="/opt/home/tsubasa.kato/E2Usd/Baselines/Time2State/images/Mocap/0722_max0.3_test"
        # create_path(addpath)
        for i in range(len(train_data_np)-1):
        #for i in range(1000):
            # print(data[i])
            # print(prediction[i])
            plt.axvspan(data_index[i], data_index[i+1], color=cmap(prediction[i]), alpha=0.2)
        plt.legend() 
        plt.savefig(addpath+"/"+fname+str(win_size)+'HDBSCAN.png')
        plt.show()
        if verbose:
            ans='HDBSCAN ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi)
            print(ans)
            with open(file_path, 'a') as file:
                file.write(ans+"\n")

    # for fname in os.listdir(base_path):
    #     dataset_path = base_path+fname
    #     df = pd.read_csv(dataset_path, sep=' ',usecols=range(0,4))
    #     data = df.to_numpy()
    #     n_state=dataset_info[fname]['n_segs']
    #     groundtruth = seg_to_label(dataset_info[fname]['label'])[:-1]
        
    #     t2s = Time2State(win_size, step, TS2Vec_Adaper(params_TS2Vec), DPGMM(10)).fit(data, win_size, step)
    #     prediction = t2s.state_seq
    #     prediction = np.array(prediction, dtype=int)
    #     result = np.vstack([groundtruth, prediction])
    #     np.save(os.path.join(out_path,fname), result)

    #     regime_list, colors=make_color(prediction)
    #     # ラベルの一覧を取得します
    #     labels = prediction
    #     #print(set(labels))
        
    #     import matplotlib.cm as cm
    #     from matplotlib.colors import ListedColormap
    #     data_test=data
    #     #data_test.shape
    #     # date=data_test.index
    #     date_list=np.arange(len(data))
    #     from sklearn.cluster import KMeans
    #     import matplotlib.cm as cm
    #     from matplotlib.colors import ListedColormap
    #     import matplotlib.pyplot as plt
    #     cmap = cm.Set1
    #     # ラベルの一覧を取得します
    #     num_clusters = len(set(labels))  # クラスタの数
    #     cmap = cm.get_cmap('tab20', num_clusters) 

    #     plt.figure(figsize=(10, 6))
    #     plt.plot(data_test,alpha=0.8)  # cパラメータにクラスタラベルを渡す
    #     for i in range(len(date_list)-1):
    #     #for i in range(1000):
    #         plt.axvspan(date_list[i], date_list[i+1], color=cmap(labels[i]), alpha=0.2)
    #     plt.legend() 
    #     create_path(save_path+"Mocap")
    #     plt.savefig(save_path+"Mocap/"+fname+'.png')
    #     plt.show()

    #     ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
    #     score_list.append(np.array([ari, anmi, nmi]))
    #     if verbose:
    #         print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi))
    # score_list = np.vstack(score_list)

    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list_shift[:,0])\
        ,np.mean(score_list_shift[:,1])
        ,np.mean(score_list_shift[:,2])))
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list_dbscan[:,0])\
        ,np.mean(score_list_dbscan[:,1])
        ,np.mean(score_list_dbscan[:,2])))
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list_hdbscan[:,0])\
        ,np.mean(score_list_hdbscan[:,1])
        ,np.mean(score_list_hdbscan[:,2])))

def exp_on_USC_HAD2(win_size, step, verbose=False):
    score_list = []
    out_path = os.path.join(output_path,'USC-HAD')
    create_path(out_path)
    params_TS2Vec['win_size'] = win_size
    params_TS2Vec['input_dim'] = 6
    params_TS2Vec['output_dim'] = 4
    for subject in range(1,15):
        for target in range(1,6):
            data, groundtruth = load_USC_HAD(subject, target, data_path)
            data = normalize(data)
            
            t2s = Time2State(win_size, step, TS2Vec_Adaper(params_TS2Vec), DPGMM(10)).fit(data, win_size, step)
            prediction = t2s.state_seq

            prediction = np.array(prediction, dtype=int)
            result = np.vstack([groundtruth, prediction])
            np.save(os.path.join(out_path,'s%d_t%d'%(subject,target)), result)

            #regime_list, colors=make_color(prediction)
            # ラベルの一覧を取得します
            labels = prediction
            #print(set(labels))
            
            import matplotlib.cm as cm
            from matplotlib.colors import ListedColormap
            data_test=data
            #data_test.shape
            # date=data_test.index
            date_list=np.arange(len(data))
            from sklearn.cluster import KMeans
            import matplotlib.cm as cm
            from matplotlib.colors import ListedColormap
            import matplotlib.pyplot as plt
            cmap = cm.Set1
            # ラベルの一覧を取得します
            num_clusters = len(set(labels))  # クラスタの数
            cmap = cm.get_cmap('tab20', num_clusters) 

            plt.figure(figsize=(10, 6))
            plt.plot(data_test,alpha=0.8)  # cパラメータにクラスタラベルを渡す
            for i in range(len(date_list)-1):
            #for i in range(1000):
                plt.axvspan(date_list[i], date_list[i+1], color=cmap(labels[i]), alpha=0.2)
            plt.legend() 
            create_path(save_path+"USC")
            plt.savefig(save_path+"USC/"+str(subject)+"_"+str(target)+'.png')
            plt.show()

            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            score_list.append(np.array([ari, anmi, nmi]))
            if verbose:
                print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %('s'+str(subject)+'t'+str(target), ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def exp_on_UCR_SEG(win_size, step, verbose=False):
    score_list = []
    out_path = os.path.join(output_path,'UCRSEG')
    create_path(out_path)
    dataset_path = os.path.join(data_path,'UCRSEG/')
    params_TS2Vec['win_size'] = win_size
    params_TS2Vec['input_dim'] = 1
    params_TS2Vec['output_dim'] = 4
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

        t2s = Time2State(win_size, step, TS2Vec_Adaper(params_TS2Vec), DPGMM(10)).fit(data, win_size, step)
        prediction = t2s.state_seq

        groundtruth = seg_to_label(seg_info)[:-1]

        labels = prediction
            #print(set(labels))
            
        import matplotlib.cm as cm
        from matplotlib.colors import ListedColormap
        data_test=data
        #data_test.shape
        # date=data_test.index
        date_list=np.arange(len(data))
        from sklearn.cluster import KMeans
        import matplotlib.cm as cm
        from matplotlib.colors import ListedColormap
        import matplotlib.pyplot as plt
        cmap = cm.Set1
        # ラベルの一覧を取得します
        num_clusters = len(set(labels))  # クラスタの数
        cmap = cm.get_cmap('tab20', num_clusters) 

        plt.figure(figsize=(10, 6))
        plt.plot(data_test,alpha=0.8)  # cパラメータにクラスタラベルを渡す
        for i in range(len(date_list)-1):
        #for i in range(1000):
            plt.axvspan(date_list[i], date_list[i+1], color=cmap(labels[i]), alpha=0.2)
        plt.legend() 
        create_path(save_path+"UCR")
        plt.savefig(save_path+"UCR/"+fname+'.png')
        plt.show()


        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(out_path,fname[:-4]), result)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def exp_on_synthetic(win_size, step, verbose=False):
    out_path = os.path.join(output_path,'synthetic')
    create_path(out_path)
    prefix = os.path.join(data_path, 'synthetic/test')
    params_TS2Vec['win_size'] = win_size
    params_TS2Vec['input_dim'] = 4
    params_TS2Vec['output_dim'] = 4
    score_list = []
    for i in range(100):
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=range(4), skiprows=1)
        data = df.to_numpy()
        print(len(data))
        df = pd.read_csv(prefix+str(i)+'.csv', usecols=[4], skiprows=1)
        groundtruth = df.to_numpy(dtype=int).flatten()
        
        t2s = Time2State(win_size, step, TS2Vec_Adaper(params_TS2Vec), DPGMM(10)).fit(data, win_size, step)
        prediction = t2s.state_seq

        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(out_path,str(i)), result)

        labels = prediction
            #print(set(labels))
            
        import matplotlib.cm as cm
        from matplotlib.colors import ListedColormap
        data_test=data
        #data_test.shape
        # date=data_test.index
        date_list=np.arange(len(data))
        from sklearn.cluster import KMeans
        import matplotlib.cm as cm
        from matplotlib.colors import ListedColormap
        import matplotlib.pyplot as plt
        cmap = cm.Set1
        # ラベルの一覧を取得します
        num_clusters = len(set(labels))  # クラスタの数
        cmap = cm.get_cmap('tab20', num_clusters) 

        plt.figure(figsize=(10, 6))
        plt.plot(data_test,alpha=0.8)  # cパラメータにクラスタラベルを渡す
        for i in range(len(date_list)-1):
        #for i in range(1000):
            plt.axvspan(date_list[i], date_list[i+1], color=cmap(labels[i]), alpha=0.2)
        plt.legend() 
        create_path(save_path+"syn")
        plt.savefig(save_path+"syn/"+str(i)+'.png')
        plt.show()

        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(i, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

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
    dataset_path = os.path.join(data_path,'PAMAP2/Protocol/subject10'+str(1)+'.dat')
    score_list = []
    params_TS2Vec['win_size'] = win_size
    params_TS2Vec['input_dim'] = 9
    params_TS2Vec['output_dim'] = 4
    for i in range(1, 9):
        dataset_path = os.path.join(data_path,'PAMAP2/Protocol/subject10'+str(i)+'.dat')
        df = pd.read_csv(dataset_path, sep=' ', header=None)
        data = df.to_numpy()
        groundtruth = np.array(data[:,1],dtype=int)
        hand_acc = data[:,4:7]
        chest_acc = data[:,21:24]
        ankle_acc = data[:,38:41]
        data = np.hstack([hand_acc, chest_acc, ankle_acc])
        data = fill_nan(data)
        data = normalize(data)
        t2s = Time2State(win_size, step, TS2Vec_Adaper(params_TS2Vec), DPGMM(10)).fit(data, win_size, step)
        prediction = t2s.state_seq
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(out_path,'10'+str(i)), result)

        labels = prediction
            #print(set(labels))
            
        import matplotlib.cm as cm
        from matplotlib.colors import ListedColormap
        data_test=data
        #data_test.shape
        # date=data_test.index
        date_list=np.arange(len(data))
        from sklearn.cluster import KMeans
        import matplotlib.cm as cm
        from matplotlib.colors import ListedColormap
        import matplotlib.pyplot as plt
        cmap = cm.Set1
        # ラベルの一覧を取得します
        num_clusters = len(set(labels))  # クラスタの数
        cmap = cm.get_cmap('tab20', num_clusters) 

        plt.figure(figsize=(10, 6))
        plt.plot(data_test,alpha=0.8)  # cパラメータにクラスタラベルを渡す
        for i in range(len(date_list)-1):
        #for i in range(1000):
            plt.axvspan(date_list[i], date_list[i+1], color=cmap(labels[i]), alpha=0.2)
        plt.legend() 
        create_path(save_path+"PAMAP")
        plt.savefig(save_path+"PAMAP/"+'subject10'+str(i)+'.png')
        plt.show()

        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        plt.savefig('1.png')
        if verbose:
            print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(i, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

# print("Mocap")
# exp_on_MoCap(512, 10, verbose=True)
# print("UCR_SEG")
# exp_on_UCR_SEG(256, 50, verbose=True)
# print("synthetic")
# exp_on_synthetic(128, 50, verbose=True)
# print("PAMAP2")
# exp_on_PAMAP2(512, 100, verbose=True)
# print("ActRecTut")
# exp_on_ActRecTut(256, 50, verbose=True)
# print("USC_HAD2")
# exp_on_USC_HAD2(256, 50, verbose=True)

#exp_on_UCR_SEG(256, 50, verbose=True)
exp_on_MoCap(256, 50, verbose=True)
#exp_on_ActRecTut(128, 50, verbose=True)
#exp_on_PAMAP2(512,100, verbose=True)
#exp_on_synthetic(128, 50, verbose=True)
#exp_on_USC_HAD2(256, 50, verbose=True)

# window_s=256
# steps=10
# print("window_s",window_s)
# print("steps",steps)

# print("Mocap")
# exp_on_MoCap(window_s, steps, verbose=True)
# print("UCR_SEG")
# exp_on_UCR_SEG(window_s, steps, verbose=True)
# print("synthetic")
# exp_on_synthetic(window_s, steps, verbose=True)
# print("PAMAP2")
# exp_on_PAMAP2(window_s, steps, verbose=True)
# print("ActRecTut")
# exp_on_ActRecTut(window_s, steps, verbose=True)
# print("USC_HAD2")
# exp_on_USC_HAD2(window_s, steps, verbose=True)