'''
Created by Chengyu on 2022/2/26.
'''

import numpy as np
from TSpy.TSpy.label import reorder_label
from TSpy.TSpy.utils import calculate_scalar_velocity_list
import umap
from collections import Counter
from scipy.stats import mode
import matplotlib.colors as mcolors

class Time2State:
    def __init__(self, win_size, step, encoder, clustering_component, verbose=False):
        """
        Initialize Time2State_backup.

        Parameters
        ----------
        win_size : even integer.
            The size of sliding window.

        step : integer.
            The step size of sliding window.

        encoder_class : object.
            The instance of encoder.

        clustering_class: object.
            The instance of clustering component.
        """

        # The window size must be an even number.
        if win_size%2 != 0:
            raise ValueError('Window size must be even.')

        self.__win_size = win_size
        self.__step = step
        self.__offset = int(win_size/2)
        self.__encoder = encoder
        self.__encoder.print_pa()
        self.__clustering_component = clustering_component

    def fit(self, X, win_size, step,label,saveplt):
        """
        Fit time2state.

        Parameters
        ----------
        X : {ndarray} of shape (n_samples, n_features)

        win_size : even integer.
            The size of sliding window.

        step : integer.
            The step size of sliding window.

        Returns
        -------
        self : object
            Fitted time2state.
        """
        #X=X['x']
        self.__length = X.shape[0]
        #print(X.shape[0])
        self.fit_encoder(X,win_size,step)

        # self.__normal_encode(X, win_size, step,label,saveplt)
        # self.__normal_cluster()

        self.__encode(X, win_size, step,label,saveplt)
        self.__cluster()
        self.__assign_label()
        
        return self

    def predict(self, X, win_size, step):
        """
        Find state sequence for X.

        Parameters
        ----------
        X : {ndarray} of shape (n_samples, n_features)

        win_size : even integer.
            The size of sliding window.

        step : integer.
            The step size of sliding window.

        Returns
        -------
        self : object
            Fitted time2state.
        """
        #X=X['x']
        self.__length = X.shape[0]
        self.__step = step
        #self.__encode(X, win_size, step)
        # self.__normal_encode(X, win_size, step)
        # self.__normal_cluster()
        self.__cluster()
        self.__assign_label()
        return self

    def set_step(self, step):
        self.__step = step

    def set_clustering_component(self, clustering_obj):
        self.__clustering_component = clustering_obj
        return self

    def fit_encoder(self, X,win_size,step):
        self.__encoder.fit(X,win_size,step)
        return self

    def predict_without_encode(self, X, win_size, step):
        self.__cluster()
        self.__assign_label()
        return self

    def __encode(self, X, win_size, step,label,savename):
        #X=X['x']
        self.__embeddings = self.__encoder.encode(X, win_size, step)
        new_list=[]
        num_window = int((len(X)-win_size)/step)+1
        i=0
        from matplotlib import cm
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink']
        # 'tab20' カラーマップから色を取得
        # cmap = cm.get_cmap('tab20', 20)

        # # カラーマップから色をリストとして抽出
        # colors = [cmap(i) for i in range(20)]
        label=label.tolist()
        for _ in range(num_window):
            window = label[i:i + win_size]
            # if isinstance(window, np.ndarray):
            #     window = window.tolist()
            #print(window)
            if len(set(window))==1:
                new_list.append(window[0])
            else:
                mode1 = Counter(window).most_common(1)[0][0]
                new_list.append(mode1)
            i+=step
        #print("label",len(new_list))
        #print("X",self.__embeddings.size)
        reducer = umap.UMAP()
        vis= reducer.fit_transform(self.__embeddings)
        #print("vis",len(vis))
        vmax1=len(np.unique(new_list))
        colors=colors[:vmax1]
        cmap = mcolors.ListedColormap(colors)
        #print(vmax1)
        import matplotlib.pyplot as plt
        # 圧縮したデータをプロット
        plt.figure(figsize=(10, 8))
        scatter=plt.scatter(vis[:, 0], vis[:, 1], c=new_list, cmap=cmap,s=7,norm=mcolors.Normalize(vmin=0, vmax=vmax1))
        #scatter=plt.scatter(vis[:, 0], vis[:, 1], c=label, cmap=cmap,s=7,norm=mcolors.Normalize(vmin=0, vmax=vmax1))
        plt.gca().set_aspect('equal', 'datalim')
        cbar = plt.colorbar(scatter, ticks=range(vmax1+1)) 
        cbar.set_label('Classes')
        plt.title('visual', fontsize=24)
        plt.savefig(savename)

    def __normal_encode(self, X, win_size, step,label,savename):
        #X=X['x']
        self.__embeddings = self.__encoder.encode_normal(X, win_size, step)
        # new_list=[]
        # num_window = int((len(X)-win_size)/step)+1
        # i=0
        from matplotlib import cm
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink']
        # 'tab20' カラーマップから色を取得
        # cmap = cm.get_cmap('tab20', 20)

        # # カラーマップから色をリストとして抽出
        # colors = [cmap(i) for i in range(20)]
        label=label.tolist()
       
        # for _ in range(num_window):
        #     window = label[i:i + win_size]
        #     # if isinstance(window, np.ndarray):
        #     #     window = window.tolist()
        #     #print(window)
        #     if len(set(window))==1:
        #         new_list.append(window[0])
        #     else:
        #         mode1 = Counter(window).most_common(1)[0][0]
        #         new_list.append(mode1)
        #     i+=step
        #print("label",len(new_list))
        #print("X",self.__embeddings.size)
        reducer = umap.UMAP()
        vis= reducer.fit_transform(self.__embeddings)
        #print("vis",len(vis))
        vmax1=len(np.unique(label))
        colors=colors[:vmax1]
        cmap = mcolors.ListedColormap(colors)

         # x, y の長さを取得
        xsize = len(vis[:, 0])

        # c のサイズを調整
        if len(label) > xsize:
            label = label[:xsize]  # 余分な要素を削除
        elif len(label) < xsize:
            label = np.pad(label, (0, xsize - len(label)), mode='constant')  # 0埋め

        #print(vmax1)
        import matplotlib.pyplot as plt
        # 圧縮したデータをプロット
        plt.figure(figsize=(10, 8))
        scatter=plt.scatter(vis[:, 0], vis[:, 1], c=label, cmap=cmap,s=7,norm=mcolors.Normalize(vmin=0, vmax=vmax1))
        plt.gca().set_aspect('equal', 'datalim')
        cbar = plt.colorbar(scatter, ticks=range(vmax1+1)) 
        cbar.set_label('Classes')
        plt.title('visual', fontsize=24)
        plt.savefig(savename)

    def __cluster(self):
        self.__embedding_label = reorder_label(self.__clustering_component.fit(self.__embeddings))

    def __normal_cluster(self):
        self.__state_seq = reorder_label(self.__clustering_component.fit(self.__embeddings))

    def __assign_label(self):
        hight = len(set(self.__embedding_label))
        # weight_vector = np.concatenate([np.linspace(0,1,self.__offset),np.linspace(1,0,self.__offset)])
        weight_vector = np.ones(shape=(2*self.__offset)).flatten()
        self.__state_seq = self.__embedding_label
        vote_matrix = np.zeros((self.__length,hight))
        print("weight_vector",weight_vector.shape)
        print("vote_matrix",vote_matrix.shape)
        print("self.__embedding_label",len(self.__embedding_label))
        i = 0
        for l in self.__embedding_label:
            vote_matrix[i:i+self.__win_size,l]+= weight_vector
            i+=self.__step
        self.__state_seq = np.array([np.argmax(row) for row in vote_matrix])

        #self.__state_seq = np.array(self.__embedding_label)

    # def __assign_label(self):
    #     hight = len(set(self.__embedding_label))  # ユニークなラベルの数を取得
    #     # weight_vectorを__win_sizeに合わせて調整
    #     weight_vector = np.ones(shape=(self.__win_size)).flatten()

    #     self.__state_seq = self.__embedding_label
    #     vote_matrix = np.zeros((self.__length, hight))  # 投票行列を初期化

    #     print("weight_vector.shape:", weight_vector.shape)
    #     print("vote_matrix.shape:", vote_matrix.shape)

    #     i = 0
    #     for l in self.__embedding_label:
    #         # vote_matrixの範囲チェックを追加
    #         end_idx = min(i + self.__win_size, self.__length)
    #         segment_len = end_idx - i

    #         if segment_len < len(weight_vector):
    #             # weight_vectorを切り詰める
    #             truncated_weight_vector = weight_vector[:segment_len]
    #         else:
    #             truncated_weight_vector = weight_vector

    #         # ラベルlに対して重みを加算
    #         vote_matrix[i:end_idx, l] += truncated_weight_vector
    #         i += self.__step

    #     # 投票行列から最頻ラベルを決定
    #     self.__state_seq = np.array([np.argmax(row) for row in vote_matrix])


    def save_encoder(self):
        pass

    def load_encoder(self):
        pass

    def save_result(self, path):
        pass

    def load_result(self, path):
        pass

    def plot(self, path):
        pass

    @property
    def embeddings(self):
        return self.__embeddings

    @property
    def state_seq(self):
        return self.__state_seq
    
    @property
    def embedding_label(self):
        return self.__embedding_label

    @property
    def velocity(self):
        return self.__velocity

    @property
    def change_points(self):
        return self.__change_points