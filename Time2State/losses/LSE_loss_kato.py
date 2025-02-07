import torch
import numpy
import math
import numpy as np
import torch
import random
import torch.nn.functional as F

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
        # CUDAを使用する場合
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        #torch.cuda.manual_seed_all(seed)  # マルチGPUの場合
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms = True

def diff_X(X):
    # スライシングを使用して隣接する要素間の差を計算
    diff = X[:, 1:, :] - X[:, :-1, :]

    # 差の二乗を計算
    squared_diff = diff ** 2

    len_sum=len(X)

    # すべての差の二乗を合計
    sum_of_squared_diff = torch.sum(squared_diff)/len_sum
    
    return sum_of_squared_diff


def hanning_tensor(X):
    length = X.size(2)
    weight = (1-np.cos(2*math.pi*np.arange(length)/length))/2
    weight = torch.tensor(weight)
    return weight.cuda()*X

# ヘルパー関数を定義
def compute_features(tensor, sampling_rate=1.0):
    """
    各次元ごとに平均値、標準偏差、スペクトルエネルギー、主要周波数成分を計算する関数

    Parameters:
    tensor (torch.Tensor): 3次元テンソル (例: [batch_size, n_channels, n_samples])
    sampling_rate (float): サンプリングレート

    Returns:
    features (torch.Tensor): 計算された特徴量を含むテンソル
    """
    device = tensor.device  # テンソルのデバイスを取得
    
    # 平均値と標準偏差を計算
    mean = torch.mean(tensor, dim=2, keepdim=True)  # 平均値を次元2に沿って計算
    std = torch.std(tensor, dim=2, keepdim=True)    # 標準偏差を次元2に沿って計算
    
    # スペクトルエネルギーと主要周波数成分を保存するためのテンソルを初期化
    batch_size, n_channels, n_samples = tensor.shape
    spectral_energy = torch.zeros((batch_size, n_channels, 1), device=device)  # デバイスに配置
    dominant_frequency = torch.zeros((batch_size, n_channels, 1), device=device)  # デバイスに配置
    
    # 各バッチと各チャネルごとにスペクトル特徴を計算
    for i in range(batch_size):
        for j in range(n_channels):
            # フーリエ変換を実行
            yf = torch.fft.fft(tensor[i, j, :])
            xf = torch.fft.fftfreq(n_samples, d=1 / sampling_rate)[:n_samples // 2]
            
            # スペクトルエネルギーを計算
            spectral_energy[i, j, 0] = torch.sum(torch.abs(yf[:n_samples // 2])**2)
            
            # 主要周波数成分を特定
            dominant_frequency[i, j, 0] = xf[torch.argmax(torch.abs(yf[:n_samples // 2]))]
    
    # すべての特徴量を結合
    features = torch.cat((mean, std, spectral_energy, dominant_frequency), dim=2)
    return features

# def hanning_tensor(X):
#     length = X.shape[2]
#     half_len = int(length/2)
#     quarter_len = int(length/4)
#     margin_weight = (1-np.cos(2*math.pi*np.arange(half_len)/half_len))/2
#     weight = np.ones((length,))
#     weight[:quarter_len] = margin_weight[:quarter_len]
#     weight[3*quarter_len:] = margin_weight[quarter_len:]
#     weight = torch.tensor(weight)
#     return weight.cuda()*X

class LSELoss(torch.nn.modules.loss._Loss):
    """
    LSE loss for representations of time series.

    Parameters
    ----------
    win_size : even integer.
        Size of the sliding window.
    
    M : integer.
        Number of inter-state samples.

    N : integer.
        Number of intra-state samples.

    win_type : {'rect', 'hanning'}.
        window function.
    """

    def __init__(self, win_size, M, N, win_type,seed):
        super(LSELoss, self).__init__()
        self.win_size = win_size
        self.win_type = win_type
        self.M = M
        self.N = N
        torch_fix_seed(seed)
        # temperature parameter
        # self.tau = 1
        # self.lambda1 = 1

    
        

    def forward(self, batch, win_size,step,encoder, save_memory=False,):
        M = self.M
        N = self.N
        length_pos_neg=self.win_size

        def encoders(self, X_reshape,encoder,num,step,win_size):
            start=num*step
            out=X_reshape[:,start:start+win_size]
                
            out = out.transpose(1, 2)
            #print("out",out.shape)
            return encoder(out)
        
        total_length = batch.size(2)
        #print(total_length)
        center_list = []
        

        length = total_length
        #win_size=numpy.random.randint(50,256)
        win_size=numpy.random.randint(win_size*3/4,win_size*5/4)
        # win_size=win_size
        # step=step
        #win_size=self.win_size
        #print("win_size",win_size)
    # ＃step=numpy.random.randint(5,20)
        step=int(win_size/4)
        num_window = int((length-win_size)/step)+1
        #print("num_window",num_window)
        #use_num=min(int(num_window*0.1),500)
        use_num=min(int(num_window*0.2),100)
        #use_num=10
        print("use_num",use_num)
        # #batch=batch.squeeze(0)
        print("batch",batch.shape)

        #windowed_data = []
        feature_data=[]
        i=0
        temp_x = list()
        
        i=0
        kato=True
        normal=False
        if kato:
            X_reshape=batch.transpose(1,2)
            for k in range(num_window):
                out=X_reshape[:,i:i+win_size]
                
                out = out.transpose(1, 2)
                #windowed_data.append(encoder(out))
                #print(out.shape)
                # 各ウィンドウの平均値と標準偏差を計算
                out = compute_features(out)
                #out=compute_mean_and_std(out)
                #print(out)
                # 再度転置して元の次元に戻す
                out = out.transpose(1, 2)
                #print(out.shape)
                # リスト内包表記を使ってフラットなリストに変換
                out = out.reshape(out.shape[0],out.shape[2]*4)
                feature_data.append(out)                    
                i+=step

            # #windowed_data = np.stack(windowed_data)
            # windowed_data = torch.cat(windowed_data, dim=0)
            
            win_size_a=numpy.random.randint(50,256)
            #win_size=self.win_size
            #print("win_size",win_size)
        #   ＃step=numpy.random.randint(5,20)
            # step_a=int(win_size_a/10)
            # num_window_a = int((length-win_size_a)/step_a)+1
            # #print("num_window",num_window)
            # #use_num=min(int(num_window*0.1),500)
            # use_num_a=min(int(num_window_a*0.2),100)

            # #windowed_data = []
            # feature_data_a=[]

            # i=0
            # X_reshape_a=batch.transpose(1,2)
            # for k in range(num_window_a):
            #     out_a=X_reshape_a[:,i:i+win_size_a]
                
            #     out_a = out_a.transpose(1, 2)
            #     #windowed_data.append(encoder(out))
            #     #print(out.shape)
            #     # 各ウィンドウの平均値と標準偏差を計算
            #     out_a= compute_features(out_a)
            #     #out=compute_mean_and_std(out)
            #     #print(out)
            #     # 再度転置して元の次元に戻す
            #     out_a = out_a.transpose(1, 2)
            #     #print(out.shape)
            #     # リスト内包表記を使ってフラットなリストに変換
            #     out_a = out_a.reshape(out_a.shape[0],out_a.shape[2]*4)
            #     feature_data_a.append(out_a)                    
            #     i+=step_a

            # 差の二乗和を計算する関数
            def squared_diff_sum(x, y):
                return torch.sum((x - y) ** 2)
            
            #print(feature_data[0].shape)
            min_diffs=[]
            max_diffs=[]
            max_diffs_2=[]
            #conf_diffs=[]
            diffs=[]
            
            num_x=random.sample(range(num_window), use_num)
            #num_x_a=random.sample(range(num_window_a), use_num_a)
            #print(num_x)
            
            # for i in num_x_a:
            #     conf_diffs_a=[]
            #     for j in range(0,num_window_a):
            #         if i==j:
            #             continue
            #         else:
            #             diff = squared_diff_sum(torch.flatten(feature_data_a[i]), torch.flatten(feature_data_a[j]))
            #             conf_diffs_a.append(((i, j), diff))
            #     #print(i)
            #     conf_diffs_a.sort(key=lambda x: x[1]) 
            #     mnas=int(len(conf_diffs_a)*0.5)
            #     #print("mnas",mnas)
            #     if 10>len(conf_diffs_a):
            #         for i in range(int(len(conf_diffs_a)*0.5)):
            #             max_diffs_2.append(conf_diffs_a[mnas+i])
                
            #     else:
            #         for i in range(10):
            #             max_diffs_2.append(conf_diffs_a[mnas+i])

            for i in num_x:
                conf_diffs=[]
                for j in range(0,num_window):
                    if i==j:
                        continue
                    else:
                        diff = squared_diff_sum(torch.flatten(feature_data[i]), torch.flatten(feature_data[j]))
                        conf_diffs.append(((i, j), diff))
                #print(i)
                conf_diffs.sort(key=lambda x: x[1]) 
                mnas=int(len(conf_diffs)*0.3)
                # print("len(conf_diffs)",len(conf_diffs))
                # print("mnas",mnas)
                if 10+mnas>len(conf_diffs):
                    for i in range(int(len(conf_diffs)*0.5)):
                        min_diffs.append(conf_diffs[i])
                        max_diffs.append(conf_diffs[-i-1])
                
                else:
                    for i in range(10):
                    
                        min_diffs.append(conf_diffs[i])
                        #max_diffs.append(conf_diffs[mnas+i])
                        max_diffs.append(conf_diffs[-i-1])
            # print(min_diffs)
            # print(max_diffs)

            loss_feature_min=0
            loss_feature_min_t=0
            #print(use_num)

            #for pair in min_pairs:
            for pair in min_diffs:
                pair_i,pair_j=pair[0][0],pair[0][1]
                windowed_data_1=encoders(self, X_reshape,encoder,pair_i,step,win_size)
                windowed_data_2=encoders(self, X_reshape,encoder,pair_j,step,win_size)
                # print("windowed_data")
                #print(windowed_data_1.size())
                # print(windowed_data_1.size(1))
                # for i in range(windowed_data[pair_i].size(0)):
                #     for j in range(windowed_data[pair_j].size(0)):
                #         size_representation=windowed_data[pair[0][0]].size(1)
                #         loss_feature_min += -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
                #             windowed_data[pair_i][i].view(1, 1, size_representation),windowed_data[pair_j][j].view(1, size_representation, 1))))

                for i in range(windowed_data_1.size(0)):
                        for j in range(windowed_data_2.size(0)):
                            size_representation=windowed_data_1.size(1)
                            loss_feature_min += -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
                                windowed_data_1[i].view(1, 1, size_representation),windowed_data_2[j].view(1, size_representation, 1))))

            loss_feature_min_t=loss_feature_min/(windowed_data_1.size(0)*windowed_data_2.size(0)*len(min_diffs))
            print("loss_pos",loss_feature_min_t)

            loss_feature_max=0
            loss_feature_max_t=0
            #for pair in max_pairs:
            for pair in max_diffs:
                #print(windowed_data[pair[0][0]].size())
                pair_i,pair_j=pair[0][0],pair[0][1]
                windowed_data_1=encoders(self, X_reshape,encoder,pair_i,step,win_size)
                windowed_data_2=encoders(self, X_reshape,encoder,pair_j,step,win_size)
                #print(pair_i,pair_j)
                for i in range(windowed_data_1.size(0)):
                    for j in range(windowed_data_2.size(0)):
                        size_representation=windowed_data_1.size(1)
                        # loss_feature_max += -torch.mean(torch.nn.functional.logsigmoid(-torch.bmm(
                        #     windowed_data[pair_i][i].view(1, 1, size_representation),windowed_data[pair_j][j].view(1, size_representation, 1))))
                        loss_feature_max += -torch.mean(torch.nn.functional.logsigmoid(-torch.bmm(
                            windowed_data_1[i].view(1, 1, size_representation),windowed_data_2[j].view(1, size_representation, 1))))
            
            loss_feature_max_t=loss_feature_max/(windowed_data_1.size(0)*windowed_data_2.size(0)*len(max_diffs))
            print("loss_neg",loss_feature_max_t)

            # # loss_feature_max=0
            # # loss_feature_max_t=0
            # # for pair in max_pairs:
            # # for pair in max_diffs_2:
            # #     print(windowed_data[pair[0][0]].size())
            # #     pair_i,pair_j=pair[0][0],pair[0][1]
            # #     windowed_data_1=encoders(self, X_reshape_a,encoder,pair_i,step_a,win_size_a)
            # #     windowed_data_2=encoders(self, X_reshape_a,encoder,pair_j,step_a,win_size_a)
            # #     print(pair_i,pair_j)
            # #     for i in range(windowed_data_1.size(0)):
            # #         for j in range(windowed_data_2.size(0)):
            # #             size_representation=windowed_data_1.size(1)
            # #             loss_feature_max += -torch.mean(torch.nn.functional.logsigmoid(-torch.bmm(
            # #                 windowed_data[pair_i][i].view(1, 1, size_representation),windowed_data[pair_j][j].view(1, size_representation, 1))))
            # #             loss_feature_max += -torch.mean(torch.nn.functional.logsigmoid(-torch.bmm(
            # #                 windowed_data_1[i].view(1, 1, size_representation),windowed_data_2[j].view(1, size_representation, 1))))
            
            # # loss_feature_max_t_2=loss_feature_max/(windowed_data_1.size(0)*windowed_data_2.size(0)*len(max_diffs))
            # # print("loss_neg",loss_feature_max_t_2)
        if normal:
            loss1 = 0
            for i in range(M):
                random_pos = numpy.random.randint(0, high=total_length - length_pos_neg*2 + 1, size=1)
                #print("random_pos",random_pos)
                rand_samples = [batch[0,:, i: i+length_pos_neg] for i in range(random_pos[0],random_pos[0]+N)]
                #print(rand_samples)
                if self.win_type == 'hanning':
                    embeddings = encoder(hanning_tensor(torch.stack(rand_samples)))
                else:
                    embeddings = encoder(torch.stack(rand_samples))
                # print("emb")
                # print(embeddings.shape)
                size_representation = embeddings.size(1)
                #print(N)
                for i in range(N):
                    for j in range(N):
                        if j<=i:
                            continue
                        else:
                            # print(embeddings[i].shape)
                            # print(embeddings[i].view(1, 1, size_representation).shape)
                            # loss1 += -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
                            #     embeddings[i].view(1, 1, size_representation),
                            #     embeddings[j].view(1, size_representation, 1))/self.tau))
                            loss1 += -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
                                embeddings[i].view(1, 1, size_representation),
                                embeddings[j].view(1, size_representation, 1))))
                center = torch.mean(embeddings, dim=0)
                center_list.append(center)
            
            loss2=0
            for i in range(M):
                for j in range(M):
                    if j<=i:
                        continue
                    # loss2 += -torch.mean(torch.nn.functional.logsigmoid(-torch.bmm(
                    #     center_list[i].view(1, 1, size_representation),
                    #     center_list[j].view(1, size_representation, 1))/self.tau))
                    loss2 += -torch.mean(torch.nn.functional.logsigmoid(-torch.bmm(
                        center_list[i].view(1, 1, size_representation),
                        center_list[j].view(1, size_representation, 1))))

        #loss = loss1/(M*N*(N-1)/2) + loss2/(M*(M-1)/2)
        #loss =loss_feature_min_t+ loss2/(M*(M-1)/2)
       #loss= loss1/(M*N*(N-1)/2)+loss_feature_max_t
        # # loss = loss2/(M*(M-1)/2)
        # loss1_t=loss1/(M*N*(N-1)/2)
        # loss2_t=loss2/(M*(M-1)/2)
        # print("loss1",loss1_t)
        # print("loss2",loss2_t)
        loss=loss_feature_max_t+loss_feature_min_t
        print("loss",loss)
        #
        #loss=loss_feature_max_t+loss_feature_min_t
        #+loss_feature_max_t_2*0.75
        #loss=(loss_feature_max_t+loss_feature_min_t)* 0.5+ (loss1_t+ loss2_t)*0.5
        #loss=loss1_t+loss2_t
        return loss
    

class LSELoss_kato(torch.nn.modules.loss._Loss):
    """
    LSE loss for representations of time series.

    Parameters
    ----------
    win_size : even integer.
        Size of the sliding window.
    
    M : integer.
        Number of inter-state samples.

    N : integer.
        Number of intra-state samples.

    win_type : {'rect', 'hanning'}.
        window function.
    """

    def __init__(self, win_size, M, N, win_type,seed):
        super(LSELoss_kato, self).__init__()
        self.win_size = win_size
        self.win_type = win_type
        self.M = M
        self.N = N
        torch_fix_seed(seed)
        # temperature parameter
        # self.tau = 1
        # self.lambda1 = 1

    
        

    def forward(self, batch, win_size,step,encoder, save_memory=False,):
        M = self.M
        N = self.N
        length_pos_neg=self.win_size

        def encoders(self, X_reshape,encoder,num,step,win_size):
            start=num*step
            out=X_reshape[:,start:start+win_size]
                
            out = out.transpose(1, 2)
            #print("out",out.shape)
            return encoder(out)
        
        total_length = batch.size(2)
        #print(total_length)
        center_list = []
        

        length = total_length
        #win_size=numpy.random.randint(50,256)
        #win_size=numpy.random.randint(512,1024)
        #win_size=win_size
        win_size=numpy.random.randint(200,300)
        step=int(win_size/4)
        #win_size=self.win_size
        #print("win_size",win_size)
    # ＃step=numpy.random.randint(5,20)
        #step=int(win_size/6)
        num_window = int((length-win_size)/step)+1
        #print("num_window",num_window)
        #use_num=min(int(num_window*0.1),500)
        #use_num=min(int(num_window*0.4),100)
        use_num=10
        print("use_num",use_num)
        # #batch=batch.squeeze(0)
        #print("batch",batch.shape)

        #windowed_data = []
        feature_data=[]
        i=0
        temp_x = list()
        
        i=0
        kato=True
        if kato:
            X_reshape=batch.transpose(1,2)
            for k in range(num_window):
                out=X_reshape[:,i:i+win_size]
                
                out = out.transpose(1, 2)
                #windowed_data.append(encoder(out))
                #print(out.shape)
                # 各ウィンドウの平均値と標準偏差を計算
                out = compute_features(out)
                #out=compute_mean_and_std(out)
                #print(out)
                # 再度転置して元の次元に戻す
                out = out.transpose(1, 2)
                #print(out.shape)
                # リスト内包表記を使ってフラットなリストに変換
                out = out.reshape(out.shape[0],out.shape[2]*4)
                feature_data.append(out)                    
                i+=step

            # #windowed_data = np.stack(windowed_data)
            # windowed_data = torch.cat(windowed_data, dim=0)
            
            #win_size_a=numpy.random.randint(50,256)
            #win_size=self.win_size
            #print("win_size",win_size)
        #   ＃step=numpy.random.randint(5,20)
            # step_a=int(win_size_a/10)
            # num_window_a = int((length-win_size_a)/step_a)+1
            # #print("num_window",num_window)
            # #use_num=min(int(num_window*0.1),500)
            # use_num_a=min(int(num_window_a*0.2),100)

            # #windowed_data = []
            # feature_data_a=[]

            # i=0
            # X_reshape_a=batch.transpose(1,2)
            # for k in range(num_window_a):
            #     out_a=X_reshape_a[:,i:i+win_size_a]
                
            #     out_a = out_a.transpose(1, 2)
            #     #windowed_data.append(encoder(out))
            #     #print(out.shape)
            #     # 各ウィンドウの平均値と標準偏差を計算
            #     out_a= compute_features(out_a)
            #     #out=compute_mean_and_std(out)
            #     #print(out)
            #     # 再度転置して元の次元に戻す
            #     out_a = out_a.transpose(1, 2)
            #     #print(out.shape)
            #     # リスト内包表記を使ってフラットなリストに変換
            #     out_a = out_a.reshape(out_a.shape[0],out_a.shape[2]*4)
            #     feature_data_a.append(out_a)                    
            #     i+=step_a

            # 差の二乗和を計算する関数
            def squared_diff_sum(x, y):
                return torch.sum((x - y) ** 2)
            
            #print(feature_data[0].shape)
            min_diffs=[]
            max_diffs=[]
            max_diffs_2=[]
            #conf_diffs=[]
            diffs=[]
            
            num_x=random.sample(range(num_window), use_num)
            #num_x_a=random.sample(range(num_window_a), use_num_a)
            #print(num_x)
            
            # for i in num_x_a:
            #     conf_diffs_a=[]
            #     for j in range(0,num_window_a):
            #         if i==j:
            #             continue
            #         else:
            #             diff = squared_diff_sum(torch.flatten(feature_data_a[i]), torch.flatten(feature_data_a[j]))
            #             conf_diffs_a.append(((i, j), diff))
            #     #print(i)
            #     conf_diffs_a.sort(key=lambda x: x[1]) 
            #     mnas=int(len(conf_diffs_a)*0.5)
            #     #print("mnas",mnas)
            #     if 10>len(conf_diffs_a):
            #         for i in range(int(len(conf_diffs_a)*0.5)):
            #             max_diffs_2.append(conf_diffs_a[mnas+i])
                
            #     else:
            #         for i in range(10):
            #             max_diffs_2.append(conf_diffs_a[mnas+i])

            for i in num_x:
                conf_diffs=[]
                for j in range(0,num_window):
                    if i==j:
                        continue
                    else:
                        diff = squared_diff_sum(torch.flatten(feature_data[i]), torch.flatten(feature_data[j]))
                        conf_diffs.append(((i, j), diff))
                #print(i)
                conf_diffs.sort(key=lambda x: x[1]) 
                mnas=int(len(conf_diffs)*0.3)
                print("len(conf_diffs)",len(conf_diffs))
                if 10>len(conf_diffs):
                    for i in range(int(len(conf_diffs)*0.5)):
                        min_diffs.append(conf_diffs[i])
                        max_diffs.append(conf_diffs[-i-1])
                
                else:
                    for i in range(10):
                    
                        min_diffs.append(conf_diffs[i])
                        max_diffs.append(conf_diffs[mnas+i])
                        #max_diffs.append(conf_diffs[-i-1])
            # print(min_diffs)
            # print(max_diffs)

            loss_feature_min=0
            loss_feature_min_t=0
            #print(use_num)

            pos_loss = 0
            total_loss=0
            #for pair in min_pairs:
            for pair in min_diffs:
                i=0
                total_loss_t=0
                # pair_i,pair_j,pair_z,pair_z_1=pair[0][0],pair[0][1],max_diffs[i][0][1],max_diffs[i][0][0]
                pair_i,pair_j=pair[0][0],pair[0][1]
                # print(pair_i)
                # print(pair_z_1)
                windowed_data_1=encoders(self, X_reshape,encoder,pair_i,step,win_size)
                windowed_data_2=encoders(self, X_reshape,encoder,pair_j,step,win_size)
                #windowed_data_3=encoders(self, X_reshape,encoder,pair_z,step,win_size)

                bmm_results = torch.mm(windowed_data_1, windowed_data_2.t())
                # シグモイド関数を適用し、損失を計算
                loss_feature_min += -torch.mean(torch.nn.functional.logsigmoid(bmm_results))
            loss_feature_min_t=loss_feature_min/len(min_diffs)

            loss_feature_max=0
            loss_feature_max_t=0
            #for pair in max_pairs:
            for pair in max_diffs:
                #print(windowed_data[pair[0][0]].size())
                pair_i,pair_j=pair[0][0],pair[0][1]
                windowed_data_1=encoders(self, X_reshape,encoder,pair_i,step,win_size)
                windowed_data_2=encoders(self, X_reshape,encoder,pair_j,step,win_size)
                bmm_results = torch.mm(windowed_data_1, windowed_data_2.t())

                # シグモイド関数と損失計算
                loss_feature_max += -torch.mean(torch.nn.functional.logsigmoid(-bmm_results))

            loss_feature_max_t=loss_feature_max/len(max_diffs)
            #loss_feature_max_t=loss_feature_max/(data1.size(0)*data2.size(0)*len(max_diffs))
            #loss_feature_max_t=loss_feature_max/(data1_waru3.size(0)*data2_waru3.size(0)*len(max_diffs))

            print("loss_feature_min",loss_feature_min_t)
            print("loss_feature_max",loss_feature_max_t)

            loss =loss_feature_max_t+loss_feature_min_t*10
        return loss



            #     #print(windowed_data_1.size())
            #     B, C, T = windowed_data_1.size()
            #     i+=1

                
                
            #     for c in range(C):

            #         # windowed_data_1_f = windowed_data_1[:, c, :].view(B, -1)
            #         # windowed_data_2_f = windowed_data_2[:, c, :].view(B, -1)
            #         # windowed_data_3_f = windowed_data_3[:, c, :].view(B, -1)
                    
            #         # # Positive similarity
            #         # pos_sim = F.cosine_similarity(windowed_data_1_f, windowed_data_2_f, dim=-1)  # (batch_size,)
                    
            #         # # Negative similarity
            #         # neg_sim = F.cosine_similarity(windowed_data_1_f.unsqueeze(1), windowed_data_3_f.unsqueeze(0), dim=2)  # (batch_size, batch_size)
                    
            #         # # Positive loss
            #         # pos_loss += -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.sum(torch.exp(neg_sim), dim=1))).mean()

            #         pos_dist = F.pairwise_distance(windowed_data_1[:, c, :].view(B, -1), windowed_data_2[:, c, :].view(B, -1), p=2)
            #         neg_dist = F.pairwise_distance(windowed_data_1[:, c, :].view(B, -1), windowed_data_3[:, c, :].view(B, -1), p=2)
            #         loss = F.relu(pos_dist - neg_dist).mean()
            #         total_loss_t += loss
            #         print("pos_loss",pos_loss)
            #     total_loss=total_loss_t / C
            #     print("total_loss",total_loss)
            #         # # 各チャネルごとに類似度を計算
            #         # pos_sim = F.cosine_similarity(windowed_data_1[:, c, :], windowed_data_2[:, c, :], dim=-1)
            #         # print("pos_sim",pos_sim)
                    
            #         # # 正のペアの損失
            #         # pos_loss += -torch.log(pos_sim)
            #         # print("pos_loss",pos_loss)
            # total_loss=total_loss/len(min_diffs)       

        #         # # 各チャネルの損失を平均
        #         # pos_loss /= C
        #     pos_loss_t/=len(min_diffs)

        #     #loss_feature_min_t=loss_feature_min/(windowed_data_1.size(0)*windowed_data_2.size(0)*len(min_diffs))
        #     print("loss_pos",pos_loss)

        #     loss_feature_max=0
        #     loss_feature_max_t=0
        #     #for pair in max_pairs:
        #     neg_loss = 0
        #     for pair in max_diffs:
        #         #print(windowed_data[pair[0][0]].size())
        #         pair_i,pair_j=pair[0][0],pair[0][1]
        #         windowed_data_1=encoders(self, X_reshape,encoder,pair_i,step,win_size)
        #         windowed_data_2=encoders(self, X_reshape,encoder,pair_j,step,win_size)
                
        #         B, C, T = windowed_data_1.size()

                
                
        #         for c in range(C):
        #             # 各チャネルごとに類似度を計算
        #             neg_sim = F.cosine_similarity(windowed_data_1[:, c, :], windowed_data_2[:, c, :], dim=-1)
                    
                    
        #             # 正のペアの損失
        #             neg_loss += -torch.log(1 - neg_sim + 1e-8).mean()
                    

        #         # 各チャネルの損失を平均
        #         neg_loss /= C
        #     neg_loss/=len(min_diffs)
        #     print("neg_loss",neg_loss)
           
        # loss=pos_loss+neg_loss
       # print("")
        #
        #loss=loss_feature_max_t+loss_feature_min_t
        #+loss_feature_max_t_2*0.75
        #loss=(loss_feature_max_t+loss_feature_min_t)* 0.5+ (loss1_t+ loss2_t)*0.5
        #loss=loss1_t+loss2_t
        # return total_loss