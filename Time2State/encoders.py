import torch
import numpy
import sklearn
import math
import sys
import os
sys.path.append(os.path.dirname(__file__))
import utils
import networks
import losses
import math
import numpy as np
import torch
from TSpy.TSpy.label import reorder_label
from Time2State.clustering import *

def hanning_numpy(X):
    length = X.shape[2]
    weight = (1-np.cos(2*math.pi*np.arange(length)/length))/2
    # weight = np.cos(2*math.pi*np.arange(length)/length)+0.5
    return weight*X

def hanning_tensor(X):
    length = X.size(2)
    weight = (1-np.cos(2*math.pi*np.arange(length)/length))/2
    weight = torch.tensor(weight)
    return weight.cuda()*X

def diff_X(X):
    # スライシングを使用して隣接する要素間の差を計算
    diff = X[1:, :] - X[:-1, :]

    # 差の二乗を計算
    squared_diff = diff ** 2

    len_sum=len(X)

    # すべての差の二乗を合計
    sum_of_squared_diff = torch.sum(squared_diff)/len_sum
    
    return sum_of_squared_diff

def diff_loss(x):
        batch_size = x.shape[0]
        train_dataset_size = x.shape[1]
        k_cluster = 6
        lambda_kmeans = 1e-3
        #x_h = x.transpose(1, 2)
        x_torch = x

        # デバイスに移動
        device = x.device
        x_torch = x_torch.to(device)
        print(device)

        # PyTorchテンソルにrequires_grad=Trueを設定
        x_torch.requires_grad_()
        HTH = torch.matmul(x_torch, x_torch.t())

        # F_copyをデバイスに移動
        F_copy = torch.empty(batch_size, k_cluster, dtype=torch.float32, device=device)
        torch.nn.init.orthogonal_(F_copy, gain=1.0)

        # PyTorchのコード
        FTHTHF = torch.matmul(torch.matmul(F_copy.t(), HTH), F_copy)
        # print("HTH",HTH)
        # print("FTHTHF",FTHTHF)
        loss_k = torch.trace(HTH) - torch.trace(FTHTHF)
        # print(loss_k)

        return loss_k

# def hanning_numpy(X):
#     length = X.shape[2]
#     half_len = int(length/2)
#     quarter_len = int(length/4)
#     margin_weight = (1-np.cos(2*math.pi*np.arange(half_len)/half_len))/2
#     weight = np.ones((length,))
#     weight[:quarter_len] = margin_weight[:quarter_len]
#     weight[3*quarter_len:] = margin_weight[quarter_len:]
#     return weight*X

class BasicEncoder():
    def encode(self, X):
        pass

    def save(self, X):
        pass

    def load(self, X):
        pass

class CausalConv_LSE(BasicEncoder):
    def __init__(self, win_size, batch_size, nb_steps, lr,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu, M, N, win_type,seed):
        self.network = self.__create_network(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size, cuda, gpu)

        self.win_type = win_type
        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss = losses.LSE_loss.LSELoss(
            win_size, M, N, win_type,seed
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.98, -1)
        self.loss_list = []
    
    def __create_network(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        network = networks.causal_cnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )
        network.double()
        if cuda:
            network.cuda(gpu)
        return network

    def fit(self, X,win_size,step, save_memory=False, verbose=True):
        # _, dim = X.shape
        # X = numpy.transpose(numpy.array(X, dtype=float)).reshape(1, dim, -1)

        train = torch.from_numpy(X)
        if self.cuda:
            train = train.cuda(self.gpu)

        train_torch_dataset = utils.Dataset(X)
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )

        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs

        # Encoder training
        while i < self.nb_steps:
            print('Epoch: ', epochs + 1)
            if True:
                print('Epoch: ', epochs + 1)
            for batch in train_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()
                loss = self.loss(batch, self.network, win_size,step,save_memory=save_memory)
                loss.backward()
                self.optimizer.step()
                i += 1
                if i >= self.nb_steps:
                    break
            # self.scheduler.step()
            epochs += 1

        return self.network

    def encode(self, X, batch_size=500):
        """
        Outputs the representations associated to the input by the encoder.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        self.network = self.network.eval()

        count = 0
        with torch.no_grad():
            for batch in test_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                # if self.win_type=='hanning':
                #     batch = hanning_tensor(batch)
                features[
                    count * batch_size: (count + 1) * batch_size
                ] = self.network(batch).cpu()
                count += 1

        self.network = self.network.train()
        return features

    def encode_window(self, X, win_size=128, batch_size=500, window_batch_size=10000, step=10):
        """
        Outputs the representations associated to the input by the encoder,
        for each subseries of the input of the given size (sliding window
        representations).

        @param X Testing set.
        @param window Size of the sliding window.
        @param step size of the sliding window.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA.
        @param window_batch_size Size of batches of windows to compute in a
               run of encode, to save RAM.
        @param step Step length of the sliding window.
        """
        # _, dim = X.shape
        # X = numpy.transpose(numpy.array(X, dtype=float)).reshape(1, dim, -1)

        num_batch, num_channel, length = numpy.shape(X)
        num_window = int((length-win_size)/step)+1
        embeddings = numpy.empty((num_batch, self.out_channels, num_window))

        for b in range(num_batch):
            for i in range(math.ceil(num_window/window_batch_size)):
                masking = numpy.array([X[b,:,j:j+win_size] for j in range(step*i*window_batch_size, step*min((i+1)* window_batch_size, num_window),step)])
                if self.win_type=='hanning':
                    masking = hanning_numpy(masking)
                # print(masking.shape,step*i*window_batch_size, step*min((i+1)* window_batch_size, num_window))
                embeddings[b,:,i * window_batch_size: (i + 1) * window_batch_size] = numpy.swapaxes(self.encode(masking[:], batch_size=batch_size), 0, 1)
        return embeddings[0].T
    
    def set_params(self, compared_length, batch_size, nb_steps, lr,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, batch_size,
            nb_steps, lr, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
        )
        return self

class LSTM_LSE(BasicEncoder):
    def __init__(self, compared_length, nb_random_samples, negative_penalty,
                   batch_size, nb_steps, lr, penalty, early_stopping,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu, M, N):
        self.network = self.__create_network(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size, cuda, gpu)

        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.penalty = penalty
        self.early_stopping = early_stopping
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss = losses.LSE_loss.LSELoss(
            compared_length, nb_random_samples, negative_penalty, M, N
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        # self.optimizer = torch.optim.Adagrad(self.network.parameters(), lr=lr)
        # self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr=lr)
        self.loss_list = []
    
    def __create_network(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        network = networks.rnn.RnnEncoder(256, in_channels, out_channels, num_layers=2, cell_type='GRU', device='cuda', dropout=0.1)
        # network.double()
        if cuda:
            network.cuda(gpu)
        return network

    def fit(self, X, y=None, save_memory=False, verbose=False):
        """
        Trains the encoder unsupervisedly using the given training data.

        @param X Training set.
        @param y Training labels, used only for early stopping, if enabled. If
               None, disables early stopping in the method.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """

        # _, dim = X.shape
        # X = numpy.transpose(numpy.array(X, dtype=float)).reshape(1, dim, -1)

        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        train = torch.from_numpy(X)
        if self.cuda:
            train = train.cuda(self.gpu)

        if y is not None:
            nb_classes = numpy.shape(numpy.unique(y, return_counts=True)[1])[0]
            train_size = numpy.shape(X)[0]
            ratio = train_size // nb_classes

        train_torch_dataset = utils.Dataset(X)
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )

        max_score = 0
        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs
        count = 0  # Count of number of epochs without improvement
        # Will be true if, by enabling epoch_selection, a model was selected
        # using cross-validation
        found_best = False

        # Encoder training
        while i < self.nb_steps:
            if verbose:
                print('Epoch: ', epochs + 1)
            for batch in train_generator:
                # print(batch.size(2))
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()
                if not varying:
                    loss = self.loss(
                        batch, self.network, train, save_memory=save_memory
                    )
                else:
                    loss = self.loss_varying(
                        batch, self.network, train, save_memory=save_memory
                    )
                self.loss_list.append(loss.detach().cpu().numpy())
                loss.backward()
                self.optimizer.step()
                i += 1
                if i >= self.nb_steps:
                    break
            epochs += 1
            # Early stopping strategy
            if self.early_stopping is not None and y is not None and (
                ratio >= 5 and train_size >= 50
            ):
                # Computes the best regularization parameters
                features = self.encode(X)
                self.classifier = self.fit_classifier(features, y)
                # Cross validation score
                score = numpy.mean(sklearn.model_selection.cross_val_score(
                    self.classifier, features, y=y, cv=5, n_jobs=5
                ))
                count += 1
                # If the model is better than the previous one, update
                if score > max_score:
                    count = 0
                    found_best = True
                    max_score = score
                    best_encoder = type(self.network)(**self.params)
                    best_encoder.double()
                    if self.cuda:
                        best_encoder.cuda(self.gpu)
                    best_encoder.load_state_dict(self.network.state_dict())
            if count == self.early_stopping:
                break

        # If a better model was found, use it
        if found_best:
            self.encoder = best_encoder

        return self.network

    def encode(self, X, batch_size=5000):
        """
        Outputs the representations associated to the input by the encoder.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        self.network = self.network.eval()

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    features[
                        count * batch_size: (count + 1) * batch_size
                    ] = self.network(batch).cpu()
                    count += 1
            else:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    length = batch.size(2) - torch.sum(
                        torch.isnan(batch[0, 0])
                    ).data.cpu().numpy()
                    features[count: count + 1] = self.network(
                        batch[:, :, :length]
                    ).cpu()
                    count += 1

        self.network = self.network.train()
        return features

    def encode_window(self, X, win_size=128, batch_size=50, window_batch_size=1000, step=10):
        """
        Encode a time series.

        Parameters
        ----------
        X : {ndarray} of shape (n_samples, n_features).
        
        win_size : even integer.
            Size of window.
        
        batch_size : integer.
            Batch size when encoding.

        window_batch_size : integer.

        step : integer.
            Step size of sliding window.
        """
        # _, dim = X.shape
        # X = numpy.transpose(numpy.array(X, dtype=float)).reshape(1, dim, -1)

        num_batch, num_channel, length = numpy.shape(X)
        num_window = int((length-win_size)/step)+1
        embeddings = numpy.empty((num_batch, self.out_channels, num_window))
        #print(X.shape)
        for b in range(num_batch):
            for i in range(math.ceil(num_window/window_batch_size)):
                masking = numpy.array([X[b,:,j:j+win_size] for j in range(step*i*window_batch_size, step*min((i+1)* window_batch_size, num_window),step)])
                # print(masking.shape,step*i*window_batch_size, step*min((i+1)* window_batch_size, num_window))
                embeddings[b,:,i * window_batch_size: (i + 1) * window_batch_size] = numpy.swapaxes(self.encode(masking[:], batch_size=batch_size), 0, 1)
        #print(embeddings.shape)
        return embeddings[0].T
    
    def set_params(self, compared_length, nb_random_samples, negative_penalty,
                   batch_size, nb_steps, lr, penalty, early_stopping,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
        )
        return self

class CausalConv_Triplet(BasicEncoder):
    def __init__(self, compared_length, nb_random_samples, negative_penalty,
                   batch_size, nb_steps, lr, penalty, early_stopping,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.network = self.__create_network(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size, cuda, gpu)

        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.penalty = penalty
        self.early_stopping = early_stopping
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss = losses.triplet_loss.TripletLoss(
            compared_length, nb_random_samples, negative_penalty
        )
        self.loss_varying = losses.triplet_loss.TripletLossVaryingLength(
            compared_length, nb_random_samples, negative_penalty
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.loss_list = []
        self.kmeans=[]
    
    def __create_network(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        network = networks.causal_cnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )
        network.double()
        if cuda:
            network.cuda(gpu)
        return network

    def fit(self, X, y=None, save_memory=False, verbose=False):
        """
        Trains the encoder unsupervisedly using the given training data.

        @param X Training set.
        @param y Training labels, used only for early stopping, if enabled. If
               None, disables early stopping in the method.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """

        # _, dim = X.shape
        # X = numpy.transpose(numpy.array(X, dtype=float)).reshape(1, dim, -1)

        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        train = torch.from_numpy(X)
        if self.cuda:
            train = train.cuda(self.gpu)

        if y is not None:
            nb_classes = numpy.shape(numpy.unique(y, return_counts=True)[1])[0]
            train_size = numpy.shape(X)[0]
            ratio = train_size // nb_classes

        train_torch_dataset = utils.Dataset(X)
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )

        max_score = 0
        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs
        count = 0  # Count of number of epochs without improvement
        # Will be true if, by enabling epoch_selection, a model was selected
        # using cross-validation
        found_best = False

        # Encoder training
        while i < self.nb_steps:
            if verbose:
                print('Epoch: ', epochs + 1)
            for batch in train_generator:
                # print(batch.size(2))
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()
                if not varying:
                    loss = self.loss(
                        batch, self.network, train, save_memory=save_memory
                    )
                else:
                    loss = self.loss_varying(
                        batch, self.network, train, save_memory=save_memory
                    )
                self.loss_list.append(loss.detach().cpu().numpy())
                loss.backward()
                self.optimizer.step()
                i += 1
                if i >= self.nb_steps:
                    break
            epochs += 1
            # Early stopping strategy
            if self.early_stopping is not None and y is not None and (
                ratio >= 5 and train_size >= 50
            ):
                # Computes the best regularization parameters
                features = self.encode(X)
                self.classifier = self.fit_classifier(features, y)
                # Cross validation score
                score = numpy.mean(sklearn.model_selection.cross_val_score(
                    self.classifier, features, y=y, cv=5, n_jobs=5
                ))
                count += 1
                # If the model is better than the previous one, update
                if score > max_score:
                    count = 0
                    found_best = True
                    max_score = score
                    best_encoder = type(self.network)(**self.params)
                    best_encoder.double()
                    if self.cuda:
                        best_encoder.cuda(self.gpu)
                    best_encoder.load_state_dict(self.network.state_dict())
            if count == self.early_stopping:
                break

        # If a better model was found, use it
        if found_best:
            self.encoder = best_encoder

        return self.network

    def encode(self, X, batch_size=500):
        """
        Outputs the representations associated to the input by the encoder.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        self.network = self.network.eval()

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    features[
                        count * batch_size: (count + 1) * batch_size
                    ] = self.network(batch).cpu()
                    count += 1
            else:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    length = batch.size(2) - torch.sum(
                        torch.isnan(batch[0, 0])
                    ).data.cpu().numpy()
                    features[count: count + 1] = self.network(
                        batch[:, :, :length]
                    ).cpu()
                    count += 1

        self.network = self.network.train()
        return features

    def encode_window(self, X, win_size=128, batch_size=50, window_batch_size=1000, step=10):
        """
        Outputs the representations associated to the input by the encoder,
        for each subseries of the input of the given size (sliding window
        representations).

        @param X Testing set.
        @param window Size of the sliding window.
        @param step size of the sliding window.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA.
        @param window_batch_size Size of batches of windows to compute in a
               run of encode, to save RAM.
        @param step Step length of the sliding window.
        """
        # _, dim = X.shape
        # X = numpy.transpose(numpy.array(X, dtype=float)).reshape(1, dim, -1)

        num_batch, num_channel, length = numpy.shape(X)
        num_window = int((length-win_size)/step)+1
        embeddings = numpy.empty((num_batch, self.out_channels, num_window))

        for b in range(num_batch):
            for i in range(math.ceil(num_window/window_batch_size)):
                masking = numpy.array([X[b,:,j:j+win_size] for j in range(step*i*window_batch_size, step*min((i+1)* window_batch_size, num_window),step)])
                # print(masking.shape,step*i*window_batch_size, step*min((i+1)* window_batch_size, num_window))
                embeddings[b,:,i * window_batch_size: (i + 1) * window_batch_size] = numpy.swapaxes(self.encode(masking[:], batch_size=batch_size), 0, 1)
        return embeddings[0].T
    
    def set_params(self, compared_length, nb_random_samples, negative_penalty,
                   batch_size, nb_steps, lr, penalty, early_stopping,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
        )
        return self


class CausalConv_LSE(BasicEncoder):
    def __init__(self, win_size, batch_size, nb_steps, lr,
                 channels, depth, reduced_size, out_channels, kernel_size,
                 in_channels, cuda, gpu, M, N, win_type,seed):
        self.network = self.__create_network(in_channels, channels, depth, reduced_size,
                                             out_channels, kernel_size, cuda, gpu)

        self.win_type = win_type
        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.loss = losses.LSE_loss.LSELoss(
            win_size, M, N, win_type,seed
        )
        
        #katoバージョンのロス
        # self.loss = losses.LSE_loss_kato.LSELoss(
        #     win_size, M, N, win_type,seed
        # )

        # # #ここを追記
        # self.loss = losses.LSE_loss.LSELoss_kato(
        #     win_size, M, N, win_type,seed
        # )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.98, -1)
        self.loss_list = []

    def __create_network(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        network = networks.causal_cnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )

        #ここを追記
        # network = networks.causal_cnn.CausalCNNEncoder_kato(
        #     in_channels, channels, depth, reduced_size, out_channels,
        #     kernel_size
        # )
        #ここを追記
        # network = networks.causal_cnn.CausalCNNEncoder_kato2(
        #     in_channels, channels, depth, reduced_size, out_channels,
        #     kernel_size
        # )
        network.double()
        if cuda:
            print("gpu")
            network.cuda(gpu)
        return network
    
    def encode(self, X, batch_size=500):
        """
        Outputs the representations associated to the input by the encoder.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        #追記
        #features = numpy.zeros((numpy.shape(X)[0], self.out_channels*2))
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        self.network = self.network.eval()

        count = 0
        with torch.no_grad():
            for batch in test_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                # if self.win_type=='hanning':
                #     batch = hanning_tensor(batch)
                features[
                count * batch_size: (count + 1) * batch_size
                ] = self.network(batch).cpu()
                count += 1

        self.network = self.network.train()
        return features
    
    def encode_torch(self, X, batch_size=500):
        """
        Outputs the representations associated to the input by the encoder.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
            avoid out of memory errors when using CUDA. Ignored if the
            testing set contains time series of unequal lengths.
        """
        # Check if the given time series have unequal lengths
        varying = bool(torch.isnan(torch.sum(X)))

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        #out_channelを変更, 追記
        #features = torch.zeros((X.size(0), self.out_channels*2))
        features = torch.zeros((X.size(0), self.out_channels))
        self.network = self.network.eval()

        count = 0
        with torch.no_grad():
            for batch in test_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                features[
                    count * batch_size: (count + 1) * batch_size
                ] = self.network(batch).cpu()
                count += 1

        self.network = self.network.train()
        return features


    def encode_window(self, X, win_size=128, batch_size=500, window_batch_size=10000, step=10):
        """
        Outputs the representations associated to the input by the encoder,
        for each subseries of the input of the given size (sliding window
        representations).

        @param X Testing set.
        @param window Size of the sliding window.
        @param step size of the sliding window.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA.
        @param window_batch_size Size of batches of windows to compute in a
               run of encode, to save RAM.
        @param step Step length of the sliding window.
        """
        # _, dim = X.shape
        # X = numpy.transpose(numpy.array(X, dtype=float)).reshape(1, dim, -1)

        num_batch, num_channel, length = numpy.shape(X)
        num_window = int((length - win_size) / step) + 1
        #追記
        #embeddings = numpy.empty((num_batch, self.out_channels*2, num_window))
        embeddings = numpy.empty((num_batch, self.out_channels, num_window))

        for b in range(num_batch):
            for i in range(math.ceil(num_window / window_batch_size)):
                masking = numpy.array([X[b, :, j:j + win_size] for j in range(step * i * window_batch_size,
                                                                              step * min((i + 1) * window_batch_size,
                                                                                         num_window), step)])
                if self.win_type == 'hanning':
                    masking = hanning_numpy(masking)
                # print(masking.shape,step*i*window_batch_size, step*min((i+1)* window_batch_size, num_window))
                embeddings[b, :, i * window_batch_size: (i + 1) * window_batch_size] = numpy.swapaxes(
                    self.encode(masking[:], batch_size=batch_size), 0, 1)
        return embeddings[0].T
    
    def encode_window_torch(self, X, win_size=128, batch_size=500, window_batch_size=10000, step=10):
        """
        Outputs the representations associated to the input by the encoder,
        for each subseries of the input of the given size (sliding window
        representations).

        @param X Testing set.
        @param window Size of the sliding window.
        @param step size of the sliding window.
        @param batch_size Size of batches used for splitting the test data to
            avoid out of memory errors when using CUDA.
        @param window_batch_size Size of batches of windows to compute in a
            run of encode, to save RAM.
        @param step Step length of the sliding window.
        """
        num_batch, num_channel, length = X.size()
        num_window = int((length - win_size) / step) + 1
        #追記
        #embeddings = torch.empty((num_batch, self.out_channels*2, num_window))
        embeddings = torch.empty((num_batch, self.out_channels, num_window))

        for b in range(num_batch):
            for i in range(math.ceil(num_window / window_batch_size)):
                masking = torch.stack([X[b, :, j:j + win_size] for j in range(step * i * window_batch_size,
                                                                            step * min((i + 1) * window_batch_size, num_window), step)])
                if self.win_type == 'hanning':
                    masking = hanning_tensor(masking)
                embeddings[b, :, i * window_batch_size: (i + 1) * window_batch_size] = torch.swapaxes(
                    self.encode_torch(masking[:], batch_size=batch_size), 0, 1)
        
        return embeddings[0].T

    def lossalf(x):
        from sklearn.cluster import KMeans
        from scipy.optimize import linear_sum_assignment as linear_assignment
        n_clusters = 6  # クラスタの数
        #x = torch.rand(171, 2)  # 171個の2次元データ点を生成
        #y = torch.randint(0, n_clusters, (x.shape[0],))  # クラスタリングの真のラベル（テスト用）
        update_interval = 1  # 更新間隔
        batch_size = 32  # バッチサイズ
        hidden_units = 5  # モデルの隠れユニット数（例）

        # 初期化
        assignment = torch.full((x.shape[0],), -1, dtype=torch.int64)
        index_array = torch.arange(x.shape[0])
        kmeans_n_init = 100
        loss_value = 0
        index = 0

        for ite in range(int(140 * 100)):
            if ite % update_interval == 0:
                H = x.numpy()
                ans_kmeans = KMeans(n_clusters=n_clusters, n_init=kmeans_n_init).fit(H)
                kmeans_n_init = int(ans_kmeans.n_iter_ * 2)

                U = ans_kmeans.cluster_centers_
                assignment_new = ans_kmeans.labels_

                w = np.zeros((n_clusters, n_clusters), dtype=np.int64)
                for i in range(len(assignment_new)):
                    w[assignment_new[i], assignment[i]] += 1

                ind = linear_assignment(-w)
                temp = np.array(assignment)
                for i in range(n_clusters):
                    assignment[temp == ind[1][i]] = i
                n_change_assignment = np.sum(assignment_new != assignment.numpy())
                assignment = torch.tensor(assignment_new)

                S_i = []
                for i in range(n_clusters):
                    temp = H[assignment == i] - U[i]
                    temp = np.matmul(np.transpose(temp), temp)
                    S_i.append(temp)
                S_i = np.array(S_i)
                S = np.sum(S_i, 0)
                Evals, V = np.linalg.eigh(S)
                H_vt = np.matmul(H, V)  # 171,5
                U_vt = np.matmul(U, V)  # 3,5

                loss = np.round(np.mean(loss_value), 5)
                acc = None  # ここで正しい値に置き換える必要があります
                nmi = None  # ここで正しい値に置き換える必要があります

                log_str = f'iter {ite // update_interval}; acc, nmi, ri = {acc, nmi, loss}; loss:' \
                        f'{loss:.5f}; n_changed_assignment:{n_change_assignment}'
                print(log_str)

            if n_change_assignment <= len(x) * 0.005:
                print('End of training.')
                break

            idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]
            y_true = torch.tensor(H_vt[idx], dtype=torch.float32)
            temp = assignment[idx]
            y_true[:, -1] = torch.tensor(U_vt[temp.numpy(), -1], dtype=torch.float32)

            y_pred = x[idx]
            y_pred_cluster = torch.matmul(y_pred[:, :hidden_units], torch.tensor(V, dtype=torch.float32))
            loss_value = loss_fn(y_true, y_pred_cluster)
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0


    def fit(self, X, win_size,step,y=None, save_memory=False, verbose=False):
        # _, dim = X.shape
        # X = numpy.transpose(numpy.array(X, dtype=float)).reshape(1, dim, -1)

        train = torch.from_numpy(X)
        if self.cuda:
            train = train.cuda(self.gpu)

        train_torch_dataset = utils.Dataset(X)
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=False
        )


       
        
        epoch_i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs

        # Encoder training
        while epoch_i < self.nb_steps:
            #print(self.nb_steps)
            print('Epoch: ', epochs + 1)
            #if verbose:
                #print('Epoch: ', epochs + 1)
            #print("train_generator",train_generator)
            for batch in train_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()
                loss = self.loss(batch, win_size,step,self.network, save_memory=save_memory)
                #ここにkmeansロスを入れられる
                emb=self.encode_window_torch(X=batch, win_size=win_size, batch_size=500, window_batch_size=10000, step=step)
                emb=emb.to(self.gpu)
                print(self.gpu)

                #print(prediction)
                #ここからクラスタリングの結果を用いる
                X_a = emb.clone()  # .clone() メソッドを使ってテンソルをコピー
                # PyTorchテンソルをNumPy配列に変換
                X_a_numpy = X_a.cpu().numpy()
                clust=DPGMM(None)
                numm=DPGMM(None).fit(X=X_a_numpy)
                #numm=numm.labels_
                #print(len(numm))
                __embedding_label=reorder_label(numm)
                hight = len(set(__embedding_label))
                #hight=X.shape[1]
                __length = X.shape[2]
                
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


                # loss_k=diff_loss(emb)
                loss_diff=diff_X(emb)
                loss=loss+loss_diff*0.5
                # #print("emb",emb.shape,win_size,step)
                # print('loss_k',loss_k)
                print("loss_diff",loss_diff)
                # print('loss',loss)
                loss.backward()
                self.optimizer.step()
                #print("np_steps",self.nb_steps)
                if i >= self.nb_steps:
                    break
            # self.scheduler.step()
            epoch_i += 1
            #print("epoch_i",epoch_i)
            epochs += 1
            #print("epochs",epochs)

        return self.network

    

    def set_params(self, compared_length, batch_size, nb_steps, lr,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, batch_size,
            nb_steps, lr, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
        )
        return self
