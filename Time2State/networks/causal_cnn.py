import torch


class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.

    @param chomp_size Number of elements to remove.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).

    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param dilation Dilation parameter of non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(CausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(
            conv1, chomp1, relu1, conv2, chomp2, relu2
        )

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)


class CausalCNN(torch.nn.Module):
    """
    Causal CNN, composed of a sequence of causal convolution blocks.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).

    @param in_channels Number of input channels.
    @param channels Number of channels processed in the network and of output
           channels.
    @param depth Depth of the network.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels, depth, out_channels,
                 kernel_size):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size
            )]
            dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size
        )]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CausalCNNEncoder(torch.nn.Module):
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).

    @param in_channels Number of input channels.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels, depth, reduced_size,
                 out_channels, kernel_size):
        super(CausalCNNEncoder, self).__init__()
        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        linear = torch.nn.Linear(reduced_size, out_channels)
        self.network = torch.nn.Sequential(
            causal_cnn, reduce_size, squeeze, linear
        )

    def forward(self, x):
        return self.network(x)



# # Licensed to the Apache Software Foundation (ASF) under one
# # or more contributor license agreements.  See the NOTICE file
# # distributed with this work for additional information
# # regarding copyright ownership.  The ASF licenses this file
# # to you under the Apache License, Version 2.0 (the
# # "License"); you may not use this file except in compliance
# # with the License.  You may obtain a copy of the License at

# #   http://www.apache.org/licenses/LICENSE-2.0

# # Unless required by applicable law or agreed to in writing,
# # software distributed under the License is distributed on an
# # "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# # KIND, either express or implied.  See the License for the
# # specific language governing permissions and limitations
# # under the License.


# # Implementation of causal CNNs partly taken and modified from
# # https://github.com/locuslab/TCN/blob/master/TCN/tcn.py, originally created
# # with the following license.

# # MIT License

# # Copyright (c) 2018 CMU Locus Lab

# # Permission is hereby granted, free of charge, to any person obtaining a copy
# # of this software and associated documentation files (the "Software"), to deal
# # in the Software without restriction, including without limitation the rights
# # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of the Software, and to permit persons to whom the Software is
# # furnished to do so, subject to the following conditions:

# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.

# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.


# import torch

# import torch
# from torch import nn
# import torch.nn.functional as F
# import numpy as np

# class SamePadConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
#         super().__init__()
#         self.receptive_field = (kernel_size - 1) * dilation + 1
#         padding = self.receptive_field // 2
#         self.conv = nn.Conv1d(
#             in_channels, out_channels, kernel_size,
#             padding=padding,
#             dilation=dilation,
#             groups=groups
#         )
#         self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
#     def forward(self, x):
#         out = self.conv(x)
#         if self.remove > 0:
#             out = out[:, :, : -self.remove]
#         return out
    
# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
#         super().__init__()
#         self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
#         self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
#         self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
#     def forward(self, x):
#         residual = x if self.projector is None else self.projector(x)
#         x = F.gelu(x)
#         x = self.conv1(x)
#         x = F.gelu(x)
#         x = self.conv2(x)
#         return x + residual

# class DilatedConvEncoder(nn.Module):
#     def __init__(self, in_channels, channels, kernel_size):
#         super().__init__()
#         self.net = nn.Sequential(*[
#             ConvBlock(
#                 channels[i-1] if i > 0 else in_channels,
#                 channels[i],
#                 kernel_size=kernel_size,
#                 dilation=2**i,
#                 final=(i == len(channels)-1)
#             )
#             for i in range(len(channels))
#         ])
        
#     def forward(self, x):
#         return self.net(x)

# class Chomp1d(torch.nn.Module):
#     """
#     Removes the last elements of a time series.

#     Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
#     batch size, `C` is the number of input channels, and `L` is the length of
#     the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
#     is the number of elements to remove.

#     @param chomp_size Number of elements to remove.
#     """
#     def __init__(self, chomp_size):
#         super(Chomp1d, self).__init__()
#         self.chomp_size = chomp_size

#     def forward(self, x):
#         return x[:, :, :-self.chomp_size]


# class SqueezeChannels(torch.nn.Module):
#     """
#     Squeezes, in a three-dimensional tensor, the third dimension.
#     """
#     def __init__(self):
#         super(SqueezeChannels, self).__init__()

#     def forward(self, x):
#         return x.squeeze(2)


# class CausalConvolutionBlock(torch.nn.Module):
#     """
#     Causal convolution block, composed sequentially of two causal convolutions
#     (with leaky ReLU activation functions), and a parallel residual connection.

#     Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
#     batch size, `C` is the number of input channels, and `L` is the length of
#     the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).

#     @param in_channels Number of input channels.
#     @param out_channels Number of output channels.
#     @param kernel_size Kernel size of the applied non-residual convolutions.
#     @param dilation Dilation parameter of non-residual convolutions.
#     @param final Disables, if True, the last activation function.
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, dilation,
#                  final=False):
#         super(CausalConvolutionBlock, self).__init__()

#         # Computes left padding so that the applied convolutions are causal
#         padding = (kernel_size - 1) * dilation

#         # First causal convolution
#         conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
#             in_channels, out_channels, kernel_size,
#             padding=padding, dilation=dilation
#         ))
#         # The truncation makes the convolution causal
#         chomp1 = Chomp1d(padding)
#         relu1 = torch.nn.LeakyReLU()

#         # Second causal convolution
#         conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
#             out_channels, out_channels, kernel_size,
#             padding=padding, dilation=dilation
#         ))
#         chomp2 = Chomp1d(padding)
#         relu2 = torch.nn.LeakyReLU()

#         # Causal network
#         self.causal = torch.nn.Sequential(
#             conv1, chomp1, relu1, conv2, chomp2, relu2
#         )

#         # Residual connection
#         self.upordownsample = torch.nn.Conv1d(
#             in_channels, out_channels, 1
#         ) if in_channels != out_channels else None

#         # Final activation function
#         self.relu = torch.nn.LeakyReLU() if final else None

#     def forward(self, x):
#         out_causal = self.causal(x)
#         res = x if self.upordownsample is None else self.upordownsample(x)
#         if self.relu is None:
#             return out_causal + res
#         else:
#             return self.relu(out_causal + res)


# class CausalCNN(torch.nn.Module):
#     """
#     Causal CNN, composed of a sequence of causal convolution blocks.

#     Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
#     batch size, `C` is the number of input channels, and `L` is the length of
#     the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).

#     @param in_channels Number of input channels.
#     @param channels Number of channels processed in the network and of output
#            channels.
#     @param depth Depth of the network.
#     @param out_channels Number of output channels.
#     @param kernel_size Kernel size of the applied non-residual convolutions.
#     """
#     def __init__(self, in_channels, channels, depth, out_channels,
#                  kernel_size):
#         super(CausalCNN, self).__init__()

#         layers = []  # List of causal convolution blocks
#         dilation_size = 1  # Initial dilation size

#         for i in range(depth):
#             in_channels_block = in_channels if i == 0 else channels
#             layers += [CausalConvolutionBlock(
#                 in_channels_block, channels, kernel_size, dilation_size
#             )]
#             dilation_size *= 2  # Doubles the dilation size at each step

#         # Last layer
#         layers += [CausalConvolutionBlock(
#             channels, out_channels, kernel_size, dilation_size
#         )]

#         self.network = torch.nn.Sequential(*layers)

#     def forward(self, x):
#         return self.network(x)


# class CausalCNNEncoder(torch.nn.Module):
#     """
#     Encoder of a time series using a causal CNN: the computed representation is
#     the output of a fully connected layer applied to the output of an adaptive
#     max pooling layer applied on top of the causal CNN, which reduces the
#     length of the time series to a fixed size.

#     Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
#     batch size, `C` is the number of input channels, and `L` is the length of
#     the input. Outputs a three-dimensional tensor (`B`, `C`).

#     @param in_channels Number of input channels.
#     @param channels Number of channels manipulated in the causal CNN.
#     @param depth Depth of the causal CNN.
#     @param reduced_size Fixed length to which the output time series of the
#            causal CNN is reduced.
#     @param out_channels Number of output channels.
#     @param kernel_size Kernel size of the applied non-residual convolutions.
#     """
#     def __init__(self, in_channels, channels, depth, reduced_size,
#                  out_channels, kernel_size):
#         super(CausalCNNEncoder, self).__init__()
#         causal_cnn = CausalCNN(
#             in_channels, channels, depth, reduced_size, kernel_size
#         )
#         reduce_size = torch.nn.AdaptiveMaxPool1d(1)
#         squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
#         # Fully Connected Layer for final output
#         linear_nom=nn.Linear(reduced_size , out_channels)
#         self.linear = nn.Linear(reduced_size + out_channels, out_channels)
#         self.network = torch.nn.Sequential(
#             causal_cnn, reduce_size, squeeze
#         )
#         self.network_norm = torch.nn.Sequential(
#             causal_cnn, reduce_size, squeeze,linear_nom
#         )
#         self.input_fc = nn.Linear(in_channels, 10)

#         self.feature_extractor = DilatedConvEncoder(
#             10,
#             [10] * depth + [out_channels],
#             kernel_size=3
#         )
#         self.reduce_size_dilated = nn.AdaptiveMaxPool1d(1)
#         self.squeeze_dilated = SqueezeChannels()
#         self.period_len=4
#         self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
#                                 stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)

#     def forward(self, x):

#         # print(x.shape)
#         # x_dilated = x.transpose(1, 2)  # B x Ch x T
#         # x_dilated=self.input_fc(x_dilated)
#         # x_dilated = x_dilated.transpose(1, 2)  # B x Ch x T
#         # x_dilated=self.feature_extractor(x_dilated)
#         # x_dilated = self.reduce_size_dilated(x_dilated)
#         # x_dilated = self.squeeze_dilated(x_dilated)
#         # print(x_dilated.shape)
#         # x_causal=self.network(x)
#         # print(x_causal.shape)
#         # # Concatenate the outputs from both branches
#         # x_combined = torch.cat((x_dilated, x_causal), dim=-1)  # B x (Co + Co)

#         # # Final linear layer
#         # x_out = self.linear(x_combined)
#         "ここでインスタンスノーマライゼーションの前処理を行っている"
#         # # インスタンスノーマライゼーション
#         # # 平均値計算
#         seq_mean = torch.mean(x, dim=2, keepdim=True)  # 平均値を (B, C, 1) の形状で計算

#         # 平均値を引き算して正規化
#         x = x - seq_mean  # (B, C, S)

#         # 1D畳み込みの入力形状に変換
#         x_conv_input = x.reshape(-1, 1, x.size(2))  # (B * C, 1, S)

#         # 畳み込み適用
#         x_conv_output = self.conv1d(x_conv_input)  # (B * C, 1, S)

#         # 畳み込み出力を元の形状に戻す
#         x_conv_output = x_conv_output.reshape(x.size(0), x.size(1), x.size(2))  # (B, C, S)

#         # スキップ接続（畳み込み出力と元の入力を足し算）
#         x = x + x_conv_output
#         x_out=self.network_norm(x)
        
#         return x_out


#         # return self.network(x)

# class CausalCNNEncoder_kato(torch.nn.Module):
#     """
#     Encoder of a time series using a causal CNN: the computed representation is
#     the output of a fully connected layer applied to the output of an adaptive
#     max pooling layer applied on top of the causal CNN, which reduces the
#     length of the time series to a fixed size.

#     Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
#     batch size, `C` is the number of input channels, and `L` is the length of
#     the input. Outputs a three-dimensional tensor (`B`, `C`).

#     @param in_channels Number of input channels.
#     @param channels Number of channels manipulated in the causal CNN.
#     @param depth Depth of the causal CNN.
#     @param reduced_size Fixed length to which the output time series of the
#            causal CNN is reduced.
#     @param out_channels Number of output channels.
#     @param kernel_size Kernel size of the applied non-residual convolutions.
#     """
#     def __init__(self, in_channels, channels, depth, reduced_size,
#                  out_channels, kernel_size):
#         super(CausalCNNEncoder_kato, self).__init__()
#         causal_cnn = CausalCNN(
#             in_channels*2, channels, depth, reduced_size, kernel_size
#         )
#         reduce_size = torch.nn.AdaptiveMaxPool1d(1)
#         squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
#         # Fully Connected Layer for final output
#         linear_nom=nn.Linear(reduced_size , out_channels*2)
#         self.linear = nn.Linear(reduced_size + out_channels, out_channels)
#         self.network = torch.nn.Sequential(
#             causal_cnn, reduce_size, squeeze
#         )
#         self.network_norm = torch.nn.Sequential(
#             causal_cnn, reduce_size, squeeze,linear_nom
#         )


#         self.input_fc = nn.Linear(in_channels, 64)

#         self.feature_extractor = DilatedConvEncoder(
#             64,
#             [64] * depth + [in_channels*2],
#             kernel_size=3
#         )
#         self.repr_dropout = nn.Dropout(p=0.1)

#         self.reduce_size_dilated = nn.AdaptiveMaxPool1d(1)
#         self.squeeze_dilated = SqueezeChannels()


#     def forward(self, x):
#         #print(x.shape)

#         x_dilated = x.transpose(1, 2)  # B x Ch x T
#         #print(x_dilated.shape)
#         x_dilated=self.input_fc(x_dilated)
#         #print('fc')
#         #print(x_dilated.shape)
#         x_dilated = x_dilated.transpose(1, 2)  # B x Ch x T
#         #print(x_dilated.shape)
#         x_dilated=self.feature_extractor(x_dilated)
#         #print("feat")
#         #print(x_dilated.shape)

#         #x_dilated = x_dilated.transpose(1, 2)

#         x_out=self.network_norm(x_dilated)
#         #print(x_out.shape)
        
#         # x_causal=self.network(x)
#         # #print(x_causal)
#         # # Concatenate the outputs from both branches
#         # x_combined = torch.cat((x_dilated, x_causal), dim=-1)  # B x (Co + Co)

#         # # Final linear layer
#         # x_out = self.linear(x_combined)

#         #x_out=self.network_norm(x)
        
#         return x_out



#     #     def generate_binomial_mask(B, T, p=0.5):
#     #         return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)
#     #     print("A")
#     #     print(x.shape)
#     #     x_dilated = x.transpose(1, 2)  # B x Ch x T
#     #     x_dilated=self.input_fc(x)
#     #     nan_mask = ~x_dilated.isnan().any(axis=-1)
#     #     mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
#     #     mask &= nan_mask
#     #     x_dilated[~mask] = 0
#     #     x_dilated = self.repr_dropout(self.feature_extractor(x_dilated))  # B x Co x T
#     #     #x, _ = self.attention(x, x, x)
#     #     x_dilated = x_dilated.transpose(1, 2)  # B x T x Co
#     #     #x_dilated = x_dilated.transpose(1, 2)  # B x Ch x T
#     #     print(x_dilated.shape)


#     #     #x_dilated=self.feature_extractor(x_dilated)
#     #     #x_dilated = self.reduce_size_dilated(x_dilated)
#     #    # x_dilated = self.squeeze_dilated(x_dilated)
#     #     #print(x_dilated.shape)
#     #     #x_causal=self.network(x)
#     #     #print(x_causal)
#     #     # Concatenate the outputs from both branches
#     #     #x_combined = torch.cat((x_dilated, x_causal), dim=-1)  # B x (Co + Co)

#     #     # Final linear layer
#     #     #x_out = self.linear(x_combined)

#     #     x_out=self.network_norm(x_dilated)
#     #     print(x_out.shape)
        
#     #     return x_out


#     #     # return self.network(x)

# class CausalCNNEncoder_kato2(torch.nn.Module):
#     """
#     Encoder of a time series using a causal CNN: the computed representation is
#     the output of a fully connected layer applied to the output of an adaptive
#     max pooling layer applied on top of the causal CNN, which reduces the
#     length of the time series to a fixed size.

#     Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
#     batch size, `C` is the number of input channels, and `L` is the length of
#     the input. Outputs a three-dimensional tensor (`B`, `C`).

#     @param in_channels Number of input channels.
#     @param channels Number of channels manipulated in the causal CNN.
#     @param depth Depth of the causal CNN.
#     @param reduced_size Fixed length to which the output time series of the
#            causal CNN is reduced.
#     @param out_channels Number of output channels.
#     @param kernel_size Kernel size of the applied non-residual convolutions.
#     """
#     def __init__(self, in_channels, channels, depth, reduced_size,
#                  out_channels, kernel_size):
#         super(CausalCNNEncoder_kato2, self).__init__()
#         causal_cnn = CausalCNN(
#             in_channels*2, channels, depth, reduced_size, kernel_size
#         )
#         #reduce_size = torch.nn.AdaptiveMaxPool1d(1)
#         #squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
#         # Fully Connected Layer for final output
#         self.linear_nom=nn.Linear(reduced_size , in_channels*2)
#         #self.linear = nn.Linear(reduced_size + out_channels, out_channels)
#         # self.network = torch.nn.Sequential(
#         #     causal_cnn, reduce_size, squeeze
#         # )
#         self.network_norm = torch.nn.Sequential(
#             causal_cnn
#         )


#         self.input_fc = nn.Linear(in_channels, 64)

#         self.feature_extractor = DilatedConvEncoder(
#             64,
#             [64] * depth + [in_channels*2],
#             kernel_size=3
#         )
#         self.repr_dropout = nn.Dropout(p=0.1)

#         self.reduce_size_dilated = nn.AdaptiveMaxPool1d(1)
#         self.squeeze_dilated = SqueezeChannels()


#     def forward(self, x):
#         #print(x.shape)
#         x_dilated = x.transpose(1, 2)  # B x Ch x T
#         #print(x_dilated.shape)
#         x_dilated=self.input_fc(x_dilated)
#         #print('fc')
#         #print(x_dilated.shape)
#         x_dilated = x_dilated.transpose(1, 2)  # B x Ch x T
#         #print(x_dilated.shape)
#         x_dilated=self.feature_extractor(x_dilated)
#         #print("feat")
#         #print(x_dilated.shape)

#         #x_dilated = x_dilated.transpose(1, 2)

#         x_out=self.network_norm(x_dilated)
#         #print(x_out.shape)
#         x_out=x_out.transpose(1, 2) 
#         x_out=self.linear_nom(x_out)
#         x_out=x_out.transpose(1, 2)
#         #print(x_out.shape)
        
#         # x_causal=self.network(x)
#         # #print(x_causal)
#         # # Concatenate the outputs from both branches
#         # x_combined = torch.cat((x_dilated, x_causal), dim=-1)  # B x (Co + Co)

#         # # Final linear layer
#         # x_out = self.linear(x_combined)

#         #x_out=self.network_norm(x)
        
#         return x_out



#     #     def generate_binomial_mask(B, T, p=0.5):
#     #         return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)
#     #     print("A")
#     #     print(x.shape)
#     #     x_dilated = x.transpose(1, 2)  # B x Ch x T
#     #     x_dilated=self.input_fc(x)
#     #     nan_mask = ~x_dilated.isnan().any(axis=-1)
#     #     mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
#     #     mask &= nan_mask
#     #     x_dilated[~mask] = 0
#     #     x_dilated = self.repr_dropout(self.feature_extractor(x_dilated))  # B x Co x T
#     #     #x, _ = self.attention(x, x, x)
#     #     x_dilated = x_dilated.transpose(1, 2)  # B x T x Co
#     #     #x_dilated = x_dilated.transpose(1, 2)  # B x Ch x T
#     #     print(x_dilated.shape)


#     #     #x_dilated=self.feature_extractor(x_dilated)
#     #     #x_dilated = self.reduce_size_dilated(x_dilated)
#     #    # x_dilated = self.squeeze_dilated(x_dilated)
#     #     #print(x_dilated.shape)
#     #     #x_causal=self.network(x)
#     #     #print(x_causal)
#     #     # Concatenate the outputs from both branches
#     #     #x_combined = torch.cat((x_dilated, x_causal), dim=-1)  # B x (Co + Co)

#     #     # Final linear layer
#     #     #x_out = self.linear(x_combined)

#     #     x_out=self.network_norm(x_dilated)
#     #     print(x_out.shape)
        
#     #     return x_out


#     #     # return self.network(x)