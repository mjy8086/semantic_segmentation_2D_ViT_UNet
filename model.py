import torch
from torch import nn
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _triple
import torch.nn.functional
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm, Conv2d
import torchvision
import math
import copy
import torchsummary
from torchsummary import summary
import GPUtil
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# 1) CNN Encoder
# 2) CNN Decoder
# 3) ViT


# CNN Encoder

class CNNencoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(out_c)
        )

    def forward(self, x):
        out = self.model(x)
        return out


# CNN Decoder

class CNNdecoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(out_c)
        )

    def forward(self, x, skip_x):
        x = self.model(x)
        if x.size() != skip_x.size():
            x = match_size(x, skip_x.shape[2:])
        x = torch.cat((x, skip_x), 1)
        return x



# Making ViT

# Patch Embedding

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size):
        super(Embeddings, self).__init__()
        down_factor = 2
        # patch_size = _triple(8, 8, 8)
        patch_size = (8, 8)
        n_patches = int((img_size[0]/2**down_factor// patch_size[0]) * (img_size[1]/2**down_factor// patch_size[1]))
        self.hybrid_model = CNNencoder(in_c=128, out_c=128)
        self.patch_embeddings = Conv2d(in_channels=128,
                                       # 우선 in channels는 128로 설정하자
                                       out_channels=252,
                                       # out_channels = hidden size D = 252
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, 252))

        self.dropout = Dropout(0.1)

    def forward(self, x):
        x = self.hybrid_model(x)
        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


# Multi-head self attention (MSA) - layer norm not included

class MSA(nn.Module):
    def __init__(self):
        super(MSA, self).__init__()
        self.num_attention_heads = 12
        # Number of head = 12
        self.attention_head_size = int(252 / self.num_attention_heads)
        # Attention Head size = Hidden size(D)(252) / Number of head(12) = 21
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # All Head size = (12 * 21) = 252 = Hidden size
        self.query = Linear(252, self.all_head_size)
        self.key = Linear(252, self.all_head_size)
        self.value = Linear(252, self.all_head_size)
        self.out = Linear(252, 252)
        self.attn_dropout = Dropout(0.0)
        self.proj_dropout = Dropout(0.0)
        self.softmax = Softmax(dim=-1)

    # def transpose_for_scores(self, x):
    #     new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    #     x = x.view(*new_x_shape)
    #     return x.permute(0, 2, 1, 3)

    def transpose_for_scores(self, x):
        x = x.view([x.size()[0], -1, self.num_attention_heads, self.attention_head_size])
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


# MLP - layer norm not included

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = Linear(252, 3072)
        self.fc2 = Linear(3072, 252)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# Block - incorporating MSA, MLP, Layer Norm

class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.hidden_size = 252
        self.attention_norm = LayerNorm(252, eps=1e-6)
        self.ffn_norm = LayerNorm(252, eps=1e-6)
        self.ffn = MLP()
        self.attn = MSA()

    def forward(self, x):
        h = x

        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


#  ViTencoder - ViT Encoder with Blocks

class ViTencoder(nn.Module):
    def __init__(self):
        super(ViTencoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(252, eps=1e-6)
        for _ in range(12):
            layer = Block()
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):


        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)

        encoded = self.encoder_norm(hidden_states)
        return encoded


#  ViT 마지막에 나온 latent를 CNNdecoder에 넣기 위해 변환시키기위한 Conv

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


#  ViT

class ViT(nn.Module):
    def __init__(self, img_size):
        super(ViT, self).__init__()
        self.embeddings = Embeddings(img_size=img_size)
        self.encoder = ViTencoder()
        self.img_size = img_size
        self.patch_size = (8, 8)
        self.down_factor = 2
        self.conv_more = Conv2dReLU(252, 128, kernel_size=3, padding=1, use_batchnorm=True)

    def forward(self, input_ids):
        # embedding_output, features = self.embeddings(input_ids)
        # encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        # return encoded, attn_weights, features

        embedding_output = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)

        B, n_patch, hidden = encoded.size()
        h, w = (self.img_size[0] // 2**self.down_factor // self.patch_size[0]), (self.img_size[1] // 2**self.down_factor // self.patch_size[0])
        x = encoded.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        return x

    # 여기서 x = ViT의 결과값 feature(x)이
    # features = skip connection에 사용될 feature인듯고
    # 그러면 난 features는 필요없을듯! CNN encoder에서 갖고오니깐

# 그리고 이 ViT의 output은 3개 dimension ex) [2, 64, 252]가 나오는데 = [B, n_patch, hidden]
# 이거를 CNN decoder에 넣으려면 5-dimension으로 변환시켜야해
# Class ViT 안에 3 -> 5 dimension 변환 과정을 추가하자!



# Matching size

def match_size(x, size):
    _, _, h1, w1, d1 = x.shape
    h2, w2, d2 = size

    while d1 != d2:
        if d1 < d2:
            x = nn.functional.pad(x, (0, 1), mode='constant', value=0)
            d1 += 1
        else:
            x = x[:, :, :, :, :d2]
            break
    while w1 != w2:
        if w1 < w2:
            x = nn.functional.pad(x, (0, 0, 0, 1), mode='constant', value=0)
            w1 += 1
        else:
            x = x[:, :, :, :w2, :]
            break
    while h1 != h2:
        if h1 < h2:
            x = nn.functional.pad(x, (0, 0, 0, 0, 0, 1), mode='constant', value=0)
            h1 += 1
        else:
            x = x[:, :, :h2, :, :]
            break
    return x


# Generator

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, img_size=(256, 256)):
        super().__init__()
        nf = 16

        self.pooling = nn.MaxPool3d(kernel_size=2)

        self.conv1 = CNNencoder(in_channels, nf)
        self.conv2 = CNNencoder(nf, nf*2)
        self.conv3 = CNNencoder(nf*2, nf*4)
        self.conv4 = CNNencoder(nf*4, nf*8)
        self.conv5 = CNNencoder(nf*8, nf*8)

        self.vit = ViT(img_size)

        self.up1 = CNNdecoder(nf*8, nf*8)
        self.up2 = CNNdecoder(nf*8*2, nf*4)
        self.up3 = CNNdecoder(nf*4*2, nf*2)
        self.up4 = CNNdecoder(nf*2*2, nf)

        self.out = nn.Sequential(
            nn.Conv3d(nf*2, out_channels, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(inplace=True)
        )


    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pooling(c1)

        c2 = self.conv2(p1)
        p2 = self.pooling(c2)

        c3 = self.conv3(p2)
        p3 = self.pooling(c3)

        c4 = self.conv4(p3)
        p4 = self.pooling(c4)

        c5 = self.conv5(p4)

        v1 = self.vit(c5)

        u1 = self.up1(v1, c4)
        u2 = self.up2(u1, c3)
        u3 = self.up3(u2, c2)
        u4 = self.up4(u3, c1)
        out = self.out(u4)

        if x.size() != out.size():
            out = match_size(out, x.shape[2:])

        return u1, u2, u3, u4, out



# model1 = Generator(in_channels=1, out_channels=1, img_size=(160, 192, 224)).cuda()

model1 = Generator(in_channels=1, out_channels=1, img_size=(256, 256)).cuda()

# print(list(model1.parameters()))

# print(list(model1.named_parameters()))

# summary(model1, (1, 160, 192, 224))

summary(model1, (1, 256, 256))
