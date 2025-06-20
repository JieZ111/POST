"""Self Attention Module


"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_channel, out_channel=None, attn_dropout=0.1):
        """
        :param in_channel: previous layer's output feature dimension
        :param out_channel: size of output vector, defaults to in_channel
        """
        super(SelfAttention, self).__init__()
        self.in_channel = in_channel

        if out_channel is not None:
            self.out_channel = out_channel
        else:
            self.out_channel = in_channel

        self.temperature = self.out_channel ** 0.5

        self.q_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.k_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.v_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)

        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        """
        :param x: the feature maps from previous layer,
                      shape: (batch_size, in_channel, num_points)
        :return: y: attentioned features maps,
                        shape： (batch_size, out_channel, num_points)
        """
        q = self.q_map(x)  # (batch_size, out_channel, num_points)
        k = self.k_map(x)  # (batch_size, out_channel, num_points)
        v = self.v_map(x)  # (batch_size, out_channel, num_points)

        attn = torch.matmul(q.transpose(1,2) / self.temperature, k)

        attn = self.dropout(F.softmax(attn, dim=-1))
        y = torch.matmul(attn, v.transpose(1,2)) # (batch_size, num_points, out_channel)

        return y.transpose(1, 2)


class QGPA(nn.Module):
    def __init__(self, attn_dropout=0.1):

        super(QGPA, self).__init__()
        self.in_channel = self.out_channel = 192

        self.temperature = self.out_channel ** 0.5
        self.layer_norm = nn.LayerNorm(self.in_channel)
        proj_dim = 512*5
        self.q_map = nn.Conv1d(2048*5, proj_dim, 1, bias=False)
        self.k_map = nn.Conv1d(2048*5, proj_dim, 1, bias=False)

        self.v_map = nn.Linear(self.in_channel, self.out_channel)
        self.fc = nn.Conv1d(self.in_channel, self.out_channel, 1, bias=False)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, support, prototype):

        batch, dim = query.shape[0], query.shape[1]
        way = support.shape[0] + 1
        residual = prototype
        q = self.q_map(query.transpose(1, 2))
        if len(support.shape) == 4:
            support = support.squeeze()
        support = torch.cat([support.mean(0).unsqueeze(0), support], dim=0)
        k = self.k_map(support.transpose(1, 2))
        v = self.v_map(prototype)
        q = q.view(q.shape[1], q.shape[2] * q.shape[0])
        k = k.view(k.shape[1], k.shape[2] * k.shape[0])

        attn = torch.matmul(q.transpose(0, 1) / self.temperature, k)
        attn = attn.reshape(batch, way, dim, dim)
        attn = F.softmax(attn, dim=-1)
        v = v.unsqueeze(2)
        output = torch.matmul(attn, v.transpose(-2, -1)).squeeze(-1).transpose(1, 2)
        output = self.dropout(self.fc(output)).transpose(1, 2)
        output = self.layer_norm(output + residual)

        return output
class MultiHeadAttention1(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention1, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.fc_out = nn.Linear(d_model, d_model)
        self.fc=nn.Sequential(nn.Linear(d_model, d_model),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(d_model, d_model))
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.transpose(2, 1)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        energy = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim ** 0.5

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = self.dropout(torch.softmax(energy, dim=-1))

        out = torch.matmul(attention, value)
        out = out.transpose(2, 1).contiguous().view(batch_size, -1, self.d_model)
        out = self.fc(out)
        out=query+out

        return out
