import torch
import torch.nn as nn

class MySelfAttentionWithoutMultiHead(nn.Module):
    """
    无多头的Self Attention实现
    """
    def __init__(self, input_dim, hidden_dim):
        super(MySelfAttentionWithoutMultiHead, self).__init__()
        self.Q = nn.Linear(input_dim, hidden_dim)
        self.K = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(input_dim, hidden_dim)

    @staticmethod
    def attention_without_masked(q, k, v):
        # batch_size, head, length, d_tensor = k.size()
        batch_size, length, d_tensor = k.size()
        attn = torch.softmax(torch.matmul(q, k) / torch.sqrt(torch.tensor(d_tensor)), dim=-1)
        score = attn @ v
        return score

    def forward(self, x):
        q = self.Q(x)
        k = self.K(x).transpose(1, 2)
        v = self.V(x)

        o = self.attention_without_masked(q, k, v)
        return o

class MyMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0, bias=True):
        super(MyMultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dk = embed_dim // num_heads  # (head_dim)
        self.dropout_rate = dropout_rate

        self.Q = nn.Linear(embed_dim, embed_dim, bias)
        self.K = nn.Linear(embed_dim, embed_dim, bias)
        self.V = nn.Linear(embed_dim, embed_dim, bias)

        self.OutLayer = nn.Linear(embed_dim, embed_dim, bias)

    def _scaled_dot_product_attention(self, q, k, v, attn_mask=None, dropout_rate=0.):
        pass
        # if dropout_p > 0.0:
        #     attn = dropout(attn, p=dropout_p)

    def forward(self, x):
        # x: [batch_size, seq_length, embed_dim]
        q = self.Q(x)  # [batch_size, seq_length, num_heads * dk]
        k = self.K(x)  # [batch_size, seq_length, num_heads * dk]
        v = self.V(x)  # [batch_size, seq_length, num_heads * dk]

        scaling = 1 / torch.sqrt(torch.tensor(self.dk, device='cuda' if next(self.parameters()).is_cuda else 'cpu'))
        batch_size, seq_length, _ = q.shape

        q = q.transpose(1, 2)  # [batch_size, num_heads * dk, seq_length]
        q = q.reshape((batch_size * self.num_heads, self.dk, seq_length))  # [batch_size * num_heads, dk, seq_length]
        k = k.transpose(1, 2)  # [batch_size, num_heads * dk, seq_length]
        k = k.reshape((batch_size * self.num_heads, self.dk, seq_length)).transpose(1, 2)  # [batch_size * num_heads, seq_length, dk]
        v = v.transpose(1, 2).reshape((batch_size * self.num_heads, self.dk, seq_length))  # [batch_size * num_heads, dk, seq_length]
        attn = torch.softmax(torch.bmm(q, k) * scaling, dim=-1)  # [batch_size * num_heads, dk, dk]
        output = torch.bmm(attn, v)  # [batch_size * num_heads, dk, seq_length]
        output = output.reshape((batch_size, self.embed_dim, seq_length)).transpose(1, 2)  # [batch_size * num_heads, seq_length, dk]
        return output


class MyTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, head_num):
        super(MyTransformer, self).__init__()
        assert hidden_dim % head_num == 0
        self.dk = hidden_dim / head_num
        # self


if __name__ == '__main__':
    x = torch.randn(32, 15, 128)
    t = MyMultiHeadAttention(128, 4)
    print(t(x).shape)