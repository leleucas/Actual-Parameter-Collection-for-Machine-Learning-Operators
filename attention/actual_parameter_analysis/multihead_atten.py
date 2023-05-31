import torch
import json
import numpy as np
from flash_attn.flash_attention import FlashMHA


def multihead_atten(embed_dim, num_heads, query, key, value):
    multihead_attn = torch.nn.MultiheadAttention(embed_dim, num_heads, device='cuda', batch_first=True, dtype=torch.float16)
    # print(query.device, key.device, value.device)
    attn_output, attn_output_weights = multihead_attn(query, key, value)
    return attn_output


def flash_atten(batch_size, s, embed_dim, num_heads):
    device = "cuda:0"
    dtype = torch.float16
    flash_mha = FlashMHA(
        embed_dim=embed_dim, # total channels (= num_heads * head_dim)
        num_heads=num_heads, # number of heads
        device=device,
        dtype=dtype,
    )
    x = torch.randn(
        (batch_size, s, embed_dim), # (batch, seqlen, embed_dim)
        device=device,
        dtype=dtype
    )
    output = flash_mha(x)[0]
    return output


def gen_args(query_, key_, value_):
    query = torch.rand(query_).cuda()
    key = torch.rand(key_).cuda()
    value = torch.rand(value_).cuda()

    return [query, key, value]


if __name__ == '__main__':
    #Bert
    embed_dim = 256
    num_heads = 8

    torch_arg = gen_args([2, 20, 256], [2, 20, 256], [2, 20, 256])
    multihead_atten(embed_dim, num_heads, torch_arg[0], torch_arg[1], torch_arg[2])

    #GPT-2
    # num_heads = 12
    # features = 64
    # seq = 32
    # batch_size = 2
    # embed_dim = num_heads * features
    # torch_arg = gen_args([batch_size, seq, embed_dim], [batch_size, seq, embed_dim], [batch_size, seq, embed_dim])
    # multihead_atten(embed_dim, num_heads, torch_arg[0], torch_arg[1], torch_arg[2])

    # SAM
    # num_heads = 8
    # s = 7
    # batch_size = 64
    # embed_dim = 256
    # torch_arg = gen_args([batch_size, s, embed_dim], [batch_size, s, embed_dim], [batch_size, s, embed_dim])
    # multihead_atten(embed_dim, num_heads, torch_arg[0], torch_arg[1], torch_arg[2])

    # t = 4096
    # torch_arg = gen_args([batch_size, s, embed_dim], [batch_size, t, embed_dim], [batch_size, t, embed_dim])
    # multihead_atten(embed_dim, num_heads, torch_arg[0], torch_arg[1], torch_arg[2])

    # torch_arg = gen_args([batch_size, t, embed_dim], [batch_size, s, embed_dim], [batch_size, s, embed_dim])
    # multihead_atten(embed_dim, num_heads, torch_arg[0], torch_arg[1], torch_arg[2])


    # Bert
    # num_heads = 8
    # embed_dim = 256
    # batch_size = 2
    # s = 20

    # flash_atten(batch_size, s, embed_dim, num_heads)

    # num_heads = 12
    # features = 64
    # seq = 32
    # batch_size = 2
    # embed_dim = num_heads * features
    # flash_atten(batch_size, seq, embed_dim, num_heads)

    # SAM
    # num_heads = 8
    # embed_dim = 256
    # batch_size = 64
    # s = 7

    # flash_atten(batch_size, s, embed_dim, num_heads)

    # test case
    # query = torch.rand(20, 32, 512).cuda()
    # key = torch.rand(10, 32, 512).cuda()
    # value = torch.rand(10, 32, 512).cuda()
    # print(query.device, key.device, value.device)
    # multihead_attn = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8, device='cuda')
    # output, attn_output_weights = multihead_attn(query, key, value)
    # print(output.shape)