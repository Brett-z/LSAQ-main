import torch

def pack_to_uint8(quant_x, in_f, out_f):
    # out_f 为行，in_f 为列
    i = 0
    rounds = int(out_f/2)

    new_weight = torch.zeros(rounds, in_f, dtype=torch.uint8)

    while i < rounds:

        row0 = quant_x[i] 
        row1 = quant_x[i + rounds] 

        packed_row0_row1 = ((row0 + 8).to(torch.uint8) << 4) | (row1 + 8).to(torch.uint8)  # 第一列（高4位）+第三列（低4位）

        new_weight[i] = packed_row0_row1

        i += 1
    
    return new_weight

def unpack_to_int8(packed_x, in_f, out_f):

    i = 0
    rounds = int(out_f/2)

    new_weight = torch.zeros(out_f, in_f, dtype=torch.int8)

    while i < rounds:

        packed_row0_row1 = packed_x[i]
        row0 = ((packed_row0_row1 >> 4) & 0x0F).to(torch.int8) - 8  # 高4位→原第一列
        row1 = (packed_row0_row1 & 0x0F).to(torch.int8) - 8        # 低4位→原第三列

        new_weight[i] = row0
        new_weight[i + rounds] = row1

        i += 1
    
    return new_weight