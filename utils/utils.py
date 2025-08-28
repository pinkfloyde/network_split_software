import torch
import struct
import os
import numpy as np
import random
import torch
import torch.nn as nn

def makeDir(filepath):
    filepath = filepath.strip()  # 去除首位空格
    parent_path, _ = os.path.split(filepath)
    isExists = os.path.exists(parent_path)  # 判断路径是否存在，存在则返回true
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(parent_path)


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 设置PyTorch在CUDA上生成随机数的种子。
    torch.cuda.manual_seed_all(seed)  # 在多GPU环境中，设置所有可见的CUDA设备的随机种子。
    torch.backends.cudnn.deterministic = True  # 设置使用CuDNN时，使得卷积算法保持确定性。这是为了确保在使用GPU加速时，卷积操作的结果是确定性的。
    torch.backends.cudnn.benchmark = False  # 关闭CuDNN的自动调整，确保在不同批次上卷积的性能一致。这也是为了保证实验的可重复性。
    torch.backends.cudnn.enabled = False  # 禁用CuDNN。这是因为CuDNN的实现可能会引入一些非确定性的因素，而在某些情况下禁用它可以获得可重复的结果。

    # 这段代码是用来设置PyTorch和其他相关库的随机种子（random
    # seed）的。在机器学习中，设置随机种子是为了使实验可重复，即每次运行代码时得到的随机结果相同，这样有助于调试和比较不同模型或算法的性能。
    # 具体来说，这段代码做了以下几个操作：
    # random.seed(seed)：设置Python的random模块的种子，确保在使用random模块生成随机数时，得到的结果是可复现的。
    # os.environ['PYTHONHASHSEED'] = str(seed)：设置Python中hash的种子，这同样有助于使得使用哈希的操作变得可重复。
    # np.random.seed(seed)：设置NumPy库的随机种子，以确保NumPy生成的随机数也是可重复的。
    # torch.manual_seed(seed)：设置PyTorch的随机种子，包括CPU上的随机数生成。
    # torch.cuda.manual_seed(seed)：

    # 总体来说，这些设置是为了确保在使用PyTorch进行深度学习实验时，通过控制随机性使实验结果具有可重复性。这在实验、调试和结果比较时都是非常重要的。


# Convert a number to hex
# BF16, FP32, INT8, INT32 can be convert to hex one by one
# BIN can be convert to hex 4 bits (MSB at index 3) at a time
def num_to_hex(num, type):
    # Convert the float to bytes
    if (type == "FP32"):
        num_bytes = struct.pack('f', num.to(torch.float32))
    elif (type == "BF16"):
        num_bytes = struct.pack('f', num.to(torch.float32))
    elif (type == "INT8"):
        num_bytes = struct.pack('b', num.to(torch.int8))
    elif (type == "INT32"):
        num_bytes = struct.pack('i', num.to(torch.int32))

    # Convert the bytes to an integer
    if (type == "BIN"):
        num_int = num[0] * 1 + num[1] * 2 + num[2] * 4 + num[3] * 8
    else:
        num_int = int.from_bytes(num_bytes, byteorder='little', signed=False)

    # output = hex(num_int)

    if (type == "BF16"):
        output = "{:08x}".format(num_int)
        output = output[:-4]
    elif (type == "FP32"):
        output = "{:08x}".format(num_int)
    elif (type == "INT8"):
        output = "{:02x}".format(num_int)
    elif (type == "INT32"):
        output = "{:08x}".format(num_int)
    elif (type == "BIN"):
        output = "{:01x}".format(int(num_int))

    return output


def tensor_output(x: torch.Tensor, type: str, order: tuple, filename: str):
    assert (len(x.shape) == len(order))
    # print('len(x.shape)', x.shape)
    # print('len(order)', len(order))
    assert (type == 'BF16' or type == 'FP32' or type == 'INT8' or type == 'INT32' or type == 'BIN')
    # 将输入张量转换为指定的数据类型
    if (type == 'BF16'):
        x.to(torch.bfloat16)
    elif (type == 'FP32'):
        x.to(torch.float32)
    elif (type == 'INT8'):
        x.to(torch.int8)
    elif (type == 'INT32'):
        x.to(torch.int32)
    elif (type == 'BIN'):
        x.to(torch.bool)

    # Permute the dimensions of x based on the order specified in the tuple
    # 根据元组中指定的顺序重新排列张量的维度
    permuted_x = x.permute(order)

    if (type == 'BF16'):
        number_in_32B = 16.0
    elif (type == 'FP32'):
        number_in_32B = 8.0
    elif (type == 'INT8'):
        number_in_32B = 32.0
    elif (type == 'INT32'):
        number_in_32B = 8.0
    elif (type == 'BIN'):
        number_in_32B = 256.0

    # 计算内部维度的填充到32位单元的数量
    inner_dimension = permuted_x.shape[-1]
    # 在Python中，负数索引表示从序列的末尾开始计数。因此，-1 表示最后一个元素，-2 表示倒数第二个元素，以此类推
    dimension_padded_to_cell = np.ceil(inner_dimension / number_in_32B) * number_in_32B

    zero_to_full_cell = torch.zeros(
        size=[dim for dim in permuted_x.shape[:-1]] + [int(dimension_padded_to_cell) - inner_dimension])
    # 补0操作   # 使用[:-1]获取除最后一个元素外的 子序列

    # 将零张量连接到 permuted_x，以确保其维度被填充到32位单元
    permuted_x = torch.cat((permuted_x, zero_to_full_cell), dim=-1)

    # Flatten the tensor and get the sorted indices
    flat_x = torch.flatten(permuted_x)  # 使用torch.flatten函数将张量permuted_x展平为一维张量flat_x。这意味着所有维度上的元素都被拉平，得到一个包含所有元素的一维张量
    if (not type == 'BF16' and not type == 'FP32'):
        flat_x = flat_x.to(torch.int32)
    length = flat_x.shape[0]  # 因为拉平所以用 shape[0] 表示长度个数

    # Iterate over each element of x in the order specified by order
    with open(filename + '.num.txt', 'w') as f:
        for indices in range(length):
            element = flat_x[indices]
            if (type == 'BF16' or type == 'FP32'):
                f.write(('%.20g' % element.item()) + '\n')
            else:
                f.write(str(element.item()) + '\n')
        # 如果数据类型是 'BF16'
        # 或 'FP32'，则使用科学计数法将元素写入文件。否则，将元素转换为字符串并写入文件。
    x_in_mem = torch.reshape(flat_x, (int(length / number_in_32B), int(number_in_32B)))
    with open(filename + '.mem.txt', 'w') as f:
        for cell in x_in_mem:
            cell_str = ""
            if (type == 'BIN'):
                for ii in range(int(len(cell) / 4)):
                    cell_str = num_to_hex(cell[ii * 4:ii * 4 + 4], type) + cell_str
            else:
                for element in cell:
                    cell_str = num_to_hex(element, type) + cell_str
            f.write(cell_str + '\n')
    print("over")


def tensor_split(x: torch.Tensor, type: str, order: tuple, test_name: str, filename: str, h_Dimensionsize: int,
                 h_Stacksize: int, h_num: int,
                 w_Dimensionsize: int, w_Stacksize: int, w_num: int, c_Dimensionsize: int, c_Stacksize: int, c_num: int,
                 padding: int):
    pad = nn.ZeroPad2d(padding)
    x = pad(x)
    for t in range(h_num):
        x_split_h = x.narrow(2, t * h_Dimensionsize, padding * 2 + h_Dimensionsize)
        for i in range(w_num):
            x_split_w = x_split_h.narrow(3, i * w_Dimensionsize, padding * 2 + w_Dimensionsize)
            for j in range(c_num):
                x_split_c = x_split_w.narrow(1, j * c_Dimensionsize, c_Stacksize * 2 + c_Dimensionsize)
                print(filename, x_split_c.shape)
                print(filename, x_split_c.dtype)
                tensor_output(x_split_c, type, order,
                              test_name + '/' + filename + '_{}'.format(t) + '_{}'.format(i) + '_{}'.format(j))


def in_feature_split(x: torch.Tensor, test_name: str, filename: str, 
                     b_dimensionsize: int, b_num: int,
                     h_dimensionsize: int, h_num: int,
                     w_dimensionsize: int, w_num: int,
                     c_dimensionsize: int, c_num: int, padding: int):
    pad = nn.ZeroPad2d(padding)
    x = pad(x)
    # for k in range(b_num):
    #     x_split_b = x.narrow(0, k * b_dimensionsize, b_dimensionsize)
    #     for t in range(h_num):
    #         if h_num!=1:
    #             if t==0:
    #                 x_split_h = x_split_b.narrow(2, t * h_dimensionsize, padding + h_dimensionsize)
    #             else:
    #                 if t==(h_num-1):
    #                     x_split_h = x_split_b.narrow(2, t * h_dimensionsize-padding, padding + h_dimensionsize)
    #                 else:
    #                     x_split_h = x_split_b.narrow(2, t * h_dimensionsize-padding, padding*2 + h_dimensionsize)
    #         else:
    #             x_split_h = x_split_b.narrow(2, t * h_dimensionsize, h_dimensionsize)
    #         for i in range(c_num):
    #             x_split_c = x_split_h.narrow(1, i * c_dimensionsize, c_dimensionsize)
    #             for j in range(w_num):
    #                 if w_num!=1:
    #                     if j==0:
    #                         x_split_w = x_split_c.narrow(3, j * w_dimensionsize, padding + w_dimensionsize)
    #                     else:
    #                         if j==(h_num-1):
    #                             x_split_w = x_split_c.narrow(3, j * w_dimensionsize-padding, padding + w_dimensionsize)
    #                         else:
    #                             x_split_w = x_split_c.narrow(3, j * w_dimensionsize-padding, padding*2 + w_dimensionsize)
    #                 else:
    #                     x_split_w = x_split_c.narrow(2, j * w_dimensionsize, w_dimensionsize)

    #                 print(filename, x_split_w.shape)
    #                 print(filename, x_split_w.dtype)
    #                 tensor_output(x_split_w, "INT8", (0, 2, 1, 3),
    #                               test_name + '/' + filename + '_{}'.format(k) + '_{}'.format(t) + '_{}'.format(
    #                                   i) + '_{}'.format(j))
    for k in range(b_num):
        x_split_b = x.narrow(0, k * b_dimensionsize, b_dimensionsize)
        for t in range(h_num):
            x_split_h = x_split_b.narrow(2, t * h_dimensionsize, padding*2 + h_dimensionsize)
            for i in range(c_num):
                x_split_c = x_split_h.narrow(1, i * c_dimensionsize, c_dimensionsize)
                for j in range(w_num):
                    x_split_w = x_split_c.narrow(3, j * w_dimensionsize, padding*2 + w_dimensionsize)

                    print(filename, x_split_w.shape)
                    print(filename, x_split_w.dtype)
                    # tensor_output(x_split_w, "INT8", (0, 2, 1, 3),
                    #               test_name + '/' + filename + '_{}'.format(k) + '_{}'.format(t) + '_{}'.format(
                    #                   i) + '_{}'.format(j))
                    tensor_output(x_split_w, 'INT8', (0, 2, 1, 3),
                                  test_name + '/' + filename + '_{}'.format(k) + '_{}'.format(t) + '_{}'.format(
                                      i) + '_{}'.format(j))


def deep_feature_split(x: torch.Tensor, type: str,test_name: str, filename: str, 
                       b_dimensionsize: int, b_num: int,
                       h_dimensionsize: int, h_num: int,
                       w_dimensionsize: int, w_num: int, 
                       c_dimensionsize: int, c_num: int, 
                       padding: int):
    pad = nn.ZeroPad2d(padding)
    x = pad(x)
    # for k in range(b_num):
    #     x_split_b = x.narrow(0, k * b_dimensionsize, b_dimensionsize)
    #     for t in range(h_num):
    #         if h_num!=1:
    #             if t==0:
    #                 x_split_h = x_split_b.narrow(2, t * h_dimensionsize, padding + h_dimensionsize)
    #             else:
    #                 if t==(h_num-1):
    #                     x_split_h = x_split_b.narrow(2, t * h_dimensionsize-padding, padding + h_dimensionsize)
    #                 else:
    #                     x_split_h = x_split_b.narrow(2, t * h_dimensionsize-padding, padding*2 + h_dimensionsize)
    #         else:
    #             x_split_h = x_split_b.narrow(2, t * h_dimensionsize, h_dimensionsize)
    #         for i in range(w_num):
    #             if w_num!=1:
    #                 if i==0:
    #                     x_split_w = x_split_h.narrow(3, i * w_dimensionsize, padding + w_dimensionsize)
    #                 else:
    #                     if i==(h_num-1):
    #                         x_split_w = x_split_h.narrow(3, i * w_dimensionsize-padding, padding + w_dimensionsize)
    #                     else:
    #                         x_split_w = x_split_h.narrow(3, i * w_dimensionsize-padding, padding*2 + w_dimensionsize)
    #             else:
    #                 x_split_w = x_split_h.narrow(3, i * w_dimensionsize, w_dimensionsize)

    #             for j in range(c_num):
    #                 x_split_c = x_split_w.narrow(1, j * c_dimensionsize, c_dimensionsize)
    #                 print(filename, x_split_c.shape)
    #                 print(filename, x_split_c.dtype)
    #                 tensor_output(x_split_c, type,(0, 2, 3, 1),
    #                               test_name + '/' + filename + '_{}'.format(k) + 
    #                               '_{}'.format(t) + '_{}'.format(i) + '_{}'.format(j))
    for k in range(b_num):
        x_split_b = x.narrow(0, k * b_dimensionsize, b_dimensionsize)
        for t in range(h_num):
            x_split_h = x_split_b.narrow(2, t * h_dimensionsize , padding*2 + h_dimensionsize)
            for i in range(w_num):
                x_split_w = x_split_h.narrow(3, i * w_dimensionsize, padding*2 + w_dimensionsize)
                for j in range(c_num):
                    x_split_c = x_split_w.narrow(1, j * c_dimensionsize, c_dimensionsize)
                    print(filename, x_split_c.shape)
                    print(filename, x_split_c.dtype)
                    tensor_output(x_split_c, type,(0, 2, 3, 1),
                                  test_name + '/' + filename + '_{}'.format(k) + 
                                  '_{}'.format(t) + '_{}'.format(i) + '_{}'.format(j))


def weight_split(x: torch.Tensor, type: str, test_name: str, filename: str, cin_dimensionsize: int,
                 cin_num: int, cout_dimensionsize: int, cout_num: int):
    for t in range(cin_num):
        x_split_cin = x.narrow(1, t * cin_dimensionsize, cin_dimensionsize)
        for i in range(cout_num):
            x_split_cout = x_split_cin.narrow(0, i * cout_dimensionsize, cout_dimensionsize)
            print(filename, x_split_cout.shape)
            print(filename, x_split_cout.dtype)
            tensor_output(x_split_cout, type, (2, 3, 1, 0),
                          test_name + '/' + filename + '_{}'.format(t) + '_{}'.format(i))


def bias_split(x: torch.Tensor, type: str, test_name: str, filename: str, cout_dimensionsize: int, cout_num: int):
    for i in range(cout_num):
        x_split_cout = x.narrow(0, i * cout_dimensionsize, cout_dimensionsize)
        print(filename, x_split_cout.shape)
        print(filename, x_split_cout.dtype)
        tensor_output(x_split_cout, type, (0,),
                    test_name + '/' + filename + '_{}'.format(i))

