import random

def generate_256bit_binary():
    # 生成一个256位的二进制字符串（由'0'和'1'组成）
    return ''.join(random.choice('01') for _ in range(256))

def binary_to_hex(binary_str):
    # 将二进制字符串转换为16进制字符串
    # 首先确保长度是4的倍数（因为每4位二进制对应1位16进制）
    padding = (4 - len(binary_str) % 4) % 4
    padded_binary = binary_str.zfill(len(binary_str) + padding)
    
    # 转换为16进制并去掉前导的0
    hex_str = hex(int(padded_binary, 2))[2:]
    
    # 确保长度为64个字符（因为256位二进制=64位16进制）
    return hex_str.zfill(64)

def write_hex_numbers_to_file(filename, count):
    with open(filename, 'w') as file:
        for _ in range(count):
            binary_num = generate_256bit_binary()
            hex_num = binary_to_hex(binary_num)
            file.write(hex_num + '\n')

# 参数设置
output_filename = 'hex_numbers.txt'
number_of_numbers = 10  # 想要生成的16进制数数量

# 生成并写入文件
write_hex_numbers_to_file(output_filename, number_of_numbers)

print(f"已生成 {number_of_numbers} 个256位二进制数(64位16进制)并写入 {output_filename}")