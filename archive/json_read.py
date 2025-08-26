import json
import os
from typing import Dict, List, Any

def wirte_to_file(input_file: str, binary_array):
    bit_64_string = ''.join(['0' for _ in range(64)])
    with open(input_file, 'w', encoding='utf-8') as f:
            for i in range(len(binary_array)):
                if binary_array[i] is not None:
                    binary_str = binary_array[i]
                    # 逐个字符处理：将每个二进制字符转换为对应的16进制字符
                    hex_chars = []
                    for j in range(0, len(binary_str), 4):
                        # 取4位二进制
                        four_bits = binary_str[j:j+4]
                        # 将4位二进制转换为1位16进制
                        if four_bits == "0000":
                            hex_chars.append("0")
                        elif four_bits == "0001":
                            hex_chars.append("1")
                        elif four_bits == "0010":
                            hex_chars.append("2")
                        elif four_bits == "0011":
                            hex_chars.append("3")
                        elif four_bits == "0100":
                            hex_chars.append("4")
                        elif four_bits == "0101":
                            hex_chars.append("5")
                        elif four_bits == "0110":
                            hex_chars.append("6")
                        elif four_bits == "0111":
                            hex_chars.append("7")
                        elif four_bits == "1000":
                            hex_chars.append("8")
                        elif four_bits == "1001":
                            hex_chars.append("9")
                        elif four_bits == "1010":
                            hex_chars.append("a")
                        elif four_bits == "1011":
                            hex_chars.append("b")
                        elif four_bits == "1100":
                            hex_chars.append("c")
                        elif four_bits == "1101":
                            hex_chars.append("d")
                        elif four_bits == "1110":
                            hex_chars.append("e")
                        elif four_bits == "1111":
                            hex_chars.append("f")
                    
                    # 将16进制字符列表连接成字符串
                    hex_str = ''.join(hex_chars)
                    f.write(hex_str + '\n')
                else:
                    f.write(bit_64_string+'\n')

def process_prim_list_data(input_file: str, output_dir: str = "output"):
    """
    处理包含main_prim_list的JSON数据，按二进制字符串内容分类写入文件
    
    Args:
        input_file (str): 输入的JSON文件路径
        output_dir (str): 输出目录，默认为"output"
    """
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查是否存在main_prim_list字段
        if 'main_prim_list' not in data:
            print("错误：JSON文件中没有找到 'main_prim_list' 字段")
            return
        
        prim_list = data['main_prim_list']
        private_SPM = data['private_data_pool']
        shared_SPM = data['shared_data_pool']
        
        if not isinstance(prim_list, list):
            print("错误：main_prim_list 不是列表格式")
            return
        
        instruction_pool_max_address = 100
        instruction_pool_array_0 = [None] * (instruction_pool_max_address + 1)
        instruction_pool_array_1 = [None] * (instruction_pool_max_address + 1)
        instruction_pool_array_2 = [None] * (instruction_pool_max_address + 1)
        instruction_pool_array_3 = [None] * (instruction_pool_max_address + 1)

        Score_max_address = 10000
        Score_array_0 = [None] * (Score_max_address + 1)
        Score_array_1 = [None] * (Score_max_address + 1)
        Score_array_2 = [None] * (Score_max_address + 1)
        Score_array_3 = [None] * (Score_max_address + 1)

        Bcore_max_address = 100
        Bcore_array_0 = [None] * (Bcore_max_address + 1)

        for item in prim_list:
            if (isinstance(item, list) and len(item) >= 2 and 
                isinstance(item[0], list) and len(item[0]) >= 2 and
                isinstance(item[1], list) and len(item[1]) >= 1):
                
                bank = item[0][0]  # 坐标信息 [x, y]
                address=item[0][1]
                binary_string = item[1][0]  # 二进制字符串
                if(bank==0):
                    instruction_pool_array_0[address] = binary_string
                elif(bank==1):
                    instruction_pool_array_1[address] = binary_string
                elif(bank==2):
                    instruction_pool_array_2[address] = binary_string
                elif(bank==3):
                    instruction_pool_array_3[address] = binary_string

        for item in private_SPM:
            if (isinstance(item, list) and len(item) >= 2 and 
                isinstance(item[0], list) and len(item[0]) >= 2 and
                isinstance(item[1], list) and len(item[1]) >= 1):
                
                bank = item[0][0]  # 坐标信息 [x, y]
                address=item[0][1]  
                if(bank==0):
                    for i in range(len(item[1])):
                        Score_array_0[address+i] = item[1][i]
                elif(bank==1):
                    for i in range(len(item[1])):
                        Score_array_1[address+i] = item[1][i]
                elif(bank==2):
                    for i in range(len(item[1])):
                        Score_array_2[address+i] = item[1][i]
                elif(bank==3):
                    for i in range(len(item[1])):
                        Score_array_3[address+i] = item[1][i]
        
        for item in shared_SPM:
            if (isinstance(item, list) and len(item) >= 2 and 
                isinstance(item[0], list) and len(item[0]) >= 2 and
                isinstance(item[1], list) and len(item[1]) >= 1):
                
                bank = item[0][0]  # 坐标信息 [x, y]
                address=item[0][1]  
                if(bank==0):
                    for i in range(len(item[1])):
                        Bcore_array_0[address+i] = item[1][i]
        
        # 创建输出文件
        instruction_pool_0 = "D:/eyethink_project/network_split_software/test_data/test_single_prim/instruction_pool_0.txt"
        wirte_to_file(instruction_pool_0,instruction_pool_array_0)
        instruction_pool_1 = "D:/eyethink_project/network_split_software/test_data/test_single_prim/instruction_pool_1.txt"
        wirte_to_file(instruction_pool_1,instruction_pool_array_1)
        instruction_pool_2 = "D:/eyethink_project/network_split_software/test_data/test_single_prim/instruction_pool_2.txt"
        wirte_to_file(instruction_pool_2,instruction_pool_array_2)
        instruction_pool_3 = "D:/eyethink_project/network_split_software/test_data/test_single_prim/instruction_pool_3.txt"
        wirte_to_file(instruction_pool_3,instruction_pool_array_3)

        Score_0 = "D:/eyethink_project/network_split_software/test_data/test_single_prim/Score_0.txt"
        wirte_to_file(Score_0,Score_array_0)
        Score_1 = "D:/eyethink_project/network_split_software/test_data/test_single_prim/Score_1.txt"
        wirte_to_file(Score_1,Score_array_1)
        Score_2 = "D:/eyethink_project/network_split_software/test_data/test_single_prim/Score_2.txt"
        wirte_to_file(Score_2,Score_array_2)
        Score_3 = "D:/eyethink_project/network_split_software/test_data/test_single_prim/Score_3.txt"
        wirte_to_file(Score_3,Score_array_3)

        Bcore_0 = "D:/eyethink_project/network_split_software/test_data/test_single_prim/Bcore_0.txt"
        wirte_to_file(Bcore_0,Bcore_array_0)
        
    except FileNotFoundError:
        print(f"错误：文件 {input_file} 不存在")
    except json.JSONDecodeError:
        print(f"错误：文件 {input_file} 不是有效的JSON格式")
    except Exception as e:
        print(f"处理过程中发生错误：{str(e)}")

# 使用示例
if __name__ == "__main__":
    # 示例1：按二进制字符串模式分组
    process_prim_list_data("D:/eyethink_project/network_split_software/test_data/json/Bcore_test.json")
