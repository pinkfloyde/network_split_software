import onnx
from onnx import helper

def extract_node_with_all_connections(onnx_model_path, target_node_name, output_path="cropped_model.onnx"):
    """
    导出目标节点、所有输入对应的上游节点和输出对应的下游节点
    
    参数:
    - onnx_model_path: 输入模型路径
    - target_node_name: 目标节点名称
    - output_path: 输出模型路径
    """
    # 加载模型
    model = onnx.load(onnx_model_path)
    all_nodes = model.graph.node
    
    # 构建映射关系
    output_to_node = {}  # 输出张量名称 -> 节点
    input_to_nodes = {}  # 输入张量名称 -> 使用该输入的节点列表
    name_to_node = {}    # 节点名称 -> 节点
    
    for node in all_nodes:
        # 按节点名称索引
        name_to_node[node.name] = node
        
        # 按输出张量名称索引
        for output_name in node.output:
            output_to_node[output_name] = node
        
        # 按输入张量名称索引
        for input_name in node.input:
            if input_name not in input_to_nodes:
                input_to_nodes[input_name] = []
            input_to_nodes[input_name].append(node)
    
    # 方法1：通过节点名称搜索
    if target_node_name in name_to_node:
        target_node = name_to_node[target_node_name]
        print("通过节点名称找到目标节点")
    
    # 收集需要保留的节点
    nodes_to_keep = []
    nodes_to_keep.append(target_node)
    
    # 1. 找到所有输入对应的上游节点
    upstream_nodes = []
    for input_name in target_node.input:
        if input_name in output_to_node:
            upstream_node = output_to_node[input_name]
            nodes_to_keep.append(upstream_node)
            upstream_nodes.append(upstream_node)
            print(f"找到上游节点: {upstream_node.name}")
        else:
            print(f"输入 {input_name} 可能是模型输入或常量")
    
    # 2. 找到所有输出对应的下游节点
    downstream_nodes = []
    for output_name in target_node.output:
        if output_name in input_to_nodes:
            for downstream_node in input_to_nodes[output_name]:
                nodes_to_keep.append(downstream_node)
                downstream_nodes.append(downstream_node)
                print(f"找到下游节点: {downstream_node.name}")

    
    print(f"\n导出完成！")
    print(f"保留了 {len(nodes_to_keep)} 个节点:")
    print(f"  - 目标节点: {target_node.name}")
    print(f"  - 上游节点: {len(upstream_nodes)} 个")
    print(f"  - 下游节点: {len(downstream_nodes)} 个")
    print(f"模型已保存到: {output_path}")   
    return 0

def list_all_node_names(onnx_model_path):
    """列出模型中所有节点的名称和输出名称"""
    model = onnx.load(onnx_model_path)
    
    print("=== 所有节点名称和输出 ===")
    for i, node in enumerate(model.graph.node):
        print(f"节点 {i}: {node.name}")
        print(f"  输入: {node.input}")
        print(f"  输出: {node.output}")
        print("---")
    
    print("\n=== 所有输出张量名称 ===")
    for output in model.graph.output:
        print(f"模型输出: {output.name}")

# 使用示例
#list_all_node_names("D:/eyethink_project/network_split_software/Yolov8n/test_relu200_cle_bias.onnx")

# 使用示例
extract_node_with_all_connections(
    "D:/eyethink_project/network_split_software/Yolov8n/test_relu200_cle_bias.onnx", 
    "/dark2/dark2.1/cv1/conv/Conv", 
    output_path="example.onnx"
)