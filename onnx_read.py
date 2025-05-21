"D:/eyethink project/PTQ/quantization/YOLOv8/test_relu200_cle_bias.onnx"
"D:/eyethink project/eyethink_func_sim/data/yolov8/input.jpg"

import os
import onnx
import copy
import numpy as np
import logging
import onnxruntime
from collections import OrderedDict
from onnx import shape_inference
logging.basicConfig(level=logging.INFO)
from onnx import shape_inference, TensorProto, version_converter, numpy_helper
logger = logging.getLogger("[ONNXOPTIMIZER]")

import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from utils import *

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_test_transform(): 
    return transforms.Compose([
        transforms.Resize([640, 640]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_tensor_by_name(tensor_name, ort_outs, output_names):
    for name, tensor in zip(output_names, ort_outs):
        if name == tensor_name:
            return tensor
    return None

def test_model_by_onnxruntime(model):

    logger.info("Test model by onnxruntime")

    input_shape = model.graph.input[0].type.tensor_type.shape.dim

    image_shape = [x.dim_value for x in input_shape]
    image_shape_new = []
    for x in image_shape:
        if x == 0:
            image_shape_new.append(1)
        else:
            image_shape_new.append(x)
    image_shape = image_shape_new
    image = Image.open("D:/eyethink project/eyethink_func_sim/data/yolov8/input.jpg").convert('RGB')

    img = get_test_transform()(image)
    img = img.unsqueeze_(0)  # -> NCHW, 1,3,224,224

    #img_array = np.array(np.random.random(image_shape), dtype = np.float32)

    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
        for output in node.input:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
            
    ort_session = onnxruntime.InferenceSession(model.SerializeToString())
    # ort_inputs = {}
    # for i, input_ele in enumerate(ort_session.get_inputs()):
    #     ort_inputs[input_ele.name] = to_numpy(img)

    # outputs = [x.name for x in ort_session.get_outputs()]
    # ort_outs = ort_session.run(outputs, ort_inputs)
    # ort_outs = OrderedDict(zip(outputs, ort_outs))
    input_name = ort_session.get_inputs()[0].name

    conv_layers = [node for node in model.graph.node if node.op_type == "Conv"]

    # upstream_input_nodes = []
    # for node in model.graph.node:
    #     for output in node.output:
    #         if output in conv_layers.input[0]:
    #             upstream_input_nodes.append(node)

    # upstream_weight_nodes = []
    # for node in model.graph.node:
    #     for output in node.output:
    #         if output in conv_layers.input[1]:
    #             upstream_weight_nodes.append(node)

    # upstream_bias_nodes = []
    # for node in model.graph.node:
    #     for output in node.output:
    #         if output in conv_layers.input[2]:
    #             upstream_bias_nodes.append(node)

    output_names = [output.name for output in ort_session.get_outputs()]

    for node in conv_layers:
        #print(node.output)
        conv_layer_name = node.name


        #获取上游节点
        #upstream_input_nodes = []
        for a in model.graph.node:
            if node.input[0] in a.output:
                upstream_input_nodes = a
                    

        #upstream_weight_nodes = []
        for a in model.graph.node:
            if node.input[1] in a.output:
                upstream_weight_nodes = a
        
        #upstream_bias_nodes = []
        if(len(node.input)>2):
            for a in model.graph.node:
                if node.input[2] in a.output:
                    upstream_bias_nodes = a

        #downstream_output_nodes = []
        for a in model.graph.node:
            if node.output[0] in a.input:
                downstream_output_nodes = a

        
        
        # 获取卷积层的输入和输出张量名称
        conv_input_name = upstream_input_nodes.input[0]
        conv_input_scale= upstream_input_nodes.input[2]
        conv_weight_name = upstream_weight_nodes.input[0]

        if(len(node.input)>2):
            conv_bias_name = upstream_bias_nodes.input[0]

        conv_output_name = downstream_output_nodes.output[0]
        conv_output_name_fp=node.output[0]

        

        # 获取这些张量的输出
        run_options = onnxruntime.RunOptions()
        run_options.log_severity_level = 3
        ort_outs = ort_session.run([conv_input_name,conv_weight_name, conv_bias_name, conv_output_name,conv_input_scale,conv_output_name_fp], {input_name:to_numpy(img)}, run_options)

        # for i, inp in enumerate(node.input):
        #     print(f"Input {i} of {conv_layer_name}: {inp}")

        #提取weight和bias的值
        # weights = None
        # bias = None
        # for initializer in model.graph.initializer:
        #     if initializer.name == node.input[1]:
        #         weights = np.array(onnx.numpy_helper.to_array(initializer))
        #     if len(node.input) > 2 and initializer.name == node.input[2]:
        #         bias = np.array(onnx.numpy_helper.to_array(initializer))

        base_folder_path = "./data/yolov8/conv"
        file_name_with_dirs = f"{conv_layer_name}_input.txt"
        file_name_with_dirs_weight = f"{conv_layer_name}_weight.txt"
        file_name_with_dirs_bias = f"{conv_layer_name}_bias.txt"
        file_name_with_dirs_output = f"{conv_layer_name}_output.txt"
        #print("file_name_with_dirs:"+file_name_with_dirs)

        # 生成完整的文件路径
        file_path = base_folder_path + file_name_with_dirs
        file_path_weight = base_folder_path + file_name_with_dirs_weight
        file_path_bias = base_folder_path + file_name_with_dirs_bias
        file_path_output= base_folder_path + file_name_with_dirs_output
        #print("file_path:"+file_path)

        folder_path = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        file_name_without_extension = os.path.splitext(file_name)[0]
        file_name_weight = os.path.basename(file_path_weight)
        file_name_weight_without_extension = os.path.splitext(file_name_weight)[0]
        file_name_bias = os.path.basename(file_path_bias)
        file_name_bias_without_extension = os.path.splitext(file_name_bias)[0]
        file_name_output = os.path.basename(file_path_output)
        file_name_output_without_extension = os.path.splitext(file_name_output)[0]
        #print("folder_path:"+folder_path)
        os.makedirs(folder_path, exist_ok=True)
        # 保存输入和输出到各自的 txt 文件
        ort_outs_0_tensor = torch.from_numpy(ort_outs[0]).to(torch.int8)
        ort_outs_1_tensor = torch.from_numpy(ort_outs[1]).to(torch.int8)
        ort_outs_2_tensor = torch.from_numpy(ort_outs[2]).to(torch.int32)
        ort_outs_3_tensor = torch.from_numpy(ort_outs[3]).to(torch.int8)
        ort_outs_4_tensor = torch.from_numpy(ort_outs[4]).to(torch.int8)
        ort_outs_5_tensor = torch.from_numpy(ort_outs[5])

        print("fp",ort_outs_5_tensor)
        print("int",ort_outs_3_tensor)
        if ort_outs[0] is not None:
            #print(ort_outs[0].shape)
            print(ort_outs_0_tensor)
            a=ort_outs_4_tensor
            ort_outs_0_tensor=ort_outs_0_tensor-a
            print("a:",a)
            #print("fp",ort_outs_0_tensor)
            if(ort_outs_1_tensor.shape[0]<=32):
                #input_y拆8份
                c_in=ort_outs_0_tensor.shape[1]
                print("c_in:",c_in)
                input_y=int(ort_outs_0_tensor.shape[2]/8)
                print("input_y:" , input_y)
                input_x=ort_outs_0_tensor.shape[3]
                print("input_x:" , input_x)
                if(ort_outs_0_tensor.shape[1]<32):
                    if(ort_outs_1_tensor.shape[2]==1):
                        in_feature_split(ort_outs_0_tensor, folder_path, file_name_without_extension,1,1,input_y,8,input_x,1,c_in,1,0)
                    else:
                        in_feature_split(ort_outs_0_tensor, folder_path, file_name_without_extension,1,1,input_y,8,input_x,1,c_in,1,1)
                else:
                    if(ort_outs_1_tensor.shape[2]==1):
                        deep_feature_split(ort_outs_0_tensor,'INT8', folder_path, file_name_without_extension,1,1,input_y,8,input_x,1,c_in,1,0)
                    else:
                        deep_feature_split(ort_outs_0_tensor,'INT8', folder_path, file_name_without_extension,1,1,input_y,8,input_x,1,c_in,1,1)
            elif(ort_outs_1_tensor.shape[0]<=64):
                #input_y拆4份
                c_in=int(ort_outs_0_tensor.shape[1])
                print("c_in:",c_in)
                input_y=int(ort_outs_0_tensor.shape[2]/4)
                print("input_y:" , input_y)
                input_x=ort_outs_0_tensor.shape[3]
                print("input_x:" , input_x)
                if(ort_outs_0_tensor.shape[1]<32):
                    if(ort_outs_1_tensor.shape[2]==1):
                        in_feature_split(ort_outs_0_tensor, folder_path, file_name_without_extension,1,1,input_y,4,input_x,1,c_in,1,0)
                    else:
                        in_feature_split(ort_outs_0_tensor, folder_path, file_name_without_extension,1,1,input_y,4,input_x,1,c_in,1,1)
                else:
                    if(ort_outs_1_tensor.shape[2]==1):
                        deep_feature_split(ort_outs_0_tensor,'INT8', folder_path, file_name_without_extension,1,1,input_y,4,input_x,1,c_in,1,0)
                    else:
                        deep_feature_split(ort_outs_0_tensor, 'INT8',folder_path, file_name_without_extension,1,1,input_y,4,input_x,1,c_in,1,1)
            elif(ort_outs_1_tensor.shape[0]<=128):
                #input_y拆2份
                c_in=int(ort_outs_0_tensor.shape[1])
                print("c_in:",c_in)
                input_y=int(ort_outs_0_tensor.shape[2]/2)
                print("input_y:" , input_y)
                input_x=ort_outs_0_tensor.shape[3]
                print("input_x:" , input_x)
                if(ort_outs_0_tensor.shape[1]<32):
                    if(ort_outs_1_tensor.shape[2]==1):
                        in_feature_split(ort_outs_0_tensor, folder_path, file_name_without_extension,1,1,input_y,2,input_x,1,c_in,1,0)
                    else:
                        in_feature_split(ort_outs_0_tensor, folder_path, file_name_without_extension,1,1,input_y,2,input_x,1,c_in,1,1)
                else:
                    if(ort_outs_1_tensor.shape[2]==1):
                        deep_feature_split(ort_outs_0_tensor,'INT8', folder_path, file_name_without_extension,1,1,input_y,2,input_x,1,c_in,1,0)
                    else:
                        deep_feature_split(ort_outs_0_tensor, 'INT8',folder_path, file_name_without_extension,1,1,input_y,2,input_x,1,c_in,1,1)
            else:
                #input_y不拆分，C_in拆8份
                c_in=int(ort_outs_0_tensor.shape[1]/1)
                print("c_in:",c_in)
                input_y=int(ort_outs_0_tensor.shape[2]/1)
                print("input_y:" , input_y)
                input_x=ort_outs_0_tensor.shape[3]
                print("input_x:" , input_x)
                if(ort_outs_0_tensor.shape[1]<32):
                    if(ort_outs_1_tensor.shape[2]==1):
                        in_feature_split(ort_outs_0_tensor, folder_path, file_name_without_extension,1,1,input_y,1,input_x,1,c_in,1,0)
                    else:
                        in_feature_split(ort_outs_0_tensor, folder_path, file_name_without_extension,1,1,input_y,1,input_x,1,c_in,1,1)
                else:
                    if(ort_outs_1_tensor.shape[2]==1):
                        deep_feature_split(ort_outs_0_tensor,'INT8', folder_path, file_name_without_extension,1,1,input_y,1,input_x,1,c_in,1,0)
                    else:
                        deep_feature_split(ort_outs_0_tensor, 'INT8',folder_path, file_name_without_extension,1,1,input_y,1,input_x,1,c_in,1,1)

            if(ort_outs_0_tensor.shape[1]<32):
                tensor_output(x = ort_outs_0_tensor, type='INT8', order=(0,2,1,3), filename=f"./data/yolov8/conv{conv_layer_name}_input")
            else:
                tensor_output(x = ort_outs_0_tensor, type='INT8', order=(0,2,3,1), filename=f"./data/yolov8/conv{conv_layer_name}_input")

        if ort_outs[1] is not None:
            #print(ort_outs[1].shape)
            #print(ort_outs[1])
            if(ort_outs_1_tensor.shape[0]<=32):
                #c_out不拆分
                c_out=ort_outs_1_tensor.shape[0]
                print("c_out:",c_out)
                kernel_y=int(ort_outs_1_tensor.shape[2])
                print("kernel_y:" , kernel_y)
                kernel_x=ort_outs_1_tensor.shape[3]
                print("kernel_x:" , kernel_x)
                c_in=ort_outs_1_tensor.shape[1]
                print("c_in:" , c_in)
                weight_split(ort_outs_1_tensor, 'INT8',folder_path, file_name_weight_without_extension,c_in,1,c_out,1)
            elif(ort_outs_1_tensor.shape[0]<=64):
                #c_out拆2份
                c_out=int(ort_outs_1_tensor.shape[0]/2)
                print("c_out:",c_out)
                kernel_y=int(ort_outs_1_tensor.shape[2])
                print("kernel_y:" , kernel_y)
                kernel_x=ort_outs_1_tensor.shape[3]
                print("kernel_x:" , kernel_x)
                c_in=ort_outs_1_tensor.shape[1]
                print("c_in:" , c_in)
                weight_split(ort_outs_1_tensor, 'INT8',folder_path, file_name_weight_without_extension,c_in,1,c_out,2)
            elif(ort_outs_1_tensor.shape[0]<=128):
                #c_out拆4份
                c_out=int(ort_outs_1_tensor.shape[0]/4)
                print("c_out:",c_out)
                kernel_y=int(ort_outs_1_tensor.shape[2])
                print("kernel_y:" , kernel_y)
                kernel_x=ort_outs_1_tensor.shape[3]
                print("kernel_x:" , kernel_x)
                c_in=ort_outs_1_tensor.shape[1]
                print("c_in:" , c_in)
                weight_split(ort_outs_1_tensor, 'INT8',folder_path, file_name_weight_without_extension,c_in,1,c_out,4)
            else:
                #c_out拆8份
                c_out=int(ort_outs_1_tensor.shape[0]/8)
                print("c_out:",c_out)
                kernel_y=int(ort_outs_1_tensor.shape[2])
                print("kernel_y:" , kernel_y)
                kernel_x=ort_outs_1_tensor.shape[3]
                print("kernel_x:" , kernel_x)
                c_in=ort_outs_1_tensor.shape[1]
                print("c_in:" , c_in)
                weight_split(ort_outs_1_tensor, 'INT8',folder_path, file_name_weight_without_extension,c_in,1,c_out,8)

            tensor_output(x = ort_outs_1_tensor, type='INT8', order=(2,3,1,0), filename=f"./data/yolov8/conv{conv_layer_name}_weight")
            
        if ort_outs[2] is not None:
            print(ort_outs[2].shape)
            if(ort_outs_1_tensor.shape[0]<=32):
                #c_out不拆分
                c_out=ort_outs_2_tensor.shape[0]
                print("c_out:",c_out)
                bias_split(ort_outs_2_tensor, 'INT32',folder_path, file_name_bias_without_extension,c_out,1)
            elif(ort_outs_1_tensor.shape[0]<=64):
                #c_out拆2份
                c_out=int(ort_outs_2_tensor.shape[0]/2)
                print("c_out:",c_out)
                bias_split(ort_outs_2_tensor, 'INT32',folder_path, file_name_bias_without_extension,c_out,2)
            elif(ort_outs_1_tensor.shape[0]<=128):
                #c_out拆4份
                c_out=int(ort_outs_2_tensor.shape[0]/4)
                print("c_out:",c_out)
                bias_split(ort_outs_2_tensor, 'INT32',folder_path, file_name_bias_without_extension,c_out,4)
            else:
                #c_out拆8份
                c_out=int(ort_outs_2_tensor.shape[0]/8)
                print("c_out:",c_out)
                bias_split(ort_outs_2_tensor, 'INT32',folder_path, file_name_bias_without_extension,c_out,8)
            tensor_output(x = ort_outs_2_tensor, type='INT8', order=(0,), filename=f"./data/yolov8/conv{conv_layer_name}_bias")

        if ort_outs[3] is not None:
            print(ort_outs[3])
            if(ort_outs_1_tensor.shape[0]<=32):
                #output_y拆8份
                print(ort_outs_3_tensor.shape)
                c_out=ort_outs_3_tensor.shape[1]
                print("c_out:",c_out)
                output_y=int(ort_outs_3_tensor.shape[2]/8)
                print("output_y:" , output_y)
                output_x=ort_outs_3_tensor.shape[3]
                print("output_x:" , input_x)
                if(ort_outs_0_tensor.shape[1]<32):
                    in_feature_split(ort_outs_3_tensor, folder_path, file_name_output_without_extension,1,1,output_y,8,output_x,1,c_out,1,0)
                else:
                    deep_feature_split(ort_outs_3_tensor,'INT8', folder_path, file_name_output_without_extension,1,1,output_y,8,output_x,1,c_out,1,0)
            elif(ort_outs_1_tensor.shape[0]<=64):
                #output_y拆4份,c_out拆2份
                c_out=int(ort_outs_3_tensor.shape[1]/2)
                print("c_out:",c_out)
                output_y=int(ort_outs_3_tensor.shape[2]/4)
                
                print("output_y:" , output_y)
                output_x=ort_outs_3_tensor.shape[3]
                print("output_x:" , input_x)
                if(ort_outs_0_tensor.shape[1]<32):
                    in_feature_split(ort_outs_3_tensor, folder_path, file_name_output_without_extension,1,1,output_y,4,output_x,1,c_out,2,0)
                else:
                    deep_feature_split(ort_outs_3_tensor,'INT8', folder_path, file_name_output_without_extension,1,1,output_y,4,output_x,1,c_out,2,0)
            elif(ort_outs_1_tensor.shape[0]<=128):
                #output_y拆2份,c_out拆4份
                c_out=int(ort_outs_3_tensor.shape[1]/4)
                print("c_out:",c_out)
                output_y=int(ort_outs_3_tensor.shape[2]/2)
                print("output_y:" , output_y)
                output_x=ort_outs_3_tensor.shape[3]
                print("output_x:" , input_x)
                if(ort_outs_0_tensor.shape[1]<32):
                    in_feature_split(ort_outs_3_tensor, folder_path, file_name_output_without_extension,1,1,output_y,2,output_x,1,c_out,4,0)
                else:
                    deep_feature_split(ort_outs_3_tensor,'INT8', folder_path, file_name_output_without_extension,1,1,output_y,2,output_x,1,c_out,4,0)
            else:
                #output_y不拆分,c_out拆8份
                c_out=int(ort_outs_3_tensor.shape[1]/8)
                print("c_out:",c_out)
                output_y=int(ort_outs_3_tensor.shape[2]/1)
                print("output_y:" , output_y)
                output_x=ort_outs_3_tensor.shape[3]
                print("output_x:" , input_x)
                if(ort_outs_0_tensor.shape[1]<32):
                    in_feature_split(ort_outs_3_tensor, folder_path, file_name_output_without_extension,1,1,output_y,1,output_x,1,c_out,8,0)
                else:
                    deep_feature_split(ort_outs_3_tensor,'INT8', folder_path, file_name_output_without_extension,1,1,output_y,1,output_x,1,c_out,8,0)

            tensor_output(x = ort_outs_3_tensor, type='INT8', order=(0,2,3,1), filename=f"./data/yolov8/conv{conv_layer_name}_output")


        print(f"Saved {conv_layer_name} input and weight and bias and output to txt files")
 
    logger.info("Test model by onnxruntime success")

    return ort_outs


def test_model_by_onnxruntime_fp32(model):

    logger.info("Test model by onnxruntime")

    input_shape = model.graph.input[0].type.tensor_type.shape.dim

    image_shape = [x.dim_value for x in input_shape]
    image_shape_new = []
    for x in image_shape:
        if x == 0:
            image_shape_new.append(1)
        else:
            image_shape_new.append(x)
    image_shape = image_shape_new
    image = Image.open("D:/eyethink project/eyethink_func_sim/data/yolov8/input.jpg").convert('RGB')

    img = get_test_transform()(image)
    img = img.unsqueeze_(0)  # -> NCHW, 1,3,224,224

    #img_array = np.array(np.random.random(image_shape), dtype = np.float32)

    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
        for output in node.input:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
            
    ort_session = onnxruntime.InferenceSession(model.SerializeToString())
    # ort_inputs = {}
    # for i, input_ele in enumerate(ort_session.get_inputs()):
    #     ort_inputs[input_ele.name] = to_numpy(img)

    # outputs = [x.name for x in ort_session.get_outputs()]
    # ort_outs = ort_session.run(outputs, ort_inputs)
    # ort_outs = OrderedDict(zip(outputs, ort_outs))
    input_name = ort_session.get_inputs()[0].name

    conv_layers = [node for node in model.graph.node if node.op_type == "Conv"]

    # upstream_input_nodes = []
    # for node in model.graph.node:
    #     for output in node.output:
    #         if output in conv_layers.input[0]:
    #             upstream_input_nodes.append(node)

    # upstream_weight_nodes = []
    # for node in model.graph.node:
    #     for output in node.output:
    #         if output in conv_layers.input[1]:
    #             upstream_weight_nodes.append(node)

    # upstream_bias_nodes = []
    # for node in model.graph.node:
    #     for output in node.output:
    #         if output in conv_layers.input[2]:
    #             upstream_bias_nodes.append(node)

    output_names = [output.name for output in ort_session.get_outputs()]

    for node in conv_layers:
        #print(node.output)
        conv_layer_name = node.name


        #获取上游节点
        
        
        # 获取卷积层的输入和输出张量名称
        conv_input_name = node.input[0]
        conv_weight_name = node.input[1]

        if(len(node.input)>2):
            conv_bias_name = node.input[2]

        conv_output_name = node.output[0]

        

        # 获取这些张量的输出
        run_options = onnxruntime.RunOptions()
        run_options.log_severity_level = 3
        ort_outs = ort_session.run([conv_input_name,conv_weight_name, conv_bias_name, conv_output_name], {input_name:to_numpy(img)}, run_options)

        # for i, inp in enumerate(node.input):
        #     print(f"Input {i} of {conv_layer_name}: {inp}")

        #提取weight和bias的值
        # weights = None
        # bias = None
        # for initializer in model.graph.initializer:
        #     if initializer.name == node.input[1]:
        #         weights = np.array(onnx.numpy_helper.to_array(initializer))
        #     if len(node.input) > 2 and initializer.name == node.input[2]:
        #         bias = np.array(onnx.numpy_helper.to_array(initializer))

        base_folder_path = "./data/yolov8/conv_fp32"
        file_name_with_dirs = f"{conv_layer_name}_input.txt"
        file_name_with_dirs_weight = f"{conv_layer_name}_weight.txt"
        file_name_with_dirs_bias = f"{conv_layer_name}_bias.txt"
        file_name_with_dirs_output = f"{conv_layer_name}_output.txt"
        #print("file_name_with_dirs:"+file_name_with_dirs)

        # 生成完整的文件路径
        file_path = base_folder_path + file_name_with_dirs
        file_path_weight = base_folder_path + file_name_with_dirs_weight
        file_path_bias = base_folder_path + file_name_with_dirs_bias
        file_path_output= base_folder_path + file_name_with_dirs_output
        #print("file_path:"+file_path)

        folder_path = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        file_name_without_extension = os.path.splitext(file_name)[0]
        file_name_weight = os.path.basename(file_path_weight)
        file_name_weight_without_extension = os.path.splitext(file_name_weight)[0]
        file_name_bias = os.path.basename(file_path_bias)
        file_name_bias_without_extension = os.path.splitext(file_name_bias)[0]
        file_name_output = os.path.basename(file_path_output)
        file_name_output_without_extension = os.path.splitext(file_name_output)[0]
        #print("folder_path:"+folder_path)
        os.makedirs(folder_path, exist_ok=True)
        # 保存输入和输出到各自的 txt 文件
        ort_outs_0_tensor = torch.from_numpy(ort_outs[0])
        ort_outs_1_tensor = torch.from_numpy(ort_outs[1])
        ort_outs_2_tensor = torch.from_numpy(ort_outs[2])
        ort_outs_3_tensor = torch.from_numpy(ort_outs[3])

        print("int",ort_outs_3_tensor)
        if ort_outs[0] is not None:
            #print(ort_outs[0].shape)
            print(ort_outs_0_tensor)
            #print("fp",ort_outs_0_tensor)
            if(ort_outs_1_tensor.shape[0]<=32):
                #input_y拆8份
                c_in=ort_outs_0_tensor.shape[1]
                print("c_in:",c_in)
                input_y=int(ort_outs_0_tensor.shape[2]/8)
                print("input_y:" , input_y)
                input_x=ort_outs_0_tensor.shape[3]
                print("input_x:" , input_x)
                if(ort_outs_0_tensor.shape[1]<32):
                    if(ort_outs_1_tensor.shape[2]==1):
                        in_feature_split(ort_outs_0_tensor, folder_path, file_name_without_extension,1,1,input_y,8,input_x,1,c_in,1,0)
                    else:
                        in_feature_split(ort_outs_0_tensor, folder_path, file_name_without_extension,1,1,input_y,8,input_x,1,c_in,1,1)
                else:
                    if(ort_outs_1_tensor.shape[2]==1):
                        deep_feature_split(ort_outs_0_tensor,'FP32', folder_path, file_name_without_extension,1,1,input_y,8,input_x,1,c_in,1,0)
                    else:
                        deep_feature_split(ort_outs_0_tensor,'FP32', folder_path, file_name_without_extension,1,1,input_y,8,input_x,1,c_in,1,1)
            elif(ort_outs_1_tensor.shape[0]<=64):
                #input_y拆4份
                c_in=int(ort_outs_0_tensor.shape[1])
                print("c_in:",c_in)
                input_y=int(ort_outs_0_tensor.shape[2]/4)
                print("input_y:" , input_y)
                input_x=ort_outs_0_tensor.shape[3]
                print("input_x:" , input_x)
                if(ort_outs_0_tensor.shape[1]<32):
                    if(ort_outs_1_tensor.shape[2]==1):
                        in_feature_split(ort_outs_0_tensor, folder_path, file_name_without_extension,1,1,input_y,4,input_x,1,c_in,1,0)
                    else:
                        in_feature_split(ort_outs_0_tensor, folder_path, file_name_without_extension,1,1,input_y,4,input_x,1,c_in,1,1)
                else:
                    if(ort_outs_1_tensor.shape[2]==1):
                        deep_feature_split(ort_outs_0_tensor,'FP32', folder_path, file_name_without_extension,1,1,input_y,4,input_x,1,c_in,1,0)
                    else:
                        deep_feature_split(ort_outs_0_tensor, 'FP32',folder_path, file_name_without_extension,1,1,input_y,4,input_x,1,c_in,1,1)
            elif(ort_outs_1_tensor.shape[0]<=128):
                #input_y拆2份
                c_in=int(ort_outs_0_tensor.shape[1])
                print("c_in:",c_in)
                input_y=int(ort_outs_0_tensor.shape[2]/2)
                print("input_y:" , input_y)
                input_x=ort_outs_0_tensor.shape[3]
                print("input_x:" , input_x)
                if(ort_outs_0_tensor.shape[1]<32):
                    if(ort_outs_1_tensor.shape[2]==1):
                        in_feature_split(ort_outs_0_tensor, folder_path, file_name_without_extension,1,1,input_y,2,input_x,1,c_in,1,0)
                    else:
                        in_feature_split(ort_outs_0_tensor, folder_path, file_name_without_extension,1,1,input_y,2,input_x,1,c_in,1,1)
                else:
                    if(ort_outs_1_tensor.shape[2]==1):
                        deep_feature_split(ort_outs_0_tensor,'FP32', folder_path, file_name_without_extension,1,1,input_y,2,input_x,1,c_in,1,0)
                    else:
                        deep_feature_split(ort_outs_0_tensor, 'FP32',folder_path, file_name_without_extension,1,1,input_y,2,input_x,1,c_in,1,1)
            else:
                #input_y不拆分，C_in拆8份
                c_in=int(ort_outs_0_tensor.shape[1]/1)
                print("c_in:",c_in)
                input_y=int(ort_outs_0_tensor.shape[2]/1)
                print("input_y:" , input_y)
                input_x=ort_outs_0_tensor.shape[3]
                print("input_x:" , input_x)
                if(ort_outs_0_tensor.shape[1]<32):
                    if(ort_outs_1_tensor.shape[2]==1):
                        in_feature_split(ort_outs_0_tensor, folder_path, file_name_without_extension,1,1,input_y,1,input_x,1,c_in,1,0)
                    else:
                        in_feature_split(ort_outs_0_tensor, folder_path, file_name_without_extension,1,1,input_y,1,input_x,1,c_in,1,1)
                else:
                    if(ort_outs_1_tensor.shape[2]==1):
                        deep_feature_split(ort_outs_0_tensor,'FP32', folder_path, file_name_without_extension,1,1,input_y,1,input_x,1,c_in,1,0)
                    else:
                        deep_feature_split(ort_outs_0_tensor, 'FP32',folder_path, file_name_without_extension,1,1,input_y,1,input_x,1,c_in,1,1)

            if(ort_outs_0_tensor.shape[1]<32):
                tensor_output(x = ort_outs_0_tensor, type='FP32', order=(0,2,1,3), filename=f"./data/yolov8/conv{conv_layer_name}_input")
            else:
                tensor_output(x = ort_outs_0_tensor, type='FP32', order=(0,2,3,1), filename=f"./data/yolov8/conv{conv_layer_name}_input")

        if ort_outs[1] is not None:
            #print(ort_outs[1].shape)
            #print(ort_outs[1])
            if(ort_outs_1_tensor.shape[0]<=32):
                #c_out不拆分
                c_out=ort_outs_1_tensor.shape[0]
                print("c_out:",c_out)
                kernel_y=int(ort_outs_1_tensor.shape[2])
                print("kernel_y:" , kernel_y)
                kernel_x=ort_outs_1_tensor.shape[3]
                print("kernel_x:" , kernel_x)
                c_in=ort_outs_1_tensor.shape[1]
                print("c_in:" , c_in)
                weight_split(ort_outs_1_tensor, 'FP32',folder_path, file_name_weight_without_extension,c_in,1,c_out,1)
            elif(ort_outs_1_tensor.shape[0]<=64):
                #c_out拆2份
                c_out=int(ort_outs_1_tensor.shape[0]/2)
                print("c_out:",c_out)
                kernel_y=int(ort_outs_1_tensor.shape[2])
                print("kernel_y:" , kernel_y)
                kernel_x=ort_outs_1_tensor.shape[3]
                print("kernel_x:" , kernel_x)
                c_in=ort_outs_1_tensor.shape[1]
                print("c_in:" , c_in)
                weight_split(ort_outs_1_tensor, 'FP32',folder_path, file_name_weight_without_extension,c_in,1,c_out,2)
            elif(ort_outs_1_tensor.shape[0]<=128):
                #c_out拆4份
                c_out=int(ort_outs_1_tensor.shape[0]/4)
                print("c_out:",c_out)
                kernel_y=int(ort_outs_1_tensor.shape[2])
                print("kernel_y:" , kernel_y)
                kernel_x=ort_outs_1_tensor.shape[3]
                print("kernel_x:" , kernel_x)
                c_in=ort_outs_1_tensor.shape[1]
                print("c_in:" , c_in)
                weight_split(ort_outs_1_tensor, 'FP32',folder_path, file_name_weight_without_extension,c_in,1,c_out,4)
            else:
                #c_out拆8份
                c_out=int(ort_outs_1_tensor.shape[0]/8)
                print("c_out:",c_out)
                kernel_y=int(ort_outs_1_tensor.shape[2])
                print("kernel_y:" , kernel_y)
                kernel_x=ort_outs_1_tensor.shape[3]
                print("kernel_x:" , kernel_x)
                c_in=ort_outs_1_tensor.shape[1]
                print("c_in:" , c_in)
                weight_split(ort_outs_1_tensor, 'FP32',folder_path, file_name_weight_without_extension,c_in,1,c_out,8)

            tensor_output(x = ort_outs_1_tensor, type='FP32', order=(2,3,1,0), filename=f"./data/yolov8/conv{conv_layer_name}_weight")
            
        if ort_outs[2] is not None:
            print(ort_outs[2].shape)
            if(ort_outs_1_tensor.shape[0]<=32):
                #c_out不拆分
                c_out=ort_outs_2_tensor.shape[0]
                print("c_out:",c_out)
                bias_split(ort_outs_2_tensor, 'FP32',folder_path, file_name_bias_without_extension,c_out,1)
            elif(ort_outs_1_tensor.shape[0]<=64):
                #c_out拆2份
                c_out=int(ort_outs_2_tensor.shape[0]/2)
                print("c_out:",c_out)
                bias_split(ort_outs_2_tensor, 'FP32',folder_path, file_name_bias_without_extension,c_out,2)
            elif(ort_outs_1_tensor.shape[0]<=128):
                #c_out拆4份
                c_out=int(ort_outs_2_tensor.shape[0]/4)
                print("c_out:",c_out)
                bias_split(ort_outs_2_tensor, 'FP32',folder_path, file_name_bias_without_extension,c_out,4)
            else:
                #c_out拆8份
                c_out=int(ort_outs_2_tensor.shape[0]/8)
                print("c_out:",c_out)
                bias_split(ort_outs_2_tensor, 'FP32',folder_path, file_name_bias_without_extension,c_out,8)
            tensor_output(x = ort_outs_2_tensor, type='FP32', order=(0,), filename=f"./data/yolov8/conv{conv_layer_name}_bias")

        if ort_outs[3] is not None:
            print(ort_outs[3])
            if(ort_outs_1_tensor.shape[0]<=32):
                #output_y拆8份
                print(ort_outs_3_tensor.shape)
                c_out=ort_outs_3_tensor.shape[1]
                print("c_out:",c_out)
                output_y=int(ort_outs_3_tensor.shape[2]/8)
                print("output_y:" , output_y)
                output_x=ort_outs_3_tensor.shape[3]
                print("output_x:" , input_x)
                if(ort_outs_0_tensor.shape[1]<32):
                    in_feature_split(ort_outs_3_tensor, folder_path, file_name_output_without_extension,1,1,output_y,8,output_x,1,c_out,1,0)
                else:
                    deep_feature_split(ort_outs_3_tensor,'FP32', folder_path, file_name_output_without_extension,1,1,output_y,8,output_x,1,c_out,1,0)
            elif(ort_outs_1_tensor.shape[0]<=64):
                #output_y拆4份,c_out拆2份
                c_out=int(ort_outs_3_tensor.shape[1]/2)
                print("c_out:",c_out)
                output_y=int(ort_outs_3_tensor.shape[2]/4)
                print("output_y:" , output_y)
                output_x=ort_outs_3_tensor.shape[3]
                print("output_x:" , input_x)
                if(ort_outs_0_tensor.shape[1]<32):
                    in_feature_split(ort_outs_3_tensor, folder_path, file_name_output_without_extension,1,1,output_y,4,output_x,1,c_out,2,0)
                else:
                    deep_feature_split(ort_outs_3_tensor,'FP32', folder_path, file_name_output_without_extension,1,1,output_y,4,output_x,1,c_out,2,0)
            elif(ort_outs_1_tensor.shape[0]<=128):
                #output_y拆2份,c_out拆4份
                c_out=int(ort_outs_3_tensor.shape[1]/4)
                print("c_out:",c_out)
                output_y=int(ort_outs_3_tensor.shape[2]/2)
                print("output_y:" , output_y)
                output_x=ort_outs_3_tensor.shape[3]
                print("output_x:" , input_x)
                if(ort_outs_0_tensor.shape[1]<32):
                    in_feature_split(ort_outs_3_tensor, folder_path, file_name_output_without_extension,1,1,output_y,2,output_x,1,c_out,4,0)
                else:
                    deep_feature_split(ort_outs_3_tensor,'FP32', folder_path, file_name_output_without_extension,1,1,output_y,2,output_x,1,c_out,4,0)
            else:
                #output_y不拆分,c_out拆8份
                c_out=int(ort_outs_3_tensor.shape[1]/8)
                print("c_out:",c_out)
                output_y=int(ort_outs_3_tensor.shape[2]/1)
                print("output_y:" , output_y)
                output_x=ort_outs_3_tensor.shape[3]
                print("output_x:" , input_x)
                if(ort_outs_0_tensor.shape[1]<32):
                    in_feature_split(ort_outs_3_tensor, folder_path, file_name_output_without_extension,1,1,output_y,1,output_x,1,c_out,8,0)
                else:
                    deep_feature_split(ort_outs_3_tensor,'FP32', folder_path, file_name_output_without_extension,1,1,output_y,1,output_x,1,c_out,8,0)

            tensor_output(x = ort_outs_3_tensor, type='FP32', order=(0,2,3,1), filename=f"./data/yolov8/conv{conv_layer_name}_output")


        print(f"Saved {conv_layer_name} input and weight and bias and output to txt files")
 
    logger.info("Test model by onnxruntime success")

    return ort_outs

onnx_model = onnx.load("D:/eyethink project/PTQ/quantization/YOLOv8/test_relu200_cle_bias.onnx")
ort_outs = test_model_by_onnxruntime(onnx_model)
#ort_outs = test_model_by_onnxruntime_fp32(onnx_model)


