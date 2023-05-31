from asyncore import write
import csv
import json
import os
import random
import pandas as pd
csv.field_size_limit(500 * 1024 * 1024)

TOPK = 1000

def read_csv(r_path):
    content_list = []
    with open(r_path) as r:
        reader = csv.reader(r)
        for row in reader:
            if isinstance(row, (list, tuple)) and len(row) == 1:
                row = row[0]
            content_list.append(row)
    return content_list

def parentheses_change(s):
    # change '(x, y)' to '[x, y]'
    s_length = len(s)
    ret = ""
    for i in range(s_length):
        if s[i] == '(' and s[i+1] != '[':
            ret += "["
        elif s[i] == ')' and s[i-1] != ']':
            ret += "]"
        else:
            ret += s[i]
    return ret

#分离参数组
def split_arg(conv2d_info):
    ret = []
    info_length = len(conv2d_info)
    start = 1
    i = 0
    while i < info_length:
        if conv2d_info[i] == ',' and conv2d_info[i-1] == ']':
            ret.append(parentheses_change(conv2d_info[start: i-1]))
            i += 2
            start = i+1
        i += 1
    ret.append(parentheses_change(conv2d_info[start: info_length-1]))
    return ret

#提取算子参数
def find_conv2d_index(content_list):
    content_list_length = len(content_list)
    for i in range(content_list_length):
        if content_list[i][0] == 'conv2d':
            return i
    return -1

def string2list(s):
    ret = []
    s_l = len(s)
    start = 1
    for i in range(s_l):
        if s[i] == ',':
            ret.append(int(s[start:i]))
            if s[i + 1] == ' ':
                start = i + 2
            else:
                start = i + 1
    ret.append(int(s[start:s_l-1]))
    return ret

def update_rankk(topk_info, item_info_num):
    min = item_info_num[topk_info[0]]
    ret_idx = -1
    ret_num = -1

    for i in topk_info:
        if item_info_num[i] <= min:
            ret_idx = topk_info.index(i)
            min = item_info_num[i]
            ret_num = min

    assert ret_idx != -1 and ret_num != -1, "update rank k error."
    
    return ret_num, ret_idx

def sort_conv2d_info(conv2d_info):
    topk_info = []
    rankk_idx = -1
    rankk_num = -1

    item_info_set = []
    item_info_num = []

    for item in conv2d_info:
        if item in item_info_set:
            idx = item_info_set.index(item)
            item_info_num[idx] += 1
            if rankk_num != -1 and rankk_num < item_info_num[idx]:
                if idx not in topk_info:
                    topk_info[rankk_idx] = idx
                rankk_num, rankk_idx = update_rankk(topk_info, item_info_num)
        else:
            # deal with '[]'
            if (len(item)) == 0:
                print("sort_conv2d_info : item = " + item + " too short!\n")
                continue

            item_info_set.append(item)
            item_info_num.append(1)
            if len(topk_info) < TOPK:
                topk_info.append(len(item_info_set) - 1)
                rankk_idx = len(topk_info) - 1
                rankk_num = 1

    ret_info = []
    ret_calls = []
    ret_raw_info = []
    for i in topk_info:
        item_dict = {"input_size": [], "kernel_size": [], \
                "bias": [], "stride": [], "padding": [],  \
                "dilation": [], "groups": []}
        item_info = ["input_size", "kernel_size", "bias", \
            "stride", "padding", "dilation", "groups"]
        get_info_dict(item_dict, item_info, [item_info_set[i]])
        ret_info.append(item_dict)
        ret_calls.append(item_info_num[i])
        ret_raw_info.append(item_info_set[i])

    return ret_raw_info, ret_info, ret_calls

def sort_kernelsize_info(topk_info, topk_calls, raw_topk_info):
    kernelsize_set = []
    kernelsize_num = []
    new_topk_info = []
    new_topk_calls = []
    rawkernelsize_topk_info = []
    # print(topk_calls)
    print("--------------------------------")
    print(len(topk_info))
    print("--------------------------------")
    for i in range(len(topk_info)):
        if len(topk_info[i]["kernel_size"]) > 0:
            kernel_size = str(topk_info[i]["kernel_size"][0][2])+','+str(topk_info[i]["kernel_size"][0][3])
        else:
            print("sort_kernelsize_info : " + str(
                    i) + " : " + str(topk_info[i]["kernel_size"]) + " too short!\n")
            continue

        if kernel_size not in kernelsize_set:
            kernelsize_set.append(kernel_size)
            kernelsize_num.append(topk_calls[i])
        else:
            idx = kernelsize_set.index(kernel_size)
            kernelsize_num[idx] += topk_calls[i]
        if topk_info[i] not in new_topk_info:
            new_topk_info.append(topk_info[i])
            new_topk_calls.append(topk_calls[i])
            rawkernelsize_topk_info.append(raw_topk_info[i])
        else:
            idx = new_topk_info.index(topk_info[i])
            new_topk_calls[idx] += topk_calls[idx]
    return kernelsize_set, kernelsize_num, new_topk_info, new_topk_calls, rawkernelsize_topk_info

def sort_inputchannel_info(topk_info, topk_calls):
    inputchannel_set = []
    inputchannel_num = []
    print(len(topk_info))
    for i in range(len(topk_info)):

        inputchannel = topk_info[i]["input_size"][0][1]
        # print(inputchannel, ' ')
        if inputchannel not in inputchannel_set:
            inputchannel_set.append(inputchannel)
            inputchannel_num.append(topk_calls[i])
        else:
            idx = inputchannel_set.index(inputchannel)
            inputchannel_num[idx] += topk_calls[i]
    # print(inputchannel_set)
    # print(inputchannel_num)
    return inputchannel_set, inputchannel_num

def sort_outputchannel_info(topk_info, topk_calls):
    outputchannel_set = []
    outputchannel_num = []
    for i in range(len(topk_info)):

        outputchannel = topk_info[i]["kernel_size"][0][0]
        if outputchannel not in outputchannel_set:
            outputchannel_set.append(outputchannel)
            outputchannel_num.append(topk_calls[i])
        else:
            idx = outputchannel_set.index(outputchannel)
            outputchannel_num[idx] += topk_calls[i]
    
    return outputchannel_set, outputchannel_num

def sort_inoutchannel_info(topk_info, topk_calls):
    inoutchannel_set = []
    inoutchannel_num = []
    for i in range(len(topk_info)):
        inoutchannel = str(topk_info[i]["input_size"][0][1]) + ',' + str(topk_info[i]["kernel_size"][0][0])
        if inoutchannel not in inoutchannel_set:
            inoutchannel_set.append(inoutchannel)
            inoutchannel_num.append(topk_calls[i])
        else:
            idx = inoutchannel_set.index(inoutchannel)
            inoutchannel_num[idx] += topk_calls[i]
    
    return inoutchannel_set, inoutchannel_num

def sort_iH_info(topk_info, topk_calls):
    iH_set = []
    iH_num = []
    print(len(topk_info))
    for i in range(len(topk_info)):

        iH = topk_info[i]["input_size"][0][2]
        # print(inputchannel, ' ')
        if iH not in iH_set:
            iH_set.append(iH)
            iH_num.append(topk_calls[i])
        else:
            idx = iH_set.index(iH)
            iH_num[idx] += topk_calls[i]

    return iH_set, iH_num

def get_info_dict(item_dict, item_info, conv2d_info):
    for conv2d_info_iter in conv2d_info:
        item_i = 0
        i = 0
        info_length = len(conv2d_info_iter)
        start = 0
        while i < info_length:
            if conv2d_info_iter[i] == '[':
                start = i
            elif conv2d_info_iter[i] == ']':
                item_dict[item_info[item_i]].append(string2list(conv2d_info_iter[start:i+1]))
                i += 2
                start = i+1
                item_i += 1
            elif conv2d_info_iter[i] == '/' and conv2d_info_iter[i-1] != ')' and conv2d_info_iter[i-1] != ']': # bias
                if conv2d_info_iter[i-4: i] == "None":
                    item_dict[item_info[item_i]].append([False])
                else:
                    item_dict[item_info[item_i]].append([True])
                item_i += 1
            i += 1

        data = conv2d_info_iter[start:]
        if len(data) == 0:
            print("get_info_dict : conv2d_info_iter[start:] : " + str(
                start) + " : " + str(conv2d_info_iter[start:]) + " item_dict not full!\n")
            continue
        else:
            i_start = 0
            for i in range(len(data)):
                if data[i] == '/':
                    item_dict[item_info[item_i]].append(int(data[i_start:i]))
                    # i += 2
                    i_start = i + 1
                    item_i += 1
            item_dict[item_info[item_i]].append([int(data[i_start:])])  # groups
        
    return item_dict

# 将所有文件的路径放入到listcsv列表中
def list_dir(file_dir):
    # list_csv = []
    dir_list = os.listdir(file_dir)
    for cur_file in dir_list:
        path = os.path.join(file_dir, cur_file)
        # 判断是文件夹还是文件
        if os.path.isfile(path):
            # print("{0} : is file!".format(cur_file))
            dir_files = os.path.join(file_dir, cur_file)
        # 判断是否存在.csv文件，如果存在则获取路径信息写入到list_csv列表中
        if os.path.splitext(path)[1] == '.csv':
            csv_file = os.path.join(file_dir, cur_file)
            # print(os.path.join(file_dir, cur_file))
            # print(csv_file)
            list_csv.append(cur_file)
        if os.path.isdir(path):
            # print("{0} : is dir".format(cur_file))
            # print(os.path.join(file_dir, cur_file))
            list_dir(path)
    return list_csv

    
if __name__ == "__main__":
    conv2d_info_dict = {"input_size": [], "kernel_size": [], "bias": [], "stride": [], "padding": [], "dilation": [], "groups": []}
    conv2d_item_info = ["input_size", "kernel_size", "bias", "stride", "padding", "dilation", "groups"]
    item_info = ["id", "source_neural_network", "num_of_calls", "parameters"]

    dirpath = './seperate_network_3x5/'

    dir_list = os.listdir(dirpath)

    for dir_name in dir_list:
        print("---------------read csv in " + dir_name)

        list_csv = []
        list_csv = os.listdir(dirpath + dir_name)

        for filepath_ori in list_csv:
            for filepath_ori in [filepath_ori]:

                conv2d_info_split = []
                print("---------------read csv and split args---------------------")
                processed_info = []
                all_topk_info = []
                all_topk_calls = []
                all_rawtopk_info = []

                filepath = '../new_arg/new_args_' + filepath_ori + '.csv'
                print(filepath)
                if os.path.exists(filepath):
                    content_list = read_csv(filepath)
                else:
                    print("main : " + filepath + " not exit!\n")
                    continue
                conv2d_index = find_conv2d_index(content_list)
                if conv2d_index != -1: # if the model contains conv2d
                    conv2d_info = content_list[conv2d_index][2]

                    conv2d_info_network, topk_info, topk_calls = sort_conv2d_info(split_arg(conv2d_info))
                    all_topk_info += topk_info
                    all_topk_calls += topk_calls
                    all_rawtopk_info += conv2d_info_network


            kernelsize_info, kernelsize_num, all_topk_info_set, all_topk_num_set, \
                raw_topk_info_set = sort_kernelsize_info(all_topk_info, all_topk_calls, all_rawtopk_info)
            inputchannel_info, inputchannel_num = sort_inputchannel_info(all_topk_info, all_topk_calls)
            outputchannel_info, outputchannel_num = sort_outputchannel_info(all_topk_info, all_topk_calls)
            inoutchannel_info, inoutchannel_num = sort_inoutchannel_info(all_topk_info, all_topk_calls)

            ##########################################
            iH_info, iH_num = sort_iH_info(all_topk_info, all_topk_calls)
            with open(dirpath + dir_name + "/" + filepath_ori + "/" + "iH.csv", "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["iH", "totalnumofcalls"])
                for i in range(len(iH_info)):
                    writer.writerow([iH_info[i], iH_num[i]])

            data = pd.read_csv(dirpath + dir_name + "/" + filepath_ori + "/" + "iH.csv")
            sorted_data = data.sort_values(by="iH", ascending=True)
            sorted_data.to_csv(dirpath + dir_name + "/" + filepath_ori + "/" + "new_iH.csv", index=False)

            ##########################################

            # print(kernelsize_info)
            # print("--------------------------------------")
            # print(kernelsize_num)
            # print("--------------------------------------")
            # print(all_topk_info_set)
            # print(len(all_topk_info_set))
            # print("--------------------------------------")
            # print(all_topk_num_set)
            # print(len(all_topk_num_set))
            # print("--------------------------------------")
            # print(raw_topk_info_set)
            # print(len(raw_topk_info_set))
            # print("--------------------------------------")
            with open(dirpath + dir_name + "/" + filepath_ori + "/" + "kernelsize.csv", "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["kernel size", "totalnumofcalls"])
                for i in range(len(kernelsize_info)):
                    writer.writerow([kernelsize_info[i], kernelsize_num[i]])

            data = pd.read_csv(dirpath + dir_name + "/" + filepath_ori + "/" + "kernelsize.csv")
            sorted_data = data.sort_values(by="kernel size", ascending=True)
            sorted_data.to_csv(dirpath + dir_name + "/" + filepath_ori + "/" + "new_kernelsize.csv", index=False)


            with open(dirpath + dir_name + "/" + filepath_ori + "/" + "inputchannel.csv", "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["inputchannel", "totalnumofcalls"])
                for i in range(len(inputchannel_info)):
                    writer.writerow([inputchannel_info[i], inputchannel_num[i]])

            data = pd.read_csv(dirpath + dir_name + "/" + filepath_ori + "/" + "inputchannel.csv")
            sorted_data = data.sort_values(by="inputchannel", ascending=True)
            sorted_data.to_csv(dirpath + dir_name + "/" + filepath_ori + "/" + "new_inputchannel.csv", index=False)



            with open(dirpath + dir_name + "/" + filepath_ori + "/" + "outputchannel.csv", "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["outputchannel", "totalnumofcalls"])
                for i in range(len(outputchannel_info)):
                    writer.writerow([outputchannel_info[i], outputchannel_num[i]])

            data = pd.read_csv(dirpath + dir_name + "/" + filepath_ori + "/" + "outputchannel.csv")
            sorted_data = data.sort_values(by="outputchannel", ascending=True)
            sorted_data.to_csv(dirpath + dir_name + "/" + filepath_ori + "/" + "new_outputchannel.csv", index=False)



            with open(dirpath + dir_name + "/" + filepath_ori + "/" + "inoutchannel.csv", "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["inoutchannel", "totalnumofcalls"])
                for i in range(len(inoutchannel_info)):
                    writer.writerow([inoutchannel_info[i], inoutchannel_num[i]])

            data = pd.read_csv(dirpath + dir_name + "/" + filepath_ori + "/" + "inoutchannel.csv")
            sorted_data = data.sort_values(by="inoutchannel", ascending=True)
            sorted_data.to_csv(dirpath + dir_name + "/" + filepath_ori + "/" + "new_inoutchannel.csv", index=False)

            names = ["inoutchannel", "rawdata", "parameter", "numofcalls"]
            with open(dirpath + dir_name + "/" + filepath_ori + "/" + "inoutchannel_rank.csv", "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(names)
                for i in range(len(all_topk_info_set)):
                    inoutchannel = all_topk_info_set[i]["kernel_size"][0][0]
                    writer.writerow([inoutchannel, raw_topk_info_set[i], all_topk_info_set[i], all_topk_num_set[i]])

            data = pd.read_csv(dirpath + dir_name + "/" + filepath_ori + "/" + "inoutchannel_rank.csv")
            column_list = data.columns.values
            sorted_data = data.sort_values([column_list[0], column_list[3]], ascending=[1, 0])
            sorted_data.to_csv(dirpath + dir_name + "/" + filepath_ori + "/" + "new_inoutchannel_rank.csv", index=False)
            raw_topk_info_set = sorted_data["rawdata"].values
            conv2d_inoutchannelinfo_dict = get_info_dict(conv2d_info_dict, conv2d_item_info, raw_topk_info_set)
            conv2d_inoutchannelinfo_json = json.dumps(conv2d_inoutchannelinfo_dict)
            with open(dirpath + dir_name + "/" + filepath_ori + "/" + 'inoutchannel.json', 'w') as f:
                f.write(conv2d_inoutchannelinfo_json)
