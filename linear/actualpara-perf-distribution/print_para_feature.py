import json
import csv
import pandas as pd
import numpy as np

dir = './data/'
files = ['perfrangeabove70', 'perfrange60-70', 'perfrange50-60', 'perfrange40-50', 
                            'perfrange30-40', 'perfrange20-30', 'perfrangeunder20']


def print_para_info(cin_list, cout_list, h_list, w_list):
    cincout_multiplier_dict = {'1/4':[], '1/3':[], '1/2':[], '1':[], '2':[], '3':[], '4':[]}
    hw_multiplier_dict = {'1/4':[], '1/3':[], '1/2':[], '1':[], '2':[], '3':[], '4':[]}
    nchw_multiplier = []
    
    total_cnt = 0
    cin_avg = 0
    cout_avg = 0
    h_avg = 0
    w_avg = 0

    for idx in range(len(cin_list)):
        cin = cin_list[idx]
        cout = cout_list[idx]
        h = h_list[idx]
        w = w_list[idx]

        total_cnt += 1
        cin_avg += cin
        cout_avg += cout
        h_avg += h
        w_avg += w

        if cin*4 == cout:
            cincout_multiplier_dict['1/4'].append((cin, cout))
        elif cin*3 == cout:
            cincout_multiplier_dict['1/3'].append((cin, cout))
        elif cin*2 == cout:
            cincout_multiplier_dict['1/2'].append((cin, cout))
        elif cin == cout:
            cincout_multiplier_dict['1'].append((cin, cout))
        elif cin == cout*2:
            cincout_multiplier_dict['2'].append((cin, cout))    
        elif cin == cout*3:
            cincout_multiplier_dict['3'].append((cin, cout))
        elif cin == cout*4:
            cincout_multiplier_dict['4'].append((cin, cout))

        if h*4 == w:
            hw_multiplier_dict['1/4'].append((h, w))
        elif h*3 == w:
            hw_multiplier_dict['1/3'].append((h, w))
        elif h*2 == w:
            hw_multiplier_dict['1/2'].append((h, w))
        elif h == w:
            hw_multiplier_dict['1'].append((h, w))
        elif h*2 == w:
            hw_multiplier_dict['2'].append((h, w))
        elif h*3 == w:
            hw_multiplier_dict['3'].append((h, w))
        elif h*4 == w:
            hw_multiplier_dict['4'].append((h, w))

        if cin == cout and h == w:
            nchw_multiplier.append((cin, cout, h, w))
    
    print('==========average info=====================')
    print('cin_avg: ', cin_avg/total_cnt, '\ncout_avg: ', cout_avg/total_cnt, 
                        '\nh_avg: ', h_avg/total_cnt, '\nw_avg: ', w_avg/total_cnt)
    print('==========cincoutmultiplifier=====================\n', 
    '1/4:', len(cincout_multiplier_dict['1/4']), len(cincout_multiplier_dict['1/4'])/total_cnt, '***',
    '1/3: ', len(cincout_multiplier_dict['1/3']), len(cincout_multiplier_dict['1/3'])/total_cnt, '***',
    '1/2: ', len(cincout_multiplier_dict['1/2']), len(cincout_multiplier_dict['1/2'])/total_cnt, '***',
    '1: ', len(cincout_multiplier_dict['1']), len(cincout_multiplier_dict['1'])/total_cnt, '***',
    '2: ', len(cincout_multiplier_dict['2']), len(cincout_multiplier_dict['2'])/total_cnt, '***',
    '3: ', len(cincout_multiplier_dict['3']), len(cincout_multiplier_dict['3'])/total_cnt, '***',
    '4: ', len(cincout_multiplier_dict['4']),  len(cincout_multiplier_dict['4'])/total_cnt, '***')
    print('==========hwmultiplifier=====================\n', 
    '1/4: ', len(hw_multiplier_dict['1/4']), len(hw_multiplier_dict['1/4'])/total_cnt, '***', 
    '1/3: ', len(hw_multiplier_dict['1/3']), len(hw_multiplier_dict['1/3'])/total_cnt, '***',
    '1/2: ', len(hw_multiplier_dict['1/2']), len(hw_multiplier_dict['1/2'])/total_cnt, '***',
    '1: ', len(hw_multiplier_dict['1']), len(hw_multiplier_dict['1'])/total_cnt, '***',
    '2: ', len(hw_multiplier_dict['2']), len(hw_multiplier_dict['2'])/total_cnt, '***',
    '3: ', len(hw_multiplier_dict['3']), len(hw_multiplier_dict['3'])/total_cnt, '***',
    '4: ', len(hw_multiplier_dict['4']), len(hw_multiplier_dict['4'])/total_cnt, '***')
    print('==========info of cin==cout=====================')
    print(cincout_multiplier_dict['1'])
    print('==========info of h==w=====================')
    print(hw_multiplier_dict['1'])
    print(len(nchw_multiplier),total_cnt,len(nchw_multiplier)/total_cnt,'==========info of cin==cout and h==w=====================')
    print(nchw_multiplier)
    print('\n\n')


def get_para_feature():
    cin_list = []
    cout_list = []
    h_list = []
    w_list = []

    for file in files:
        data_pd = pd.read_csv(dir+file+'.csv')
        cin_split_list = list(data_pd['C_in'].values)
        cout_split_list = list(data_pd['C_out'].values)
        h_split_list = list(data_pd['H'].values)
        w_split_list = list(data_pd['W'].values)

        cin_list += cin_split_list
        cout_list += cout_split_list
        h_list += h_split_list
        w_list += w_split_list
        print('****************'+file+'***********************\n')
        print_para_info(cin_split_list, cout_split_list, h_split_list, w_split_list)

    print('\n\n****************total data statistics**********************\n')
    print_para_info(cin_list, cout_list, h_list, w_list)


def print_statitical_feature():
    for file in files:
        data_pd = pd.read_csv(dir+file+'.csv')
        n_split_list = list(data_pd['N'].values)
        cin_split_list = list(data_pd['cin'].values)
        cout_split_list = list(data_pd['cout'].values)

        print('****************'+file+'***********************\n')
        print(np.median(n_split_list), np.median(cin_split_list), np.median(cout_split_list))
        print(np.mean(n_split_list), np.mean(cin_split_list), np.mean(cout_split_list))


if __name__ == '__main__':
    # get_para_feature()
    print_statitical_feature()