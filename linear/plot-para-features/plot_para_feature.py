import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
import math
import os

def nestedpie_inout_features(filename, featurename, ax):
    featurename_dict = {'k':'ifea', 'n':'ofea'}
    def my_outerautopct(pct, dict_for_labels):
        # print(pct)
        label = ''
        for key, value in dict_for_labels.items():
            if math.fabs(value-pct) < 1e-5:
                label = key
                break
        # print(label)
        return (featurename_dict[featurename] + '=' + str(int(label)) + ',\n%.1f' % pct + '%') if pct > 4 else ''
    
    def my_innerautopct(pct, dict_for_labels):
        # print(pct)
        label = ''
        for key, value in dict_for_labels.items():
            if math.fabs(value-pct) < 1:
                label = key
                break
        # print(label)
        return (str(label) + ' %.1f' % pct + '%') if pct > 8 else ''

    # fig, ax = plt.subplots(1, 1)
    rcParams['font.size'] = 50
    data = pd.read_excel(filename, usecols=[1, 2])  # usecols = [], sheet_name='Cin'
    data.head()
    # print(data)
    size = 0.8
    labels = data[featurename].values
    vals = np.array(data['totalnumofcalls'].values)
    vals = [int(i) for i in vals]
    inner_vals = np.array([sum(vals[0:12]), sum(vals[13:])])
    vals_dict = {}
    total_feq = sum(vals)
    for i in range(len(vals)):
        vals_dict[labels[i]] = vals[i]/total_feq * 100
    innervals_dict = {}
    innervals_dict['power \n of two,\n'] = inner_vals[0]/total_feq * 100
    innervals_dict['others,\n'] = inner_vals[1]/total_feq * 100

    # print(vals_dict,'\n--------------\n', innervals_dict,'\n--------------\n')
    # exit()

    cmap = plt.colormaps['tab20c']
    cmap2 = plt.colormaps['tab20b']
    cmap3 = plt.colormaps['Greens']
    cmap4 = plt.colormaps['tab20']
    outer_colors = np.concatenate([cmap3([20, 20, 20, 20]), cmap3((np.arange(0, 6)+2)*20), cmap3([140]*3), cmap(np.arange(len(vals)-13)+20)],axis=0)
    inner_colors = np.concatenate([cmap4([4]), cmap([14])])

    ax.pie(vals, radius=2.0, colors=outer_colors, autopct=lambda pct:my_outerautopct(pct, vals_dict), startangle=60,
        pctdistance=0.78, textprops={'fontsize': 19, 'color': 'k'}, wedgeprops=dict(width=size, edgecolor='w')) # wedgeprops=dict(width=size, edgecolor='w')
    ax.pie(inner_vals, radius=2.0-size, colors=inner_colors, autopct=lambda pct:my_innerautopct(pct, innervals_dict), startangle=60,
        pctdistance=0.67, textprops={'fontsize': 20, 'color': 'w'}, wedgeprops=dict(width=size, edgecolor='w'))

    ax.set(aspect='equal')

    # plt.savefig(featurename + '.png', format='png', bbox_inches='tight') # bbox_inches  pad_inches  os.path.splitext(filename)
    # plt.savefig(featurename + '.pdf', format='pdf', bbox_inches='tight')


def plt_para_fig(featurename, figsize):
    # fm = mpl.font_manager
    # fm.get_cachedir()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # config = {
    #     # 'font.family': 'serif',
    #     'font.serif': ['Times New Roman'],
    #     'font.size': 20,
    #     'mathtext.fontset': 'stix',
    # }
    # rcParams.update(config)
    rcParams['font.serif'] = ['Times New Roman']
    nestedpie_inout_features(featurename+'.xlsx', featurename, ax)
    plt.savefig('linear.'+featurename+'.png', format='png', bbox_inches='tight')
    plt.savefig('linear.'+featurename+'.pdf', format='pdf', bbox_inches='tight')
    # plt.show()
    

if __name__ == '__main__':
    # plot_para_fig()
    plt_para_fig('k', (5, 5))
    plt_para_fig('n', (5, 5))
    # plt_para_fig('iH', (5, 5))
    # plt_para_fig('kernelsize', (6, 6))
    # pie_kernelsize()
    # nestedpie_cin_cout_ih('Cin.xlsx', 'Cin')
    # nestedpie_cin_cout_ih('Cout.xlsx', 'Cout')
    # nestedpie_cin_cout_ih('iH.xlsx', 'iH')