import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
import math
import os

def pie_kernelsize(ax):
    def my_autopct(pct):
        return ('%.1f' % pct + '%') if pct > 2 else ''

    # fig, ax = plt.subplots(1, 1)

    tinydict = {
        '1x1': 290989,
        '1x3': 515,
        '1x5': 140,
        '1x7': 5,
        '2x2': 470,
        '3x1': 531,
        '3x3': 420297,
        '4x4': 1436,
        '5x1': 140,
        '5x5': 1778,
        '7x1': 5,
        '7x7': 3649,
        '8x8': 105,
        '9x1': 1000,
        '9x9': 215,
        '11x11': 455,
        '16x16': 81,
    }
    labels = list(tinydict.keys())
    sizes = tinydict.values()
    cmap = plt.colormaps['tab20c']
    wedge_colors = cmap(np.arange(len(sizes)*4))

    wedges, texts, autotexts  = ax.pie(sizes, labels=None, 
        radius=1, colors=wedge_colors, autopct=lambda pct:my_autopct(pct), startangle=90,
        pctdistance=0.5, textprops={'fontsize': 20, 'color': 'k'})
    ax.legend(wedges, labels, title="r x r", loc='center right', bbox_to_anchor=(0.98, 0.35, 0.2, 0.3)) 
    plt.setp(autotexts, size=20)
    ax.axis('equal')
    # plt.savefig('conv2d.kernelsize.pdf', format='pdf', bbox_inches='tight') # bbox_inches  pad_inches
    # plt.savefig('conv2d.kernelsize.png', format='png', bbox_inches='tight') # bbox_inches  pad_inches


def nestedpie_cin_cout_ih(filename, featurename, ax):
    featurename_dict = {'Cin':'cin', 'Cout':'cout', 'iH':'ih'}
    def my_outerautopct(pct, dict_for_labels):
        # print(pct)
        label = ''
        for key, value in dict_for_labels.items():
            if math.fabs(value-pct) < 1e-5:
                label = key
                break
        # print(label)
        return (featurename_dict[featurename] + '=' + str(label) + ',\n%.1f' % pct + '%') if pct > 4 else ''
    
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
    cmap3 = plt.colormaps['Blues']
    outer_colors = np.concatenate([cmap(np.arange(0, 4)+20), cmap3((np.arange(0, 7)+2)*20), cmap(np.arange(len(vals)-11)+20)],axis=0)
    inner_colors = np.concatenate([cmap2([2]), cmap2([6])])

    ax.pie(vals, radius=2.0, colors=outer_colors, autopct=lambda pct:my_outerautopct(pct, vals_dict), startangle=60,
        pctdistance=0.78, textprops={'fontsize': 19, 'color': 'k'}, wedgeprops=dict(width=size, edgecolor='w')) # wedgeprops=dict(width=size, edgecolor='w')
    ax.pie(inner_vals, radius=2.0-size, colors=inner_colors, autopct=lambda pct:my_innerautopct(pct, innervals_dict), startangle=60,
        pctdistance=0.67, textprops={'fontsize': 20, 'color': 'w'}, wedgeprops=dict(width=size, edgecolor='w'))

    ax.set(aspect='equal')

    # plt.savefig(featurename + '.png', format='png', bbox_inches='tight') # bbox_inches  pad_inches  os.path.splitext(filename)
    # plt.savefig(featurename + '.pdf', format='pdf', bbox_inches='tight')


# def plot_para_fig():
#     fig, axs = plt.subplots(2, 2, figsize=(10, 10), layout='constrained') # figsize=()
#     pie_kernelsize(axs[0][0])
#     nestedpie_cin_cout_ih('iH.xlsx', 'iH', axs[0][1])
#     nestedpie_cin_cout_ih('Cin.xlsx', 'Cin', axs[1][0])
#     nestedpie_cin_cout_ih('Cout.xlsx', 'Cout', axs[1][1])
#     # nestedpie_iH(axs[0][1])
#     # nestedpie_cin(axs[1][0])
#     # nestedpie_cout(axs[1][1])

# #     # plt.show()
# #     # plt.savefig('conv2d.para.pdf', format='pdf') # bbox_inches  pad_inches
#     plt.savefig('conv2d.para.png', format='png', bbox_inches='tight') # bbox_inches  pad_inches

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
    if featurename == 'kernelsize':
        pie_kernelsize(ax)
    else:
        nestedpie_cin_cout_ih(featurename+'.xlsx', featurename, ax)
    plt.savefig('conv2d.'+featurename+'.png', format='png', bbox_inches='tight')
    plt.savefig('conv2d.'+featurename+'.pdf', format='pdf', bbox_inches='tight')
    # plt.show()
    

if __name__ == '__main__':
    plt_para_fig('Cin', (5, 5))
    plt_para_fig('Cout', (5, 5))
    plt_para_fig('iH', (5, 5))
    
    
    # plt_para_fig('kernelsize', (6, 6))