import json
realPf = open("linear1381.json", 'r')
hmcf = open("hmc.csv", 'r')
realP = json.load(realPf)
hmc = hmcf.readlines()

for i in range(len(realP)):
    realP[i]["ID"] = i

hmcSta = []
hmcStaC = -1
for line in hmc:
    if "####################" in line:
        hmcStaC += 1
        hmcSta.append([])
    else:
        hmcSta[hmcStaC].append(line.split('""')[1])

kernelTypeSet = set()
for kernels in hmcSta:
    for kernel in kernels:
        kernelTypeSet.add(kernel)

kernelTypeDict = {}
for kernel in kernelTypeSet:
    kernelTypeDict[kernel] = []

for kernelsN in range(len(hmcSta)):
    kernels = hmcSta[kernelsN]
    for kernel in kernels:
        for key in kernelTypeDict.keys():
            if key in kernel:
                kernelTypeDict[key].append(realP[kernelsN])
for (key, values) in kernelTypeDict.items():
    print(key)
    for value in values:
        print('\t',value)



realPf.close()
hmcf.close()