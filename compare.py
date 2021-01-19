import matplotlib.pyplot as plt


# 防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

file = open('Lenet5.txt', 'r')
lineList = file.readlines()
Lenet5 = []
Lenet5_t = []
for line in lineList:
    line = line[1:-2]
    # print(line)
    curList = line.split(', ')
    # print(curList)
    Lenet5.append(float(curList[1]))
    Lenet5_t.append(10*float(curList[0]))

file = open('resnet18.txt', 'r')
lineList = file.readlines()
Res18 = []
Res18_t = []
for line in lineList:
    line = line[1:-2]
    # print(line)
    curList = line.split(', ')
    # print(curList)
    Res18.append(float(curList[1]))
    Res18_t.append(float(curList[0]))


fig, ax = plt.subplots()
ax.plot(Lenet5_t, Lenet5, 'b', marker='x', label='Lenet5')
ax.plot(Res18_t, Res18, 'r', marker='o', label='ResNet18')
ax.set_xlabel('时间(s)')
ax.set_ylabel('分类准确率')
ax.set_title('Lenet5和ResNet18训练过程比较')
plt.legend()
# plt.show()
plt.savefig('Lenet5与ResNet18对比.png', dpi=150)