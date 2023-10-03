import numpy as np
import matplotlib.pyplot as plt

# Data for Q1.4
# Hopper environment
data = [[100, 142.95816040039062, 142.95816040039062],
        [200, 336.55291748046875, 121.07298278808594],
        [300, 387.7551574707031, 9.493345260620117],
        [400, 580.3482055664062, 25.68195343017578],
        [500, 660.9523315429688, 115.50784301757812],
        [600, 755.2334594726562, 141.81686401367188],
        [700, 1156.9278564453125, 40.51673126220703],
        [800, 1266.6204833984375, 102.9871597290039],
        [900, 487.38336181640625, 209.52560424804688],
        [1000, 947.5502319335938, 168.98275756835938],
        [1100, 904.0809326171875, 121.47179412841797],
        [1200, 727.9732666015625, 39.37499237060547],
        [1300, 985.794189453125, 79.57086181640625],
        [1400, 871.7406005859375, 14.051033973693848],
        [1500, 1101.6658935546875, 50.76871871948242],
        [1600, 1370.4005126953125, 172.72586059570312],
        [1700, 1096.470458984375, 36.88032531738281],
        [1800, 888.8737182617188, 10.034189224243164],
        [1900, 1141.50830078125, 35.89675521850586],
        [2000, 1073.1192626953125, 90.9300765991211]]

data = np.array(data)
hyperparameter = data[:, 0]
average_return = data[:, 1]
std_return = data[:, 2]

plt.figure()
plt.errorbar(hyperparameter, average_return, std_return, linestyle='None', marker='o')
plt.xlabel('Number of gradient steps')
plt.ylabel('Average return')
plt.title('Hopper default environment with atleast 5 rollouts and 1 hidden layer')
plt.savefig('Q1.4.png')

# Q2.2
# Ant environment
dagger_data = [[0, 3287.592041015625, 2010.4427490234375],
               [1, 4531.81005859375, 349.3808288574219],
               [2, 4605.6171875, 135.2548370361328],
               [3, 4806.06396484375, 76.90137481689453],
               [4, 4810.76953125, 80.6139907836914],
               [5, 4518.4658203125, 137.10829162597656],
               [6, 4819.5712890625, 47.333614349365234],
               [7, 4813.73291015625, 72.05846405029297],
               [8, 4285.8828125, 797.1912231445312],
               [9, 4675.7109375, 51.43892288208008]]

dagger_data = np.array(dagger_data)
dagger_iteration = dagger_data[:, 0]
dagger_average_return = dagger_data[:, 1]
dagger_std_return = dagger_data[:, 2]

plt.figure()
plt.plot([0, 9], [4713.6533203125, 4713.6533203125], color='yellow')
plt.plot([0, 9], [3287.592041015625, 3287.592041015625], color='red')
plt.errorbar(dagger_iteration, dagger_average_return, dagger_std_return, linestyle='None', marker='o')
plt.xlabel('Dagger iteration')
plt.ylabel('Average return')
plt.title('Ant default environment with atleast 5 rollouts and 1 hidden layer')
plt.legend(['Expert policy', 'BC policy', 'DAgger policy'], loc='lower right')
plt.savefig('Q2.2.png')

# Hopper environment
dagger_data = [[0, 947.5502319335938, 168.98275756835938],
               [1, 872.18115234375, 34.51455307006836],
               [2, 2355.405517578125, 906.754638671875],
               [3, 3560.546142578125, 20.9877986907959],
               [4, 1808.51318359375, 152.63877868652344],
               [5, 3772.24560546875, 1.293013334274292],
               [6, 3774.57177734375, 4.79896879196167],
               [7, 3767.768798828125, 3.2689242362976074],
               [8, 3776.86083984375, 4.2803449630737305],
               [9, 3766.19287109375, 3.9293315410614014]]

dagger_data = np.array(dagger_data)
dagger_iteration = dagger_data[:, 0]
dagger_average_return = dagger_data[:, 1]
dagger_std_return = dagger_data[:, 2]

plt.figure()
plt.plot([0, 9], [3772.67041015625, 3772.67041015625], color='yellow')
plt.plot([0, 9], [947.5502319335938, 947.5502319335938], color='red')
plt.errorbar(dagger_iteration, dagger_average_return, dagger_std_return, linestyle='None', marker='o')
plt.xlabel('Dagger iteration')
plt.ylabel('Average return')
plt.title('Hopper default environment with atleast 5 rollouts and 1 hidden layer')
plt.legend(['Expert policy', 'BC policy', 'DAgger policy'], loc='lower right')
plt.savefig('Q2.3.png')
