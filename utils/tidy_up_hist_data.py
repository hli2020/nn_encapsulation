import numpy as np
folder = 'non_target'


def sort_up_data(epoch):
    x_axis = np.linspace(-1, 1, 21)
    with open('{:s}/avglen_{:d}.txt'.format(folder, epoch)) as f:
        avglen = f.readlines()
    avglen = [float(x.strip()) for x in avglen]
    out = np.zeros((21, 3))
    out[:, 0] = x_axis
    out[:, 1] = avglen

    with open('{:s}/cosine_{:d}.txt'.format(folder, epoch)) as f:
        content = f.readlines()
    content = [x.split() for x in content]

    for item in content:
        value = float(item[0])
        if value == 1.0:
            ind = -1
        elif value == -1:
            ind = 0
        else:
            ind = sum(value > x_axis) - 1
        out[ind, 2] += float(item[1])

    out[:, 2] /= 0.01*sum(out[:, 2])

    print('epoch: ', epoch)
    for i in range(len(x_axis)):
        print('{:.1f}\t{:.8f}\t{:.8f}'.format(out[i, 0], out[i, 1], out[i, 2]))

sort_up_data(1)
sort_up_data(20)
sort_up_data(80)
sort_up_data(200)
sort_up_data(300)
