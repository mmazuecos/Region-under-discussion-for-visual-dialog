import numpy as np
import scipy.stats as stats

if __name__ == '__main__':

    for prefix in ['human', 'cl1', 'rl1', 'bl', 'sl']:
        data = np.loadtxt(prefix+'correlation-data.txt')

        cross1 = np.take(data, [0,1], axis=1)
        cross2 = np.take(data, [0,2], axis=1)
        cross3 = np.take(data, [1,2], axis=1)

        c1_vals = np.corrcoef(cross1[:,0], cross1[:,1])
        c2_vals = np.corrcoef(cross2[:,0], cross2[:,1])
        c3_vals = np.corrcoef(cross3[:,0], cross3[:,1])

        print(prefix)
        print('S/F vs MU')
        print(c1_vals)
        print('-----')
        print('S/F vs # of objs')
        print(c2_vals)
        print('-----')
        print('MU vs # of objs')
        print(c3_vals)
        print('=======')
