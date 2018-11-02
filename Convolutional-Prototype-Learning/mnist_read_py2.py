import pickle
import pdb

with open('mnist.pkl', 'r') as f:
    pdb.set_trace()
    temp = pickle.load(f)
    # temp : list(3)
    # temp[0] : list(2)
    # temp[0][0] : list(50000)
    # temp[0][0][0] : list(784)
    # temp[0][1] : list(50000)
    # temp[1][0] : list(10000)
    # temp[2][0] : list(10000)
