import os
import numpy as np

UNSUP_PATH = os.getcwd() + '\\supervised data\\'
SUPER_PATH = os.getcwd() + '\\supervised data\\'
FAKE_PATH = os.getcwd() + '\\fake data\\'
PKL_PATH = os.getcwd() + '\\pickled data\\'
FREQ_CORRECTION = 6.2
FAKE_SIGNALS = 50
LOC1 = np.array([[-0.5,1.0],[-0.8,0.5],[-0.3,0.6],[-0.6,0.25],[-1.0,0.0],[-0.6,-0.7],[-0.3,-1.0],[0.3,-1.0],[0.6,-0.7],[1.0,0.0],[0.6,0.25],[0.3,0.6],[0.8,0.5],[0.5,1.0]])
LOC = np.array([[-2.45, 4.81], [-4.00, 2.87], [-1.27, 3.15], [-3.03, 1.75], [-4.70, 0.16], [-2.95, -3.40], [-1.40, -5.22],
                [1.40, -5.22], [2.95, -3.40], [4.70, 0.16], [3.03, 1.75], [1.27, 3.15], [4.00, 2.87], [2.45, 4.81]])
