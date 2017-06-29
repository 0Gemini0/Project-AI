import os
import numpy as np

UNSUP_PATH = os.getcwd() + '\\unsupervised data\\'
SUPER_PATH = os.getcwd() + '\\supervised data\\'
FAKE_PATH = os.getcwd() + '\\fake data\\'
PKL_PATH = os.getcwd() + '\\pickled data\\'
FREQ_CORRECTION = 6.2
FAKE_SIGNALS = 50
LOC = np.array([[-0.5,1.0],[-0.8,0.5],[-0.3,0.6],[-0.6,0.25],[-1.0,0.0],[-0.6,-0.7],[-0.3,-1.0],[0.3,-1.0],[0.6,-0.7],[1.0,0.0],[0.6,0.25],[0.3,0.6],[0.8,0.5],[0.5,1.0]])