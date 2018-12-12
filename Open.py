#Import packages
import scipy.io as sio
import sklearn.cluster as cl
import matplotlib.pyplot as plt
import pandas as pd

#Read file
data = pd.read_csv('Data/SC_expression.csv')
