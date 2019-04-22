#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import pickle
import scipy.io

fr = open('wacgan-history.pkl')    
inf = pickle.load(fr)       
fr.close()                       

# print(type(inf))
scipy.io.savemat('trainingLog.mat', inf)


