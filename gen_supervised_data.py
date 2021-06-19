# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:32:56 2020

@author: lingzhang0319
"""

import numpy as np
from pdn_class import PDN, connect_1decap
import os
import numpy.random as random
from copy import deepcopy
import time


t1 = time.time()

brd = PDN()

BASE_PATH = 'brd_data/'
NEW_DATA_PATH = 'supervised_data/'

if not os.path.exists(NEW_DATA_PATH):
    os.mkdir(NEW_DATA_PATH)

repeat = 5

file_list = list(range(0,10000))

for n in file_list:
    z_orig = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['z']
    brd_shape_ic = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['brd_shape_ic']
    ic_xy_indx = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['ic_xy_indx']
    top_decap_xy_indx = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['top_decap_xy_indx']
    bot_decap_xy_indx = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['bot_decap_xy_indx']
    stackup = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['stackup']
    die_t = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['die_t']
    sxy = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['sxy']
    
    for r in range(0, repeat):
        for i in range(0, z_orig.shape[1]):
            z = deepcopy(z_orig)
            map2orig_output = list(range(0,z.shape[1]))
            x1 = np.zeros((brd_shape_ic.shape[0],brd_shape_ic.shape[0],3))
            x2 = np.zeros((9+8))
            x2[0:stackup.shape[0]] = stackup + 1          # 1 for gnd, 2 for power
            x2[9:9+die_t.shape[0]] = die_t * 1e3    # unit for thickness becomes mm
            x1[:,:,0] = brd_shape_ic
            
            if i == 0:
                y = 20 * np.log10(np.abs(z[:,0,0])) # no decap
            else:
                loc_indx = np.sort(random.choice(range(1,z.shape[1]), i, replace=False))
                
                for j in list(loc_indx):
                    decap_indx = random.randint(0, len(brd.decap_list))
                    z, map2orig_output = connect_1decap(z, map2orig_output, 
                                                        map2orig_output.index(j), 
                                                        brd.decap_list[decap_indx])
                    
                    if j <= top_decap_xy_indx.shape[0]:
                        x1[int(top_decap_xy_indx[j-1,0]),int(top_decap_xy_indx[j-1,1]),1] = decap_indx + 1
                    else:
                        x1[int(bot_decap_xy_indx[j-1-top_decap_xy_indx.shape[0],0]),
                           int(bot_decap_xy_indx[j-1-top_decap_xy_indx.shape[0],1]),2] = decap_indx + 1
                y = 20 * np.log10(np.abs(z[:,0,0]))
            np.savez(os.path.join(NEW_DATA_PATH, str(n*(z_orig.shape[1]*repeat)+r*z_orig.shape[1]+i)+'.npz'), 
                     x1=x1, x2=x2, y=y, sxy=sxy)

print(time.time()-t1)