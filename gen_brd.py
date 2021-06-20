# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 00:20:16 2020

@author: Ling Zhang, lzd76@mst.edu
"""

'''
Board thickness: 1mm ~ 10mm ...
Number of layers: 4 ~ 9 ...
Locations: 20, in which 1 is for IC and 19 for decaps
Via distance: 1mm ...
'''


import numpy as np
import numpy.random as random
from scipy.special import binom
import matplotlib.pyplot as plt
from math import sqrt, pi
import cv2
from copy import deepcopy
from pdn_class import PDN, connect_1decap
import time
import os

bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.1, edgy=0.05):
    """ given an array of points *a*, create a curve through
    those points. 
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x,y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)
    
    
def min_transition_angle(bd):
    # a is the sorted locations of boundary nodes
    angle = np.zeros((bd.shape[0]))
    for i in range(0, bd.shape[0]):
        if i == 0:
            a = np.array([bd[-1,0], bd[-1,1]])
            b = np.array([bd[i,0], bd[i,1]])
            c = np.array([bd[i+1,0], bd[i+1,1]])
        elif i == bd.shape[0]-1:
            a = np.array([bd[i-1,0], bd[i-1,1]])
            b = np.array([bd[i,0], bd[i,1]])
            c = np.array([bd[0,0], bd[0,1]])
        else:
            a = np.array([bd[i-1,0], bd[i-1,1]])
            b = np.array([bd[i,0], bd[i,1]])
            c = np.array([bd[i+1,0], bd[i+1,1]])
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle[i] = np.arccos(cosine_angle)*180/pi
    return np.min(angle)


def find_bd_pixels(a):
    a_bd = np.zeros(a.shape)
    for i in range(0,a.shape[0]):
        for j in range(0,a.shape[1]):
            if (i==0 or i==a.shape[0]-1 or j==0 or j==a.shape[1]-1) and a[i,j]==1:
                a_bd[i,j] = 1
            elif (i==0 or i==a.shape[0]-1 or j==0 or j==a.shape[1]-1) and a[i,j]==0:
                a_bd[i,j] = 0
            elif (a[i-1,j-1]==0 or a[i-1,j]==0 or a[i-1,j+1]==0 or a[i,j-1]==0 or a[i,
                  j+1]==0 or a[i+1,j-1]==0 or a[i+1,j]==0 or a[i+1,j+1]==0) and a[i,j]==1:
                a_bd[i,j] = 1
    return a_bd


# generate arbitrary board shapes and related matrices
def gen_random_brd_shape(max_len, img_size, num_locations, n=8, thre_angle=30, 
                         rad=0.1, edgy=0.05, num_pts=10):
    num_available_locations = 0
    
    while num_available_locations < num_locations:
        angle = 0
        while angle < 30:
            a = get_random_points(n=n, scale=1)
            angle = min_transition_angle(ccw_sort(a))
            
        x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)
        x = x[range(0,n*100,num_pts)]
        y = y[range(0,n*100,num_pts)]
        bxy = np.ndarray((x.shape[0],2))
        bxy[:,0] = x*max_len
        bxy[:,1] = y*max_len
        
        contour = np.ndarray((x.shape[0],2))
        contour[:,0] = x*img_size
        contour[:,1] = y*img_size
        contour = contour.astype(int)
        
        img = np.zeros( (img_size,img_size) )
        cv2.fillPoly(img, pts =[contour], color=(1,1,1))
        
        # the coordinates has been changed. We need to change it back
        img = np.transpose(img)
        
        img_contour = find_bd_pixels(img)
        # randomly select pixels
        i,j = np.nonzero(img-img_contour)   # exclude the boundary pixels
        
        num_available_locations = len(i)
    
    ix = random.choice(len(i), num_locations, replace=False)
    
    # randomly select IC location, decap top and bottom locations
    top_loc_number = random.randint(3, num_locations-1)
    top_indx = random.choice(num_locations, top_loc_number, replace=False)
    ic_local_local_indx = random.choice(top_loc_number, 1, replace=False)
    ic_local_indx = top_indx[ic_local_local_indx]
    ic_indx = ix[ic_local_indx]
    top_decap_indx = ix[np.delete(top_indx, ic_local_local_indx)]
    bot_decap_indx = np.delete(ix, top_indx)
    
    brd_shape_ic = deepcopy(img)
    brd_shape_ic[i[ic_indx],j[ic_indx]] = 2
    
    ic_xy_indx = np.array([i[ic_indx], j[ic_indx]])
    
    top_decap_xy_indx = np.zeros((top_decap_indx.shape[0], 2))
    bot_decap_xy_indx = np.zeros((bot_decap_indx.shape[0], 2))
    top_decap_xy_indx[:,0] = i[top_decap_indx]
    top_decap_xy_indx[:,1] = j[top_decap_indx]
    bot_decap_xy_indx[:,0] = i[bot_decap_indx]
    bot_decap_xy_indx[:,1] = j[bot_decap_indx]
    
    sxy = np.ndarray((bxy.shape[0],4))
    sxy[:,0:2] = bxy
    sxy[0:-1,2:4] = bxy[1:,:]
    sxy[-1,2:4] = bxy[0,:]
    
    dist2ic = np.zeros((top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0]))
    for i in range(0, top_decap_xy_indx.shape[0]):
        dist2ic[i] = (ic_xy_indx[0] - top_decap_xy_indx[i,0])**2 + (ic_xy_indx[1] - top_decap_xy_indx[i,1])**2
    for i in range(top_decap_xy_indx.shape[0], top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0]):
        dist2ic[i] = (ic_xy_indx[0] - bot_decap_xy_indx[i-top_decap_xy_indx.shape[0],0])**2 + (ic_xy_indx[1] - bot_decap_xy_indx[i-top_decap_xy_indx.shape[0],1])**2
        
    vrm_indx = np.argmax(dist2ic)
    if vrm_indx < top_decap_xy_indx.shape[0]:
        vrm_xy_indx = top_decap_xy_indx[vrm_indx, :]
        top_decap_xy_indx = np.delete(top_decap_xy_indx, (vrm_indx), axis=0)
        vrm_loc = 1
    else:
        vrm_xy_indx = bot_decap_xy_indx[vrm_indx - top_decap_xy_indx.shape[0], :]
        bot_decap_xy_indx = np.delete(bot_decap_xy_indx, (vrm_indx - top_decap_xy_indx.shape[0]), axis=0)
        vrm_loc = 0
    
    return brd_shape_ic, ic_xy_indx, top_decap_xy_indx, bot_decap_xy_indx, vrm_xy_indx, vrm_loc, sxy


# only support one power layer for now
def gen_random_stackup(min_t, max_t, min_dt, min_n_layer, max_n_layer):
    t = random.uniform(min_t, max_t, size=1)
    n_tot = int(t/min_dt)
    n_layer = random.randint(min_n_layer, max_n_layer+1)
    while n_layer * min_dt > t: # to prevent that total thickness is too small
        n_layer = random.randint(min_n_layer, max_n_layer+1)
    mid_layers = np.array(range(1,n_tot))
    mid_layer_indx = np.sort(random.choice(mid_layers, n_layer-2, replace=False))
    layer_indx = np.zeros((n_layer))
    layer_indx[1:n_layer-1] = mid_layer_indx
    layer_indx[n_layer-1] = n_tot
    die_t = np.zeros((n_layer-1))
    die_t[0:n_layer-1] = (layer_indx[1:n_layer]-layer_indx[0:n_layer-1]) * min_dt
    power_layer_indx = random.randint(1, n_layer-1)
    stackup = np.zeros((n_layer))
    stackup[power_layer_indx] = 1
    d2top = np.sum(die_t[0:power_layer_indx])
    return stackup, die_t, t, d2top
    

def gen_brd_data(brd, max_len, img_size, num_locations=20, n=8, thre_angle=30, rad=0.1, edgy=0.05, 
                 num_pts=10, min_t=1e-3, max_t=5e-3, min_dt=0.1e-3, min_n_layer=4, max_n_layer=9, 
                 er=4.4, via_dist=2e-3, vrm_r=3e-3, vrm_l=2.5e-9):
    brd_shape_ic, ic_xy_indx, top_decap_xy_indx, bot_decap_xy_indx, vrm_xy_indx, vrm_loc, sxy = gen_random_brd_shape(max_len=max_len, img_size=img_size,
                                                                           num_locations=num_locations)
    stackup, die_t, t, d2top = gen_random_stackup(min_t, max_t, min_dt, min_n_layer, max_n_layer)
    brd.sxy = sxy
    brd.er = er
    brd.stackup = stackup
    brd.die_t = die_t
    unit_size = max_len/img_size
    
    # map the ic and decap locations and calculate Z parameter
    ic_x_indx, ic_y_indx = np.where(brd_shape_ic == 2)
    ic_xy_indx = np.array([ic_x_indx, ic_y_indx])
    
    # now all the cases, the power and gnd via pair are horizontally oriented
    # create ic pins
    brd.ic_via_xy = np.zeros((2,2))
    brd.ic_via_xy[0,0] = (ic_xy_indx[0,0]+0.5)*unit_size - via_dist/2
    brd.ic_via_xy[0,1] = (ic_xy_indx[1,0]+0.5)*unit_size
    brd.ic_via_xy[1,0] = (ic_xy_indx[0,0]+0.5)*unit_size + via_dist/2
    brd.ic_via_xy[1,1] = (ic_xy_indx[1,0]+0.5)*unit_size
    
    brd.ic_via_type = np.array([1,0])
    
    # matrix that mark the location of the ports
    brd.decap_via_xy = np.zeros(((top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0])*2, 2))
    brd.decap_via_type = np.zeros(((top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0])*2))
    brd.decap_via_loc = np.zeros(((top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0])*2))
    
    for i in range(0, top_decap_xy_indx.shape[0]):
        brd.decap_via_xy[2*i,0] = (top_decap_xy_indx[i,0]+0.5)*unit_size - via_dist/2
        brd.decap_via_xy[2*i,1] = (top_decap_xy_indx[i,1]+0.5)*unit_size
        brd.decap_via_xy[2*i+1,0] = (top_decap_xy_indx[i,0]+0.5)*unit_size + via_dist/2
        brd.decap_via_xy[2*i+1,1] = (top_decap_xy_indx[i,1]+0.5)*unit_size
        brd.decap_via_type[i*2] = 1
        brd.decap_via_type[i*2+1] = 0
        brd.decap_via_loc[i*2] = 1
        brd.decap_via_loc[i*2+1] = 1
    for j in range(0, bot_decap_xy_indx.shape[0]):
        k = j + top_decap_xy_indx.shape[0]
        brd.decap_via_xy[2*k,0] = (bot_decap_xy_indx[j,0]+0.5)*unit_size - via_dist/2
        brd.decap_via_xy[2*k,1] = (bot_decap_xy_indx[j,1]+0.5)*unit_size
        brd.decap_via_xy[2*k+1,0] = (bot_decap_xy_indx[j,0]+0.5)*unit_size + via_dist/2
        brd.decap_via_xy[2*k+1,1] = (bot_decap_xy_indx[j,1]+0.5)*unit_size
        brd.decap_via_type[k*2] = 1
        brd.decap_via_type[k*2+1] = 0
        brd.decap_via_loc[k*2] = 0
        brd.decap_via_loc[k*2+1] = 0
        
    # brd.decap_via_xy[(top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0] + 1)*2-2, 0] = (vrm_xy_indx[0]+0.5)*unit_size - via_dist/2
    # brd.decap_via_xy[(top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0] + 1)*2-2, 1] = (vrm_xy_indx[1]+0.5)*unit_size
    # brd.decap_via_xy[(top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0] + 1)*2-1, 0] = (vrm_xy_indx[0]+0.5)*unit_size + via_dist/2
    # brd.decap_via_xy[(top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0] + 1)*2-1, 1] = (vrm_xy_indx[1]+0.5)*unit_size
    # brd.decap_via_type[(top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0] + 1)*2-2] = 1
    # brd.decap_via_type[(top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0] + 1)*2-1] = 0
    # brd.decap_via_loc[(top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0] + 1)*2-2] = vrm_loc
    # brd.decap_via_loc[(top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0] + 1)*2-1] = vrm_loc
    
    z = brd.calc_z_fast()
    
    # Connect VRM
    # z11_vrm = np.zeros((z.shape[0], 1, 1), dtype=complex)
    # z11_vrm[:,0,0] = vrm_r + 1j*2*pi* brd.freq.f * vrm_l
    
    # z, _ = connect_1decap(z, list(range(0,z.shape[1])), z.shape[1]-1, z11_vrm)
    return z, brd_shape_ic, ic_xy_indx, top_decap_xy_indx, bot_decap_xy_indx, vrm_xy_indx, vrm_loc, stackup, die_t, t, d2top, sxy, brd.ic_via_xy, brd.ic_via_type, brd.decap_via_xy, brd.decap_via_type, brd.decap_via_loc, max_len


if __name__ == '__main__':
    
    N = 10000
    n = 0
    BASE_PATH = 'brd_data/'
    
    if not os.path.exists(BASE_PATH):
        os.mkdir(BASE_PATH)
    
    t0 = time.time()
    
    num_error = 0
    brd = PDN()
    for i in range(0,N):
        print(str(n))
        t1 = time.time()
        z, brd_shape_ic, ic_xy_indx, top_decap_xy_indx, bot_decap_xy_indx, vrm_xy_indx, vrm_loc, stackup, die_t, t, d2top, sxy, ic_via_xy, ic_via_type, decap_via_xy, decap_via_type, decap_via_loc, max_len = gen_brd_data(brd=brd, max_len=200e-3, img_size=16)
        if np.isnan(np.sum(z)):
            num_error += 1
            continue
        np.savez(os.path.join(BASE_PATH, str(n)+'.npz'), z=z, brd_shape_ic=brd_shape_ic, 
                 ic_xy_indx=ic_xy_indx, top_decap_xy_indx=top_decap_xy_indx, 
                 bot_decap_xy_indx=bot_decap_xy_indx, vrm_xy_indx=vrm_xy_indx, vrm_loc=vrm_loc, 
                 stackup=stackup, die_t=die_t, 
                 t=t, d2top=d2top, sxy=sxy, ic_via_xy=ic_via_xy, ic_via_type=ic_via_type, 
                 decap_via_xy=decap_via_xy, decap_via_type=decap_via_type, decap_via_loc=decap_via_loc, 
                 max_len=max_len)
        n += 1
        print("Number of layers is " + str(stackup.shape[0]) + ", number of ports is " 
              + str(z.shape[1]))
        print("Consumed time is " + str(time.time()-t1) + " s")
        print("Total consumed time is " + str(time.time()-t0) + " s")
        print("\n")
    
