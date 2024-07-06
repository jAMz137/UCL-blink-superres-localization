# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 21:45:53 2023

@author: THINK-JZ
"""

import numpy as np
import copy
import scipy.optimize   as opt 
import matplotlib.pylab as plt

from scipy.ndimage import maximum_filter, label

#%%数组操作

def consec_T(arr, n):
    len_    = len(arr)
    result  = [False] *len_
    for i in range(len_):
        # if cnt >=10: break
        if all(arr[i: i+n]) and (i+n <=len_):
            # cnt +=1
            for j in range(i, i+n): 
                result[j] = True 
    return np.array(result)

def consec_T2(arr, n):
    len_    = len(arr)
    result  = [False] *len_
    for i in range(len_-2 ):
        if all(arr[i: i+3]):
            result[i +1] = True
    return np.array(result)

def consec_T3(arr,arrr, n):
    len_    = len(arr)
    result  = [False] *len_
    for i in range(len_-2 ):
        if all(arrr[i: i+3])and arr[i+1]:
            result[i +1] = True
    return np.array(result)

#%%拟合与画图

def _Gaussian2D1(xdata_tuple, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple                                                        
    x0 = float(x0)                                                              
    y0 = float(y0)
    x  = x-x0; y = y-y0;
    x1 =  x*np.cos(theta) + y*np.sin(theta)
    y1 = -x*np.sin(theta) + y*np.cos(theta)
    z  = amplitude * np.exp(-((x1/sigma_x)**2+(y1/sigma_y)**2)/2) + offset
    return z.ravel()

def _Gaussian2D2(xy, a, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    r = a*np.exp(-((x-x0)**2/(2*sigma_x**2)+(y-y0)**2/(2*sigma_y**2)))+offset
    return r.ravel()    


def sci_opt_fit(Image1: np.ndarray, pixel_size: float, sigma_xy: float):
    c0      = Image1.shape[0]
    c1      = Image1.shape[1]
    axis0   = np.linspace(0,c1-1,c1)*pixel_size
    axis1   = np.linspace(0,c0-1,c0)*pixel_size
    xy      = np.meshgrid(axis0, axis1)
    
    Image_max1  = np.max(Image1) 
    Image_min1  = np.min(Image1)
    position = np.where(Image1 == np.max(Image1))
    Pos_x1 = position[0][0]
    Pos_y1 = position[1][0]
    ini_guess1  = (Image_max1, Pos_x1*pixel_size, 
                   Pos_y1*pixel_size, 
                   pixel_size*sigma_xy, 
                   pixel_size*sigma_xy, Image_min1) 
    try:
        popt1, pcov1 = opt.curve_fit(_Gaussian2D2, 
                                     xy, 
                                     Image1.ravel(),
                                     maxfev = 2000, 
                                     p0=ini_guess1)
    except Exception:
        hint='fail'
        # RuntimeError: Optimal parameters not found: Number of calls to function 
        # has reached maxfev = 1600.
        # OptimizeWarning: Covariance of the parameters could not be estimated
        return [], hint, []
    if popt1[1]<0 or popt1[1]>c1 or popt1[2]<0 or popt1[2]>c0: hint='fail'
    else: hint='success'
    perr = np.sqrt(np.diag(pcov1))
    return popt1, hint, perr



def fit_plot(Image0, popt0, pixel_size, filename='test0000', ix=0, labelp='', show = 0):
    c0 = Image0.shape[0];  c1 = Image0.shape[1]
    
    axis0   = np.linspace(0,c1-1,c1)*pixel_size
    axis1   = np.linspace(0,c0-1,c0)*pixel_size
    xy      = np.meshgrid(axis0, axis1)
    
    data_fitted1    = _Gaussian2D2(xy, *popt0)
    
    imgtofit_fitted1 = data_fitted1.reshape(Image0.shape)
    Vmax0   = np.max(Image0)
    Vmax1   = np.max(imgtofit_fitted1)
    Vmax    = max(Vmax0,Vmax1)
    ymaxfit_posi1 = np.where(imgtofit_fitted1==Vmax1)[1][0]       
    # Vmin1=np.min(imgtofit_fitted1)
    
    f1      = plt.figure(figsize=(18,4))
    
    f1ax1   = f1.add_subplot(131)
    fig1    = f1ax1.pcolorfast(axis0, axis1, Image0, 
                               vmin =0, vmax =Vmax, cmap ='Blues')
    f1ax1.set_xlabel('$x$ ($\mu$$m$)')
    f1ax1.set_ylabel('$y$ ($\mu$$m$)')
    plt.colorbar(fig1)
    
    f1ax2   = f1.add_subplot(132)
    fig2    = f1ax2.pcolorfast(axis0, axis1, imgtofit_fitted1, 
                               vmin =0, vmax =Vmax, cmap ='Blues')
    f1ax2.set_xlabel('$x$ ($\mu$$m$)')
    f1ax2.set_ylabel('$y$ ($\mu$$m$)')
    plt.colorbar(fig2)

    f1ax3   = f1.add_subplot(133)
    f1ax3.plot(Image0[:,ymaxfit_posi1], label ='data')
    f1ax3.plot(imgtofit_fitted1[:, ymaxfit_posi1], label ='fit')
    f1ax3.legend(loc ='best') 
    f1ax3.set_ylim(0, Vmax1*1.1)
    if show: return
    else: 
        f1.savefig(filename +labelp +'_'+str('%03d'%ix) +'.png')
        plt.close(f1)
        return imgtofit_fitted1

#%% 密度矩阵操作函数

def imloc_max(matrix: np.ndarray, tr0: float, tr1:float, 
              win_size: tuple, enl0: int)\
    -> tuple[list[tuple[np.ndarray,np.ndarray]], list[np.ndarray]]: 
    if tr0 < 0: tr0=-tr0; matrix = -matrix
    # coordinates = np.array(group)
    # min_coordi  = coordinates.min(axis=0) 
    # max_coordi  = coordinates.max(axis=0)
    d3matr = matrix
    result_range = []
    result_pos = []
    local_maxima = maximum_filter(d3matr, size=win_size)
    # 找到局部最大值的位置
    max_pos = np.argwhere((d3matr==local_maxima) & (d3matr>tr0))
    # m_pos   = np.where   ((d3matr==local_maxima) & (d3matr>tr0))
    # maxes   = d3matr[m_pos]
    # if len(max_pos)>1:tr3 = np.mean(d3matr)
    # else: tr3 = tr0
    # 遍历每个局部最大值位置
    for pos in max_pos:                
        # 找到在三个轴的正负方向上数值单调递减且大于tr0的范围
        range_z = mono_range(matrix[:,pos[1],pos[2]],
                             pos[0],enl0,direction='both',
                             threshold=tr0)
        range_y = mono_range(matrix[pos[0],:,pos[2]],
                             pos[1],enl0,direction='both',
                             threshold=tr0)
        range_x = mono_range(matrix[pos[0],pos[1],:],
                             pos[2],enl0,direction='both',
                             threshold=tr0)
        
        center_region2d = np.sum(matrix[range_z[0]: range_z[1]+1,
                                        max(0,pos[1]-1):pos[1]+2, 
                                        max(0,pos[2]-1):pos[2]+2],
                                        axis=0)        
        average_value = np.mean(center_region2d)
        if  average_value >= tr1:
            result_range.append((np.array([
                                    range_z[0], 
                                    range_y[0],
                                    range_x[0]]),
                                 np.array([
                                    range_z[1], 
                                    range_y[1], 
                                    range_x[1]])
                                    ))
            result_pos.append(pos)
    return (result_range, result_pos)

def mono_range(array0: np.ndarray, pos0: int, 
               enlx: int, direction: str, threshold):
    current_value = array0[pos0]
    # 寻找正方向的范围
    if direction in ['both', 'positive']:
        i = 1
        while (pos0 +i < array0.shape[0]) and (array0[pos0 +i] >threshold) and \
                                              (array0[pos0 +i] <=current_value):
            current_value = array0[pos0 + i]
            i += 1
        if i > enlx: 
            i = enlx +1
        range_positive = pos0 +i-1
    else:
        range_positive = pos0
        
    current_value = array0[pos0]
    # 寻找负方向的范围
    if direction in ['both', 'negative']:
        i = 1
        while (pos0-i >= 0) and (array0[pos0-i] > threshold) and \
        (array0[pos0-i] <= current_value):
            current_value = array0[pos0 - i]
            i += 1
        if i > enlx: 
            i = enlx +1
        range_negative = pos0 -i+1
    else:
        range_negative = pos0

    return (range_negative, range_positive)


#%%高斯光斑形状判断

def gen_circle(center, radius, mshape):
    """生成以给定中心和半径的圆内的整数坐标点"""
    cy, cx = center
    x_coords, y_coords = np.meshgrid(
        np.arange(max(int(cx-radius),0), 
                  min(int(cx+radius)+1,
                  mshape[1])),
        np.arange(max(int(cy-radius),0), 
                  min(int(cy+radius)+1,
                  mshape[0])))

    distances   = np.sqrt((x_coords-cx)**2 \
                         +(y_coords-cy)**2)
    circle_mask = distances <= radius

    x_circle    = x_coords[circle_mask]
    y_circle    = y_coords[circle_mask]

    return x_circle, y_circle 