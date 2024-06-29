# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 21:45:53 2023

@author: THINK-JZ
"""

import numpy as np

import scipy.optimize as opt 

from scipy.ndimage import maximum_filter, label

import matplotlib.pylab as plt

import copy

#%%数组操作

def Comb_array(a1,a2):
    ##Build a combined array
    L1 = len(a1)
    L2 = len(a2)
    KK = L1 + L2
    combined = np.zeros(KK)
    Id1 = []
    n = 0
    Id2 = []
    m = 0
    for ii in range(KK):
        if (n < L1) & (m<L2):
            if a1[n] < a2[m]:
                combined[ii] = a1[n]
                Id1.append(ii)
                n = n+1
            else:
                combined[ii] = a2[m]
                Id2.append(ii)
                m = m+1            
        else:
            if n > L1-1:
                combined[ii] = a2[m]
                Id2.append(ii)
                m = m+1
            else:
                combined[ii] = a1[n]
                Id1.append(ii)
                n = n+1
    return Id1,Id2,combined

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

def sci_opt_fit(Image1,pixel_size,sigma_xy):
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
        return 0,hint,0
    if popt1[1]<0 or popt1[1]>c1 or popt1[2]<0 or popt1[2]>c0: hint='fail'
    else: hint='success'
    perr = np.sqrt(np.diag(pcov1))
    
    return popt1,hint,perr


def sci_opt_fit2(Image1,pixel_size,sigma_xy,popt00):
    def _Gaussian2D3(xy, a1, x0, y0, sigma_x, sigma_y, offset, a2):
        x, y = xy
        r = a1*np.exp(-((x-x0)**2/(2*sigma_x**2)+(y-y0)**2/(2*sigma_y**2)))+offset\
           +a2*np.exp(-((x-popt00[0])**2/(2*popt00[2]**2)+(y-popt00[1])**2/(2*popt00[3]**2)))
        return r.ravel()    
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
                   pixel_size*sigma_xy, Image_min1, Image_max1)
    
    try:
        popt1, pcov1 = opt.curve_fit(_Gaussian2D3, 
                                     xy, 
                                     Image1.ravel(),
                                     maxfev = 2000, 
                                     p0=ini_guess1)
    except Exception:
        hint='fail'
        # RuntimeError: Optimal parameters not found: Number of calls to function 
        # has reached maxfev = 1600.
        # OptimizeWarning: Covariance of the parameters could not be estimated
        return 0,hint,0
    if popt1[1]<0 or popt1[1]>c1 or popt1[2]<0 or popt1[2]>c0: hint='fail'
    else: hint='success'
    perr = np.sqrt(np.diag(pcov1))
    
    return popt1,hint,perr


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
    fig1    = f1ax1.pcolorfast(axis0, axis1, Image0, vmin =0, vmax =Vmax, cmap ='Blues')
    f1ax1.set_xlabel('$x$ ($\mu$$m$)')
    f1ax1.set_ylabel('$y$ ($\mu$$m$)')
    plt.colorbar(fig1)
    
    f1ax2   = f1.add_subplot(132)
    fig2    = f1ax2.pcolorfast(axis0, axis1, imgtofit_fitted1, vmin=0, vmax=Vmax, cmap ='Blues')
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

def spots_fit(spotsR, imden_loc, weight, 
              listx, listy, listpx, listpy, weight_l,
              pixel_size, label, file='test', drift_arr=[]):
    for i,arr in enumerate(spotsR):
        Image1      = arr
        Image_label = drift_arr[i]
        P  =  sci_opt_fit(Image1, pixel_size)
        
        if P[1]=='fail':
            continue
        popt1   = P[0]
        perr    = P[2]
        ellip=abs(popt1[3]/popt1[4])
        if (ellip<1.3)and(ellip>0.70):#and(perr[1]<0.2)and(perr[2]<0.2):
            
            '''此处绘图方便检查'''
            fit_plot(Image1, popt1, pixel_size, file, i, label)
            
            
            l11 = np.shape(Image_label)[0]
            l22 = np.shape(Image_label)[1]
            '''高斯拟合中心作为基准'''
            popt_lab = sci_opt_fit(Image_label, pixel_size)
            if popt_lab[1]=='fail':
                continue    
            popt_lab = popt_lab[0]
            x_com = popt_lab[1]
            y_com = popt_lab[2]
            
            fit_plot(Image_label, popt_lab, pixel_size, file,
                      i,labelp='label_fit_'+label)
            plt.close()
            
            '''相对拟合图片中心的距离'''
            x = popt1[1]-3 +imden_loc[i][0][2] -x_com +l11/2*pixel_size
            y = popt1[2]-3 +imden_loc[i][0][1] -y_com +l22/2*pixel_size
            listx.append(x)
            listy.append(y)
            listpx.append(perr[1])
            listpy.append(perr[2])
            weight_l.append(weight[i])
            # stack_num =weight[i]
            # listx += [x for i in range(stack_num)]
            # listy += [y for i in range(stack_num)]
            # listpx += [perr[1] for i in range(stack_num)]
            # listpy += [perr[2] for i in range(stack_num)]
    return listx, listy, listpx, listpy, weight_l

def spots_fit_sim(spotsR, weight, 
              listx, listy, listpx, listpy, weight_l,
              pixel_size, label, file='test', drift_arr=[]):
    for i,arr in enumerate(spotsR):
        Image1      = arr
        Image_label = drift_arr[i]
        P  =  sci_opt_fit(Image1, pixel_size)
        
        if P[1]=='fail':
            continue
        popt1   = P[0]
        perr    = P[2]
        ellip=abs(popt1[3]/popt1[4])
        if (ellip<1.3)and(ellip>0.70)and(perr[1]<0.1)and(perr[2]<0.1):
            
            '''此处绘图方便检查'''
            fit_plot(Image1, popt1, pixel_size, file, i, label)
            
            
            l11 = np.shape(Image_label)[0]
            l22 = np.shape(Image_label)[1]
            '''高斯拟合中心作为基准'''
            popt_lab = sci_opt_fit(Image_label, pixel_size)
            if popt_lab[1]=='fail':
                continue    
            popt_lab = popt_lab[0]
            x_com = popt_lab[1]
            y_com = popt_lab[2]
            
            fit_plot(Image_label, popt_lab, pixel_size, file,
                      i,labelp='label_fit_'+label)
            plt.close()
            
            '''相对拟合图片中心的距离'''
            x = popt1[1]-x_com +l11/2*pixel_size
            y = popt1[2]-y_com +l22/2*pixel_size
            listx.append(x)
            listy.append(y)
            listpx.append(perr[1])
            listpy.append(perr[2])
            weight_l.append(weight[i])

    return listx, listy, listpx, listpy, weight_l


#%% 密度矩阵操作函数

def den_method0(imd0,shape0,thres0,mark='pos'):
    if mark == 'neg': imd0 = -imd0
    
    ind_tar    = np.where(imd0> thres0)
    im_den     = np.zeros(shape0)
    
    # 定义局域密度的范围，这里是27个位置
    local_range = 1  # 这意味着沿x、y、z轴各有1个位置，总共27个位置
    
    # 遍历所有坐标
    for coord in zip(*ind_tar):
        # 计算局域密度的范围
        x_start, x_end = max(0, coord[0] - local_range), min(shape0[0], coord[0] + local_range + 1)
        y_start, y_end = max(0, coord[1] - local_range), min(shape0[1], coord[1] + local_range + 1)
        z_start, z_end = max(0, coord[2] - local_range), min(shape0[2], coord[2] + local_range + 1)
    
        # 在新矩阵中对应坐标周围的27个点加一
        im_den[x_start:x_end, y_start:y_end, z_start:z_end] += 1
    return im_den
    

def den_method1(imd0,shape0,thres0):
    im_den10    = np.zeros(shape0)
    im_den20    = np.zeros(shape0)
    for i in range(1,shape0[0]-1):
        for j in range(1,shape0[1]-1):
            for k in range(1,shape0[2]-1):
                im_den10[i,j,k] = len(np.where(imd0[i-1:i+2,j-1:j+2,k-1:k+2]> thres0)[0])
                im_den20[i,j,k] = len(np.where(imd0[i-1:i+2,j-1:j+2,k-1:k+2]<-thres0)[0])
    return im_den10, im_den20

def DFS_imden(matrix, shape0, tr, tz0, enl0, tr2 = 7): 
    '''
    tr : num, optional
        Chosen threshold. 
    tr2 : num, optional
        Event minium pixel amount. The default is 7.
    '''
    # 遍历整个矩阵, 检查当前位置是否越界或已访问过, 查找大于tr(默认为3)的元素并进行DFS搜索
    # 创建一个用于标记已访问元素的矩阵，初始化为False
    if tr < 0: tr=-tr; matrix = -matrix;
    visited = np.zeros(shape0, dtype=bool)
    coordinate_groups = []
    
    def dfs(i, j, k, group):
        if (i < 0 or i >= shape0[0] or j < 0 or j >= shape0[1] or
            k < 0 or k >= shape0[2] or visited[i, j, k] or matrix[i, j, k] <= tr):
            return
        visited[i, j, k]    = True
        group.append((i, j, k)) 
        for di, dj, dk in [( 1, 0, 0), (-1, 0, 0), ( 0, 1, 0), 
                           ( 0,-1, 0), ( 0, 0, 1), ( 0, 0,-1)]:
            dfs(i+di, j+dj, k+dk, group)
            
    for i in range(shape0[0]): 
        for j in range(shape0[1]):
            for k in range(shape0[2]):
                if not visited[i,j,k] and matrix[i,j,k]>tr:
                    group = []
                    dfs(i, j, k, group)
                    if group: coordinate_groups.append(group)                  
    win_size = (tz0, enl0, enl0)
    result_ranges = []
          
    for group in coordinate_groups:
        if len(group) > tr2:
            coordinates     = np.array(group)
            min_coordi  = coordinates.min(axis=0) 
            max_coordi  = coordinates.max(axis=0)
            d3matr      = np.abs(matrix[
                            # min_coordi[0]+1: max_coordi[0],
                            min_coordi[0]: max_coordi[0]+1,
                            min_coordi[1]: max_coordi[1]+1,
                            min_coordi[2]: max_coordi[2]+1])
            
            local_maxima = maximum_filter(d3matr, size=win_size)
            # 找到局部最大值的位置
            max_pos = np.argwhere((d3matr==local_maxima) & (d3matr>tr)) + min_coordi
            # if len(max_pos)>1:tr3 = np.mean(d3matr)
            # else: tr3 = tr
            # 遍历每个局部最大值位置
            for pos in max_pos:                
                # 找到在三个轴的正负方向上数值单调递减且大于tr的范围
                range_z = mono_range(matrix[:,pos[1],pos[2]], pos[0], enl0,
                                     direction='both', threshold = tr)
                range_y = mono_range(matrix[pos[0],:,pos[2]], pos[1], enl0,
                                     direction='both', threshold = tr)
                range_x = mono_range(matrix[pos[0],pos[1],:], pos[2], enl0,
                                     direction='both', threshold = tr)
                
                # 将范围添加到结果列表
                result_ranges.append((np.array([range_z[0], range_y[0], range_x[0]]),
                                      np.array([range_z[1], range_y[1], range_x[1]])))
                
    return result_ranges
    # min_max_values.append((min_coordi, max_coordi))

def imloc_max(matrix, tr0, tr1, tz0, enl0):
    win_size = (tz0, enl0, enl0)
    if tr0 < 0: tr0=-tr0; matrix = -matrix;
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
                                        max(0,pos[2]-1):pos[2]+2],axis=0)

        
        average_value = np.mean(center_region2d)
        
        if  average_value >= tr1:
            result_range.append((np.array([range_z[0], range_y[0], range_x[0]]),
                                 np.array([range_z[1], range_y[1], range_x[1]])))
            result_pos.append(pos)
    return result_range, result_pos

def mono_range(array0, pos0, enlx, direction, threshold):
    current_value = array0[pos0]
    # 寻找正方向的范围
    if direction in ['both', 'positive']:
        i = 1
        while (pos0 + i < array0.shape) \
          and (array0[pos0 + i] > threshold) \
          and (array0[pos0 + i] <= current_value):
            current_value = array0[pos0 + i]
            i += 1
        if i > enlx: i = enlx +1
        range_positive = pos0 +i-1
    else:
        range_positive = pos0
        
    current_value = array0[pos0]
    # 寻找负方向的范围
    if direction in ['both', 'negative']:
        i = 1
        while (pos0-i >= 0) \
          and (array0[pos0-i] > threshold) \
          and (array0[pos0-i] <= current_value):
            current_value = array0[pos0 - i]
            i += 1
        if i > enlx: i = enlx +1
        range_negative = pos0 -i+1
    else:
        range_negative = pos0

    return (range_negative, range_positive)


#%%高斯光斑形状判断
def Dcentr_test(matrix,min_r=2):
    '''
    matrix : np.array; Light spots to test if it is a proper guassian spot.
    '''
    matrix = np.abs(matrix)
    row_indices, col_indices = np.where(matrix > np.max(matrix)/3)
    # bin_matrix = np.zeros(np.shape(matrix))
    # bin_matrix[row_indices, col_indices] = 1
    
    # 计算这些像素的几何中心
    center_x = np.mean(col_indices)
    center_y = np.mean(row_indices)
    size = len(row_indices)
    
    # 此处粗略判断光斑直径应大于4pixel
    if size < np.pi * min_r**2: return 0
    
    # 计算每个像素到中心的距离
    distances = np.sqrt((col_indices - center_x)**2 + (row_indices - center_y)**2)
    
    # 定义半径阈值，根据需要调整
    radius_threshold = np.sqrt(size/3)  # 根据需要调整
    
    # 检查是否所有距离都接近于同一半径，并且中心在图像中心附近
    ind = np.where(distances > radius_threshold)
    ins = np.sum(distances[ind] - radius_threshold)#/radius_threshold/size
    return ins    


def updown_test(l):
    thres = np.max(l)/np.size(l)*2
    smo_l = np.convolve(l,np.ones(3)/3,mode='same')
    dif_l = np.diff(smo_l)
    smodif_l = np.convolve(dif_l,np.ones(3)/3,mode='same')
    is_increase = False
    tes = 0
    for s in smodif_l:
        if s > thres:
            if not is_increase: is_increase = True; tes+=1
        elif s < -thres:
            if is_increase: is_increase = False; tes+=10

    return tes


def cir_test2(matrix):
    if matrix.size==0: return 0
    # 此处暂时（粗略地）使用对维度的大小的判断
    if not all(dim>=9 for dim in matrix.shape): return 0
    
    matrix = np.abs(matrix)
    # row_indices, col_indices = np.where(matrix > np.max(matrix)/3)
    # bin_matrix = np.zeros(np.shape(matrix))
    # bin_matrix[row_indices, col_indices] = 1
    l00 = np.sum(matrix,axis=0)
    l11 = np.sum(matrix,axis=1)
    
    l0t = updown_test(l00)
    l1t = updown_test(l11)
    
    if l0t==11 and l1t==11: return 1
    else: return 0 
    

def cir_test2r(spots_,break_,dim_lim):
    for i,array_ in enumerate(spots_):
        # if break_[i] != 0: continue
        # array_=np.array(array_)
        # if not any(dim>=dim_lim 
        #            for dim in array_.shape): 
        #     break_[i] = 3
        #     continue
        array_ = np.abs(array_)
        l00 = np.sum(array_,axis=0)
        l11 = np.sum(array_,axis=1)
        l0t = updown_test(l00)
        l1t = updown_test(l11)
        
        if not (l0t==11 and l1t==11): 
            break_[i] = 1
    return break_

def extract_circular_region(img, diameter_x, diameter_y, center,theta = 0):
    """选择指定椭圆范围内的像素点"""
    '''theta 是与坐标轴正向夹角'''
    # X_Cor = np.zeros((img.shape[0], img.shape[1]))
    # Y_Cor = np.zeros((img.shape[0], img.shape[1]))
    img_test   = copy.copy(img)
    idx_pick   = []
    inten_sum  = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            X = j - center[0]
            Y = i - center[1]
            # X_Cor[i,j] = X*np.cos(theta)-Y*np.sin(theta)
            # Y_Cor[i,j] = Y*np.cos(theta)+X*np.sin(theta)         
            # X = X_Cor[i,j]
            # Y = Y_Cor[i,j]
            if ((X*np.cos(theta)-Y*np.sin(theta))**2/(diameter_x/2)**2 + 
                (X*np.sin(theta)+Y*np.cos(theta))**2/(diameter_y/2)**2)<=1:  
                
                inten_sum = inten_sum + img[i,j]
                img_test[i,j]=0
                idx_pick.append((i,j))
    return idx_pick, img_test#, inten_sum


def generate_circle(center, radius, mshape):
    """生成以给定中心和半径的圆内的整数坐标点"""
    cy, cx = center
    x_coords, y_coords = np.meshgrid(np.arange(max(int(cx-radius),0), 
                                    min(int(cx+radius)+1,mshape[2])),
                                     np.arange(max(int(cy-radius),0), 
                                    min(int(cy+radius)+1,mshape[1])))

    distances   = np.sqrt((x_coords-cx)**2+(y_coords-cy)**2)
    circle_mask = distances <= radius

    x_circle    = x_coords[circle_mask]
    y_circle    = y_coords[circle_mask]

    return x_circle, y_circle









    