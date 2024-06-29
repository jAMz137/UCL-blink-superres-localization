# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 11:36:24 2023 
@uthor: James Ma
For picking slices from 3D arraya rbitarily without using timetraces. 
(including time T as 3rd axis) 
重新选择 distz0; 排除非连续的三帧以下的部分; 处理多颗粒同时闪烁的情况
""" 

import sys, os, re, warnings; import numpy as np
import tifffile as tifile


from   scipy.ndimage import maximum_filter,\
                            gaussian_filter

import scipy.signal  as signal
import matplotlib.pyplot as plt

from   scipy.optimize import OptimizeWarning
warnings.simplefilter("ignore", 
                      category=OptimizeWarning)
sys.path.append('D:\\ProGrm\\python3\\library')
from   super_resoToolBoxV1 import imloc_max,\
       consec_T, consec_T3, generate_circle, fit_plot, sci_opt_fit

root        =   'G:\\data2024'
data        =   '\\20240220_2'
targ        =           '53d0'

target      =   targ + 'stack'
tarmrk      =   targ + 'stamark'

Path   = root + data + '\\stack'
loc    = "E:\\PrGm_tempfile" + data + targ
if not os.path.exists(loc): os.mkdir(loc)
os.chdir(loc)
Path_Dat    =   loc + '\\Dat\\'
if not os.path.exists(Path_Dat): 
    os.mkdir(Path_Dat)
    
# 使用正则表达式提取数字
numbers     =   re.findall(r'\d+',tarmrk)
Filename    =   target + '.tif'
Filemark    =   tarmrk + '.tif'
im0 = np.float32(tifile.imread(Path+'\\'+Filename))

#%% ex:参数统一设置

pixel_size = 1; Bin_time = 0.25 # s

sigma_s = 5*pixel_size  # key parameter, FWHM=2.355*σ
enl = int(2*sigma_s)    # spots_n/p outlier pixels

# 判断是否接触边界的范围
ex  = int(np.ceil(sigma_s/2))+1 
# ex2 = int(np.ceil(sigma_s))

sigma_f     = sigma_s   # sigma of guassian filter(xy)
sigma_t     = 0.5       # sigma of guassian filter(tz)

# 一个能与背景产生区分的值
thres0      = 3.5       # blinking event marking
# 这个参数实际上规定的的是TR中能被探测到的两帧之间变化
thres1      = 9         # 

mx_filt0    = 3         # sigma of maximum filters(tz)
# 此处取2仅对当前帧与+1帧做统一区域最大值滤波, 理想取值3
mx_filt12   = int(np.ceil(enl)) # sigma of maximum filter(xy)

distA0      = 2         # tz glitches range
distAs      = 5         # tz glitches search range
distA12     = enl+1     # xy glitches range
thres2      = thres0    # glitch two side matching

frame_tr    = 4         # min frame to be a state
distz0      = 30        # max frame to be a state
distxy      = enl *0.8  # to judge the overlapping 
                        # range of light spots on xy axis

radius      = int(ex+enl)
int_dif     = 10
# int_std     = lambda a : np.sqrt(a-95)/2 +0.3 # std error of state
'''
imlabel为后续校准漂移使用, 需要包含整个宽场区域; 
im主要是局部的对象点
'''
slice_z     = slice(0, -1)
im0         = im0[slice_z]

#%% ex:截取研究区域

# 框选对象点的正方形范围, 标注HCI上显示的像素值
# 读取后x/y轴会调换(0指标对应y轴,1指标对应x轴)

# Center_x    = 115
# Center_y    = 135
# width       = 70
# im = im0[
#         :,
#         int(Center_y-width):int(Center_y+width),
#         int(Center_x-width):int(Center_x+width)]

Center_x2    = 97
Center_y2    = 91
width2       = 12
imtr = im0[
        :,
        int(Center_y2-width2):int(Center_y2+width2),
        int(Center_x2-width2):int(Center_x2+width2)]
Attest = np.mean(imtr,axis=(1,2))

im     = imtr

###############################################################################
import time
start_time = time.time()
###############################################################################

#%% ex:漂移校准(correlated)

drift_stp   = 20    # 漂移校准步长
mark_num    = 1     # 校准标记个数

Pmrk        = [None] *mark_num
Pmrk[0]     = root +data +'\\driftmark5V'
# Pmrk[1]     = root +data +'\\driftmark2'
# Pmrk[2]     = root +data +'\\driftmark3'

drift_stm   = [None] *mark_num


for j in range(0,mark_num):    
    imlabel = np.float32(tifile.imread(Pmrk[j]+'\\'+Filemark))[:,0:28,0:28]
    shape0  = np.shape(imlabel)
    
    drift_stm[j] = []
    dir_drift_i = Pmrk[j]+'\\drift'+str('%03d'% int(numbers[0])  ) + '.tif'
    dir_drift_1 = Pmrk[j]+'\\drift'+str('%03d'%(int(numbers[0])-3))+ '.tif'
    dir_drift50 = Pmrk[j]+'\\drifA'+str('%03d'% int(50)) + '.tif'
    
    dft_recordi = Pmrk[j]+'\\dfting'+str('%03d'% int(numbers[0])  ) +'.txt'
    dft_record1 = Pmrk[j]+'\\dfting'+str('%03d'%(int(numbers[0])-3))+'.txt'
    
    if int(numbers[0]) == 50: 
        label0  = np.mean(imlabel[:drift_stp],axis=0)
        
        label0 -= np.min(label0)
        label30 = label0/np.max(label0)
        
        # label30 = gaussian_filter(label30, sigma =(sigma_f, sigma_f))
        # label30     = np.zeros(shape0[1:])
        # label30[label0>=np.mean(label0)*2] = 1
        
        tifile.imwrite(dir_drift50, label30)
    else:
        label30 = tifile.imread(dir_drift_1)
    
    labelst = []; imlabel = imlabel[slice_z] 
    
    for i in range(0, np.shape(imlabel)[0]//drift_stp): 
        
        label   = np.mean(imlabel[i*drift_stp:(i+1)*drift_stp],axis=0)
        label  -= np.min(label)
        label3  = label/np.max(label)
        
        # label3  = gaussian_filter(label3, sigma =(sigma_f, sigma_f))
        # label3  = np.zeros(shape0[1:])
        # label3[label >= np.mean(label)*1] = 1
        
        label   = signal.convolve(label3,label30[::-1,::-1],mode='same')                                                        
        
        label   = label[int(shape0[1]/4*1):
                        int(shape0[1]/4*3),
                        int(shape0[2]/4*1):
                        int(shape0[2]/4*3)]
        labelst.append(label)
        l11     = np.shape(label)[0]
        l22     = np.shape(label)[1]
        
        fitout = sci_opt_fit(label, pixel_size,7.5)
        
        if fitout[1] == 'fail': continue    
        
        l1_com = fitout[0][1]- l11/2*pixel_size
        l2_com = fitout[0][2]- l22/2*pixel_size
        
        drift_stm[j].append([l1_com, l2_com])
    
    drift_stm[j].insert(-1, [l1_com, l2_com])
    drift_stm[j] = np.array(drift_stm[j])
    
    if int(numbers[0]) == 50:
        np.savetxt(dft_recordi, drift_stm[j])
    else: 
        dft_r   = np.loadtxt(dft_record1)
        drift_stm[j] += dft_r[-1]
        np.savetxt(dft_recordi, drift_stm[j])
        
    tifile.imwrite(dir_drift_i, label3)

Adrift_xy = drift_stm[0]

#%% ex:漂移校准(multifit)

# drift_stp   = 20 # 漂移校准步长
# mark_numb   = 3  # 校准标记个数

# drift_stm   = [None] *mark_numb
# Pmrk        = [None] *mark_numb
# Pmrk[0]     = root +data +'\\driftmark1s'
# Pmrk[1]     = root +data +'\\driftmark2s'
# Pmrk[2]     = root +data +'\\driftmark3V'

# for j in range(0, mark_numb):    
#     drift_stm[j] = []
#     imlabel = np.float32(tifile.imread(Pmrk[j]+'\\'+Filemark))
#     int_lbl = np.sum(imlabel-86,axis=(1,2))
#     int_trd = np.max(int_lbl) -np.sqrt(np.max(int_lbl))/9*75
    
#     imlabel-= np.min(imlabel)
#     imlabel = imlabel /np.max(imlabel)
#     dir_drift50 = Pmrk[j] +'\\drifA'+str('%03d'% int(50))+'.tif'
#     dft_50cord  = Pmrk[j] +'\\drift050.txt'
    
#     if int(numbers[0]) == 50: 
#         flg0 = 1
#         int_lbl30 = int_lbl[:drift_stp]
#         imlabel30 = imlabel[:drift_stp]\
#                             [np.where(int_lbl30 > int_trd)]
#         if len(imlabel30)<5: 
#             print('frist50 blinking!')
#             os._exit()
#         label30 = np.mean(imlabel30, axis=0)
#         tifile.imwrite(dir_drift50, label30)
#         fitout  = sci_opt_fit(label30, pixel_size)
#         if fitout[1] == 'fail': 
#             print('frist50 errorfail') 
#             os._exit()
#         l1_0    = fitout[0][1]
#         l2_0    = fitout[0][2]
#         np.savetxt(dft_50cord, np.array([l1_0, l2_0])) 
#     else:
#         l1_0, l2_0 = np.loadtxt(dft_50cord)    
#     imlabel = imlabel[slice_z]
#     for ii in range(0, np.shape(imlabel)[0]//drift_stp): 
#         flag = 1
#         int_lbl3    = int_lbl[ii*drift_stp:(ii+1)*drift_stp]
#         imlabel3    = imlabel[ii*drift_stp:(ii+1)*drift_stp]\
#                                 [np.where(int_lbl3>int_trd)]
#         if len(imlabel3)<5: flag = 0  
#         else: 
#             label3  = np.mean(imlabel3, axis=0)
#             fitout  = sci_opt_fit(label3, pixel_size)
#             if (fitout[1]=='fail'): flag=0
#             else:
#                 popt1   = fitout[0]
#                 ellip   = abs(popt1[3]/popt1[4])
#                 if ellip>1.3 or ellip<0.7: flag = 0
#         if not flag:
#             drift_stm[j].append([50, 50])
#         else:
#             l1_com0 = popt1[1] - l1_0
#             l2_com0 = popt1[2] - l2_0
#             drift_stm[j].append([l1_com0, l2_com0])
    
#     drift_stm[j].insert(-1, drift_stm[j][-1])   
# drift_id    = np.full_like(drift_stm, 
#                         fill_value=False, dtype=bool)
# drift_stm   = np.array(drift_stm)
# drift_id[np.where(drift_stm==50)] =True
# marked_stmp = np.ma.masked_array(drift_stm,drift_id)
# Adrift_xy   = np.array(np.mean(marked_stmp, axis=0))
# Adrift_xy[:,0] = savgol_filter(Adrift_xy[:,0], 99, 1, mode= 'nearest')
# Adrift_xy[:,1] = savgol_filter(Adrift_xy[:,1], 99, 1, mode= 'nearest')

        
#%% 1. 整理闪烁事件点
# 查找整个含时间的三维矩阵中发生闪烁的帧数坐标, 定位完整闪烁
# 两个数组中分别记录了变暗与变亮两种事件

# sigma_f     = sigma_s
# sigma_t     = 0.5

img     = gaussian_filter( im, sigma = (sigma_t, sigma_f, sigma_f))
imgd    = np.diff(img,axis=0)
imdi    = np.diff(im, axis=0)
immx    = maximum_filter(imgd, size=(mx_filt0,mx_filt12,mx_filt12))
shape1  = np.shape(imgd)

# %% 1.1.

# from super_resoToolBoxV1 import *
# thres0      = 3
# thres1      = 10
# mx_filt0    = 3
# mx_filt12   = int(np.ceil(enl))
slice_p, posp   = imloc_max(imgd,  thres0, thres1, mx_filt0, mx_filt12)
slice_n, posn   = imloc_max(imgd, -thres0, thres1, mx_filt0, mx_filt12)

## %% 1.2.
def blink_mark (slice_, max_pos, enl0=enl):
    '''
    断点标记分类：
        1.硬断点(范围过大or轮廓异常) 2.范围过小 3.靠近边缘 4.信噪比低
    '''
    centr_  = [] # 中心位置
    spots_  = [] # 光斑图貌
    slc_    = []
    brk_    = [] # 断点标记
    rng_    = [] # 时间中心范围
    std_    = [] # 断点标记
    
    for j, (min_coord, max_coord) in enumerate(slice_):
        brk = 0; std = 10        
        # 事件时间中心
        tz      = (max_coord[0] +1 +min_coord[0]) /2
        # 事件时间范围
        trg     = (max_coord[0] +1 -min_coord[0]) /2    
        # den_method0(后略)
        # trg     = (max_coord[0] -min_coord[0]-1)/2
        
        # YX轴上的大小
        shapR   = np.array([max_coord[1] -min_coord[1],
                            max_coord[2] -min_coord[2]])
        # 外扩ex光斑tz加和后形貌
        spotA   = np.abs(np.sum(imdi[
                         # min_coord[0]+1: max_coord[0],
                           min_coord[0]: max_coord[0]+1,
                           max(min_coord[1]-ex,0): max_coord[1]+ex+1, 
                           max(min_coord[2]-ex,0): max_coord[2]+ex+1],axis=0))
        # 无外扩光斑tz加和后形貌
        spotB   = np.abs(np.sum(imgd[
                         # min_coord[0]+1: max_coord[0],
                           min_coord[0]: max_coord[0]+1,
                           min_coord[1]: max_coord[1]+1,
                           min_coord[2]: max_coord[2]+1],axis=0))
        maxi0   = np.argwhere(spotB == np.max(spotB))[0]
        cntr0   = maxi0 + np.array([min_coord[1], min_coord[2]])
        cntr1   = max_pos[j][1:]
        maxi1   = cntr1 - np.array([min_coord[1], min_coord[2]])
         
        if np.any(np.abs(maxi1 - maxi0)>enl/2):
            cntr0 = cntr1; maxi0 = maxi1
            brk = 1
        if np.any(np.abs(maxi0+1-shapR/2)>enl):
            brk = 1
        if brk <= 1:        
            if np.any(shapR < enl): 
                brk = 2
        if brk <= 1:
            if not(cntr0[0]-(enl0)>=0 and cntr0[1]-(enl0)>=0  
                and cntr0[0]+(enl0)<=shape1[1]-1 
                and cntr0[1]+(enl0)<=shape1[2]-1): 
                brk = 3   
        if brk <= 1:
            # with warnings.catch_warnings():
            #     warnings.filterwarnings('ignore')
            ppcc = sci_opt_fit(spotA, pixel_size,7.5)
               # try: except Warning: brk=4
            popt = ppcc[0]
            perr = ppcc[2]
            if ppcc[1] == 'fail': brk = 4
            else:
                cntr0 = np.array([popt[2]+max(min_coord[1]-ex,0),
                                  popt[1]+max(min_coord[2]-ex,0)])
                std = sum(perr[1:3])
        if brk != 2 and brk != 3:
            cntr0 =np.append(tz,cntr0)
            centr_.append(cntr0)
            spots_.append(spotA)
            slc_.append(slice_[j])
            brk_.append(brk); rng_.append(trg); std_.append(std)
                 
    return np.array(centr_), spots_, slc_, \
            np.array(brk_), np.array(rng_),np.array(std_)
 
centrp, spotsp, slice_p2, breakp, rngp, stdp =blink_mark(slice_p, posp)
centrn, spotsn, slice_n2, breakn, rngn, stdn =blink_mark(slice_n, posn)

L1 = len(slice_p2)
L2 = len(slice_n2)
ttzn = centrn[:,0]
ttzp = centrp[:,0]

###############################################################################
end_time1 = time.time()
print("time consuming: {:.2f}s".format(end_time1 - start_time))
###############################################################################

#%% 2. 标记大型毛刺
# 具体为标记方块区域, 后续TR经过方块时排除标记True占比大于10%的帧
 
visited1 = np.zeros(len(centrp), dtype=bool)
visited2 = np.zeros(len(centrn), dtype=bool)

# 存储点配对的结果
point_pairs = []
# 标记配对区域
marked_id = np.full_like(im,fill_value=False, dtype=bool)

for i, point0 in enumerate(centrp):
    # if visited1[i]: continue
    center_z, center_y, center_x = np.int32(point0)
    x_range = range(center_x-distA12, center_x+distA12+1)
    y_range = range(center_y-distA12, center_y+distA12+1)
    points_in_cuboid = []
    points_in_cuboid.append(i)
    
    localid = np.where((centrn[:,0]>center_z-distAs)
                      &(centrn[:,0]<center_z+distAs))[0]   
    for ii in localid:
        if visited2[ii]: continue
        z0 = np.floor(center_z-rngp[i]-rngn[ii]-distA0)
        z1 = np.ceil (center_z+rngp[i]+rngn[ii]+distA0+1)
        z_range = range(np.int32(z0), np.int32(z1))
        
        z, y, x = np.int32(centrn[ii])
        if x in x_range and y in y_range and z in z_range:
            li  = np.array(slice_p2[i])
            lii = np.array(slice_n2[ii])
            lmi = np.min(np.vstack((li[0],lii[0])),axis=0)
            lma = np.max(np.vstack((li[1],lii[1])),axis=0)
            spoti = np.sum(imdi[
                                # min_coords[0]+1:max_coords[0], 
                                li [0][0]:li [1][0]+1,
                                lmi[1]:  lma[1]+1, 
                                lmi[2]:  lma[2]+1],axis=0) 
            spotii= np.sum(imdi[
                                # min_coords[0]+1:max_coords[0], 
                                lii[0][0]:lii[1][0]+1,
                                lmi[1]:   lma[1]+1, 
                                lmi[2]:   lma[2]+1],axis=0)
            # FIXME: 此处应该更全面详尽比较这两帧的区别
            if  np.abs(np.mean(spoti+spotii))<thres2: 
                visited1[i ] = True
                visited2[ii] = True
                points_in_cuboid.append(ii) 
                point_pairs.append(points_in_cuboid)
                marked_id[lmi[0]+1:lma[0]+1,
                          lmi[1]:  lma[1]+1,
                          lmi[2]:  lma[2]+1]=True
                break
# marked_im   = np.ma.masked_array(im, marked_id)

###############################################################################
end_time2 = time.time()
print("time consuming: {:.2f}s".format(end_time2 - end_time1))
###############################################################################

#%% 3. 事件中心与截断点
# 作为中心事件的点
slinc_n  = np.where((~visited2))[0]#(stdn<1) &
slinc_p1 = np.where((~visited1))[0]#(stdp<1) &
# slinc_n  = np.arange(len(centrn), dtype=int)
slinc_p  = np.arange(len(centrp), dtype=int)

# 作为截断事件的点
slind_p = np.where( ~visited1 )[0]
slind_n = np.where( ~visited2 )[0]
# slind_p = slinc_p
# slind_n = slinc_n

slind   = np.append( slind_p,slind_n+L1)
centrs  = np.vstack((centrp[slind_p], centrn[slind_n]))

slice_tot = slice_p2 + slice_n2
ttzn1 = centrn[slind_n][:,0]
ttzp1 = centrp[slind_p][:,0]
###############################################################################
end_time3 = time.time()
print("time consuming: {:.2f}s".format(end_time3 - end_time2))
###############################################################################

#%% 4. 定位前后的置信帧

# frame_tr = 4

def fig_spots(slinc, slinc1, slice_, centr, neg=0):
    for tc in slinc:
        if tc in slinc1: glch_s.append(1)
        else: glch_s.append(0)
        center_z, center_y, center_x = np.int32(centr[tc])
        x_range1 = np.int32(center_x -distxy)
        x_range2 = np.int32(center_x +distxy)+1
        y_range1 = np.int32(center_y -distxy)
        y_range2 = np.int32(center_y +distxy)+1
        z_range1 = np.int32(center_z -distz0) 
        z_range2 = np.int32(center_z +distz0)
        # itmc    = []
        # itmcid  = []
        itmcid0 = np.where((centrs[:,2] >= x_range1) 
                          &(centrs[:,2] <  x_range2) 
                          &(centrs[:,1] >= y_range1) 
                          &(centrs[:,1] <  y_range2) 
                          &(centrs[:,0] >= z_range1) 
                          &(centrs[:,0] <  z_range2)) 
        itmc = np.int32(centrs[itmcid0,0]).T
        itmcid = itmcid0[0]
        '''
                    d3_____d4
                   /
                  /
          _______/
        d1       d2
        '''
        idgreater = np.where(itmc > center_z)[0]
        try:
            id1     = np.argmin(itmc[idgreater])
            id11    = idgreater[id1]
            id111   = itmcid[id11]
            id1111  = slind[id111]
            d4 = slice_tot[id1111][0][0]
        except ValueError:
            d4 = min(center_z + distz0,shape0[0]) 
            
        idless  = np.where(itmc < center_z)[0]
        try:
            id2     = np.argmax(itmc[idless])
            id22    = idless[id2]
            id222   = itmcid[id22]
            id2222  = slind[id222]
            d1  = slice_tot[id2222][1][0] +1
        except ValueError:
            d1  = center_z -distz0
            if d1 <0: d1 =0        
        
        slitc   = slice_[tc]
        d2      = slitc [0][0]
        d3      = slitc [1][0]+1
        # if d2-d1<=0 or d4-d3<=0:
        if d2-d1<=frame_tr or d4-d3<=frame_tr: 
            continue
        rng_t0.append([d1,d2,d3,d4]) 
        cnt_12.append(centr[tc][1:])
                  
rng_t0  = [] # tz 时间节点
cnt_12  = [] # xy 中心坐标
glch_s  = [] # 用以区分该事件是否来自于毛刺
# distz0  = 20
fig_spots(slinc_p, slinc_p1, slice_p2, centrp)
fig_spots(slinc_n, slinc_n , slice_n2, centrn)


#%% 4.2 intTR与tar_area
radius2 = 5
Imspt   = [] # 数据备选的区间
Indn    = [] # mark_id筛选的无效帧标记
Imint1  = [] 
Imint2  = [] # 对应区间TR
Dint    = [] # 闪烁幅度
cn_All  = [] # 未筛选的区域位置标记

for ii in range(len(cnt_12)):
    d1, d2, d3, d4 = rng_t0[ii]
    # 制作mask: 0代表无关区域 1代表拟合区域 2代表边缘区域
    # 初始化maski, 成品mask1
    maski   = np.zeros(shape1[1:]) 
    x_circle, y_circle = generate_circle(cnt_12[ii], radius, shape1)
    maski[y_circle, x_circle] = 1
    ind_1   = np.where(maski==1) 
    Visited = np.zeros(shape1[1: ], dtype=bool)
    for j in range(len(ind_1[0])):
        for x_offset in [-1, 0, 1]: 
            for y_offset in [-1, 0, 1]:
                yind = ind_1[0][j] + y_offset
                xind = ind_1[1][j] + x_offset
                if (xind >= 0 and xind < shape1[2] and  
                    yind >= 0 and yind < shape1[1] and maski[yind,xind] == 0):
                    if  not Visited[yind,xind]: 
                        maski  [yind, xind] =2 
                        Visited[yind, xind] =True
    # 根据mask的相关区域重新截取局域区间
    min1,min2   = np.array(ind_1).min(axis=1) 
    max1,max2   = np.array(ind_1).max(axis=1)
    mask1   = maski[min1:max1+1, min2:max2+1]
    # 用于最终的待拟合图片生成
    imspt   = im[d1:d4+1, min1:max1+1, min2:max2+1].copy()
    Imspt.append(imspt)
    # 用于计算区域的亮度timetrace
    immsk   = im[d1:d4+1, min1:max1+1, min2:max2+1].copy()
    # imnsk   = im[d1:d4+1, int(cnt_12[ii][0]-radius):int(cnt_12[ii][0]+radius), 
    #                       int(cnt_12[ii][1]-radius):int(cnt_12[ii][1]+radius)].copy()
    len_    = len(imspt)
    # 用于标记该帧是否采用
    indn    = np.ones(len_, dtype=bool)
    ind02   = np.array(np.where((mask1 ==2)|(mask1==0))).T
    ind_2   = np.array(np.where( maski ==2)).T
    # midval  = np.median(im[d1:d4+1, ind_2[:,0], 
    #                                 ind_2[:,1]], axis=1)
    markn   = marked_id[d1:d4+1 ,min1:max1+1, min2:max2+1]
    for k in range(len_):
        falid_pt =  np.count_nonzero(markn[k])/\
                                     markn[k].size*100
        if falid_pt >= 10: indn[k] = False
        immsk[k, ind02[:,0], ind02[:,1]] = 95 #midval[k]
    immsk[immsk < 95] = 95
    imint1  = np.mean(immsk, axis=(1,2))
    # imint2  = np.mean(imnsk, axis=(1,2))
    int0    = imint1[d2 -d1]
    int1    = imint1[d3 -d1]
    dint    = np.abs(int0-int1)
    
    Dint.append(dint); cn_All.append([min1,min2])
    Indn.append(indn); 
    Imint1.append(imint1) 
    # Imint2.append(imint2)
    
Dint    = np.array(Dint) 
int_dif = np.mean(Dint[np.where(np.abs(Dint -np.mean(Dint)) <2*np.std(Dint))])
# int_dif = 10

# %%4.3. 得到拟合图形

Ant_id  = []; Ant_12  = []; Ang_t0  = []
Ang_x0  = []; Ang_x1  = []

Cn_pos  = []    # 区域位置标记
SpotsI  = []    # 漂移校准标记
SpotsB  = []    # 待拟合光斑
validC  = []    # 有效光子数
Weight  = []    # 涉及帧数(权重)

perr_all_x = [10]*len(cnt_12)
perr_all_y = [10]*len(cnt_12)

def std_ix(xi, intx, dd):
    ix   = np.where(xi)[0]
    ditx = np.diff(intx)
    stdx = np.std(np.abs(ditx))
    try:
        ixa = np.where(np.abs(ditx)>2*stdx)[0]
        ixx = [ix0 for ix0 in ixa 
               if ix[ix0+1]-ix[ix0]>1 and ix[ix0]>=dd+4][0]
    except IndexError:
        return xi
    return [False if i >= ix[ixx]+1 else elem for i, elem in enumerate(xi)]

def brk_ix(xi, dd):
    flag = 0
    for i, elem in enumerate(xi[dd:]):
        if flag:  xi[i+dd] = False
        elif i > 15 and xi[i+dd-1] == False:
            xi[i+dd] = False; flag = 1
    return xi
            

for ii in range(len(cnt_12)):  
    if Dint[ii] >2*int_dif or Dint[ii] <0.3*int_dif: 
        continue 
    d1, d2, d3, d4 = rng_t0[ii]; cnt_I = (d4+d1)/2
    flno  = 0; flen  = 10; flag  = False
    int0    = Imint1[ii] [d2-d1]; int00   = int0
    int1    = Imint1[ii] [d3-d1]; int11   = int1
    len_    = len(Imspt[ii])
    
    std00   = int_dif/5
    # std00   = max(Dint [ii]/5, 1)
    # std01   = max(Dint [ii]/5, 1)
    
    # 确定掩膜下开始和结束态的亮度, 确定亮度范围
    while flno <= flen: 
        # if flno == flen:
        #     if np.abs(int1-int11)>=std00: int11 = int1
        #     if np.abs(int0-int00)>=std00: int00 = int0
        flno += 1
        x0 = (Imint1[ii]>=int00-std00) &(Imint1[ii]<=int00+std00)\
            &(range(len_)<=d2-d1) # &Indn[ii] 
        x1 = (Imint1[ii]>=int11-std00) &(Imint1[ii]<=int11+std00)\
            &(range(len_)>=d3-d1) # &Indn[ii] 
        valid_fr0 =np.count_nonzero(x0)
        valid_fr1 =np.count_nonzero(x1)
        # if  valid_fr0 <frame_tr or valid_fr1 <frame_tr: 
        if valid_fr0 <=0 or valid_fr1 <=0:
            flag = True; break;
        # int00 = np.mean(Imint1[ii][x0])
        # int11 = np.mean(Imint1[ii][x1])
        int00 = (np.max(Imint1[ii][x0]) +np.min(Imint1[ii][x0]))/2
        int11 = (np.max(Imint1[ii][x1]) +np.min(Imint1[ii][x1]))/2
    
    std000 = np.std (Imint1[ii][x0])
    std001 = np.std (Imint1[ii][x1])
    int000 = np.mean(Imint1[ii][x0])
    int111 = np.mean(Imint1[ii][x1])
    if np.abs(int0-int00)>=std00\
    or(std000<=std00/2 and np.abs(int000-int0)>=std00): 
          int00 = int0
    if np.abs(int1-int11)>=std00\
    or(std001<=std00/2 and np.abs(int111-int1)>=std00): 
          int11 = int1
    
    x0  = (Imint1[ii] >= int00 -std00)&(Imint1[ii] <= int00 +std00)
    x1  = (Imint1[ii] >= int11 -std00)&(Imint1[ii] <= int11 +std00)
    x00 = (Imint1[ii] >= int00 -std00*2)\
        & (Imint1[ii] <= int00 +std00*2)
    x11 = (Imint1[ii] >= int11 -std00*2)\
        & (Imint1[ii] <= int11 +std00*2)

    x0 = consec_T(x0, 3); x0 = consec_T3(x0,x00, 3)
    x1 = consec_T(x1, 3); x1 = consec_T3(x1,x11, 3)
    
    # x0 = brk_ix(x0[::-1], len(x0)-d2+d1)[::-1]
    # x1 = brk_ix(x1, d3-d1)
    
    if not(any(x0) and any(x1)): flag = True
    if flag: continue
    st0 = np.mean(Imspt[ii][x0], axis=0)
    st1 = np.mean(Imspt[ii][x1], axis=0)
    wgi = np.count_nonzero(x0) + np.count_nonzero(x1)
    # validc = np.sum(Imint1[ii][x0]-95) +np.sum(Imint1[ii][x1]-95)
    validc = np.abs(np.sum(Imint1[ii][x0]-95)*np.count_nonzero(x1)\
                   -np.sum(Imint1[ii][x1]-95)*np.count_nonzero(x0))
    if int0>int1: spt = st0 -st1
    else:         spt = st1 -st0
    
    Ant_id.append(ii); validC.append(validc)
    Ang_x0.append(x0); Ang_x1.append(x1)
    Ant_12.append(cnt_12[ii])
    Ang_t0.append(rng_t0[ii]) 
    Cn_pos.append(cn_All[ii])
    Weight.append(wgi); 
    SpotsB.append(spt); SpotsI.append(cnt_I)
    
    
    # P = sci_opt_fit(spt, pixel_size,4.6)
    # if P[1] =='fail': print('ss')
    # popt1   = P[0]
    # min1, min2 = cn_All[ii]
    # fit_plot(spt, popt1, pixel_size, targ, 
    #                       990, 'No'+str('%03d'%ii))
    # x_com, y_com = Adrift_xy[int(cnt_I//drift_stp)]
    # x = popt1[1] + min2 -3 -x_com
    # y = popt1[2] + min1 -3 -y_com
    
###############################################################################
end_time4 = time.time()
print("time consuming: {:.2f}s".format(end_time4 - end_time3))
###############################################################################

##%%5. 拟合与画图

Position_x = []; Position_y = []
Posiperr_x = []; Posiperr_y = []
Weight_arr = []
indexi_arr = []
sigma_xy   = []

ii = 1
for iii, Img in enumerate(SpotsB):
    # Img = np.pad(Img, ((3, 3), (3, 3)), 
    #              mode='constant', constant_values=Img[-1, -1])
    ind0 = SpotsI[iii]; min1,min2 = Cn_pos[iii]
    P = sci_opt_fit(Img, pixel_size,4.6)
    if P[1] =='fail': continue
    popt1   = P[0]
    perr1   = P[2]
    perr_all_x[Ant_id[iii]] = perr1[1]
    perr_all_y[Ant_id[iii]] = perr1[2]
    ellip   = abs(popt1[3]/popt1[4] -1)
    totjd   = perr1[1] +perr1[2]
    
    if ellip < 0.1 and totjd < 0.2 and perr1[1]< 0.1 and perr1[2]< 0.1 and\
       validC[iii] >=100:
        # 此处绘图方便检查        
        # fit_plot(Img, popt1, pixel_size, targ, 
        #                    iii, 'No0'+str('%03d'%ii))
        ii += 1
        # 相对拟合图片中心的距离
        x_com, y_com = Adrift_xy[int(ind0 //drift_stp)]
        x = popt1[1] + min2 -3 - x_com
        y = popt1[2] + min1 -3 - y_com
        Position_x.append(x)
        Position_y.append(y)
        Weight_arr.append(Weight[iii])
        indexi_arr.append(iii)
        sigma_xy.append((popt1[3], popt1[4]))
        Posiperr_x.append(perr1[1])
        Posiperr_y.append(perr1[2])
        
Position_x  = np.array(Position_x)
Position_y  = np.array(Position_y)
Posiperr_x  = np.array(Posiperr_x)
Posiperr_y  = np.array(Posiperr_y)
Weight_arr  = np.array(Weight_arr)
 
Andexi_arr  = np.array(indexi_arr)
Aerr_all_x  = np.array(perr_all_x)
Aerr_all_y  = np.array(perr_all_y)

aXY     = np.vstack((Position_x, Position_y)).T
dXY     = np.vstack((Posiperr_x, Posiperr_y))
AdXY    = np.mean(dXY, axis=0)
aXYT    = np.vstack((aXY.T,      Weight_arr))
adXYT   = np.vstack((aXYT,       AdXY      )).T
