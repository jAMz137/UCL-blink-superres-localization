# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 11:36:24 2023 
@uthor: James Ma
    For picking slices from 3D array arbitarily impliedly using timetraces. 

V20240429:重新选择 distz0; 排除非连续的三帧以下的部分; 处理多颗粒同时闪烁的情况. 
V20240704:格式规范, 使用typing进行类型标注. 
""" 

import numpy as np; import tifffile as tifile
import os, re
from scipy.ndimage import gaussian_filter
if __name__ == '__main__':
    script_dir =os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)   
from utils_toolbox import\
    imloc_max, sci_opt_fit, consec_T, consec_T3, gen_circle 
from drift_correction import drift_correction
from TRIM_events import blk_events
from TRIM_traces import blk_traces

Root        = 'G:\\data2024\\'
Data        =             '20240220_2'
Path        =  Root + Data + '\\stack'

Targ        = '53d0'
numbers     =  re.findall(r'\d+',Targ)
Filename    =  Targ + 'stack.tif'

Dat         =  Data +'_'+Targ +'Dat\\'
if not os.path.exists(Dat): os.mkdir(Dat)

im0  = np.array(np.float32(tifile.imread(Path +'\\' +Filename)))[ :-1]

#%% ex:参数统一设置

pixel_size = 1; bintime = 0.25 # s

sigma_s = 5 *pixel_size     # float, key parameter, FWHM=2.355*σ
enl     = int(2*sigma_s)    # int, spots_n/p outlier 
                            # (max) enl, (min) enl/2

ex      = int(np.ceil(sigma_s/2) +1) # int, 判断是否接触边界的范围

sigma_f     = sigma_s       # float, sigma of guassian filter(xy)
sigma_t     = bintime*2     # float, sigma of guassian filter(tz)

# 一个能与背景产生区分的值
thres0      = 3.5           # float, blinking event marking
# 这个参数实际上规定的的是TR中能被探测到的两帧之间变化
thres1      = 9             # float 

mx_filt0    = bintime*12    # float, sigma of maximum filter(tz)
# 此处取2仅对当前帧与+1帧做统一区域最大值滤波, 理想取值3
mx_filt12   = enl           # float, sigma of maximum filter(xy)
mxfilt_size = (mx_filt0, mx_filt12, mx_filt12)

distA0      = int(bintime*8)    # int, tz glitches range
distAs      = int(2.5*distA0)   # int, tz glitches search range
distA12     = enl+1             # int, xy glitches range
thres2      = thres0            # float, glitch match threshold

frame_tr    = 3             # int, min frame to be a state
distz0      = 30            # int, max frame to be a state
distxy      = enl *0.8      # float, to judge the overlapping 
                            # range of light spots on xy axis

radius      = int(ex+enl)
int_dif     = 10

#%% ex:截取研究区域
'''
框选对象点的正方形范围, 标注HCI上显示的像素值, 
读取后 x/y轴会调换(0指标对应y轴,1指标对应x轴)
'''
Center_x    = 115
Center_y    = 135
width       = 70
im = im0[
        :,
        int(Center_y-width):int(Center_y+width),
        int(Center_x-width):int(Center_x+width)]

###############################################################################
import time
start_time = time.time()
###############################################################################

#%% ex:漂移校准(correlated)
dft_step    = 20    # 漂移校准步长
mark_num    = 1     # 校准标记个数
mark_names  = ['driftmark5V', 'driftmark2', 'driftmark3']
Adrift_xy   = drift_correction(
            Root + Data, 
            Targ +'stamark.tif',
            numbers[0], 
            dft_step, mark_num,
            pixel_size, 
            mark_names).correlated()
        
#%% 1. 整理闪烁事件点
# 查找整个含时间的三维矩阵中发生闪烁的帧数坐标, 定位完整闪烁
# 两个数组中分别记录了变暗与变亮两种事件

img     = gaussian_filter( im, sigma = (sigma_t, sigma_f, sigma_f))
imgd    = np.diff(img,axis=0)
imdi    = np.diff(im, axis=0)
shape1  = np.shape(imgd)

slice_p, posp = imloc_max(imgd,  thres0, thres1, mxfilt_size, enl)
slice_n, posn = imloc_max(imgd, -thres0, thres1, mxfilt_size, enl)

...

eventp  = blk_events(slice_p, posp)
eventn  = blk_events(slice_n, posn)

#Aventp  = [item for item in eventp if item['brk'] <= 2]
#Aventn  = [item for item in eventn if item['brk'] <= 2]
#atzp0 = np.array([item['centr'] for item in Aventp])[:,0]
#atzn0 = np.array([item['centr'] for item in Aventn])[:,0]

###############################################################################
end_time1 = time.time()
print("time consuming: {:.2f}s".format(end_time1 - start_time))
###############################################################################

#%% 3. 定位前后的置信帧

eventpT = [item for item in Aventp if not item['glced']]
eventnT = [item for item in Aventn if not item['glced']]
centrpT = np.array([item['centr'] for item in eventpT])
centrnT = np.array([item['centr'] for item in eventnT])
atzp1 = centrpT[:,0]
atzn1 = centrnT[:,0]

##%% 作为截断事件的点
cntrAll = np.vstack((centrpT, centrnT))
eventpnT= eventpT + eventnT

'''
abort: 0,通过 1,Dint 2,3,可用帧为0 4,拟合失败 5,拟合参数不通过

glch_s:  是否来自毛刺;    rng_t0:  tz时间节点
cnt_12:  xy中心坐标;    corner:  区域位置标记
Dint:  中心事件闪烁幅度;   Indn:  mark_id标记的无效帧
Imspt:  数据备选的区间;   Imint1:  对应区间TR
'''
fig_spots(Aventp )
fig_spots(eventnT)
#%% 4. intTR与tar_area
# radius2 = 5
...

###############################################################################
end_time2 = time.time()
print("time consuming: {:.2f}s".format(end_time2 - end_time1))
###############################################################################
#%%5. 拟合与画图

ii = 1
for iii,item2 in enumerate(Avent1):
    if item2['abort'] != 0 and item2['abort'] != 5: 
        continue
    min1, min2 = item2['corner']; Img = item2['SpotsB']
    # Img = np.pad(Img, ((3, 3), (3, 3)), 
    #                   mode = 'constant', 
    #                   constant_values=Img[-1,-1])
    P = sci_opt_fit(Img, pixel_size, 4.6)
    if P[1] =='fail': 
        item2['abort'] = 4
        item2['stdx'] = 'NaN'
        item2['stdy'] = 'NaN'
        continue
    popt1 = P[0]; perr1 = P[2]
    item2['stdx']   = perr1[1]
    item2['stdy']   = perr1[2]
    item2['popt']   = popt1
    x_com, y_com = Adrift_xy[int(item2['SpotsI'] //dft_step)]
    x = popt1[1] + min2 -3 - x_com
    y = popt1[2] + min1 -3 - y_com
    item2['x'] = x; item2['y'] = y
    item2['sigma_xy'] = (popt1[3], popt1[4])
    item2['drift'] = (x_com, y_com)
    
    ellip   = abs(popt1[3]/popt1[4] -1)
    totjd   = perr1[1] + perr1[2]
    if  1 and\
        item2['validC'] >=100 and ellip< 0.1 and\
        totjd< 0.2 and perr1[1]< 0.1 and perr1[2]< 0.1:
        # 此处绘图方便检查        
        # fit_plot(Img, popt1, pixel_size, Targ, iii,
        #                      'No0' + str('%03d'%ii))
        # ii += 1
        # 相对拟合图片中心的距离
        pass
    else: item2['abort'] = 5

        
Position_x  = np.array([item['x'] 
                        for item in Avent1 if item['abort']==0]) 
Position_y  = np.array([item['y'] 
                        for item in Avent1 if item['abort']==0]) 

# Posiperr_x  = np.array([item['stdx'] 
#                         for item in Avent1 if item['abort']==0]) 
# Posiperr_y  = np.array([item['stdy'] 
#                         for item in Avent1 if item['abort']==0]) 
# Weight_arr  = np.array([item['wght'] 
#                         for item in Avent1 if item['abort']==0]) 

aXY     = np.vstack((Position_x, Position_y)).T
# dXY     = np.vstack((Posiperr_x, Posiperr_y))
# AdXY    = np.mean  (dXY, axis=0)
# aXYT    = np.vstack((aXY.T,      Weight_arr))
# adXYT   = np.vstack((aXYT,       AdXY      )).T

Avent2  = [item for item in Avent1 if item['abort']==0]
import pickle
Avent   = pickle.load(open("E:\\PrGm_tempfile\\Avent.p", "rb"))
Avent  += Avent2
pickle.dump(Avent, open("E:\\PrGm_tempfile\\Avent.p", "wb"))
