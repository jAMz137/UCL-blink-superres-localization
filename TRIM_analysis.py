"""
@uthor: James Ma
    For picking slices from 3D array arbitarily impliedly using timetraces. 

V20240429:重新选择 frame_max; 排除非连续的三帧以下部分; 处理多颗粒同时闪烁的情况. 
V20240704:格式规范, 使用typing进行类型标注. 
""" 

import numpy as np; import tifffile as tifile
import os, re
from scipy.ndimage import gaussian_filter

if __name__ == '__main__':
    script_dir =os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)   

from utils_toolbox import sci_opt_fit
 
from drift_correction import drift_correction
from TRIM_events import blk_events
from TRIM_traces import blk_traces, blk_trace

Root        = 'G:\\data2024\\'
Data        =             '20240220_2'
Path        =  Root + Data + '\\stack'

Targ        = '53d0'
numbers     =  re.findall(r'\d+',Targ)
Filename    =  Targ + 'stack.tif'

Dat         =  Data +'_'+Targ +'Dat\\'

if not os.path.exists(Dat): 
    os.mkdir(Dat)

im0  = np.array(np.float32(tifile.imread(Path +'\\' +Filename)))[ :-1]


#%% ex:参数统一设置

parameters  = {}
parameters['excitation']  = numbers[0]
parameters['thres0']  = 3.5 # float, 一个能与背景产生区分的值
parameters['thres1']  = 9   # float, TR能被探测到的两帧间变化

parameters['pixel_size'] = 1; bintime = 0.25
# float, sigma parameter, FWHM=2.355*σ
sigma_s = 5 *parameters['pixel_size'] 
# 
# int, spots_n/p outlier, (max)enl (min)enl/2
parameters['enl'] = int(2*sigma_s) 
# int, Determining if the boundary is reached
parameters['elx'] = int(np.ceil(sigma_s/2)+1) 

# int, tz glitches search range
parameters['dist_s']    = int(2.5*parameters['distA0']) 
# int, glitches range
parameters['dist00']    = int(bintime *8) 
parameters['dist12']    = parameters['enl'] +1
# float, glitch match threshold
parameters['thres2']    = parameters['thres0'] 

# int, min/max frame to be a state
parameters['frame_min'] = 3             
parameters['frame_max'] = 30
# float, to judge the overlapping(xy)
parameters['distxy']    = parameters['enl'] *0.8      
                            
# radius of spot mask to extract TR
parameters['radius']   = int(parameters['elx'] 
                            +parameters['enl'])   

# float, sigma of guassian filter(xy& tz)
sigma_t         = bintime*2
sigma_f         = sigma_s       
guasfilt_size   = (sigma_t, sigma_f, sigma_f)

# float, sigma of maximum  filter(xy& tz)
# 此处2对当前帧与+1帧做最大值滤波, 理想取值3
mx_filt0        = bintime*12    
mx_filt12       = parameters['enl']
parameters['mxfilt_size'] = (mx_filt0, mx_filt12, mx_filt12)


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

##################################################################
import time
start_time = time.time()
##################################################################

#%% ex:漂移校准(correlated)
dft_step    = 20    # 漂移校准步长
mark_num    = 1     # 校准标记个数
mark_names  = ['driftmark5V', 'driftmark2', 'driftmark3']
Adrift_xy   = drift_correction(
            Root + Data, 
            Targ +'stamark.tif',
            parameters['excitation'], 
            dft_step, mark_num,
            parameters['pixel_size'], 
            mark_names).correlated()
        
#%% 1. 整理闪烁事件点
# 查找整个含时间的三维矩阵中发生闪烁的帧数坐标, 定位完整闪烁
# 两个数组中分别记录了变暗与变亮两种事件

img     = gaussian_filter( im, sigma = guasfilt_size)
imgd    = np.diff(img,axis=0)
imdi    = np.diff(im, axis=0)

Events = blk_events(imdi, imgd, parameters)

##################################################################
end_time1 = time.time()
print("time consuming: {:.2f}s".format(end_time1 - start_time))
##################################################################

#%% 3. 定位前后的置信帧

Intervals = blk_traces(Events, im, parameters)

##################################################################
end_time2 = time.time()
print("time consuming: {:.2f}s".format(end_time2 - end_time1))
##################################################################
#%%5. 拟合与画图

ii = 1
for iii, item2 in enumerate(Intervals.Traces):
    item2: blk_trace
    outfit = {}
    if item2.abort != 0 and item2.abort != 5: 
        continue
    min1, min2 = item2.corner; Img = item2.SpotB
    # Img = np.pad(Img, ((3, 3), (3, 3)), 
    #                   mode = 'constant', 
    #                   constant_values=Img[-1,-1])
    P = sci_opt_fit(Img, parameters['pixel_size'], 4.6)
    outfit['result'] = 'fail'
    if P[1] == 'fail': 
        item2.abort = 4
        continue
    popt1 = P[0]; perr1 = P[2]
    outfit['stdxy'] = [perr1[1], perr1[2]]
    # outfit['popt' ] = popt1
    x_com, y_com = Adrift_xy[int(item2.SpotI //dft_step)]
    x = popt1[1] + min2 -3 - x_com
    y = popt1[2] + min1 -3 - y_com
    
    outfit['sigma_xy'] = (popt1[3], popt1[4])
    item2. xy = [x,y]
    item2. drift = [x_com, y_com]
    item2. fit_o = outfit
    
    ellip   = abs(popt1[3]/popt1[4] -1)
    totjd   = perr1[1] + perr1[2]
    if  item2.ValidC >=100 and ellip< 0.1 and\
        totjd< 0.2 and perr1[1]< 0.1 and perr1[2]< 0.1:
        # 此处绘图方便检查        
        # fit_plot(Img, popt1, pixel_size, Targ, iii,
        #                      'No0' + str('%03d'%ii))
        # ii += 1
        # 相对拟合图片中心的距离
        pass
    else: item2.abort = 5

        
Position_x  = np.array([item.x 
                        for item in Intervals.Traces if item.abort==0]) 
Position_y  = np.array([item.y 
                        for item in Intervals.Traces if item.abort==0]) 

aXY     = np.vstack((Position_x, Position_y)).T
# dXY     = np.vstack((Posiperr_x, Posiperr_y))
# AdXY    = np.mean  (dXY, axis=0)
# aXYT    = np.vstack((aXY.T,      Weight_arr))
# adXYT   = np.vstack((aXYT,       AdXY      )).T
Avent2  = [item for item in Intervals.Traces if item.abort==0]

import pickle
Avent = pickle.load(open("E:\\PrGm_tempfile\\Avent.p", "rb"))
Avent+= Avent2
pickle. dump(Avent, open("E:\\PrGm_tempfile\\Avent.p", "wb"))
