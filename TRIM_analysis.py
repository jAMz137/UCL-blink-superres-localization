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
if not os.path.exists(Dat): 
    os.mkdir(Dat)

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

radius      = int(ex+enl)   # radius of spot mask to extract TR

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

########################################################################
import time
start_time = time.time()
########################################################################

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

##%% 1.2.
def blink_mark (slice_, max_pos, enl0=enl):
    '''
    prooerties: 
        centr: 中心位置  rng: 时间中心  spots: 光斑图貌  std: 拟合误差    
    断点标记分类：
        1.硬断点(范围过大 or 轮廓异常) 2.信噪比低(拟合失败)  3.范围过小  4.靠近边缘 
    '''
    Avent0 = []
    for j, (min_coord, max_coord) in enumerate(slice_):
        brk     = 0
        # 事件时间中心与时间范围
        tz      = (max_coord[0] +1 +min_coord[0]) /2
        ng      = (max_coord[0] +1 -min_coord[0]) /2    
        # YX轴上的大小
        shapR   = np.array([max_coord[1] -min_coord[1],
                           max_coord[2] -min_coord[2]])
        # 外扩ex光斑tz加和后形貌
        spotA   = np.abs(np.sum(imdi[
                            min_coord[0]: max_coord[0]+1,
                            max(min_coord[1]-ex,0): max_coord[1]+ex+1, 
                            max(min_coord[2]-ex,0): max_coord[2]+ex+1],axis=0))
        # 无外扩光斑tz加和后形貌
        spotB   = np.abs(np.sum(imgd[
                            min_coord[0]: max_coord[0]+1,
                            min_coord[1]: max_coord[1]+1,
                            min_coord[2]: max_coord[2]+1],axis=0))
        maxi0   = np.argwhere(spotB == np.max(spotB))[0]
        cntr0   = maxi0 + np.array([min_coord[1],min_coord[2]])
        cntr1   = max_pos[j][1:]
        maxi1   = cntr1 - np.array([min_coord[1],min_coord[2]])
        if np.any(np.abs(maxi1 - maxi0)>enl/2):
            cntr0 = cntr1; maxi0 = maxi1; brk = 1
        if np.any(np.abs(maxi0+1-shapR/2)>enl): brk = 1
        if np.any(shapR < enl): 
            brk = 3; std = 'None'
        elif not(cntr0[0]-(enl0)>=0 and cntr0[1]-(enl0)>=0  
             and cntr0[0]+(enl0)<=shape1[1]-1 and cntr0[1]+(enl0)<=shape1[2]-1): 
            brk = 4; std = 'None'   
        if brk <= 1:
            ppcc = sci_opt_fit(spotA, pixel_size, 4.6)
            popt = ppcc[0]
            perr = ppcc[2]
            if ppcc[1] =='fail': brk = 2; std ='Wrong'
            else:
                # 此处绘图方便检查        
                # fit_plot(spotA, popt, pixel_size, show =1)
                std     = sum(perr[1 :3])
                if np.all(std < 1): 
                    cntr0 = np.array([popt[2]+max(min_coord[1]-ex,0), 
                                     popt[1]+max(min_coord[2]-ex,0)])
        item000 = {}        
        item000['slc']  = slice_[j]
        item000['brk']  = brk
        item000['zng']  = ng 
        item000['std']  = std
        # item00['spots'] = spotA
        item000['glced'] = False
        item000['centr'] = np.append(tz,cntr0)
        Avent0.append(item000)  
    return Avent0
             
eventp  = blink_mark(slice_p, posp)
eventn  = blink_mark(slice_n, posn)

Aventp  = [item for item in eventp if item['brk'] <= 2]
Aventn  = [item for item in eventn if item['brk'] <= 2]
atzp0 = np.array([item['centr'] for item in Aventp])[:,0]
atzn0 = np.array([item['centr'] for item in Aventn])[:,0]


#%% 2. 标记大型毛刺
# 具体为标记方块区域, 后续TR经过方块时排除标记True占比大于10%的帧
centrn      = np.array([item['centr'] for item in Aventn ])
# 标记配对区域
marked_id   = np.full_like(im,fill_value=False, dtype=bool)

def pairing(itm0, itm1):
    cntrz0 = np.int32(itm0['centr'])[0]
    cntrz1 = np.int32(itm1['centr'])[0]
    z0 = np.floor(cntrz0 -itm0['zng'] -itm1['zng'] -distA0)
    z1 = np.ceil (cntrz0 +itm0['zng'] +itm1['zng'] +distA0+1)
    z_range = range(np.int32(z0), np.int32(z1))
    
    if cntrz1 in z_range:
        li  = np.array(itm0['slc'])
        lii = np.array(itm1['slc'])
        lmi = np.min(np.vstack((li[0],lii[0])),axis=0)
        lma = np.max(np.vstack((li[1],lii[1])),axis=0)
        spoti = np.sum(imdi[li[0][0]:li[1][0]+1,
                            lmi[1]:  lma[1]+1, 
                            lmi[2]:  lma[2]+1],axis=0) 
        spotii= np.sum(imdi[lii[0][0]:lii[1][0]+1,
                            lmi[1]:  lma[1]+1, 
                            lmi[2]:  lma[2]+1],axis=0)
        # FIXME: 此处应该更全面详尽比较这两帧的区别
        diff = np.abs(np.mean(spoti+spotii)) 
        slca = [slice(lmi[0]+1,lma[0]+1), slice(lmi[1],lma[1]+1), slice(lmi[2],lma[2]+1)]
        if diff < thres2: return diff, slca 
    
for item00 in Aventp:
    if item00['brk']==1: continue
    center_z,center_y,center_x =np.int32(item00['centr'])
    x_range = range(center_x-distA12, center_x+distA12+1)
    y_range = range(center_y-distA12, center_y+distA12+1)  
    locidxy = np.where((centrn[:, 0] > center_z -distAs)
                      &(centrn[:, 0] < center_z +distAs)
                      &(centrn[:, 1] >=np.floor(center_y -distA12))
                      &(centrn[:, 1] <=np.ceil (center_y +distA12))
                      &(centrn[:, 2] >=np.floor(center_x -distA12))
                      &(centrn[:, 2] <=np.ceil (center_x +distA12)))
    if len(locidxy[0]) == 0: continue
    z_value     = np.int32(centrn[locidxy, 0]).T
    locidgrt    = np.where(z_value >= center_z)[0]
    pairn       = 0
    if len(locidgrt) != 0:
        locid1      = np.argmin(z_value[locidgrt])
        locid11     = locidgrt  [locid1]
        locidz1     = locidxy[0][locid11]
        item11      = Aventn[ locidz1]
        if (not item11['glced']) and item11['brk'] !=1: 
            try: diff1, _slc1 = pairing(item00, item11); pairn += 1
            except: pass
    locidles   = np.where(z_value <= center_z)[0]
    if len(locidles) != 0:
        locid2      = np.argmax(z_value[locidles])
        locid22     = locidles[locid2]
        locidz2     = locidxy[0][locid22]
        item12      = Aventn [locidz2]
        if (not item12['glced']) and item12['brk'] !=1: 
            try: diff2, _slc2 = pairing(item00, item12); pairn += 2
            except: pass
    if   pairn == 0: continue
    elif pairn == 1 or (pairn ==3 and diff1<=diff2): 
        item00['glced'] = True; item11['glced'] = True
        marked_id[_slc1[0], _slc1[1], _slc1[2]] = True
    elif pairn == 2 or (pairn ==3 and diff1> diff2): 
        item00['glced'] = True; item12['glced'] = True
        marked_id[_slc2[0], _slc2[1], _slc2[2]] = True

# marked_im   = np.ma.masked_array(im, marked_id)

########################################################################
end_time1 = time.time()
print("time consuming: {:.2f}s".format(end_time1 - start_time))
########################################################################

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

########################################################################
end_time2 = time.time()
print("time consuming: {:.2f}s".format(end_time2 - end_time1))
########################################################################
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
Avent = pickle.load(open("E:\\PrGm_tempfile\\Avent.p", "rb"))
Avent+= Avent2
pickle. dump(Avent, open("E:\\PrGm_tempfile\\Avent.p", "wb"))
