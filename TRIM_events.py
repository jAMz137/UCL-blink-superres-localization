import numpy as np
from utils_toolbox import sci_opt_fit

class blk_event():
    def __init__(self, slc: tuple[np.ndarray,np.ndarray] ):
        (self. min_coord, self. max_coord) = slc
        # 事件时间范围
        self. zng   = (self.max_coord[0] +1 -self.min_coord[0]) /2 
        # self. spots = spotA
        self. glced = False
        self. brk   = 0
    
    @property
    def centr(self):
        # 事件时间中心
        tz  = (self.max_coord[0] +1 +self.min_coord[0]) /2

        return np.append(tz, cntr0)
    
    def shapeR(self):
        # YX轴上的大小
        return np.array([self.max_coord[1] -self.min_coord[1],
                        self.max_coord[2] -self.min_coord[2]])
        


        self. std  = std


class blk_events():
    def __init__(self, slice__: list[tuple[np.ndarray,np.ndarray]], 
                       max_pos: list[np.ndarray]):
        self.slice_ = slice__
        self.maxpos = max_pos

    def blink_mark (self, enl0: int, enl: int, ex: int, shape1, pixel_size,
                    imdi: np.ndarray, imgd: np.ndarray):
        '''
        prooerties: 
            centr: 中心位置  rng: 时间中心  spots: 光斑图貌  std: 拟合误差    
        断点标记分类：
            1.硬断点(范围过大 or 轮廓异常) 2.信噪比低(拟合失败)  3.范围过小  4.靠近边缘 
        '''
        Avent0 = []
        for j, slc_ in enumerate(self.slice_):
            event   = blk_event(slc_)
            
            shapR   = event. shapeR()
            # 外扩ex光斑tz加和后形貌
            spotA   = np.abs(np.sum(imdi[
                                event.min_coord[0]: event.max_coord[0]+1,
                                max(event.min_coord[1]-ex,0): event.max_coord[1]+ex+1, 
                                max(event.min_coord[2]-ex,0): event.max_coord[2]+ex+1],axis=0))
            # 无外扩光斑tz加和后形貌
            spotB   = np.abs(np.sum(imgd[
                                event.min_coord[0]: event.max_coord[0]+1,
                                event.min_coord[1]: event.max_coord[1]+1,
                                event.min_coord[2]: event.max_coord[2]+1],axis=0))
            maxi0   = np.argwhere(spotB == np.max(spotB))[0]
            cntr0   = maxi0 + np.array([event.min_coord[1],event.min_coord[2]])
            cntr1   = self.maxpos[j][1:]
            maxi1   = cntr1 - np.array([event.min_coord[1],event.min_coord[2]])
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
                        cntr0 = np.array([popt[2]+max(event.min_coord[1]-ex,0), 
                                          popt[1]+max(event.min_coord[2]-ex,0)])
            item000 = {}        
            item000['self.slc']  = self.slc[j]
            item000['brk']  = brk
            item000['zng']  = ng 
            item000['std']  = std
            # item00['spots'] = spotA
            item000['glced'] = False
            item000['centr'] = np.append(tz,cntr0)
            Avent0.append(item000)  
        return Avent0
             
    #标记大型毛刺
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
            li  = np.array(itm0['self.slc'])
            lii = np.array(itm1['self.slc'])
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
