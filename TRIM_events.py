import numpy as np
from utils_toolbox import sci_opt_fit, imloc_max

class blk_event():
    def __init__(self, slc: tuple[np.ndarray,np.ndarray] ):
        (self. min_coord, self. max_coord) = slc
        # 事件时间范围
        self. zng   = (self.max_coord[0] +1 -self.min_coord[0]) /2 
        # self. spots = spotA
        self. glced: bool       = False
        self. brk_mrk: int      = 0
        self. std_err: float    = 10

    @property
    def centr(self): return self._centr   
    @centr.setter
    def centr(self, ary: np.ndarray):
        if np.any(ary) <= 0:
            raise ValueError("index must be positive!")
        tz  = (self.max_coord[0] +1 +self.min_coord[0]) /2
        self. _centr = np.append(tz, ary)
        
    @property
    def shapeR(self):
        # YX轴上的大小
        return np.array([self.max_coord[1] -self.min_coord[1],
                         self.max_coord[2] -self.min_coord[2]])



class blk_events():
    def __init__(self, imdi_: np.ndarray, imgd_: np.ndarray,
                       parameters_: dict):
        self. imdi = imdi_
        self. imgd = imgd_
        self. shape1 = np.shape(imgd_)
        self. parameters = parameters_
        slice_p, posp = imloc_max(imgd_,  
                                  parameters_['thres0'], 
                                  parameters_['thres1'], 
                                  parameters_['mxfilt_size'], 
                                  parameters_['enl'])
        slice_n, posn = imloc_max(imgd_, 
                                 -parameters_['thres0'], 
                                  parameters_['thres1'], 
                                  parameters_['mxfilt_size'], 
                                  parameters_['enl'])

        self. events_p = self.get_event_with_mark(slice_p, posp)
        self. events_n = self.get_event_with_mark(slice_n, posn)
        self. glitch_id = self.pair_glitch()
        

    def get_event_with_mark (self, slice_, maxpos) -> list[blk_event]:
        enl0 = self.parameters['enl']
        enl  = self.parameters['enl']
        elx  = self.parameters['elx']
        shape1 = self.shape1
        '''
        prooerties: 
            centr: 中心位置  rng: 时间中心  spots: 光斑图貌  std: 拟合误差    
        断点标记分类：
            1. 范围过大或轮廓异常
            2. 信噪比低(拟合失败)  
            3. 范围过小  
            4. 靠近边缘 
        '''
        Avents = []
        for j, slc_ in enumerate(slice_):
            event   = blk_event(slc_)
            shapR   = event. shapeR
            brk     = event.brk_mrk 
            # 外扩ex光斑tz加和后形貌
            spotA   = np.abs(np.sum(self.imdi[
                                event.min_coord[0]: event.max_coord[0]+1,
                                max(event.min_coord[1]-elx, 0): 
                                    event.max_coord[1]+elx +1, 
                                max(event.min_coord[2]-elx, 0): 
                                    event.max_coord[2]+elx +1],axis=0))
            # 无外扩光斑tz加和后形貌
            spotB   = np.abs(np.sum(self.imgd[
                                event.min_coord[0]: event.max_coord[0]+1,
                                event.min_coord[1]: event.max_coord[1]+1,
                                event.min_coord[2]: event.max_coord[2]+1],axis=0))
            maxi0   = np.argwhere(spotB == np.max(spotB))[0]
            cntr0   = maxi0 + np.array([event.min_coord[1],
                                        event.min_coord[2]])
            cntr1   = maxpos[j][1:]
            maxi1   = cntr1 - np.array([event.min_coord[1],
                                        event.min_coord[2]])
            if np.any(np.abs(maxi1 - maxi0)>enl/2):
                cntr0 = cntr1 
                maxi0 = maxi1
                brk = 1
            if np.any(np.abs(maxi0+1-shapR/2)>enl): 
                brk = 1
            if np.any(shapR < enl): 
                brk = 3
            elif not(cntr0[0]-(enl0)>=0 and cntr0[1]-(enl0)>=0  
                    and cntr0[0]+(enl0)<=shape1[1]-1 
                    and cntr0[1]+(enl0)<=shape1[2]-1): 
                brk = 4  
            if brk <= 1:
                ppcc = sci_opt_fit(spotA, self.parameters['pixel_size'], 4.6)
                popt = ppcc[0]
                perr = ppcc[2]
                if ppcc[1] =='fail': 
                    brk = 2
                else:
                    # 此处绘图方便检查        
                    # fit_plot(spotA, popt, pixel_size, show =1)
                    event.std_err = sum(perr[1 :3])
                    if np.all(event.std_err < 1): 
                        cntr0 = np.array([popt[2]+max(event.min_coord[1]-elx, 0), 
                                          popt[1]+max(event.min_coord[2]-elx, 0)])
            event.centr    = cntr0 
            event.brk_mrk  = brk

            Avents.append(event)  
        return Avents


    # enable indexing
    def __getitem__(self, idx):
        if idx ==0:
            return self.events_p
        if idx ==1:
            return self.events_n
        else:
            raise IndexError("index {} is out of range".format(idx))



    def pairing(self, itm0: blk_event, itm1: blk_event):
        cntrz0 = np.int32(itm0.centr[0])
        cntrz1 = np.int32(itm1.centr[0])
        z0 = np.floor(cntrz0 -itm0.zng -itm1.zng -self.parameters['dist00'])
        z1 = np.ceil (cntrz0 +itm0.zng +itm1.zng +self.parameters['dist00']+1)
        z_range = range(np.int32(z0), np.int32(z1))
        
        if cntrz1 in z_range:
            li0  = itm0.min_coord
            li1  = itm0.max_coord
            lii0 = itm1.min_coord
            lii1 = itm1.max_coord
            lmi = np.min(np.vstack((li0,lii0)),axis=0)
            lma = np.max(np.vstack((li1,lii1)),axis=0)
            spoti = np.sum(self.imdi[li0[0]:li1[0]+1,
                                lmi[1]:  lma[1]+1, 
                                lmi[2]:  lma[2]+1],axis=0) 
            spotii= np.sum(self.imdi[lii0[0]:lii1[0]+1,
                                lmi[1]:  lma[1]+1, 
                                lmi[2]:  lma[2]+1],axis=0)
            # FIXME: 此处应该更全面详尽比较这两帧的区别
            diff = np.abs(np.mean(spoti+spotii)) 
            slca = [slice(lmi[0]+1,lma[0]+1), slice(lmi[1],lma[1]+1), slice(lmi[2],lma[2]+1)]
            if diff < self.parameters['thres2']: 
                return diff, slca 
            else: return []
        else: return []



    def pair_glitch(self):
        #标记大型毛刺
        # 具体为标记方块区域, 后续TR经过方块时排除标记True占比大于10%的帧
        centrn      = np.array([item.centr for item in self.events_n ])
        distA12     = self.parameters['dist12']
        distAs      = self.parameters['dist_s']
        # 标记配对区域
        shape2 = (self.shape1[0]+1, self.shape1[1], self.shape1[2])
        marked_id = np.full(shape2, False, dtype=bool)
        for item00 in self.events_n:
            if item00.brk_mrk==1: continue
            center_z,center_y,center_x =np.array(np.int32(item00.centr))
            x_range = range(center_x-distA12, center_x+distA12+1)
            y_range = range(center_y-distA12, center_y+distA12+1)  
            locidxy = np.where((centrn[:, 0] > center_z -distAs)
                            &(centrn[:, 0] < center_z +distAs)
                            &(centrn[:, 1] >=np.floor(center_y -distA12))
                            &(centrn[:, 1] <=np.ceil (center_y +distA12))
                            &(centrn[:, 2] >=np.floor(center_x -distA12))
                            &(centrn[:, 2] <=np.ceil (center_x +distA12)))
            if len(locidxy[0]) == 0: continue
            z_value     = np.array(np.int32(centrn[locidxy, 0])).T
            pairn       = 0

            locidgrt    = np.where(z_value >= center_z)[0]
            if len(locidgrt) != 0:
                locid1      = np.argmin(z_value[locidgrt])
                locid11     = locidgrt  [locid1]
                locidz1     = locidxy[0][locid11]
                item11      = self.events_n[locidz1]
                if (not item11.glced) and item11.brk_mrk !=1: 
                    try: 
                        diff1, _slc1 = self.pairing(item00, item11); 
                        pairn += 1
                    except: pass

            locidles   = np.where(z_value <= center_z)[0]
            if len(locidles) != 0:
                locid2      = np.argmax(z_value[locidles])
                locid22     = locidles[locid2]
                locidz2     = locidxy[0][locid22]
                item12      = self.events_n[locidz2]
                if (not item12.glced) and item12.brk_mrk !=1: 
                    try: 
                        diff2, _slc2 = self.pairing(item00, item12); 
                        pairn += 2
                    except: pass

            if   pairn == 0: continue
            elif pairn == 1 or (pairn ==3 and diff1 <= diff2):  # type: ignore
                item00.glced = True; item11.glced = True
                marked_id[_slc1[0], _slc1[1], _slc1[2]] = True
            elif pairn == 2 or (pairn ==3 and diff1 >  diff2):  # type: ignore
                item00.glced = True; item12.glced = True
                marked_id[_slc2[0], _slc2[1], _slc2[2]] = True
        return marked_id
        # marked_im   = np.ma.masked_array(im, marked_id)
