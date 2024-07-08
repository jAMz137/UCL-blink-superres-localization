import numpy as np
from TRIM_events import blk_event, blk_events
from utils_toolbox import consec_T, consec_T3, gen_circle

class blk_trace():
    '''
    abort:  0, 通过; 
            1, Dint;
            2, 3, 可用帧为0; 
            4, 拟合失败; 
            5, 拟合参数不通过.

    glitch_or_not:      是否来自毛刺;   
    excitation:    
    Imspt:              数据备选的区间;      
    Imint:              对应区间TR
    rng_t0:             tz时间节点
    rng_ax:
    cnt_12:             xy中心坐标;         
    corner:             区域位置标记
    Dint:               中心事件闪烁幅度;     
    Indn:               标记的无效帧
    weight:
    ValidC:
    SpotB:
    '''
    __slot__ = (
        'glitch_or_not', 'excitation', 'Dint', 'Indn', 'SpotB',
         'Imspt',  'Imint', 
        'rng_t0', 'rng_ax', 
        'cnt_12', 'corner', 
        'weight', 'ValidC', 
        )
    def __init__( self, excitation_, glitchs_, centr_, TR_mrk_):
        self. glitch_or_not = glitchs_
        self. excitation = excitation_
        self. abort  = 0
        self. rng_t0 = TR_mrk_
        self. cnt_12 = centr_
        self. Imint:  np.ndarray
        self. rng_ax: np.ndarray
        

    @property
    def SpotI(self):
        return (self.rng_t0[3]+ self.rng_t0[0])/2

    @property
    def Dint(self):
        return np.abs(self.Imint[self.rng_t0[1]-self.rng_t0[0]] 
                     -self.Imint[self.rng_t0[2]-self.rng_t0[0]])

    @property
    def weight(self):
        return np.count_nonzero(self.rng_ax[0]) \
                            + np.count_nonzero(self.rng_ax[1])
    

class blk_traces():
    def __init__(self, Events_: blk_events, im_: np.ndarray, parameters_: dict):
        self. glitch_id = Events_.glitch_id
        self. parameters = parameters_
        self. im = im_
        self. shape1 = np.shape(im_)
        EventpT = [item for item in Events_[0] if not item.glced]
        EventnT = [item for item in Events_[1] if not item.glced]
        centrpT = np.array([item.centr for item in EventpT])
        centrnT = np.array([item.centr for item in EventnT])

        self. centrAll  = np.vstack((centrpT, centrnT))
        self. EventpnT  = EventpT + EventnT
        self. Traces    = self. fig_spots(Events_[0])+ self. fig_spots(EventnT)

        self. TR_area()

        Dltaint      = np.array([ item.Dint for item in self.Traces]) 
        self. int_dif   = np.mean (Dltaint[np.where(
                                    np.abs(Dltaint-np.mean(Dltaint)) 
                                                 <2*np.std(Dltaint)
                                                    )])
        # int_dif = 10


    def fig_spots(self, Events_: list[blk_event]):
        distxy = self.parameters['distxy']
        distz0 = self.parameters['distz0']
        frame_tr = self.parameters['frame_min']
        Traces_ = []
        for item1 in Events_:
            if item1.brk_mrk ==1: continue
            center_z, center_y, center_x =\
                              np.array(np.int32(item1.centr))
            x_range1 = np.int32(center_x -distxy)
            x_range2 = np.int32(center_x +distxy)+1
            y_range1 = np.int32(center_y -distxy)
            y_range2 = np.int32(center_y +distxy)+1
            z_range1 = np.int32(center_z -distz0) 
            z_range2 = np.int32(center_z +distz0)
            itmcid0 = np.where((self.centrAll[:,2] >= x_range1) 
                              &(self.centrAll[:,2] <  x_range2) 
                              &(self.centrAll[:,1] >= y_range1) 
                              &(self.centrAll[:,1] <  y_range2) 
                              &(self.centrAll[:,0] >= z_range1) 
                              &(self.centrAll[:,0] <  z_range2)) 
            itmc = np.array(np.int32(self.centrAll[itmcid0,0])).T
            itmcid = itmcid0[0]
            '''
                        d3_____d4
                       /
                      /
              _______/
            d1       d2
            '''
            idgrtr      = np.where(itmc > center_z)[0]
            try:
                id1     = np.argmin(itmc[idgrtr])
                id11    = idgrtr[id1]
                id111   = itmcid[id11]
                d4      = self.EventpnT[id111].min_coord[0]
            except ValueError:
                d4 = min(center_z +distz0, self.shape1[0]-1) 
                
            idless      = np.where(itmc < center_z)[0]
            try:
                id2     = np.argmax(itmc[idless])
                id22    = idless[id2]
                id222   = itmcid[id22]
                d1      = self.EventpnT[id222].max_coord[0]+1
            except ValueError:
                d1 = max(center_z -distz0, 0)
            d2      = item1.min_coord[0]
            d3      = item1.max_coord[0]+1
            # if d2-d1<=0 or d4-d3<=0:
            if d2-d1<=frame_tr or d4-d3<=frame_tr: 
                continue
            item    = blk_trace(self.parameters['excitation'],
                                item1.glced, item1.centr[1: ], 
                                [d1,d2,d3,d4])
            Traces_.append(item) 
        return Traces_                
    

    def TR_area(self):
        radius = self.parameters['radius']
        for item0 in self.Traces:
            d1, d2, d3, d4 = item0.rng_t0
            # 制作mask: 0代表无关区域 1代表拟合区域 2代表边缘区域
            # 初始化maski, 成品mask1
            maski   = np.zeros(self.shape1[1:]) 
            x_circle, y_circle = gen_circle(item0.cnt_12, radius, 
                                            self.shape1[ 1:])
            maski[y_circle, x_circle] = 1
            ind_1   = np.where(maski==1) 
            Visited = np.zeros(self.shape1[1: ], dtype=bool)
            for j in range(len(ind_1[0])):
                for x_offset in [-1, 0, 1]: 
                    for y_offset in [-1, 0, 1]:
                        yind = ind_1[0][j] + y_offset
                        xind = ind_1[1][j] + x_offset
                        if (xind >= 0 and xind < self.shape1[2] and  
                            yind >= 0 and yind < self.shape1[1] and 
                            maski[yind,xind] == 0):
                            if  not Visited[yind,xind]: 
                                maski  [yind, xind] =2 
                                Visited[yind, xind] =True
            # 根据mask的相关区域重新截取局域区间
            min1,min2   = np.array(ind_1).min(axis=1) 
            max1,max2   = np.array(ind_1).max(axis=1)
            mask1   = maski[min1:max1+1, min2:max2+1]
            # 用于最终的待拟合图片生成
            imspt   = self.im[d1:d4+1, min1:max1+1, min2:max2+1].copy()
            item0. Imspt = imspt
            # 用于计算区域的亮度timetrace
            immsk   = self.im[d1:d4+1, min1:max1+1, min2:max2+1].copy()
            # imnsk   = im[d1:d4+1, int(item0['cnt_12'][0]-radius2):
            #                       int(item0['cnt_12'][0]+radius2), 
            #                       int(item0['cnt_12'][1]-radius2):
            #                       int(item0['cnt_12'][1]+radius2)].copy()
            
            len_    = len(imspt)
            # 用于标记该帧是否采用
            indn    = np.ones(len_, dtype=bool)
            ind02   = np.array(np.where((mask1 ==2)|(mask1==0))).T
            ind_2   = np.array(np.where( maski ==2)).T
            # midval  = np.median(im[d1:d4+1, ind_2[:,0], 
            #                                 ind_2[:,1]], axis=1)
            markn   = self.glitch_id[d1:d4+1 ,min1:max1+1, min2:max2+1]
            for k in range(len_):
                falid_pt =  np.count_nonzero(markn[k])\
                                            /markn[k].size*100
                if falid_pt >= 10: indn[k] = False
                immsk[k, ind02[:,0], ind02[:,1]] = 95 #midval[k]
            immsk[immsk < 95] = 95
            imint1  = np.mean(immsk, axis=(1,2))
            # imint2  = np.mean(imnsk, axis=(1,2))

            item0. Imint  = imint1
            item0. Indn   = indn
            item0. corner = [min1,min2]
            # item0. Imint2 = imint2 

    
    # %%4.. 得到拟合图形

    def std_ix(self, xi, intx, dd):
        ix   = np.where(xi)[0]
        ditx = np.diff(intx)
        stdx = np.std(np.abs(ditx))
        try:
            ixa = np.where(np.abs(ditx)>2*stdx)[0]
            ixx = [ix0 for ix0 in ixa if ix[ix0+1]-ix[ix0]>1 and ix[ix0]>=dd+4][0]
        except IndexError:
            return xi
        return [False if i >= ix[ixx]+1 else elem for i, elem in enumerate(xi)]
    
    def brk_ix(self, xi, dd):
        flag = 0
        for i, elem in enumerate(xi[dd:]):
            if flag:  xi[i+dd] = False
            elif i > 15 and xi[i+dd-1] == False:
                xi[i+dd] = False; flag = 1
        return xi
                
    def find_spot(self):
        for item1 in self.Traces:
            if item1.Dint >2*self.int_dif or item1.Dint < 0.3 *self.int_dif: 
                item1.abort = 1; continue 
            flno  = 0;  flen  = 10;  flag  = False
            d1, d2, d3, _   = item1.rng_t0
            imintt  = item1.Imint
            int0    = imintt[d2-d1]; int00   = int0
            int1    = imintt[d3-d1]; int11   = int1
            len_    = len ( imintt); std00   = self.int_dif/5
            # std00   = max(Dint[ii]/5, 1)
            # std01   = max(Dint[ii]/5, 1)
            
            # 确定掩膜下开始和结束态的亮度, 确定亮度范围
            while flno <= flen: 
                # if flno == flen:
                #     if np.abs(int1-int11)>=std00: int11 = int1
                #     if np.abs(int0-int00)>=std00: int00 = int0
                flno += 1
                x0 = (imintt>=int00-std00) &(imintt<=int00+std00)\
                    &(range(len_)<=d2-d1) # &Indn[ii] 
                x1 = (imintt>=int11-std00) &(imintt<=int11+std00)\
                    &(range(len_)>=d3-d1) # &Indn[ii] 
                # valid_fr0 =np.count_nonzero(x0)
                # valid_fr1 =np.count_nonzero(x1)
                # if  valid_fr0 <frame_tr or valid_fr1 <frame_tr: 
                if not(any(x0) and any(x1)):
                    flag = True; item1['abort'] = 2; break;
                # int00 = np.mean(imintt[x0])
                # int11 = np.mean(imintt[x1])
                int00 = (np.max(imintt[x0]) +np.min(imintt[x0]))/2
                int11 = (np.max(imintt[x1]) +np.min(imintt[x1]))/2
            if flag: continue
        
            std000 = np.std (imintt[x0])
            std001 = np.std (imintt[x1])
            int000 = np.mean(imintt[x0])
            int111 = np.mean(imintt[x1])
            if np.abs(int0-int00)>=std00\
            or (std000<=std00/2 and np.abs(int000-int0)>=std00): 
                int00 = int0
            if np.abs(int1-int11)>=std00\
            or (std001<=std00/2 and np.abs(int111-int1)>=std00): 
                int11 = int1
            x0  = (imintt >= int00 -std00) & (imintt <= int00 +std00)
            x1  = (imintt >= int11 -std00) & (imintt <= int11 +std00)
            x00 = (imintt >= int00 -std00*2)\
                & (imintt <= int00 +std00*2)
            x11 = (imintt >= int11 -std00*2)\
                & (imintt <= int11 +std00*2)
            x0 = consec_T(x0,3); x0 = consec_T3(x0,x00,3)
            x1 = consec_T(x1,3); x1 = consec_T3(x1,x11,3)
            x0 = self.brk_ix(x0[::-1], len(x0)-d2+d1)[::-1]
            x1 = self.brk_ix(x1, d3-d1)
            if not(any(x0) and any(x1)): 
                flag = True; item1.abort = 3; continue
        
            st0 = np.mean(item1.Imspt[x0], axis=0)
            st1 = np.mean(item1.Imspt[x1], axis=0)
            # vlC = np.sum(imintt[x0]-95) +np.sum(imintt[x1]-95)
            vlC = np.abs(np.sum(imintt[x0]-95)*np.count_nonzero(x1)\
                        -np.sum(imintt[x1]-95)*np.count_nonzero(x0))
            if int0>int1: spt = st0 -st1; item1.SpotsD = st1
            else:         spt = st1 -st0; item1.SpotsD = st0
            
            # Ant_id.append(ii); 
            
            item1. rng_ax    = np.array([x0, x1])
            item1. validC    = vlC
            item1. SpotsB    = spt 
            
            
            # P = sci_opt_fit(spt, pixel_size,4.6)
            # if P[1] =='fail': print('ss')
            # popt1   = P[0]
            # min1, min2 = item1['corner']
            # fit_plot(spt, popt1, pixel_size, Targ, 
            #               990, 'No'+str('%03d'%ii))
            # x_com, y_com = Adrift_xy[int(cnt_I//drift_stp)]
            # x = popt1[1] + min2 -3 -x_com
            # y = popt1[2] + min1 -3 -y_com