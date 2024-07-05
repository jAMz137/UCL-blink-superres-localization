import numpy as np

class blk_traces():
    def __init__(self):
        ...
    def fig_spots(event):
        for item1 in event:
            if item1['brk']==1: continue
            center_z, center_y, center_x =\
                              np.int32(item1['centr'])
            x_range1 = np.int32(center_x -distxy)
            x_range2 = np.int32(center_x +distxy)+1
            y_range1 = np.int32(center_y -distxy)
            y_range2 = np.int32(center_y +distxy)+1
            z_range1 = np.int32(center_z -distz0) 
            z_range2 = np.int32(center_z +distz0)
            itmcid0 = np.where((cntrAll[:,2] >= x_range1) 
                              &(cntrAll[:,2] <  x_range2) 
                              &(cntrAll[:,1] >= y_range1) 
                              &(cntrAll[:,1] <  y_range2) 
                              &(cntrAll[:,0] >= z_range1) 
                              &(cntrAll[:,0] <  z_range2)) 
            itmc = np.int32(cntrAll[itmcid0,0]).T
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
                d4      = eventpnT[id111]['slc'][0][0]
            except ValueError:
                d4 = min(center_z +distz0, shape1[0]) 
                
            idless      = np.where(itmc < center_z)[0]
            try:
                id2     = np.argmax(itmc[idless])
                id22    = idless[id2]
                id222   = itmcid[id22]
                d1      = eventpnT[id222]['slc'][1][0]+1
            except ValueError:
                d1 = max(center_z -distz0, 0)
            slitc   = item1['slc']
            d2      = slitc [0][0]
            d3      = slitc [1][0]+1
            # if d2-d1<=0 or d4-d3<=0:
            if d2-d1<=frame_tr or d4-d3<=frame_tr: 
                continue
            item    = {}
            if item1['glced']: item['glch_s']= 1
            else: item['glch_s']= 0
            item['excitation'] = int(numbers[0])
            item['abort' ] = 0 
            item['rng_t0'] = [d1,d2,d3,d4]
            item['nframe'] = [d2-d1,d4-d3] 
            item['cnt_12'] = item1['centr'][1:]
            Avent1.append(item)
            
    Avent1  = []                  
    
    for item0 in Avent1:
        d1, d2, d3, d4 = item0['rng_t0']
        # 制作mask: 0代表无关区域 1代表拟合区域 2代表边缘区域
        # 初始化maski, 成品mask1
        maski   = np.zeros(shape1[1:]) 
        x_circle, y_circle = gen_circle(item0['cnt_12'], radius, shape1)
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
        item0['Imspt'] = imspt
        # 用于计算区域的亮度timetrace
        immsk   = im[d1:d4+1, min1:max1+1, min2:max2+1].copy()
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
        markn   = marked_id[d1:d4+1 ,min1:max1+1, min2:max2+1]
        for k in range(len_):
            falid_pt =  np.count_nonzero(markn[k])/\
                                         markn[k].size*100
            if falid_pt >= 10: indn[k] = False
            immsk[k, ind02[:,0], ind02[:,1]] = 95 #midval[k]
        immsk[immsk < 95] = 95
        imint1  = np.mean(immsk, axis=(1,2))
        # imint2  = np.mean(imnsk, axis=(1,2))
        dint    = np.abs(imint1[d2 -d1] -imint1[d3 -d1])
        
        item0['Dint']   = dint
        item0['Indn']   = indn
        item0['corner'] = [min1,min2]
        item0['Imint1'] = imint1 
        # item0['Imint2'] = imint2 
        
    Dint    = np.array([item['Dint'] for item in Avent1]) 
    int_dif = np.mean(Dint[np.where(np.abs(Dint -np.mean(Dint)) <2*np.std(Dint))])
    # int_dif = 10
    
    # %%4.. 得到拟合图形
    
    # Ant_id  = []
    def std_ix(xi, intx, dd):
        ix   = np.where(xi)[0]
        ditx = np.diff(intx)
        stdx = np.std(np.abs(ditx))
        try:
            ixa = np.where(np.abs(ditx)>2*stdx)[0]
            ixx = [ix0 for ix0 in ixa if ix[ix0+1]-ix[ix0]>1 and ix[ix0]>=dd+4][0]
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
                
    
    for item1 in Avent1:
        if item1['Dint'] >2*int_dif or item1['Dint'] <0.3*int_dif: 
            item1['abort'] = 1; continue 
        flno  = 0;  flen  = 10;  flag  = False
        d1, d2, d3, d4 = item1['rng_t0']; imintt = item1['Imint1']
        int0    = imintt[d2-d1]; int00   = int0
        int1    = imintt[d3-d1]; int11   = int1
        len_    = len(imintt);  std00   = int_dif/5
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
        x0 = brk_ix(x0[::-1], len(x0)-d2+d1)[::-1]
        x1 = brk_ix(x1, d3-d1)
        if not(any(x0) and any(x1)): 
            flag = True; item1['abort'] = 3; continue
    
        st0 = np.mean(item1['Imspt'][x0], axis=0)
        st1 = np.mean(item1['Imspt'][x1], axis=0)
        wgi = np.count_nonzero(x0) + np.count_nonzero(x1)
        # vlC = np.sum(imintt[x0]-95) +np.sum(imintt[x1]-95)
        vlC = np.abs(np.sum(imintt[x0]-95)*np.count_nonzero(x1)\
                    -np.sum(imintt[x1]-95)*np.count_nonzero(x0))
        if int0>int1: spt = st0 -st1; item1['SpotsD'] = st1
        else:         spt = st1 -st0; item1['SpotsD'] = st0
        
        # Ant_id.append(ii); 
        item1['validC'] = vlC
        item1['rng_x0'] =  x0
        item1['rng_x1'] =  x1
        item1['wght']   = wgi 
        item1['SpotsB'] = spt 
        item1['SpotsI'] = (d4+d1)/2
        
        # P = sci_opt_fit(spt, pixel_size,4.6)
        # if P[1] =='fail': print('ss')
        # popt1   = P[0]
        # min1, min2 = item1['corner']
        # fit_plot(spt, popt1, pixel_size, Targ, 
        #               990, 'No'+str('%03d'%ii))
        # x_com, y_com = Adrift_xy[int(cnt_I//drift_stp)]
        # x = popt1[1] + min2 -3 -x_com
        # y = popt1[2] + min1 -3 -y_com