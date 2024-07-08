import numpy as np; import tifffile as tifile
import scipy.signal as signal

from utils_toolbox import sci_opt_fit


class drift_correction():
    def __init__(self, rootpath: str, filemark: str, deg: float, 
                 dft_step: int, mark_num: int, pixel_size: float, 
                 mark_name:list[str]):
        self. Filemark = filemark
        self. exct_deg = deg
        self. pixel_size = pixel_size
        self. dft_step = dft_step    
        self. mark_num = mark_num    
        
        self. Pmrk: list = [None] *self.mark_num
        for j in range(0, self.mark_num):
            self. Pmrk[0] = rootpath +'\\' +mark_name[j]
        
    def correlated(self)-> np.ndarray: 
        '''漂移校准(correlated)'''
        Drift_stm: list = [None] *self.mark_num
        for j in range(0, self.mark_num):    
            imlabel = np.array(np.float32(tifile.imread(self.Pmrk[j] 
                                          +'\\'+self.Filemark)))[:,0:28,0:28]
            shapel  = np.shape(imlabel)
            Drift_stm[j] = []
            dir_drift_i = self.Pmrk[j]+'\\drift' \
                                    +str('%03d'% int(self.exct_deg)  ) +'.tif'
            dir_drift_1 = self.Pmrk[j]+'\\drift' \
                                    +str('%03d'%(int(self.exct_deg)-3))+'.tif'
            dir_drift50 = self.Pmrk[j]+'\\drifA' \
                                    +str('%03d'% int(50)) + '.tif'
            dft_recordi = self.Pmrk[j]+'\\dfting'\
                                    +str('%03d'% int(self.exct_deg)  ) +'.txt'
            dft_record1 = self.Pmrk[j]+'\\dfting'\
                                    +str('%03d'%(int(self.exct_deg)-3))+'.txt'
            
            if int(self.exct_deg) == 50: 
                label0  = np.mean(imlabel[:self.dft_step],axis=0)
                label0 -= np.min(label0)
                label30 = label0/np.max(label0)        
                # tifile.imwrite(dir_drift50, label30)
            else:
                label30 = np.array(tifile.imread(dir_drift_1))
            
            labelst = []; imlabel = imlabel[:-1] 
            
            for i in range(0, np.shape(imlabel)[0]//self.dft_step): 
                
                label   = np.mean(imlabel[i*self.dft_step:
                                      (i+1)*self.dft_step],axis=0)
                label  -= np.min(label)
                label3  = label/np.max(label)        
                label   = signal.convolve(label3, label30[::-1,::-1],mode='same')                                                        
                label   = label[int(shapel[1]/4*1):
                                int(shapel[1]/4*3),
                                int(shapel[2]/4*1):
                                int(shapel[2]/4*3)]
                labelst.append(label)
                l11     = np.shape(label)[0]
                l22     = np.shape(label)[1]
                fitout  = sci_opt_fit(label, 
                                      self.pixel_size, 7.5)
                if fitout[1] == 'fail': continue    
                l1_com =fitout[0][1] -l11/2*self.pixel_size
                l2_com =fitout[0][2] -l22/2*self.pixel_size
                Drift_stm[j]. append([l1_com, l2_com])
        
            Drift_stm[j].insert(-1,[l1_com,l2_com])
            Drift_stm[j]  =  np.array(Drift_stm[j])
            if int(self.exct_deg) == 50:
                np.savetxt(dft_recordi, Drift_stm[j])
            else: 
                dft_r = np.loadtxt(dft_record1)
                Drift_stm[j] += dft_r[-1]
                np.savetxt(dft_recordi, Drift_stm[j])
                
            # tifile.imwrite(dir_drift_i, label3)
        Adrift_xy   = np.array(np.mean(np.array(
                                        Drift_stm), axis=0))
        return Adrift_xy
    
    
    
    def multifit(self) -> np.ndarray:
        ''' ex:漂移校准(multifit)'''
        Drift_stm: list   = [None] *self.mark_num
        
        for j in range(0, self.mark_num):    
            Drift_stm[j] = []
            imlabel = np.array(np.float32(tifile.imread(self.Pmrk[j]
                                                       +'\\'+self.Filemark)))
            int_lbl = np.sum(imlabel-86,axis=(1,2))
            int_trd = np.max(int_lbl) -np.sqrt(np.max(int_lbl))/9*75
            
            imlabel-= np.min(imlabel)
            imlabel = imlabel /np.max(imlabel)
            dir_drift50 = self.Pmrk[j] +'\\drifA'+str('%03d'% int(50))+'.tif'
            dft_50cord  = self.Pmrk[j] +'\\drift050.txt'
            
            if int(self.exct_deg) == 50: 
                flg0 = 1
                int_lbl30 = int_lbl[:self.dft_step]
                imlabel30 = imlabel[:self.dft_step]\
                                    [np.where(int_lbl30 > int_trd)]
                if len(imlabel30)<5: 
                    print('frist50 blinking!')
                    raise 
                label30 = np.mean(imlabel30, axis=0)
                tifile.imwrite(dir_drift50, label30)
                fitout  = sci_opt_fit(label30, self.pixel_size, 4.6)
                if fitout[1] == 'fail': 
                    print('frist50 errorfail') 
                    raise
                l1_0    = fitout[0][1]
                l2_0    = fitout[0][2]
                np.savetxt(dft_50cord, np.array([l1_0, l2_0])) 
            else:
                l1_0, l2_0 = np.loadtxt(dft_50cord)    
            imlabel = imlabel[ :-1]
            for ii in range(0, np.shape(imlabel)[0]//self.dft_step): 
                flag = 1
                int_lbl3    = int_lbl[ii*self.dft_step:
                                      (ii+1)*self.dft_step]
                imlabel3    = imlabel[ii*self.dft_step:
                                      (ii+1)*self.dft_step]\
                                        [np.where(int_lbl3>int_trd)]
                if len(imlabel3)<5: flag = 0  
                else: 
                    label3  = np.mean(imlabel3, axis=0)
                    fitout  = sci_opt_fit(label3, self.pixel_size, 4.6)
                    if (fitout[1]=='fail'): flag=0
                    else:
                        popt1   = fitout[0]
                        ellip   = abs(popt1[3]/popt1[4])
                        if ellip>1.3 or ellip<0.7: flag = 0
                if not flag:
                    Drift_stm[j].append([50, 50])
                else:
                    l1_com0 = popt1[1] - l1_0
                    l2_com0 = popt1[2] - l2_0
                    Drift_stm[j].append([l1_com0, l2_com0])
            
            Drift_stm[j].insert(-1, Drift_stm[j][-1])   
        drift_id    = np.full_like(Drift_stm, 
                              fill_value=False, dtype=bool)
        drift_id[np.where(np.array(Drift_stm)==50)] =True
        marked_stmp = np.ma.masked_array(np.array(
                                       Drift_stm),drift_id)
        Adrift_xy   = np.array(np.mean(np.array(
                                       Drift_stm), axis=0))
        return Adrift_xy
        
        # Adrift_xy[:,0] = signal.savgol_filter(Adrift_xy[:,0], 
        #                                       99, 1, mode= 'nearest')
        # Adrift_xy[:,1] = signal.savgol_filter(Adrift_xy[:,1], 
        #                                       99, 1, mode= 'nearest')
