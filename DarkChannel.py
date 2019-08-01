import cv2  
import numpy as np  
   
def zmMinFilterGray(src, r=7):  
    return cv2.erode(src,np.ones((2*r-1,2*r-1)))
   
def guidedfilter(I, p, r, eps):  
    height, width = I.shape  
    m_I = cv2.boxFilter(I, -1, (r,r))  
    m_p = cv2.boxFilter(p, -1, (r,r))  
    m_Ip = cv2.boxFilter(I*p, -1, (r,r))  
    cov_Ip = m_Ip-m_I*m_p  
   
    m_II = cv2.boxFilter(I*I, -1, (r,r))  
    var_I = m_II-m_I*m_I  
   
    a = cov_Ip/(var_I+eps)  
    b = m_p-a*m_I  
   
    m_a = cv2.boxFilter(a, -1, (r,r))  
    m_b = cv2.boxFilter(b, -1, (r,r))  
    return m_a*I+m_b  
   
def getV1(m, r, eps, w, maxV1):     
    V1 = np.min(m,2)                                         
    V1 = guidedfilter(V1, zmMinFilterGray(V1,7), r, eps)     
    bins = 2000  
    ht = np.histogram(V1, bins)                               
    d = np.cumsum(ht[0])/float(V1.size)  
    for lmax in range(bins-1, 0, -1):  
        if d[lmax]<=0.999:  
            break  
    A  = np.mean(m,2)[V1>=ht[1][lmax]].max()  
           
    V1 = np.minimum(V1*w, maxV1)                   
       
    return V1,A  
   
def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=True):  
    Y = np.zeros(m.shape)  
    V1,A = getV1(m, r, eps, w, maxV1)               
    for k in range(3):  
        Y[:,:,k] = (m[:,:,k]-V1)/(1-V1/A)           
    Y =  np.clip(Y, 0, 1)  
    if bGamma:  
        Y = Y**(np.log(0.5)/np.log(Y.mean()))         
    return Y  
   
if __name__ == '__main__':

    input_folder = 'E:\\Users\\haze\\'
    output_folder = 'E:\\Users\\test_DarkChannel\\'
    for i in range(1747,1847):
        for j in range(0,10):
            m = deHaze(cv2.imread(input_folder + str(i).zfill(6) + '_' + str(j) + '_' + '.png')/255.0)*255  
            cv2.imwrite(output_folder + str(i).zfill(6) + '_' + str(j) + '_' + '.png', m)
    print('finish '+ str(i))
