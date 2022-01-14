import numpy as np
import PIL.Image as Image
import cv2
import os

def generate(ks):
    dataPath = './data/'
    dst = './scatter_k{}/'.format(round(512/ks))
    if not os.path.exists(dst):
        os.makedirs(dst)
    
    for each in os.listdir(dataPath):#########################################################################################################
        if 'angle' in each:
            continue
        file_name_1 = dataPath + each
        # generate scatter
        w = 512
        dot = np.random.rand(round(w/ks), round(w/ks)) * 2 * np.pi
        dot = np.cos(dot) + 1j * np.sin(dot)
        h = dot.shape[0]
        scatter = np.zeros((512, 512)) + 1j * np.zeros((512, 512))
        scatter[:h, :h] = dot

        fstart = abs(np.fft.fft2(scatter))**2 
        ma = fstart.max()
        mi = fstart.min()
        Istart = (fstart-mi)/(ma-mi)+ 0.01

        temp_1 = Image.open(file_name_1)
        temp_1 = temp_1.convert('L') 
        org_amp = np.array(temp_1).astype(np.float32)
        amp = np.zeros((512, 512), dtype=np.float32)
        amp[110:-110, 110:-110] = org_amp[110:-110, 110:-110]
        
        dif = Istart * amp
        dif = np.clip(dif, 0, 255)
        
        speckle_name = each.replace('amp', 'speckle')
        cv2.imwrite(os.path.join(dst, speckle_name), dif)#########################################################################################################

if __name__ == '__main__':
    k = [40, 46, 60, 73, 100, 180]
    for i, ki in enumerate(k):
       generate(ki)
       print('_____________________________scater size: '+str(round(512/ki))+' finished__________________________________')