import numpy as np

def encode(phi ):
    p= np.asarray(phi,dtype =np.float64 )
    p=np.where((p< -np.pi)|(p> np.pi),(p +np.pi)% (2 *np.pi)- np.pi,p)
    return np.clip(np.rint(255 *(p+ np.pi)/(2*np.pi) ),0 ,255).astype(np.uint8)
def decode( q):return -np.pi+ 2*np.pi*np.asarray(q,dtype=np.float64)/ 255


























