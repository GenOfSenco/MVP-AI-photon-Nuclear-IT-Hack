import math
import numpy as np
import torch

N = 128
WAVELENGTH= 532e-9
PITCH = 10e-6
DEPTHS = (24.1e-3, 25.3e-3 ,26.5e-3, 27.7e-3) # из тз

def normalize_max(a):
    if isinstance(a, torch.Tensor):
        return a/ a.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return a / np.maximum(np.max(a, axis=(-2, -1), keepdims=True),  1e-12)
def numpy_field(u,z , pitch= PITCH, wavelength =WAVELENGTH):
    n=u.shape[-1 ]
    c= (np.arange(n)-n/2)*pitch
    v, w=np.meshgrid(c,c,indexing='ij')
    f =np.fft.fftshift(np.fft.fftfreq(n,d=pitch))
    x,y =np.meshgrid(wavelength* z* f,wavelength*z *f,indexing= 'ij')
    pre=np.exp(1j*np.pi*(v *v+ w *w)/(wavelength *z))
    post =np.exp(1j*np.pi* (x *x+y* y)/(wavelength* z))
    factor= np.exp(1j*(2*np.pi*z/ wavelength))/(1j*wavelength* z)*pitch**2
    return factor *post*np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u* pre,axes=( -2 ,-1))),axes=(-2,-1))
def fresnel_one_fft_numpy(phi,z,config=None):
    return numpy_field(np.exp(1j*phi),z )
class Fresnel:
    """комплексная оптика на фурье по тз"""
    def __init__(self,device='cpu',dtype=torch.float32,n=N):
        self.n= n;self.device=device;self.dtype=dtype
        #фазы float64
        c=(torch.arange(n,dtype= torch.float64,device=device)-n/2) *PITCH
        r2=c[:,None]**2+c[None,:]**2
        f=torch.fft.fftshift(torch.fft.fftfreq(n, d= PITCH,device=device,dtype =torch.float64))
        pres=[]; posts=[]; factors =[]
        cdtype= torch.complex128 if dtype ==torch.float64 else torch.complex64
        for z in DEPTHS:
            xy=WAVELENGTH* z *f
            pres.append(torch.exp(1j*math.pi*r2/(WAVELENGTH*z)))
            posts.append(torch.exp(1j*math.pi*(xy[:,None]**2+xy[None,:]**2)/(WAVELENGTH*z)))
            factors.append(np.exp(1j*(2*np.pi*z/WAVELENGTH))/(1j*WAVELENGTH*z)*PITCH**2)
        self.pre= torch.stack(pres).to(cdtype)
        self.post=torch.stack(posts).to(cdtype)
        self.factor =torch.tensor(factors, device =device,dtype =cdtype)[:,None,None]
        self.count=0
    def field(self,u):
        if u.ndim==2:u= u[None]
        self.count+=u.shape[0]* 4
        f=torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(u[:,None]*self.pre,dim=(-2,-1))),dim=(-2,-1))
        return self.factor *self.post* f
    def inverse(self,k):
        self.count+= int(np.prod(k.shape[:-2] ))
        f=k/(self.factor* self.post)
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(f,dim= (-2,-1))),dim=(-2,-1))* self.pre.conj()
    def reconstruct(self ,phi):
        k=self.field(torch.exp(1j*phi));a =k.abs()
        return k,a ,normalize_max( a)
class UnitaryFresnel:
    def __init__(self,official):self.pre=official.pre;self.device =official.device;self.count=0
    def forward(self,u):
        if u.ndim== 2:u =u[None]
        self.count+=u.shape[0 ]*4
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(u[:,None]*self.pre,dim=(-2,-1)),norm='ortho'),dim=( -2,-1))
    def adjoint( self, v ):
        self.count+=int(v.numel()/v.shape[-1]**2)
        return self.pre.conj()*torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(v,dim=( -2,-1)),norm='ortho'),dim=(-2 ,-1) )

















    