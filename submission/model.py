import math
import torch
from torch import nn
from torch.nn import functional as F
from .physics import UnitaryFresnel, normalize_max

class Block(nn.Module ):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.body =nn.Sequential(nn.Conv2d(incoming, outgoing, 3, padding=1),
                                  nn.GroupNorm(8, outgoing), nn.SiLU(),
                                  nn.Conv2d(outgoing, outgoing, 3, padding=1),
                                  nn.GroupNorm(8, outgoing))
        self.skip = nn.Identity() if incoming == outgoing else nn.Conv2d(incoming, outgoing, 1)
    def forward(self, x):
        return F.silu(self.body(x) + self.skip(x))
class InitNet(nn.Module):
    def __init__(self, adjoint=True, width=32):
        super().__init__()
        self.adjoint= adjoint
        w = [ width * 2**j for j in range(4)]
        self.encoder= nn.ModuleList(Block(a, b) for a,b in zip([12 if adjoint else 4 ]+ w,w))
        self.decoder = nn.ModuleList(Block(w[j +1 ]+w[j],w[ j]) for j in (2, 1,0))
        self.output =nn.Conv2d(w[0],1,1)
        nn.init.normal_(self.output.weight,std=.12)
        nn.init.zeros_(self.output.bias)
    def forward(self, x):
        return math.pi *torch.tanh(self.output(self.spatial_features(x)).float()[:,0])
    def spatial_features( self, x):
        features = []
        y =x
        for j,block in enumerate(self.encoder):
            y= block(y if j ==0 else F.avg_pool2d(y,2 ))
            features.append(y)
        y =self.bottleneck( y)
        for block,skip in zip(self.decoder,reversed(features[:-1]) ):
            y = block(torch.cat((F.interpolate(y ,scale_factor= 2,mode='nearest'),skip),dim =1))
        return y
    def bottleneck(self ,y):
        return y
class UpdateNet(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.body= nn.Sequential(Block(22,width ),Block( width, width),nn.Conv2d(width,1,1))
        nn.init.normal_(self.body[-1 ].weight,std =.01)
        nn.init.zeros_(self.body[ -1].bias)
    def forward( self,features):
        return .35*torch.tanh(self.body(features).float()[:,0 ])
class OpticalUpdates:
    def __init__(self ,op):
        self.unit= UnitaryFresnel(op )
    @torch.no_grad()
    def features(self,t,phi):
        u= torch.exp(1j*phi)
        k=self.unit.forward(u)
        a=normalize_max(k.abs())
        residual=t-a
        back=self.unit.adjoint(residual *torch.exp(1j*torch.angle(k)))* u.conj()[:,None]
        scale=back.abs().square().mean((1 ,2,3), keepdim=True).sqrt().clamp_min( 1e-7)
        back =back/ scale
        return torch.cat((t,a ,residual,u.real[:,None],u.imag[:,None],back.real,back.imag),1)
    def __call__(self,model,t, phi,steps):
        for _ in range(steps):
            x= self.features(t,phi).contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda',dtype=torch.bfloat16 ,enabled=t.is_cuda):
                delta =model(x)
            phi = phi+delta
        return phi
class PhysicsFeatures:
    """фиксир оптич аналитические фильтры"""
    def __init__(self,op):
        self.unit=UnitaryFresnel(op)
        #не возвращается как кандидат
        g = torch.Generator(device='cpu').manual_seed(1821)
        probe =torch.exp((2j*torch.pi)*torch.rand((1, op.n,op.n),generator= g)).to(op.device)
        with torch.no_grad():
            self.reference = torch.exp(1j*torch.angle(self.unit.forward(probe)))
        self.unit.count= 0
    @torch.no_grad()
    def __call__(self,t):
        b = self.unit.adjoint(t*self.reference)
        b =b/ b.abs().square().mean((1, 2,3),keepdim=True).sqrt().clamp_min(1e-7)
        return torch.cat((t, b.real,b.imag),1)
def optical_features(optics,t,phi,aligned =False):
    """признаки под МНК для амплитуд"""
    u=torch.exp(1j*phi)
    k= optics.unit.forward(u)
    raw =k.abs()
    a= normalize_max(raw)
    residual= t -a
    if aligned:
        scale =(raw*t).sum((-2, -1),keepdim=True)/ t.square().sum((-2,-1),keepdim=True).clamp_min(1e-12)
        optical_residual=(scale *t-raw)/raw.square().mean((-2, -1),keepdim =True).clamp_min(1e-12).sqrt()
    else:optical_residual= residual
    carrier=k /raw.clamp_min(1e-7)
    back=optics.unit.adjoint(optical_residual*carrier)*u.conj()[:,None]
    back= back/back.abs().square().mean((1,2,3),keepdim=True).clamp_min(1e-14).sqrt()
    return torch.cat((t,a ,residual,u.real[:,None],u.imag[:,None],back.real,back.imag),1)
def infer(model,t,features=None):
    x= features(t) if model.adjoint else t
    x=x.contiguous(memory_format =torch.channels_last)
    with torch.autocast('cuda',dtype =torch.bfloat16,enabled=t.is_cuda):
        return model(x)

    















     
