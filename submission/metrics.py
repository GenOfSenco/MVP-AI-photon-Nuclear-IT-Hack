import numpy as np
import torch

def differentiable_scores(t,a):
    tc =t -t.mean((-2,-1),keepdim=True);    ac=a- a.mean((-2,-1 ),keepdim= True)
    vt= (tc *tc).mean((-2 ,-1));    va=(ac *ac).mean( ( -2,-1));cov=(tc *ac).mean((-2,-1))
    corr=cov /(vt * va).clamp_min(1e-16  ).sqrt()
    mt=t.mean((-2,  -1));ma=a.mean((-2 ,-1))
    ssim= ((2*mt*ma+.0001) *(2 *cov+.0009 ))/((mt.square() +ma.square()+.0001)*(vt +va+.0009))
    return corr,ssim
def evaluate(target, amplitude):
    r, s= differentiable_scores(torch.tensor(target, dtype=torch.float64),
                                 torch.tensor(amplitude, dtype=torch.float64))
    return dict(pearson=r.tolist(), ssim_global=s.tolist(),
                pearson_mean=float(r.mean()), ssim_global_mean= float(s.mean()),
                pearson_min= float(r.min()), ssim_global_min =float(s.min()))





















    