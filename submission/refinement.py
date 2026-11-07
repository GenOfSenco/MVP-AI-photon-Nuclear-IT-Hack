import torch
from .metrics import differentiable_scores
from .physics import normalize_max

def hybrid_loss(target, amplitude):
    scale = (amplitude *target).sum((-2 , -1), keepdim =True)
    scale= scale/ target.square().sum((-2, -1), keepdim=True).clamp_min(1e-12 )
    residual = (amplitude - scale* target).square().mean(( -2, -1))
    residual =residual /amplitude.square().mean((-2,-1)).clamp_min( 1e-12)
    pearson, ssim =differentiable_scores(target, normalize_max( amplitude))
    loss = .4 * (1- pearson) + .4 *(1- ssim) + .2 *residual
    return (loss.mean(-1 ) + .5* loss.max(-1).values).mean()
class ClosureLimit(Exception):
    pass
def refine(op,target, phase , initial, budget , export,check, score ):
    """декод/сейв"""
    adam_steps, closure_budget = budget
    p= torch.nn.Parameter(phase.detach().clone())
    winner, best_key = initial, score(check(initial))
    adam_calls= lbfgs_calls = 0
    png_checks =1
    def consider():
        nonlocal winner,best_key, png_checks
        candidate= export(p)
        key =score(check(candidate))
        png_checks += 1
        if key >best_key:
            winner,best_key =candidate, key
    def gradient(optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss= hybrid_loss(target, op.reconstruct(p)[ 1])
        loss.backward()
        return loss
    adam =torch.optim.Adam([p], lr=.08)
    for step in range(1, adam_steps + 1):
        gradient(adam)
        adam.step()
        adam_calls +=1
        if step %20 == 0 or step== adam_steps:
            consider()
    adam_winner= winner
    if closure_budget:
        optimizer =torch.optim.LBFGS([p], max_iter=1, max_eval= 16, history_size=10,
                                      line_search_fn ='strong_wolfe',
                                      tolerance_grad=1e-8 , tolerance_change=1e-10)
        def closure():
            nonlocal lbfgs_calls
            if lbfgs_calls == closure_budget:
                raise ClosureLimit
            loss = gradient(optimizer)
            lbfgs_calls+= 1
            if lbfgs_calls %10 ==0:
                consider()
            return loss
        try:
            while lbfgs_calls< closure_budget:
                optimizer.step(closure)
                consider()
        except ClosureLimit:
            pass
    return winner, dict(adam_calls =adam_calls,lbfgs_closure_calls= lbfgs_calls,
                        backward_calls=adam_calls +lbfgs_calls , png_checks=png_checks),adam_winner












    
