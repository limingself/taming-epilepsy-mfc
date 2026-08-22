"""Portable extraction of the frozen HUP060 teacher objective.

Only :func:`objective` is imported by the HUP065/HUP080 training adapters.
The original gated stage launcher remains hash-recorded in ``PORTABILITY_MAP.json``
but is not published because it contains local authorization material and
machine-specific paths.  The numerical function below is unchanged.
"""

from __future__ import annotations


def objective(sequence, controls, commands, reference, channel_weights, reference_scale, projections, actor, baseline, weights, *, parameter_regularization: bool, smooth=0.2, curvature=0.1, slew_barrier=0.0, curvature_barrier=0.0, max_first=0.35, max_second=0.5):
    import torch
    _, horizon, channels = sequence.shape
    reference_mean = reference.mean(dim=0); pooled = reference.reshape(-1,channels).var(dim=0,unbiased=False).clamp_min(1e-5)
    reference_variance = (0.25*reference.var(dim=0,unbiased=False)+0.75*pooled[None]).clamp_min(1e-5)
    tw=torch.linspace(0.25,1.0,horizon,dtype=sequence.dtype);tw=tw/tw.mean()
    mean_err=((sequence.mean(0)-reference_mean)/reference_scale[None]).square()*channel_weights[None]*tw[:,None]
    pv=sequence.var(0,unbiased=False).clamp_min(1e-5); lv=(torch.log(pv)-torch.log(reference_variance)).square()*channel_weights[None]*tw[:,None]
    probs=torch.linspace(.1,.9,9,dtype=sequence.dtype);ti=torch.arange(0,horizon,4)
    rq=torch.quantile(reference[:,ti],probs,dim=0);pq=torch.quantile(sequence[:,ti],probs,dim=0)
    qe=((pq-rq)/reference_scale[None,None]).square()*channel_weights[None,None]*tw[ti][None,:,None]
    zero=mean_err.new_zeros(()); nhm,nhq=zero,zero
    if baseline is not None:
        bm=((baseline.mean(0)-reference_mean)/reference_scale[None]).square()*channel_weights[None]*tw[:,None]
        bq=torch.quantile(baseline[:,ti],probs,dim=0); bqe=((bq-rq)/reference_scale[None,None]).square()*channel_weights[None,None]*tw[ti][None,:,None]
        mh=torch.relu(mean_err.mean(0)-bm.mean(0)); qh=torch.relu(qe.mean((0,1))-bqe.mean((0,1)))
        nhm=mh.mean()+torch.topk(mh,min(4,channels)).values.mean();nhq=qh.mean()+torch.topk(qh,min(4,channels)).values.mean()
    op=torch.linspace(.05,.95,19,dtype=sequence.dtype)
    occ=((torch.quantile(sequence.reshape(-1,channels),op,dim=0)-torch.quantile(reference.reshape(-1,channels),op,dim=0))/reference_scale[None]).square()*channel_weights[None]
    lo=torch.quantile(reference,.1,dim=0);hi=torch.quantile(reference,.9,dim=0)
    tube=((torch.relu(lo[None]-sequence)+torch.relu(sequence-hi[None]))/reference_scale[None,None]).square()*channel_weights[None,None]*tw[None,:,None]
    jt=[]; jp=torch.linspace(.1,.9,9,dtype=sequence.dtype); cs=torch.sqrt(channel_weights)[None]
    for step in torch.arange(0,horizon,16):
        a=((sequence[:,step]-reference_mean[step][None])/reference_scale[None])*cs;b=((reference[:,step]-reference_mean[step][None])/reference_scale[None])*cs
        jt.append((torch.quantile(a@projections,jp,dim=0)-torch.quantile(b@projections,jp,dim=0)).square().mean())
    normalized=controls/1.8;first=normalized[:,1:]-normalized[:,:-1];second=normalized[:,2:]-2*normalized[:,1:-1]+normalized[:,:-2]
    terms={"mean":mean_err.mean(),"log_variance":lv.mean(),"worst_log_variance":torch.topk(lv.mean(0),min(4,channels)).values.mean(),"time_quantile":qe.mean(),"worst_time_quantile":torch.topk(qe.mean((0,1)),min(4,channels)).values.mean(),"no_harm_mean":nhm,"no_harm_quantile":nhq,"occupation":occ.mean(),"tube":tube.mean(),"joint":torch.stack(jt).mean(),"energy":normalized.square().mean(),"smoothness":first.square().mean(),"curvature":second.square().mean(),"slew_excess":torch.relu(first.abs()-max_first/1.8).square().mean(),"curvature_excess":torch.relu(second.abs()-max_second/1.8).square().mean(),"saturation":torch.relu(commands.abs()/1.8-.95).square().mean()}
    total=sum(float(weights.get(k,0.0))*terms[k] for k in ("mean","log_variance","worst_log_variance","time_quantile","worst_time_quantile","no_harm_mean","no_harm_quantile","occupation","tube","joint","energy","saturation"))+smooth*terms["smoothness"]+curvature*terms["curvature"]+slew_barrier*terms["slew_excess"]+curvature_barrier*terms["curvature_excess"]
    regularization=actor.mean_gain_delta.square().mean()+actor.deviation_gain_delta.square().mean()
    if hasattr(actor,"local_deviation_gain_logits"): regularization=regularization+0.1*actor.local_deviation_gain_logits.square().mean()
    for name in ("common_markov_residual","deviation_markov_residual"):
        module=getattr(actor,name,None)
        if module is not None: regularization=regularization+0.01*sum(p.square().mean() for p in module.parameters())
    terms["gain_regularization"]=regularization
    if parameter_regularization: total=total+0.001*regularization
    terms["loss"]=total
    return total,terms
