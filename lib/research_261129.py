"""Independent candidate implementations; not official reproductions.
See RESEARCH_261129.md for exact sources, design changes and validation limits.
"""
import math
import os
import json
import torch
from torch import nn
from torch.nn import functional as F
from lib.research_261129_config import PAIR

def resize(x,size,mode="bilinear"):
    if tuple(x.shape[-2:])==tuple(size):
        return x
    if mode=="nearest":
        return F.interpolate(x,size=size,mode=mode)
    return F.interpolate(x,size=size,mode=mode,align_corners=False)

def prob(z):
    z=z.float()
    if z.shape[1]==1:
        p=z.sigmoid()
        return torch.cat((1-p,p),1)
    return z.softmax(1)

def uncertainty(p):
    t=p.topk(2,dim=1).values
    return (1-t[:,:1]+t[:,1:2]).clamp(0,1)

def contrast(x):
    return x-F.avg_pool2d(x,3,1,1,count_include_pad=False)

def rgb(x):
    return x.expand(-1,3,-1,-1) if x.shape[1]==1 else x

def small_init(layer):
    nn.init.normal_(layer.weight,std=0.001)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer

def base_grid(h,w,device):
    y=(torch.arange(h,device=device,dtype=torch.float32)+0.5)*2/h-1
    x=(torch.arange(w,device=device,dtype=torch.float32)+0.5)*2/w-1
    yy,xx=torch.meshgrid(y,x,indexing="ij")
    return torch.stack((xx,yy),-1)[None]

def warp(x,flow):
    _,_,h,w=flow.shape
    grid=base_grid(h,w,flow.device)+flow.float().permute(0,2,3,1)*flow.new_tensor([2/w,2/h]).float()
    return F.grid_sample(x.float(),grid,align_corners=False,padding_mode="border").to(x.dtype)

def neighbor(x,dy,dx):
    h,w=x.shape[-2:]
    z=F.pad(x,(1,1,1,1),mode="replicate")
    return z[:,:,1+dy:1+dy+h,1+dx:1+dx+w]

OFFSETS=[(y,x) for y in (-1,0,1) for x in (-1,0,1)]
def local_filter(x,weights):
    return sum(neighbor(x,y,z)*weights[:,i:i+1] for i,(y,z) in enumerate(OFFSETS))

def flags_from_env():
    flags={}
    for name in PAIR:
        value=os.environ.get(name.upper()+"_ENABLED","1")
        if value not in ("0","1"):
            raise ValueError(name.upper()+"_ENABLED must be 0 or 1")
        flags[name]=int(value)
    return flags

def validate_flags(flags):
    if flags is None:
        return flags_from_env()
    if set(flags)!=set(PAIR) or any(type(v) not in (int,bool) or v not in (0,1) for v in flags.values()):
        raise ValueError("Research flags do not match branch "+str(PAIR))
    return {name:int(flags[name]) for name in PAIR}

def capture_training_flags(args):
    args.research_flags=flags_from_env()
    print("RESEARCH_FLAGS "+json.dumps(args.research_flags,sort_keys=True))

def checkpoint_flags(config):
    return validate_flags(config.get("research_flags",{name:0 for name in PAIR}))

def train_forward(model,images):
    raw=model.module if hasattr(model,"module") else model
    if not hasattr(raw,"research_pair"):
        return model(images,mode="train"),{}
    return model(images,mode="train",return_research_aux=True)

def add_auxiliary_loss(loss,aux,target):
    if not aux:
        return loss
    labels=target[:,0] if target.ndim==4 else target
    if "point_logits" in aux:
        z=aux["point_logits"].float()
        y=labels.flatten(1).gather(1,aux["point_indices"])
        point=F.binary_cross_entropy_with_logits(z[:,0],y.float()) if z.shape[1]==1 else F.cross_entropy(z,y.long())
        loss=loss+0.2*point
    if "edge_logits" in aux:
        z=aux["edge_logits"].float()
        y=labels[:,None].float() if z.shape[1]==1 else F.one_hot(labels.long(),z.shape[1]).permute(0,3,1,2).float()
        edge=(F.max_pool2d(y,3,1,1)+F.max_pool2d(-y,3,1,1)).clamp(0,1)
        pixel=F.binary_cross_entropy_with_logits(z,edge,reduction="none")
        pos=edge.sum((2,3))
        neg=(1-edge).sum((2,3))
        lp=(pixel*edge).sum((2,3))/pos.clamp_min(1)
        ln=(pixel*(1-edge)).sum((2,3))/neg.clamp_min(1)
        valid=(pos>0).float()+(neg>0).float()
        loss=loss+0.2*((lp+ln)/valid.clamp_min(1)).mean()
    return loss

def functional_flops_estimate(model):
    # Supplemental estimate only, not complete or a substitute for profiling.
    return sum(int(getattr(m,"_extra_ops",0)) for m in model.modules())

class FRD(nn.Module):
    """Full-resolution detail residual conditioned on decoder semantics."""
    def __init__(self,c,k):
        super().__init__()
        n=16
        self.stem=nn.Sequential(nn.Conv2d(6,n,3,padding=1),nn.GroupNorm(4,n),nn.SiLU())
        self.detail=nn.ModuleList([nn.Sequential(nn.Conv2d(n,n,3,padding=d,dilation=d,groups=n,bias=False),nn.Conv2d(n,n,1),nn.GroupNorm(4,n),nn.SiLU()) for d in (1,2,3,2,1)])
        self.semantic=nn.Conv2d(c,n,1)
        self.gate=nn.Sequential(nn.Conv2d(3*n,n,1),nn.SiLU(),nn.Conv2d(n,n,1))
        self.out=small_init(nn.Conv2d(n,k,1))
    def forward(self,image,feature):
        x=rgb(image)
        detail=self.stem(torch.cat((x,contrast(x)),1))
        for layer in self.detail:
            detail=detail+0.2*layer(detail)
        s=resize(self.semantic(feature),detail.shape[-2:])
        g=self.gate(torch.cat((detail,s,(detail-s).abs()),1)).sigmoid()
        self._extra_ops=detail.numel()*20+x.numel()*12+s.numel()*7
        return self.out(detail*g)

class SFAlign(nn.Module):
    """Bidirectional flow with forward/backward cycle reliability."""
    def __init__(self,c):
        super().__init__()
        n=min(32,c)
        self.low=nn.Conv2d(c,n,1,bias=False)
        self.high=nn.Conv2d(c,n,1,bias=False)
        self.forward_flow=small_init(nn.Conv2d(2*n,2,3,padding=1))
        self.reverse_flow=small_init(nn.Conv2d(2*n,2,3,padding=1))
        self.blend=nn.Conv2d(2*n+1,1,1)
    def forward(self,d,skip):
        a,b=self.low(d),self.high(skip)
        f=1.5*self.forward_flow(torch.cat((a,b),1)).tanh()
        r=1.5*self.reverse_flow(torch.cat((b,a),1)).tanh()
        error=(f+warp(r,f)).abs().mean(1,keepdim=True)
        valid=torch.exp(-error.float()).to(skip.dtype)
        gate=self.blend(torch.cat((a,b,error),1)).sigmoid()*valid
        moved=warp(skip,f)
        self._extra_ops=7*(skip.numel()+r.numel())+10*f.numel()+4*skip.numel()
        return skip+gate*(moved-skip)

class DySample(nn.Module):
    """Grouped offsets from semantic and high-frequency proposals."""
    def __init__(self,c):
        super().__init__()
        self.groups=math.gcd(c,4)
        n=min(c,32)
        self.project=nn.Conv2d(c,n,1)
        self.semantic_offset=small_init(nn.Conv2d(n,8*self.groups,1))
        self.detail_offset=small_init(nn.Conv2d(n,8*self.groups,3,padding=1))
        self.scope=nn.Conv2d(2*n,8*self.groups,1)
    def forward(self,x):
        b,c,h,w=x.shape
        z=self.project(x)
        high=contrast(z)
        gate=self.scope(torch.cat((z,high.abs()),1)).sigmoid()
        strength=(high.abs().mean(1,keepdim=True)/(z.abs().mean(1,keepdim=True)+1e-3)).clamp(0,1)
        mix=0.5*gate+0.5*strength
        off=0.5*((1-mix)*self.semantic_offset(z)+mix*self.detail_offset(high)).tanh()
        off=F.pixel_shuffle(off,2).reshape(b,2,self.groups,2*h,2*w).permute(0,2,3,4,1).float()
        grid=base_grid(2*h,2*w,x.device)[:,None]+off*off.new_tensor([2/w,2/h])
        out=F.grid_sample(x.reshape(b*self.groups,c//self.groups,h,w).float(),grid.reshape(b*self.groups,2*h,2*w,2),align_corners=False,padding_mode="border")
        self._extra_ops=7*out.numel()+15*z.numel()+8*off.numel()
        return out.reshape(b,c,2*h,2*w).to(x.dtype)

class PointRend(nn.Module):
    """Disagreement/area-sensitive selection and residual point refinement."""
    def __init__(self,c,k):
        super().__init__()
        self.detail=nn.Sequential(nn.Conv2d(c,32,1),nn.SiLU())
        self.mlp=nn.Sequential(nn.Conv1d(32+3+3*k,96,1),nn.ReLU(),nn.Conv1d(96,96,1),nn.ReLU(),small_init(nn.Conv1d(96,k,1)))
    @staticmethod
    def sample(x,coords):
        return F.grid_sample(x.float(),coords[:,:,None],align_corners=False,padding_mode="border").squeeze(3).to(x.dtype)
    def forward(self,coarse,adjacent,feature,image):
        base=coarse
        refined=base
        fine=self.detail(feature)
        image=rgb(image)
        self._extra_ops=0
        result={}
        for _ in range(2):
            refined=F.interpolate(refined,scale_factor=2,mode="bilinear",align_corners=False)
            b,k,h,w=refined.shape
            prior=resize(adjacent,(h,w))
            with torch.no_grad():
                p=prob(refined)
                q=prob(prior)
                u=uncertainty(p)
                dis=(p-q).abs().mean(1,keepdim=True)
                cls=p.argmax(1).flatten(1)
                area=p.mean((2,3))
                rare=(0.05/area.clamp_min(1/(h*w))).sqrt().clamp(1,3)
                rare[:,0]=1
                weight=rare.gather(1,cls).reshape(b,1,h,w)
                score=(u*(1+0.25*(weight-1))+0.5*dis).flatten(1)
                count=min(784,h*w)
                hard=count if not self.training else max(1,int(count*0.75))
                idx=score.topk(hard,1).indices
                if hard<count:
                    random=torch.rand_like(score).scatter(1,idx,-1)
                    idx=torch.cat((idx,random.topk(count-hard,1).indices),1)
                coords=torch.stack(((idx%w+0.5)*2/w-1,(idx//w+0.5)*2/h-1),-1).float()
            current=refined.flatten(2).gather(2,idx[:,None].expand(-1,k,-1))
            original=self.sample(base,coords)
            other=self.sample(prior,coords)
            values=torch.cat((self.sample(fine,coords),self.sample(image,coords),current,original,other),1)
            updated=current+self.mlp(values).to(current.dtype)
            refined=refined.flatten(2).scatter(2,idx[:,None].expand(-1,k,-1),updated).reshape(b,k,h,w)
            result={"point_logits":updated,"point_indices":idx}
            self._extra_ops+=7*(fine.shape[1]+3+2*k)*b*count+15*p.numel()+7*refined.numel()
        return refined,result

class UACA(nn.Module):
    """Normalized class/uncertain prototypes with scale-disagreement gating."""
    def __init__(self,c,k):
        super().__init__()
        n=min(c,32)
        self.feature=nn.Conv2d(c,n,1)
        self.query=nn.Conv2d(c,n,1)
        self.key=nn.Conv1d(n,n,1,bias=False)
        self.value=nn.Conv1d(n,n,1,bias=False)
        self.out=nn.Sequential(nn.Conv2d(c+n+2,n,3,padding=1),nn.SiLU(),small_init(nn.Conv2d(n,k,1)))
    def forward(self,feature,current,coarse):
        b,_,h,w=feature.shape
        p=prob(resize(coarse,(h,w)))
        q=prob(current)
        u=uncertainty(p)
        dis=(p-q).abs().mean(1,keepdim=True)
        masks=torch.cat((p*(1-u),u),1).flatten(2)
        masks=masks/masks.sum(2,keepdim=True).clamp_min(1e-4)
        f=self.feature(feature).flatten(2)
        proto=torch.bmm(f.float(),masks.transpose(1,2)).to(f.dtype)
        query=F.normalize(self.query(feature).float().flatten(2).transpose(1,2),dim=-1,eps=1e-4)
        key=F.normalize(self.key(proto).float(),dim=1,eps=1e-4)
        attention=(5*torch.bmm(query,key)).softmax(-1)
        value=self.value(proto).float().transpose(1,2)
        context=torch.bmm(attention,value).transpose(1,2).reshape(b,-1,h,w).to(feature.dtype)
        gate=(0.25+0.5*u+0.25*dis).to(feature.dtype)
        self._extra_ops=6*b*h*w*f.shape[1]*masks.shape[1]+12*p.numel()
        return current+gate*self.out(torch.cat((feature,context,u.to(feature.dtype),dis.to(feature.dtype)),1))

class FreqFusion(nn.Module):
    """Joint adaptive kernels, displacement and semantic agreement."""
    def __init__(self,c):
        super().__init__()
        n=min(c,24)
        self.a=nn.Conv2d(c,n,1)
        self.b=nn.Conv2d(c,n,1)
        self.kernel=nn.Sequential(nn.Conv2d(3*n,n,3,padding=1),nn.SiLU(),nn.Conv2d(n,20,1))
        self.offset=small_init(nn.Conv2d(3*n,2,3,padding=1))
    def forward(self,d,skip):
        a,b=self.a(d),self.b(skip)
        joint=torch.cat((a,b,(a-b).abs()),1)
        z=self.kernel(joint)
        low=z[:,:9].float().softmax(1).to(d.dtype)
        high=z[:,9:18].float().softmax(1).to(skip.dtype)
        flow=0.75*self.offset(joint).tanh()
        agreement=(1+F.cosine_similarity(a.float(),b.float(),dim=1,eps=1e-4)[:,None])*0.5
        smooth=warp(local_filter(d,low),flow)
        detail=skip-local_filter(skip,high)
        low_gate=z[:,18:19].sigmoid()*agreement.to(d.dtype)
        high_gate=z[:,19:20].sigmoid()*agreement.to(skip.dtype)
        self._extra_ops=24*d.numel()+20*skip.numel()+10*a.numel()+5*z.numel()
        return d+low_gate*(smooth-d),skip+high_gate*detail

class WTConv(nn.Module):
    """Two-level Haar convolution, with cross-subband gates."""
    def __init__(self,c):
        super().__init__()
        self.base=nn.Conv2d(c,c,5,padding=2,groups=c,bias=False)
        self.sub=nn.ModuleList([nn.Conv2d(4*c,4*c,3,padding=1,groups=4*c,bias=False) for _ in range(2)])
        self.gate=nn.ModuleList([nn.Conv2d(4*c,4*c,1,groups=c) for _ in range(2)])
    @staticmethod
    def decompose(x):
        a,b,c,d=x[:,:,0::2,0::2],x[:,:,0::2,1::2],x[:,:,1::2,0::2],x[:,:,1::2,1::2]
        return torch.stack((a+b+c+d,a-b+c-d,a+b-c-d,a-b-c+d),2)*0.5
    @staticmethod
    def reconstruct(z):
        a,b,c,d=z.unbind(2)
        out=torch.stack((a+b+c+d,a-b+c-d,a+b-c-d,a-b-c+d),2)*0.5
        return F.pixel_shuffle(out.flatten(1,2),2)
    def forward(self,x):
        levels=[]
        current=x
        self._extra_ops=0
        for conv,gate in zip(self.sub,self.gate):
            h,w=current.shape[-2:]
            padded=F.pad(current,(0,w%2,0,h%2),mode="replicate")
            z=self.decompose(padded)
            current=z[:,:,0]
            flat=z.flatten(1,2)
            v=(conv(flat)*(0.5+gate(flat).sigmoid())).reshape_as(z)
            levels.append((v,h,w))
            self._extra_ops+=11*flat.numel()
        result=None
        for z,h,w in reversed(levels):
            ll=z[:,:,0] if result is None else z[:,:,0]+result
            result=self.reconstruct(torch.cat((ll[:,:,None],z[:,:,1:]),2))[:,:,:h,:w]
        self._extra_ops+=3*x.numel()
        return x+0.1*(self.base(x)+result)

class SAPA(nn.Module):
    """Similarity-based affiliation with adaptive temperature and entropy fallback."""
    def __init__(self,c,skip_c):
        super().__init__()
        self.key=nn.Conv2d(c,16,1)
        self.query=nn.Conv2d(skip_c,16,1)
        self.temperature=nn.Conv2d(16,1,1)
    def forward(self,x,skip):
        size=skip.shape[-2:]
        q=F.normalize(self.query(skip).float(),dim=1,eps=1e-4)
        k=F.normalize(self.key(x).float(),dim=1,eps=1e-4)
        t=1+7*self.temperature(q.to(skip.dtype)).float().sigmoid()
        score=torch.cat([(q*resize(neighbor(k,dy,dx),size,"nearest")).sum(1,keepdim=True) for dy,dx in OFFSETS],1)*t
        weight=score.softmax(1)
        entropy=-(weight*weight.clamp_min(1e-8).log()).sum(1,keepdim=True)/math.log(9)
        confidence=(1-entropy).clamp(0,1).to(x.dtype)
        mixed=sum(resize(neighbor(x,dy,dx),size,"nearest")*weight[:,i:i+1].to(x.dtype) for i,(dy,dx) in enumerate(OFFSETS))
        base=resize(x,size,"nearest")
        self._extra_ops=18*q.numel()+22*mixed.numel()+8*weight.numel()
        return base+confidence*(mixed-base)

class GSCNN(nn.Module):
    """Class-specific shape stream; differentiable contrast and scale-disagreement gates."""
    def __init__(self,c1,c2,k):
        super().__init__()
        n=16
        self.stem=nn.Sequential(nn.Conv2d(6,n,3,padding=1),nn.GroupNorm(4,n),nn.SiLU())
        self.sem=nn.ModuleList([nn.Conv2d(c2,n,1),nn.Conv2d(c1,n,1)])
        self.gates=nn.ModuleList([nn.Conv2d(2*n+1,n,1) for _ in range(2)])
        self.shape=nn.ModuleList([nn.Sequential(nn.Conv2d(n,n,3,padding=1,groups=n),nn.Conv2d(n,n,1),nn.SiLU()) for _ in range(2)])
        self.edge=nn.Conv2d(n,k,1)
        self.residual=small_init(nn.Conv2d(2*n,k,1))
    def forward(self,image,features,current,adjacent):
        x=rgb(image)
        size=current.shape[-2:]
        detail=self.stem(torch.cat((x,contrast(x)),1))
        dis=(prob(current)-prob(adjacent)).abs().mean(1,keepdim=True).to(detail.dtype)
        for projection,gate,layer,f in zip(self.sem,self.gates,self.shape,(features[-2],features[-1])):
            semantic=resize(projection(f),size)
            detail=detail+layer(detail)*gate(torch.cat((detail,semantic,dis),1)).sigmoid()
        edge=self.edge(detail)
        delta=self.residual(torch.cat((detail,semantic),1))
        self._extra_ops=12*x.numel()+22*detail.numel()+9*edge.numel()
        return current+edge.sigmoid()*(0.5+0.5*dis)*delta,edge

class ARA(nn.Module):
    """Parallel axial competition and class-wise reverse attention."""
    def __init__(self,c,k):
        super().__init__()
        n=min(c,32)
        self.qkv=nn.Conv2d(c,3*n,1)
        self.axis_gate=nn.Conv2d(2,2,1)
        self.out=nn.Sequential(nn.Conv2d(c+n,n,3,padding=1),nn.SiLU(),small_init(nn.Conv2d(n,k,1)))
    @staticmethod
    def axis(q,k,v,vertical=False):
        if vertical:
            q,k,v=(a.transpose(2,3) for a in (q,k,v))
        b,c,h,w=q.shape
        def seq(x):
            return x.permute(0,2,3,1).reshape(b*h,w,c).float()
        a=(torch.bmm(seq(q),seq(k).transpose(1,2))/math.sqrt(c)).softmax(-1)
        out=torch.bmm(a,seq(v)).reshape(b,h,w,c).permute(0,3,1,2)
        return out.transpose(2,3) if vertical else out
    def forward(self,feature,current,coarse):
        p=prob(resize(coarse,current.shape[-2:]))
        q=prob(current)
        u=uncertainty(p)
        dis=(p-q).abs().mean(1,keepdim=True)
        a,b,v=self.qkv(feature).chunk(3,1)
        weights=self.axis_gate(torch.cat((u,dis),1).to(feature.dtype)).float().softmax(1)
        context=(weights[:,:1]*self.axis(a,b,v)+weights[:,1:]*self.axis(a,b,v,True)).to(feature.dtype)
        reverse=(1-p[:,1:] if current.shape[1]==1 else 1-p).to(current.dtype)
        gate=0.25+0.5*reverse+0.25*dis.to(current.dtype)
        self._extra_ops=4*v.numel()*(v.shape[-1]+v.shape[-2])+10*p.numel()
        return current+gate*self.out(torch.cat((feature,context),1))

class ResearchPair(nn.Module):
    """Instantiate only the selected pair; disabled modules have no parameters."""
    def __init__(self,channels,k,flags):
        super().__init__()
        self.flags=validate_flags(flags)
        active={name for name,enabled in self.flags.items() if enabled}
        m=self.blocks=nn.ModuleDict()
        if "FRD" in active:
            m["FRD"]=FRD(channels[-1],k)
        if "SFAlign" in active:
            m["SFAlign"]=nn.ModuleList([SFAlign(c) for c in channels[1:]])
        if "DySample" in active:
            m["DySample"]=nn.ModuleList([DySample(c) for c in channels[:-1]])
        if "PointRend" in active:
            m["PointRend"]=PointRend(channels[-1],k)
        if "UACA" in active:
            m["UACA"]=UACA(channels[-1],k)
        if "FreqFusion" in active:
            m["FreqFusion"]=nn.ModuleList([FreqFusion(c) for c in channels[1:]])
        if "WTConv" in active:
            m["WTConv"]=nn.ModuleList([WTConv(c) for c in channels[:2]])
        if "SAPA" in active:
            m["SAPA"]=nn.ModuleList([SAPA(a,b) for a,b in zip(channels[:-1],channels[1:])])
        if "GSCNN" in active:
            m["GSCNN"]=GSCNN(channels[-1],channels[-2],k)
        if "ARA" in active:
            m["ARA"]=ARA(channels[-1],k)
    def decode(self,dec,x,skips):
        m=self.blocks
        if not any(k in m for k in ("SFAlign","DySample","FreqFusion","WTConv","SAPA")):
            return dec(x,skips)
        d=dec.cab4(x)*x
        d=dec.sab(d)*d
        d=dec.mscb4(d)
        if "WTConv" in m:
            d=m["WTConv"][0](d)
        features=[d]
        for i,(stage,skip) in enumerate(zip((3,2,1),skips)):
            up=getattr(dec,"eucb"+str(stage))
            if "DySample" in m or "SAPA" in m:
                from lib.decoders import channel_shuffle
                d=m["DySample"][i](d) if "DySample" in m else m["SAPA"][i](d,skip)
                for layer in list(up.up_dwc.children())[1:]:
                    d=layer(d)
                d=up.pwc(channel_shuffle(d,up.in_channels))
            else:
                d=up(d)
            if "SFAlign" in m:
                skip=m["SFAlign"][i](d,skip)
            if "FreqFusion" in m:
                d,skip=m["FreqFusion"][i](d,skip)
            d=d+getattr(dec,"lgag"+str(stage))(g=d,x=skip)
            d=getattr(dec,"cab"+str(stage))(d)*d
            d=dec.sab(d)*d
            d=getattr(dec,"mscb"+str(stage))(d)
            if "WTConv" in m and i==0:
                d=m["WTConv"][1](d)
            features.append(d)
        return features
    def refine(self,image,features,native,outputs):
        m=self.blocks
        current=native[-1]
        aux={}
        if "UACA" in m:
            current=m["UACA"](features[-1],current,native[-2])
        if "ARA" in m:
            current=m["ARA"](features[-1],current,native[-2])
        if "PointRend" in m:
            final,aux=m["PointRend"](current,native[-2],features[-1],image)
        else:
            final=resize(current,outputs[-1].shape[-2:])
        if "FRD" in m:
            final=final+m["FRD"](image,features[-1])
        if "GSCNN" in m:
            final,edge=m["GSCNN"](image,features,final,outputs[-2])
            aux["edge_logits"]=edge
        return outputs[:-1]+[final],aux
