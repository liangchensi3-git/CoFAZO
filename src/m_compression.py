"""GPU-local compressed FFN history; never allocate a dense layer moment.

Every method compresses the *history for the next step*. The present update
and R-AdaZO m^2 statistics use A = beta * decode(old_history) + (1-beta)*g.
All work is row-blocked. Zero signed values use the nonnegative sign; an
entirely zero magnitude is reconstructed as exactly zero.
"""
import hashlib
import torch
import torch.nn.functional as F


def pack_signs(value):
    bits=(value>=0).to(torch.uint8)
    padding=(-bits.shape[1])%8
    if padding: bits=F.pad(bits,(0,padding))
    shifts=torch.arange(8,device=bits.device,dtype=torch.uint8)
    return bits.view(bits.shape[0],-1,8).bitwise_left_shift_(shifts).sum(-1).to(torch.uint8)


def unpack_signs(packed,cols):
    shifts=torch.arange(8,device=packed.device,dtype=torch.uint8)
    bits=(packed.unsqueeze(-1).bitwise_right_shift(shifts)).bitwise_and_(1)
    return bits.reshape(packed.shape[0],-1)[:,:cols].float().mul_(2).sub_(1)


class SignedRC:
    compressed=True
    mode='signed_rc'

    def __init__(self,name,p,rank=16):
        self.rows,self.cols=p.shape
        self.row=torch.zeros(self.rows,device=p.device,dtype=torch.float32)
        self.col=torch.zeros(self.cols,device=p.device,dtype=torch.float32)
        self.signs=torch.zeros((self.rows,(self.cols+7)//8),device=p.device,dtype=torch.uint8)

    def tensors(self): return (self.row,self.col,self.signs)

    def begin(self,step):
        return dict(row=torch.zeros_like(self.row),col=torch.zeros_like(self.col))

    def decode(self,row,rows):
        magnitude=self.row[row:row+rows,None]*self.col[None,:]
        magnitude.div_(self.row.mean().clamp_min(1e-30))
        return magnitude.mul_(unpack_signs(self.signs[row:row+rows],self.cols))

    def combine(self,g,row,beta):
        return self.decode(row,g.shape[0]).mul_(beta).add_(g,alpha=1-beta)

    def accumulate(self,pending,value,row):
        absolute=value.abs()
        pending['row'][row:row+value.shape[0]].copy_(absolute.mean(1))
        pending['col'].add_(absolute.sum(0),alpha=1/self.rows)

    def finish_first(self,pending):
        pending['mean']=pending['row'].mean().clamp_min(1e-30)

    def second(self,pending,value,row,diagnostic):
        # Read old signs first (combine), then overwrite this row slice only.
        self.signs[row:row+value.shape[0]].copy_(pack_signs(value))
        if not diagnostic: return None
        approx=pending['row'][row:row+value.shape[0],None]*pending['col'][None,:]
        approx.div_(pending['mean']).mul_(unpack_signs(self.signs[row:row+value.shape[0]],self.cols))
        return (approx.double()-value.double()).square().sum()

    def commit(self,pending):
        self.row.copy_(pending['row']); self.col.copy_(pending['col'])


class LowRank:
    compressed=True
    mode='lowrank'

    def __init__(self,name,p,rank=16):
        self.rows,self.cols=p.shape
        self.rank=min(rank,self.rows,self.cols)
        self.u=torch.zeros((self.rows,self.rank),device=p.device,dtype=torch.float32)
        self.c=torch.zeros((self.rank,self.cols),device=p.device,dtype=torch.float32)
        self.seed=int.from_bytes(hashlib.sha256(name.encode()).digest()[:8],'little')%(2**63-1)

    def tensors(self): return (self.u,self.c)

    def begin(self,step):
        # Independent PRNG: never consumes model/batch/FZOO-direction RNG.
        generator=torch.Generator(device=self.u.device).manual_seed((self.seed+104729*step)%(2**63-1))
        omega=torch.randn((self.cols,self.rank),generator=generator,device=self.u.device,dtype=torch.float32)
        return dict(omega=omega,y=torch.empty_like(self.u),c=torch.zeros_like(self.c))

    def decode(self,row,rows): return self.u[row:row+rows]@self.c

    def combine(self,g,row,beta):
        return g.mul_(1-beta).addmm_(self.u[row:row+g.shape[0]],self.c,alpha=beta)

    def accumulate(self,pending,value,row):
        torch.mm(value,pending['omega'],out=pending['y'][row:row+value.shape[0]])

    def finish_first(self,pending):
        # A thin rows x rank QR, not a full moment or a full-matrix SVD.
        pending['u'],_=torch.linalg.qr(pending.pop('y'),mode='reduced')
        del pending['omega']

    def second(self,pending,value,row,diagnostic):
        pending['c'].addmm_(pending['u'][row:row+value.shape[0]].T,value)
        return None

    def commit(self,pending):
        self.u.copy_(pending['u']); self.c.copy_(pending['c'])


def tensor_bytes(tensors): return sum(t.numel()*t.element_size() for t in tensors)
