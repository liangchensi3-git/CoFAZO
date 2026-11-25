"""Signed row-column magnitude with 3-bit ratio; two packed nibbles per byte.

Only next-step history is quantized. Same row/column means, present A and A^2
as R34; ratio levels are fixed at 0,3/7,...,3. No extra gradient pass.
"""
import torch
import torch.nn.functional as F
from m_compression import SignedRC

def pack_codes(codes):
    if codes.shape[1]%2:codes=F.pad(codes,(0,1))
    return codes[:,0::2].bitwise_or(codes[:,1::2].bitwise_left_shift(4))

def unpack_codes(packed,cols):
    out=torch.stack((packed.bitwise_and(15),packed.bitwise_right_shift(4)),dim=-1)
    return out.reshape(packed.shape[0],-1)[:,:cols]

def encode(value,magnitude):
    levels=value.abs().div_(magnitude.clamp_min(1e-30)).mul_(7/3).round_().clamp_(0,7).to(torch.uint8)
    return levels.bitwise_or((value>=0).to(torch.uint8).bitwise_left_shift(3))

def reconstruct(magnitude,codes):
    sign=codes.bitwise_right_shift(3).float().mul_(2).sub_(1)
    return magnitude.mul(sign).mul_(codes.bitwise_and(7).float()).mul_(3/7)

class SignedRC4(SignedRC):
    mode='signed_rc4'
    def __init__(self,name,p,rank=16):
        self.rows,self.cols=p.shape
        self.row=torch.zeros(self.rows,device=p.device,dtype=torch.float32)
        self.col=torch.zeros(self.cols,device=p.device,dtype=torch.float32)
        self.codes=torch.zeros((self.rows,(self.cols+1)//2),device=p.device,dtype=torch.uint8)
    def tensors(self):return self.row,self.col,self.codes
    def decode(self,row,rows):
        magnitude=self.row[row:row+rows,None]*self.col[None,:]/self.row.mean().clamp_min(1e-30)
        return reconstruct(magnitude,unpack_codes(self.codes[row:row+rows],self.cols))
    def decode_samples(self,rows,cols):
        code=self.codes[rows,cols//2].bitwise_right_shift((cols%2)*4).bitwise_and(15)
        magnitude=self.row[rows]*self.col[cols]/self.row.mean().clamp_min(1e-30)
        return reconstruct(magnitude,code)
    def second(self,pending,value,row,diagnostic):
        magnitude=pending['row'][row:row+value.shape[0],None]*pending['col'][None,:]/pending['mean']
        codes=encode(value,magnitude)
        self.codes[row:row+value.shape[0]].copy_(pack_codes(codes))
        if not diagnostic:return None
        return (reconstruct(magnitude,codes).double()-value.double()).square().sum()
