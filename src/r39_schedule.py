"""Scalar-only early learning-rate multiplier; no optimizer-state changes."""
import math
ARMS=('fzoo','stored_m','r38','lr125','lr150','all_rc4')
BOOST={'fzoo':1.,'stored_m':1.,'r38':1.,'lr125':1.25,'lr150':1.5,'all_rc4':1.}

def multiplier(step,peak=1.,until=3000):
    if step<1 or until<2 or not math.isfinite(peak) or peak<1:
        raise ValueError('Invalid early learning-rate schedule')
    return 1.+(peak-1.)*max(0.,(until-step)/(until-1))

def effective_lr(base_lr,step,drop_after=0,factor=.25,*,peak=1.,until=3000):
    from r28_support import effective_lr as original
    return original(base_lr,step,drop_after,factor)*multiplier(step,peak,until)
