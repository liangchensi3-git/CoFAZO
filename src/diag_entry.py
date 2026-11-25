"""Invoke the original trainer, optionally attaching a read-only observer."""
import argparse,runpy,sys
from pathlib import Path
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--r35-observe',required=True,choices=['off','on'])
p.add_argument('--r35-output',type=Path,required=True)
p.add_argument('--r35-interval',type=int,default=50)
p.add_argument('--r35-samples',type=int,default=256)
a,remaining=p.parse_known_args()
if a.r35_observe=='on':
    from observer import install
    install(a.r35_output,a.r35_samples,a.r35_interval)
sys.argv=[str(Path(__file__).with_name('r22_train.py'))]+remaining
runpy.run_path(sys.argv[0],run_name='__main__')
