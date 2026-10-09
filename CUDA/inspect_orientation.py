#!/usr/bin/env python3
"""Inspect the density principal axes of a Sky3D checkpoint without changing it."""
import argparse
import json
from pathlib import Path

import numpy as np
from benchmark import checkpoint
from validate_extended import sha256


def orientation(path):
    state=checkpoint(path)
    nx,ny,nz=state['grid']
    dx,dy,dz=state['spacing']
    axes=[(np.arange(n)+0.5-n/2)*d for n,d in zip(state['grid'],state['spacing'])]
    basis=state['psi'].reshape(state['states'],2,nz,ny,nx)
    rho=np.sum(state['attributes'][0,:,None,None,None]*np.sum(abs(basis)**2,axis=1),axis=0)
    z,y,x=np.meshgrid(axes[2],axes[1],axes[0],indexing='ij')
    coords=np.stack([x.ravel(),y.ravel(),z.ravel()])
    weights=rho.ravel()/rho.sum()
    cm=coords@weights
    centered=coords-cm[:,None]
    tensor=(centered*weights)@centered.T
    values,vectors=np.linalg.eigh(tensor)
    order=np.argsort(values)[::-1]
    values,vectors=values[order],vectors[:,order]
    for column in range(3):
        if vectors[np.argmax(abs(vectors[:,column])),column]<0:
            vectors[:,column]*=-1
    result=dict(state_sha256=sha256(path),grid=state['grid'],spacing_fm=state['spacing'],
                particles=state['particles'],center_of_mass_fm=cm.tolist(),
                coordinate_mean_square_fm2=np.diag(tensor).tolist(),
                principal_mean_square_fm2=values.tolist(),
                principal_axes_columns_xyz=vectors.tolist(),
                longest_axis_xyz=vectors[:,0].tolist(),
                transverse_relative_splitting=float(abs(values[1]-values[2])/max(abs(values[1]),1e-300)),
                note='Axis signs are conventional. A spherical density has no preferred intrinsic axis; nearly degenerate transverse axes are not individually robust.')
    return result,axes,rho


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('state',type=Path)
    parser.add_argument('--rotated-state',type=Path,help='Optionally compare an independently rotated checkpoint')
    parser.add_argument('--json',type=Path,required=True)
    parser.add_argument('--plot',type=Path)
    args=parser.parse_args()
    original,axes,rho=orientation(args.state)
    report=dict(original=original)
    panels=[('Converged static density',original,axes,rho)]
    if args.rotated_state:
        rotated,r_axes,r_rho=orientation(args.rotated_state)
        report['rotated']=rotated
        panels.append(('After exact grid + spinor rotation',rotated,r_axes,r_rho))
    args.json.parent.mkdir(parents=True,exist_ok=True)
    args.json.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,plots=plt.subplots(1,len(panels),figsize=(5.2*len(panels),4.5),squeeze=False,constrained_layout=True)
        for ax,(title,result,a,r) in zip(plots[0],panels):
            mid=len(a[1])//2
            ax.contourf(a[0],a[2],r[:,mid,:],levels=16,cmap='Blues')
            direction=np.array(result['longest_axis_xyz'])[[0,2]]
            ax.annotate('',xy=5*direction,xytext=-5*direction,
                        arrowprops=dict(arrowstyle='<->',color='#d27622',lw=2.5))
            ax.set(xlabel='x (fm)',ylabel='z (fm)',title=title,aspect='equal',xlim=(-8,8),ylim=(-8,8))
        fig.suptitle(r'$^{20}$Ne: same static state, different Cartesian orientation',fontsize=14)
        fig.savefig(args.plot,dpi=170)
        plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
