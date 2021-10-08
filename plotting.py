import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from physics import Particle, Disk
from matplotlib import rc
import pickle as pkl
rc('text',usetex=True)
cmap = matplotlib.cm.get_cmap('Dark2')
colors = cmap(np.linspace(0,1,6))
import glob

def inspect_grid(d,dpi=200):
    if not isinstance(d,Disk):
        return NotImplemented
    else:
        R = d._R
        N = d._N
        rs = d.p.r
        v_mag = d.p.vmag
        Nrings = d.Nrings
        dr = d.dr
        theta_pl = np.linspace(0,2*np.pi,1000)
        theta_crit = d.phi_edges
        r_rings = d.r_edges
        thetas = d.p.phi
        fig,axes = plt.subplots(1,3,figsize=(9,3),dpi=dpi,facecolor='white')
        rmax = 0.1
        ax = axes[0]
        ax.set_aspect('equal')
        for ri in r_rings:
            ax.plot(ri*np.cos(theta_pl),ri*np.sin(theta_pl),'k-')
        for tc in theta_crit:
            ax.plot([0,R*np.cos(tc)],[0,R*np.sin(tc)],'k-')
        ax.grid(color='gray',linestyle='dashed',alpha=0.3)
        ax.set_xlim(-rmax,rmax);ax.set_ylim(-rmax,rmax)
        ax.set_xticks(np.linspace(-rmax,rmax,3));ax.set_yticks(np.linspace(-rmax,rmax,3))
        ax.set_xlabel('$x$',size=15)
        ax.set_ylabel('$y$',size=15)
        ax.scatter(d.p.x,d.p.y,s=3,color=colors[2])

        ax = axes[1]
        ax.set_aspect('equal')
        ax.scatter(d.p.x,d.p.y,s=0.3,color=colors[2])
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.grid(color='gray',linestyle='dashed',alpha=0.3)
        ax.set_xlabel('$x$',size=15)
        ax.set_ylabel('$y$',size=15)

        ax = axes[2]
        ax.set_aspect('equal')
        ax.quiver(d.p.x,d.p.y,d.p.vx,d.p.vy,scale=5e3/v_mag)
        ax.scatter(d.p.x,d.p.y,s=0.3,color=colors[2])
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.grid(color='gray',linestyle='dashed',alpha=0.3)
        ax.set_xlabel('$x$',size=15)
        ax.set_ylabel('$y$',size=15)
        plt.tight_layout()
        return fig,axes
    

def plot_disp(d,dpi=200):
    if not isinstance(d,Disk):
        return NotImplemented
    else:
        fig,ax = plt.subplots(1,1,figsize=(3,4),dpi=dpi,facecolor='white')
        ax.scatter(d.rpl,d.vpl,marker='o',edgecolor='black',facecolor='none',s=12)
        ax.scatter(d.ravg,d.sigmar,marker='x',c='black',s=12)
        ax.scatter(d.ravg,d.sigmaz,marker='v',edgecolor='black',facecolor='none',s=12)
        ax.grid(color='gray',linestyle='dashed',alpha=0.3)
        ax.set_ylabel('$v(r)$',size=15)
        ax.set_xlabel('$r/R$',size=15)
        ax.set_ylim(0,np.max(d.vpl)+2)
        ax.set_xlim(0,1)
        return fig,ax
    
    
    
def plot_vels(d,dpi=200):
    if not isinstance(d,Disk):
        return NotImplemented
    else:
        fig,axes = plt.subplots(1,3,figsize=(9,3),dpi=dpi,facecolor='white')
        ax = axes[0]
        ax.quiver(d.p.x,d.p.y,d.p.vperp_x,d.p.vperp_y,scale=200)
        ax.set_title('$v_\phi$',size=15)
        ax.set_ylabel('$y$',size=15)
        ax = axes[1]
        ax.quiver(d.p.x,d.p.y,d.p.vpar_x,d.p.vpar_y,scale=200)
        ax.set_title('$v_r$',size=15)
        ax.set_xlabel('$y$',size=15)
        ax = axes[2]
        ax.quiver(d.p.x,d.p.z,np.zeros_like(d.p.vz),d.p.vz,scale=200)
        ax.set_title('$v_z$',size=15)
        ax.set_ylabel('$z$',size=15)
        for ax in axes:
            ax.set_xlim(-1.2,1.2)
            ax.set_ylim(-1.2,1.2)
            ax.grid(color='gray',linestyle='dashed',alpha=0.3)
            ax.set_xlabel('$x$',size=15)
            ax.set_xticks(np.linspace(-1,1,5));ax.set_yticks(np.linspace(-1,1,5))
        plt.tight_layout()
        return fig,axes

def plot_dumps(R,c,Mh,N,outdir='/home/pdave/cosmo/disk_out/'):
    l = glob.glob(outdir + "disk_t_*_R_{:.1f}_c_{:.2f}_Mh_{:.0f}_N_{:.0f}.pkl".format(R,c,Mh,N))
    l.sort()
    for infile in l:
        with open(infile, "rb") as i:
            d = pkl.load(i)
        fig,ax = plot_vels(d)
        fig.savefig(infile[:-3]+'png')
        plt.close()
        
def dump_scalars(R,c,Mh,N,outdir='/home/pdave/cosmo/disk_out/'):
    l = glob.glob(outdir + "disk_t_*_R_{:.1f}_c_{:.2f}_Mh_{:.0f}_N_{:.0f}.pkl".format(R,c,Mh,N))
    l.sort()
    tau, zrms, tmean = np.zeros_like(l),np.zeros_like(l),np.zeros_like(l)
    result = np.zeros(len(l), dtype=[('tau', '<f8'), ('tmean', '<f8'), ('zrms', '<f8')])
    for i in range(len(l)):
        with open(l[i], "rb") as infile:
            d = pkl.load(infile)
            result['zrms'][i] = d.zrms
            result['tmean'][i] = d.tmean
            del d
    np.save(outdir+"scalars_R_{:.1f}_c_{:.2f}_Mh_{:.0f}_N_{:.0f}".format(R,c,Mh,N),result)