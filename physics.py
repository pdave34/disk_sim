#!/usr/bin/env python
import numpy as np

class Particle(object):
    def __init__(self, x, y, z, vx, vy, vz):
        if isinstance(x,(list,tuple,np.ndarray)):
            self._plural = True
        else:
            self._plural = False
        self._x  = x
        self._y  = y
        self._z  = z
        self._vx = vx
        self._vy = vy
        self._vz = vz

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def vx(self):
        return self._vx

    @property
    def vy(self):
        return self._vy

    @property
    def vz(self):
        return self._vz

    @property
    def vmag(self):
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

    @property
    def r(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def phi(self):
        return np.arctan2(self.y,self.x)

    @property
    def rhat_x(self):
        return self.x/self.r

    @property
    def rhat_y(self):
        return self.y/self.r

    @property
    def rhat_z(self):
        return self.z/self.r

    @property
    def vpar(self):
        return self.vx*self.x + self.vy*self.y

    @property
    def vpar_x(self):
        return self.vpar*self.rhat_x

    @property
    def vpar_y(self):
        return self.vpar*self.rhat_y

    @property
    def vperp_x(self):
        return self.vx - self.vpar_x

    @property
    def vperp_y(self):
        return self.vy - self.vpar_y

    @property
    def vperp(self):
        return np.sqrt(self.vperp_x**2 + self.vperp_y**2)

    @property
    def that_x(self):
        return self.vperp_x/self.vperp

    @property
    def that_y(self):
        return self.vperp_y/self.vperp


    def distance(self, p2):
        if not isinstance(p2, Particle):
            return NotImplemented
        else:
            return np.sqrt((self.x - p2.x)**2 + (self.y - p2.y)**2 + (self.z - p2.z)**2)

    def gx(self,p2,c):
        if not isinstance(p2, Particle):
            return NotImplemented
        else:
            r = self.distance(p2)
            return (p2.x - self.x)/(r**2 + c**2)**(3/2)

    def gy(self,p2,c):
        if not isinstance(p2, Particle):
            return NotImplemented
        else:
            r = self.distance(p2)
            return (p2.y - self.y)/(r**2 + c**2)**(3/2)

    def gz(self,p2,c):
        if not isinstance(p2, Particle):
            return NotImplemented
        else:
            r = self.distance(p2)
            return (p2.z - self.z)/(r**2 + c**2)**(3/2)

    def ghalo_x(self,Mh,R):
        if self._plural:
            m = self.r > R
            gx = np.zeros_like(self.r)
            gx[m] = -self.x[m] * Mh/self.r[m]**3
            gx[~m] = -(1.1)**2 * self.x[~m] * Mh / (self.r[~m] + 0.1*R)**2
            return gx
        else:
            if self.r > R:
                return -self.x * Mh / self.r**3
            else:
                return -(1.1)**2 * self.x * Mh / (self.r + 0.1*R)**2

    def ghalo_y(self,Mh,R):
        if self._plural:
            m = self.r > R
            gy = np.zeros_like(self.r)
            gy[m] = -self.y[m] * Mh/self.r[m]**3
            gy[~m] = -(1.1)**2 * self.y[~m] * Mh / (self.r[~m] + 0.1*R)**2
            return gy
        else:
            if self.r > R:
                return -self.y * Mh / self.r**3
            else:
                return -(1.1)**2 * self.y * Mh / (self.r + 0.1*R)**2

    def ghalo_z(self,Mh,R):
        if self._plural:
            m = self.r > R
            gz = np.zeros_like(self.r)
            gz[m] = -self.z[m] * Mh/self.r[m]**3
            gz[~m] = -(1.1)**2 * self.z[~m] * Mh / (self.r[~m] + 0.1*R)**2
            return gz
        else:
            if self.r > R:
                return -self.z * Mh / self.r**3
            else:
                return -(1.1)**2 * self.z * Mh / (self.r + 0.1*R)**2

    def __getitem__(self, item):
        return getattr(self, item)

class Disk(object):
    def __init__(self,N,R,c,Mh,rss):
        self._n  = 10
        self._N  = int(N)
        self._R  = R
        self._c  = c
        self._Mh = Mh
        self._rss = rss
        self._x, self._y, self._z = self.init_coordinates()
        self._vx, self._vy, self._vz = np.zeros_like(self._x), np.zeros_like(self._x), np.zeros_like(self._x)
        self._p = Particle(self._x,self._y,self._z,self._vx,self._vy,self._vz)
        self._sigmar = np.zeros(self._n-1,dtype=float)
        self._sigmaphi = np.zeros(self._n-1,dtype=float)
        self._sigmaz = np.zeros(self._n-1,dtype=float)
        self._ravg = np.zeros(self._n-1,dtype=float)
        self._rpl = np.zeros(self.Nrings,dtype=float)
        self._vpl = np.zeros(self.Nrings,dtype=float)
        print("Finished initializing disk with paramters (N,R,c,Mh) = ({},{},{},{})".format(N,R,c,Mh))

    @property
    def tau(self):
        m = np.argmax(self.p.r)
        v = self.p.vmag[m]
        return 2*np.pi*self._R/v

    @property
    def rpl(self):
        return self._rpl

    @property
    def vpl(self):
        return self._vpl

    @property
    def sigmar(self):
        return self._sigmar

    @property
    def sigmaphi(self):
        return self._sigmaphi

    @property
    def sigmaz(self):
        return self._sigmaz

    @property
    def ravg(self):
        return self._ravg

    @property
    def p(self):
        return self._p

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def vx(self):
        return self._vx

    @property
    def vy(self):
        return self._vy

    @property
    def vz(self):
        return self._vz

    @property
    def Nrings(self):
        return int(self._N / self._n)

    @property
    def dr(self):
        return self._R / self.Nrings

    @property
    def r_edges(self):
        return np.linspace(0, self._R, self.Nrings + 1)

    @property
    def r_bins(self):
        return np.linspace(0, self._R, self._n + 1)

    @property
    def phi_edges(self):
        return np.linspace(-np.pi, np.pi, self._n + 1)

    @property
    def zrms(self):
        return np.sqrt(np.mean(self.p.z**2))

    @property
    def totPE(self):
        PE = 0
        p2 = self.create_particles(self.p)
        for i in range(self._N):
            for j in range(self._N):
                if i!=j:
                    PE += 0.5/p2[i].distance(p2[j])
        PE += np.sum(self._Mh/self.p.r)
        PE += self._Mh/self._R**2
        return PE

    @property
    def totKE(self):
        KE = 0.0
        p = self.p
        r_e = np.arange(0,max(p.r)+1e-2,0.1)
        for i in range(len(r_e)-1):
            m = (p.r > r_e[i])*(p.r < r_e[i+1])
            nalpha = np.sum(m)
            if nalpha == 0:
                KE += 0
            else:
                valpha = np.mean(p.vperp[m])
                KE += 0.5*nalpha*valpha**2
        return KE

    @property
    def tmean(self):
        return self.totKE/self.totPE

    def init_coordinates(self):
        phi,r = [],[]
        for i in range(self._n):
            for j in range(self.Nrings):
                r.append(self._rss.uniform(self.r_edges[j], self.r_edges[j+1]))
                phi.append(self._rss.uniform(self.phi_edges[i], self.phi_edges[i+1]))
        phi,r = np.array(phi),np.array(r)
        x = r*np.cos(phi)
        y = r*np.sin(phi)
        z = np.zeros(self._N,dtype=float)
        return x,y,z

    def init_velocities(self):
        p = self.p
        phi = p.phi
        r = p.r
        a_dot_r = np.zeros_like(r)
        for i in range(30):
            delta = self._rss.uniform(-np.pi,np.pi,self._N)
            p2 = Particle(r*np.cos(delta+phi),r*np.sin(delta+phi),p.z,p.vx,p.vy,p.vz)
            ax,ay,az = self.get_acc(p2)
            a_dot_r += -(ax*p2.x + ay*p2.y + az*p2.z)/30
        del p2

        v_mag = np.sqrt(np.abs(a_dot_r))
        alpha = phi - np.pi/2
        p._vx, p._vy, p._vz = v_mag*np.cos(alpha), v_mag*np.sin(alpha), np.zeros_like(r)
        self._vx, self._vy, self._vz = v_mag*np.cos(alpha), v_mag*np.sin(alpha), np.zeros_like(r)
        #return alpha,phi,v_mag
        r_pl, v_pl, ravg, sigma_r, sigma_theta, sigma_z = self.get_sigmas()

        self._sigmar = sigma_r
        self._sigmaphi = sigma_theta
        self._sigmaz = sigma_z
        self._ravg = ravg
        self._rpl = r_pl
        self._vpl = v_pl

        print("Finished initializing velocities, outermost particle has an orbital period of {:.3f}".format(self.tau))

    def create_particles(self,p):
        ps = []
        for i in range(self._N):
                 ps.append(Particle(p.x[i],p.y[i],p.z[i],p.vx[i],p.vy[i],p.vz[i]))
        return ps

    def get_acc(self,p):
        N = self._N
        p2 = self.create_particles(p)
        ax = np.array([p2[i].gx(p,self._c) for i in range(N)])
        ay = np.array([p2[i].gy(p,self._c) for i in range(N)])
        az = np.array([p2[i].gz(p,self._c) for i in range(N)])
        del p2
        ax = np.array([np.sum(ax[i]) for i in range(N)])
        ay = np.array([np.sum(ay[i]) for i in range(N)])
        az = np.array([np.sum(az[i]) for i in range(N)])
        return ax + p.ghalo_x(self._Mh,self._R), ay + p.ghalo_y(self._Mh,self._R), az + p.ghalo_z(self._Mh,self._R)

    def get_sigmas(self, sigma = 0.454):
        N = self._N
        p = self._p
        r = p.r
        r_edges = self.r_edges
        v_mag = np.sqrt(p.vx**2 + p.vy**2)
        r_pl, v_pl = [],[]
        for i in range(len(r_edges)-1):
            ri,rf = r_edges[i],r_edges[i+1]
            m = (r > ri) * (r < rf)
            r_pl.append(np.mean(r[m]))
            v_pl.append(np.mean(v_mag[m]))

        vavg,ravg = [],[]
        r_bins = self.r_bins
        for i in range(self._n):
            ri,rf = r_bins[i],r_bins[i+1]
            m = (r > ri) * (r < rf)
            vavg.append(np.mean(v_mag[m]))
            ravg.append(np.mean(r[m]))

        vavg,ravg = np.array(vavg),np.array(ravg)
        dlnvavg = np.diff(np.log(vavg))
        dr = np.diff(ravg)
        r2avg = (ravg[1:]+ravg[:-1])/2.
        dd = dlnvavg/dr*r2avg
        factor = (1+dd)**(1/2)
        factor = np.append(factor,2*factor[-1])
        nu = vavg*factor

        sig_rT = sigma*N/nu
        sig_rS = 0.4*nu
        sigma_r = np.minimum(sig_rT,sig_rS)
        sigma_theta = sigma_r/2**(1/2)*factor
        sigma_z = sigma_theta

        return r_pl, v_pl, ravg, sigma_r, sigma_theta, sigma_z

    def add_dispersion(self):
        p = self.p
        r = p.r
        v_mag = p.vmag
        r_e = self.r_bins
        v_rx, v_ry, v_thx, v_thy = np.zeros_like(r),np.zeros_like(r),np.zeros_like(r),np.zeros_like(r)
        v_z = p.vz
        for i in range(len(self._sigmar)):
            m = (r > r_e[i])*(r < r_e[i+1])
            ni = np.sum(m)
            v_r = self._rss.normal(0, self.sigmar[i], ni)
            v_rx[m] = v_r*p.rhat_x[m]
            v_ry[m] = v_r*p.rhat_y[m]

            v_th = self._rss.normal(0, self.sigmaphi[i], ni)
            v_thx[m] = np.abs(v_th)*p.that_x[m]
            v_thy[m] = np.abs(v_th)*p.that_y[m]

            v_z[m] = self._rss.normal(0,self.sigmaz[i],ni)
        vx_new = p.vx + v_rx + v_thx
        vy_new = p.vy + v_ry + v_thy
        vz_new = v_z

        v_disp = np.sqrt(vx_new**2 + vy_new**2 + vz_new**2)
        v_new = np.zeros_like(r)
        for j in range(self._n):
            m = (r > r_e[j])*(r < r_e[j+1])
            ke_old = np.sum(v_mag[m]**2)
            ke_new = np.sum(v_disp[m]**2)
            f = np.sqrt(ke_old/ke_new)
            v_new[m] = v_disp[m]*f
        vx = vx_new/v_disp*v_new
        vy = vy_new/v_disp*v_new
        vz = vz_new/v_disp*v_new
        p._vx, p._vy, p._vz = vx, vy, vz

        print("Finished adding dispersion, outermost particle has an orbital period of {:.3f}".format(self.tau))



    def sim(self,tau_max,dt,outdir='disk_out/'):
        t = 0
        tau = self.tau

        N = int(tau_max*tau/dt)+1
        scalars = np.zeros(N, dtype=[('t','<f8'), ('tau','<f8'), ('zrms', '<f8'), ('KE', '<f8'), ('PE', '<f8'), ('tmean', '<f8')])
        print("Starting sim...")
        for i in range(N):
            p = self.p
            ax,ay,az = self.get_acc(p)

            k1vx, k1vy, k1vz = ax*dt, ay*dt, az*dt
            k1x, k1y, k1z = p.vx*dt, p.vy*dt, p.vz*dt

            p2 = Particle(p.x+k1x/2,p.y+k1y/2,p.z+k1z/2,p.vx,p.vy,p.vz)
            a2x,a2y,a2z = self.get_acc(p2)

            k2vx, k2vy, k2vz = a2x*dt, a2y*dt, a2z*dt
            k2x, k2y, k2z = (p.vx + k1vx/2)*dt, (p.vy + k1vy/2)*dt, (p.vz+k1vz/2)*dt

            p._vx, p._vy, p._vz = p.vx + k2vx, p.vy + k2vy, p.vz + k2vz
            p._x, p._y, p._z = p.x + k2x, p.y + k2y, p.z + k2z

            del p2
            t += dt
            scalars['t'][i] = t
            scalars['tau'][i] = t/tau
            scalars['zrms'][i] = self.zrms
            scalars['KE'][i] = self.totKE
            scalars['PE'][i] = self.totPE
            scalars['tmean'][i] = self.tmean

            if (i%10==0):
                print("Saving output at t = {:.3f}, tau = {:.3f}".format(t,t/tau))
                result = np.zeros(len(x),dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('vx', '<f8'),
                                 ('vy', '<f8'), ('vz', '<f8')])
                result['x'] = p.x
                result['y'] = p.y
                result['z'] = p.z
                result['vx'] = p.vx
                result['vy'] = p.vy
                result['vz'] = p.vz
                fname = outdir+"disk_t_{:.3f}_R_{:.1f}_c_{:.2f}_Mh_{:.0f}_N_{:.0f}".format(t,self._R,self._c,self._Mh,self._N)
                np.save(fname,result)
        print("Finished! Writing scalars...")
        fname = outdir+"scalars_R_{:.1f}_c_{:.2f}_Mh_{:.0f}_N_{:.0f}".format(self._R,self._c,self._Mh,self._N)
        np.save(fname,scalars)
