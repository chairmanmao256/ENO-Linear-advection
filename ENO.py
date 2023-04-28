import numpy as np

class ENO_advection:
    '''
    ### Description

    The class to solve the 1-D linear advection equation using ENO 
    reconstruction scheme of order 3.
    '''
    def __init__(self, a = 1.0, ni = 100, cfl = 0.5):
        '''
        ### Description

        Initialize the ENO_advection class with:

        `a`: The convection speed. Default value is 1.0.
        `ni`: The number of internal cells. Default value is 100.
        '''

        self.ib = 2 # third order scheme requires 2 ghost cells
        self.ni = ni
        self.im = ni + self.ib # to iterate through all the internal cells: for i in range(ib, im)
        self.a = a
        self.cfl = cfl
        delx = 1.0 / float(ni)
        self.delx = delx
        self.dt = cfl * delx / a

        self.u = np.zeros(2 * self.ib + ni) # solution vector
        self.u_m = np.zeros_like(self.u) # stores the middle step (previous solution) in Runge-Kutta scheme
        
        delx = 1.0 / float(ni)
        self.xc = np.linspace(-3.0/2.0 * delx - 0.5, 0.5 + 3.0/2.0 * delx, 2 * self.ib + ni)
        self.xface = np.linspace(-0.5 - 2 * delx, 0.5 + 2*delx, 2 * self.ib + ni + 1)

        # interface values
        self.ul = np.zeros(2 * self.ib + ni+1)
        self.ur = np.zeros(2 * self.ib + ni+1)
        self.flux = np.zeros(2 * self.ib + ni+1)

        # Newton's devided difference
        self.V = np.zeros((2 * self.ib + ni + 1, 3)) 

    def set_initial(self):
        '''
        ### Description

        Set the initial condition
        '''
        self.u = (self.xc < -0.25).astype(np.int32) * 0.0 + (self.xc >= -0.25).astype(np.int32) *\
                 (self.xc <= 0.25).astype(np.int32) * 1.0 + (self.xc >  0.25).astype(np.int32) * 0.0
        
    def set_bell(self):
        '''
        ### Description

        Sete the smooth initial condition
        '''
        self.u = self.xc**2

    def set_ghost(self):
        '''
        ### Description

        Set the value in the boundary ghost cells
        '''
        # left boundary
        self.u[self.ib-2] = self.u[self.im-2]
        self.u[self.ib-1] = self.u[self.im-1]

        # right boundary
        self.u[self.im] = self.u[self.ib]
        self.u[self.im+1] = self.u[self.ib+1]

    def NDD(self):
        '''
        ### Description

        Compute the Newton Divde Difference of the function value array.
        Function value array is stored on the cell centers.

            +--------+
            |        |
        i> |  u[i]  | <i+1
            |        |
            +--------+

        ib is the starting index of the internal cells. It also equals to
        the ghost cells used.

        We only compute the the Diveded Difference to the order 3: 
        V[x_{i-1/2}, x_{i+1/2}, x_{i+3/2}, x_{i+5/2}]

        We also assume uniform cell length: \Delta x = const.
        '''
        nn = self.u.shape[0] # total number of cells, including the ghost cells

        self.V = np.zeros((nn+1, 3))
        self.V[0:nn, 0] = self.u.copy()                                # order-1: V[x_{i-1/2}, x_{i+1/2}]
        self.V[0:nn-1, 1] = self.V[1:nn, 0] - self.V[0:nn-1, 0]        # order-2: V[x_{i-1/2}, x_{i+1/2}, x_{i+3/2}]
        self.V[0:nn-2, 2] = self.V[1:nn-1, 1] - self.V[0:nn-2, 1]      # order-3: V[x_{i-1/2}, x_{i+1/2}, x_{i+3/2}, x_{i+5/2}]

    def ENO_weight(self, r: int):
        '''
        ### Description:

        Compute the ENO weight based on the left shift of the stencil
        '''
        crj = np.zeros(3)
        for j in range(3):
            #crj[j]
            for m in range(j+1, 4):
                de = 1.0
                no = 0.0
                for l in range(4):
                    if l != m:
                        de = de * (m - l)
                
                for l in range(4):
                    if l != m:
                        ee = 1.0
                        for q in range(4):
                            if q != m and q != l:
                                ee *= (r - q + 1)
                        no += ee

                crj[j] += float(no)/float(de)

        return crj

    def ENO_reconstruction(self):
        '''
        ### Description:

        Perform ENO reconstruction cell-wise
        '''
        ib, im = self.ib, self.im
        # compute the NDD first
        self.NDD()

        # reconstruct on internal cell faces, cell by cell
        for i in range(ib, im):
            # initial stencil
            stencil = np.array([i, i+1])
            for k in range(2):
                L, R = stencil[0], stencil[-1]

                # determine the expanded stencil by evaluating NDD
                stencilL = np.append(L-1, stencil)
                stencilR = np.append(stencil, R+1)

                V2L = self.V[stencilL[0], k+1]
                V2R = self.V[stencilR[0], k+1]

                if abs(V2L) < abs(V2R):
                    stencil = stencilL.copy()
                else:
                    stencil = stencilR.copy()
        
            # final stencil is now stored in `stencil`. Evaluate the stencil shift.
            r = i - stencil[0]

            # obtain the ENO weight
            cL = self.ENO_weight(r)
            cR = self.ENO_weight(r-1)

            # obtain the cell-center values
            vv = self.u[stencil[0:-1]]

            self.ul[i+1] = cL @ vv
            self.ur[i] = cR @ vv

        # set the boundary state by using periodic condition
        self.ul[ib] = self.ul[im]
        self.ur[im] = self.ur[ib]

    def LAX_flux(self):
        '''
        ### Description

        Compute the L-F flux based on the reconstructed values
        '''
        self.flux = (self.ur + self.ul) * self.a / 2.0 - 0.5 * abs(self.a) * (self.ur - self.ul)

    def Runge_Kutta(self):
        self.u_m = self.u.copy()

        alpha1 = [1.0, 3.0/4.0, 1.0/3.0]
        alpha2 = [0.0, 1.0/4.0, 2.0/3.0]
        alpha3 = [1.0, 1.0/4.0, 2.0/3.0]

        for j in range(3):
            self.set_ghost()
            self.ENO_reconstruction()
            self.LAX_flux()
            self.u[self.ib:self.im] = alpha1[j] * self.u_m[self.ib:self.im] + alpha2[j] * self.u[self.ib:self.im] - \
                     alpha3[j] * self.dt/self.delx * (self.flux[self.ib+1:self.im+1] - self.flux[self.ib:self.im])
            
        return self.dt

