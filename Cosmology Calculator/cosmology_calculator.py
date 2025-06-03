import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class cosmos_calc():
    
    def __init__(self,parameters,*,H0=0.070,a0=1,dt=0.1,T_back=20,T_forward=20,c=300):
        '''
        Input for parameters: radiation, mass, lambda.
        H0 is hubble constant right now (in /Gyr), a0 is scale factor right now, 
        dt is timestep for integration, 
        T_back is the maximum time to calculate going back in time (in Gyr), 
        T_forward is the maximum time to calculate going forward in time (in Gyr),
        c is the speed of light in units Mpc/Gyr
        '''
        self._radiation = parameters[0]
        self._mass = parameters[1]
        self._lambda = parameters[2]
        self._H0=H0
        self._a0=a0
        self._dt=dt
        self._T_back=T_back
        self._T_forward=T_forward
        self._c=c
        
        #Omega0
        self._omega0=self._radiation+self._mass+self._lambda

        #definition for radius of curvature in nonflat universe
        if self._omega0!=1:
            self._R0=np.sqrt(self._c**2/(self._H0**2 * abs(1-self._omega0)))

        #defining an array of times:
        # tau: eg: 0->20 where tau is the time going back
        # t: eg: 0->20 where t is the time going forward
        # flipped tau: eg: -20->0
        # t_axis: eg: -20->20  
        self._tau = np.arange(0,self._T_back,self._dt) #goes from 0 to T_back
        self._t = np.arange(0,self._T_forward,self._dt) #goes from 0 to T_forward
        self._flipped_tau=-1*np.flip(self._tau)  #goes from -T_back to -dt
        self._t_axis = np.concatenate([self._flipped_tau,self._t[1:]]) 
            #goes from -T_back to T_forward with 0 being now
            #[1:] is there to remove double counting the 0

    def scale_factor(self,*,time=None,z=None):
        '''
        Computs the scale factor of the Universe at a given time/redshift

        Input: 
        If empty, returns a list. 
        If given time/z, returns the scale factor at that value of time/z

        output: 
        For empty input: scale factor across all time
        For a given time/z: scale factor at that time/z
        '''
        def _scale_factor_dot_tau(a,tau): #Equation of motion for scale factor in tau
            adot2 = self._H0**2*(self._radiation/a**2 + self._mass/a + self._lambda*a**2 + (1-self._omega0))
            return -1*np.sqrt(adot2) #integration is negative since t=t0-tau so dt=-dtau        
        def _scale_factor_dot_t(a,t): #Equation of motion for scale factor in t
            adot2 = self._H0**2*(self._radiation/a**2 + self._mass/a + self._lambda*a**2 + (1-self._omega0))
            return np.sqrt(adot2) #positive integration


        #Integrate the equation of motion to get scale factor
        scale_factor_tau=odeint(_scale_factor_dot_tau,self._a0,self._tau) 
        scale_factor_t=odeint(_scale_factor_dot_t,self._a0,self._t)

        #Convert into desired shape 
        self._scale_factor_tau=scale_factor_tau.flatten('F') 
        self._scale_factor_t=scale_factor_t.flatten('F')

        #Combine them into a array across all times (eg. -13 -> +13)
        flipped_tau=np.flip(scale_factor_tau)
        scale_factor=np.concatenate([flipped_tau.flatten(),self._scale_factor_t[1:]])

        #Arguments to decide the output based on the input

        #Empty input
        if time is None and z is None:
            return scale_factor.flatten('F')

        #If the input is time
        elif time is not None and z is None:
            #If the time input is out of range
            if time>self._T_forward or time< (-1)*self._T_back:
                print("time value out of range")
            else:
                index=self._get_index(time)
                return scale_factor[index]

        #If the input is z: use equation 1+z = 1/a
        elif time is None and z is not None:
            return (1/(1+z))

        #Prevent having input of both time and z
        elif time is not None and z is not None:
            print('Please only input time OR redshift, not both')
        
    
    def z(self,time=None):
        '''
        Obtains the redshift given the scale factor equation 1+z = 1/a.
        Empty input: returns array of redshifts that corresponds to the t_axis.
        Time input: returns redshift at a given time.
        '''
        if time is None:
            return 1/self.scale_factor()-np.array([1]*len(self.scale_factor()))
        else:
            return 1/self.scale_factor(time=time)-1
            
    def _get_index(self,value,type='t'):
        '''
        For a given value, finds the closest value of that inside the t_axis/scale factor list 
        and returns the index. For example, I want to know the position (index) of the value 
        -10 Byrs in the t_axis list or scale factor of 0.5 in the scale factor list.

        If type is t (default):
            Get the index of any given input time in the t_axis list 
        If type is z:
            1. Calculates the scale factor (A) based on that z value
            2. Finds the closest corresponding (A) value in the scale factor list and the index
            3. Returns index
        '''
        if type == 't':
            #round the input time to the nearest decimal based on dt
            rounded_value = round(value, int(np.log10(1/self._dt))) 
            #returns the calculated index using index = time/dt - (first value in the t_axis)/dt
            return int(rounded_value/self._dt-self._t_axis[0]/self._dt) 
        elif type == 'z':
            a = self.scale_factor(z=value) #scale factor at redshift z (input)
            z_approx, z_index = self._closest(a,self.scale_factor()) #Calls function to find the index for z
            return z_index 
        else:
            print('Please specify type="z" for redshift or leave it blank for time')
        
    def _closest (self,num, arr):
        '''
        Given a number and an array, calculates the closest value in the array to that number
        '''
        initial=self._get_index(0) #Find the index for t=0 in t_axis
        curr = arr[initial]  # we want to start the loop at t=0 to avoid nan
        index = 0
        # Loop over every value in the list to find the smallest difference between
        # the number in the list and the input value.
        for val in range(len(arr)): 
            if abs (num - arr[val]) < abs (num - curr):
                curr = arr[val]
                index = val
        return curr,index
       
    
    def D_proper(self,t_em=None,z=None): 
        '''
        Calculates the proper distance/line of sight distance in Mpc. 
        Empty input: returns an array corresponding to flipped_tau (-backtime -> now). 
                        (afterall it doesn't make sense to calculate distance to objects in the future)
        t_em input: returns the proper distance to an object that was emitted at t_em
        z input: returns the proper distance to an object with redshift of z
        '''
        # Empty input
        if t_em is None and z is None:
            index_at_t0 = self._get_index(0) #index of t=0 in t_axis
            d_proper = np.zeros(index_at_t0+1) #array of objects emitted at every timestep 
            for i in range(len(d_proper)): #Loop over each object/timestep
                #Integrate from emission time to t=0
                d_proper[i] = self._c*np.trapz(1/self.scale_factor()[i:index_at_t0],
                                            self._t_axis[i:index_at_t0])
        # t_em input
        elif t_em is not None and z is None:
            if t_em > 0: #Can't input positive time (future time)
                print("Please put a number between","(age of universe)","and 0 for time of emission")
            else:
                #Integrate from emission time to t=0
                d_proper = self._c*np.trapz(1/self.scale_factor()[self._get_index(t_em):self._get_index(0)],
                                            self._t_axis[self._get_index(t_em):self._get_index(0)])
        # z input
        elif t_em is None and z is not None:
            a_index = self._get_index(z,type='z') #Find the corresponding scale factor (index) given z
                #Integrate from scale factor at z to scale factor at t=0
            d_proper = self._c*np.trapz(1/self.scale_factor()[a_index:self._get_index(0)],
                                            self._t_axis[a_index:self._get_index(0)])
        # Can't have both t_em and z inputs.
        elif t_em is not None and z is not None:
            print("Please enter only t_em or z. Not Both!")
        return (d_proper)
    
    def D_trans(self,t_em=None,z=None): 
        '''
        Calculates the transversal distance.
        Empty input: returns an array corresponding to flipped_tau (-backtime -> now)
        t_em input: returns the transversal distance to an object that was emitted at t_em
        z input: returns the transversal distance to an object with redshift of z
        '''
        #First define d_p based on the input parameter
        #t_em input
        if t_em is not None and z is None:
            d_p = self.D_proper(t_em=t_em)
        #z input
        elif t_em is None and z is not None:
            d_p = self.D_proper(z=z)
        #prevent both t_em and z input
        elif t_em is not None and z is not None:
            print("Give value for t_em or z, not both!")
        #Empty input
        else:
            d_p = self.D_proper()

        #Account for the different cases for positive/0/negative curvatures:
        #0 curvature: 
        if self._omega0==1:
            D_trans=d_p
        #Negative (1-omega0) -> positive Kappa -> sin
        elif self._omega0>1:
            D_trans=self._R0*np.sin(d_p/self._R0)
        #Positive (1-omega0) -> negative Kappa -> sinh
        elif self._omega0<1:
            D_trans=self._R0*np.sinh(d_p/self._R0)
        return D_trans
    
    def D_A(self,t_em=None,z=None):
        '''
        Calculates D_A. Input format same as D_proper and D_trans
        '''
        if t_em is not None and z is None:
            return self.D_trans(t_em=t_em)/(1+self.z(time=t_em))
        elif t_em is None and z is not None:
            return self.D_trans(z=z)/(1+z)
        elif t_em is None and z is None:
            return self.D_trans()/(1+self.z()[:self._get_index(0)+1])
        else:
            print("Give value for t_em or z, not both!")
        
            

    def D_L(self,t_em=None,z=None):
        '''
        Calculates D_L. Input format same as D_proper and D_trans
        '''
        if t_em is not None and z is None:
            return self.D_trans(t_em=t_em)*(1+self.z(time=t_em))
        elif t_em is None and z is not None:
            return self.D_trans(z=z)*(1+z)
        elif t_em is not None and z is not None:
            print("Give value for t_em or z, not both!")
        else:
            return self.D_trans()*(1+self.z()[:self._get_index(0)+1])

    def Hubble_constant(self,t_em=None):
        '''
        Calculates the Hubble Constant using Friedman equation
        '''
        if t_em is not None:
            a=self.scale_factor(time=t_em)
            print(a)
            H=self._H0*np.sqrt(self._radiation/a**4 + self._mass/a**3 + self._lambda + (1-self._omega0)/a**2)
        else:
            a=self.scale_factor()
            H=self._H0*np.sqrt(self._radiation/a**4 + self._mass/a**3 + self._lambda + (1-self._omega0)/a**2)
        return H
    

    def plot_scale_factor(self,xaxis):
        if xaxis=='t':
            plt.plot(self._t_axis,self.scale_factor())
            plt.xlabel('time (Gyr)')
            plt.ylabel('scale factor a')
        if xaxis=='z':
            plt.plot(self.z(),self.scale_factor())
            plt.xlabel('redshift z')
            plt.ylabel('scale factor a')
        else:
            print("please enter 't' for plotting against time or 'z' for plotting against redshift ")