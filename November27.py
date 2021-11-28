# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 18:47:20 2021

@author: marcs
"""

import numpy as np
from dataclasses import dataclass
import time
import plotly.graph_objs as go
import random
import matplotlib.pyplot as plt
import csv 
import timeit, functools
import winsound

from scipy.optimize import minimize 

import scipy




#%% Definitive Classes
@dataclass
class Atom:
    def __init__(self,sct,lmbds,m_uma):
        self.h=6.62607004e-34
        self.scatEff = sct
        self.lambdas = np.array(lmbds)
        self.k=2*np.pi/self.lambdas[0]
        self.m = m_uma * 1.6605402e-27 
        self.recoils=self.h/(self.m*self.lambdas)
        self.directions = np.concatenate( (np.eye(3) , -np.eye(3)) , 0)
                 
Magnesium = Atom(sct=2e4,lmbds=[457e-9,462e-9,634e-9,285e-9],m_uma=24.305)

Calcium = Atom(sct=3e4,lmbds=[657.5e-9,452.7e-9,732.8,422.8],m_uma=40.078)

class MonteCarlo:
    def __init__(self,atom,N=100,timesteps=100,deltaL=None,sigmar=5e-4,sigmav=0.81,dt=20e-6,Omega12=5e5,Magnetic_gradient=np.array([5,5,5]),waist= 2e-3):
        self.atom=atom
        self.mag= 8.8e8 * Magnetic_gradient
        self.Omega12 = Omega12
        self.dt=dt
        self.sigmar=sigmar
        self.sigmav=sigmav
        self.N=N
        
        self.timesteps=timesteps
        self.waist=waist
        self.probcycle = 1- np.exp(-self.dt*self.atom.scatEff)
        self.directions = np.concatenate( (np.eye(3) , -np.eye(3)) , 0)
        
        self.deltaL=deltaL
        
        
        self.mean_measure=[('module',None),('mean',None)]
        self.temperature_measure=[('module',None),('temperature',None),('multiply',1000)]
        
        
        self.reset()
        
    def reset(self):
        # self.insideMask=np.ones(self.N,dtype=bool)
        # self.insideMaskMemory=np.ones((self.timesteps,self.N),dtype=bool)
        self.X=np.random.normal(0,self.sigmar,(self.N,3))
        self.V=np.random.normal(0,self.sigmav,(self.N,3))
        self.memoryX=np.zeros((self.timesteps,self.N,3))
        self.memoryV=np.zeros((self.timesteps,self.N,3))
        self.T=0
    
    def changeReset(self,N,timesteps,dL):
        self.deltaL=dL
        self.N=N
        self.timesteps=timesteps
        self.reset()
    
    def rdir2(self):
        A = np.einsum('ij,ij,jk->ik',self.X,self.X,1-np.eye(3))
        return np.concatenate((A,A),axis=1)
    def kv(self):
        return self.atom.k * np.einsum('ij,kj->ik',self.V,self.directions)
    
    def bx(self):
        return np.einsum('ij,kj,j->ik',self.X,self.directions,self.mag)
    
    def rho22(self):
        numerator  = self.Omega12**2*np.exp(-self.rdir2()/self.waist**2)
        denominator = 2*self.Omega12**2*np.exp(-self.rdir2()/self.waist**2)
        denominator += self.atom.scatEff**2
        denominator += (self.deltaL[self.T]+self.kv()+self.bx())**2
        return numerator/denominator
    
    def randomChoices(self):
        probs = np.cumsum(self.rho22(), axis=1) + 1e-15
        probs /= np.tile(probs[:,-1],(6,1)).transpose()
        choices = np.random.uniform(0,1,self.N)  
        choices = np.tile(choices,(6,1)).transpose()
        choices -= probs
        choices = choices>0
        return np.sum(choices,axis=1)
    
    def RandomDirection(self):
        alpha = np.random.uniform(0,2*np.pi,self.N)
        beta = np.random.uniform(0,np.pi,self.N)
        return np.array([np.sin(beta)*np.cos(alpha),np.sin(beta)*np.sin(alpha),np.cos(beta)]).transpose()
    
    def cycle(self):
        d=self.randomChoices()
        cyclehappens = np.random.choice(a=[False, True], size=(self.N,), p=[1-self.probcycle, self.probcycle])
        #First 2 recoils
        self.V += np.einsum('ij,i->ij',self.directions[d], cyclehappens) * self.atom.recoils[0]
        self.V += np.einsum('ij,i->ij',self.directions[d], cyclehappens) * self.atom.recoils[1]
        
        #Second 2 recoils
        self.V += np.einsum('ij,i->ij',self.RandomDirection(), cyclehappens) *self.atom.recoils[2]
        self.V += np.einsum('ij,i->ij',self.RandomDirection(), cyclehappens) *self.atom.recoils[3]
    

    #This function describes the time evolution
    def evolve(self,N,ts,dL):
        #Here we set the parameteres and reset our distributions     
        self.changeReset(N,ts,dL)
        
        
        #We are going to make time evolve for a given amount of timesteps
        while self.T<self.timesteps:
            #Here we record the postion and velocity of our system
            self.memoryX[self.T]=self.X
            self.memoryV[self.T]=self.V
            # self.insideMaskMemory[self.T] = self.insideMask
            #We will assume that the recoil process happens at half the timestep
            self.X += self.V  * self.dt/2
            
            #The recoils happen according to the semiclassical model
            self.cycle()
            #The trajectory evolves (Note that the biggest recoil is of 0.2 m/s which on average happens every 3 cycles)
            #The total error in position using this method is error = ts*dt*0.2=T*0.2 or about 0.4mm
            self.V += self.directions[-1] * 9.81 * self.dt
            self.X += self.V  * self.dt/2
            # self.insideMask &= (np.abs(self.X)<self.waist).all(axis=1)
            
            #Here we make time evolve (used to keep track of deltaL)
            self.T += 1
        
        self.mask_inside = self.mask_in()

    def mask_in(self):
        out=np.abs(self.memoryX)>self.waist
        out=out.any(axis=2)
        out=out.cumsum(axis=0,dtype=bool)
        return np.logical_not(out)
    
    def efficiency(self):
        return  self.mask_in().mean(axis=1)
    
    def derivative_efficiency(self,w=5):
        diffeff=(self.efficiency()[:-1]-self.efficiency()[1:])/self.dt
        smooth_diffeff=np.convolve(diffeff, np.ones(w), 'valid') / w
        return smooth_diffeff
        
        
    #Getting the measurables of the particles that have been inside the box at all times
    def get(self,quantity,t=-1):
            return getattr(self,'memory'+quantity)[:,self.mask_inside[t],:]

    #Here we will need to pass a series of concatenated mesures and parameters
    def observable(self,quantity,measures_params,t=-1):
        result=self.get(quantity,t)
        measures_params_get=[(getattr(self,measure[0]) , measure[1]) for measure in measures_params]
        for measure,param in measures_params_get:
            result=measure(result,param)
        return result
    
    #First Layer to reduce dimensions
    @staticmethod
    def module(quantity,param):#Returns module over 3 dimensions
        return np.linalg.norm(quantity,axis=2)
    @staticmethod
    def axis(quantity,param):
        ax={'x':0,'y':1,'z':2}
        return quantity[:,:,ax[param]]
    @staticmethod
    def maxaxis(quantity,param):
        return quantity.max(axis=2)
    @staticmethod
    def minaxis(quantity,param):
        return quantity.min(axis=2)
    
    #Second layer to reduce particles
    
    def temperature(self,quantity,param):   
        kb=1.38e-23
        return (self.atom.m*quantity**2)/(3*kb)
    
    @staticmethod 
    def mean(quantity,param):
        return quantity.mean(axis=1)
    
    @staticmethod
    def std(quantity,param):
        return quantity.std(axis=1)
    
    @staticmethod 
    def quantile(quantity,param):
        return np.quantile(quantity,param,axis=1)
    
    #Third layer to get particular times
    @staticmethod
    def timestep(quantity,param):
        return quantity[param]
    @staticmethod
    def multiply(quantity,param):
        return quantity*param
    
    
    
    def Velocity(self):
        return self.observable('V',self.mean_measure)
    
    
    #Optimization
    # def heuristic_dL(self,measure_params_V=None,measure_params_X=None):
    #     if (measure_params_V,measure_params_X)==(None,None):
    #         measure_params_V=[('module',None),('mean',None)]
    #         measure_params_X=[('module',None),('mean',None)]
        
    #     heuristic = np.zeros(self.timesteps)
    #     for t in range(self.timesteps):
    #         heuristic[t]=self.atom.k * self.observable('V',measure_params_V + [('timestep',t)],t=t)  + self.mag[0] * self.observable('X',measure_params_X + [('timestep',t)] ,t=t)
        
    #     return heuristic
     
    def heuristic_dL(self,measure_params_V=None,measure_params_X=None):
        if (measure_params_V,measure_params_X)==(None,None):
            measure_params_V=[('module',None),('mean',None)]
            measure_params_X=[('module',None),('mean',None)]
        v=self.observable('V',measure_params_V)
        x=self.observable('X',measure_params_X)
        return self.atom.k * v + self.mag[0] * x    
    
        
        
    
            


#%%General Plots


#Optimization
class Optimizer:
    def __init__(self,monty_opt):
        self.monty_opt = monty_opt
        self.erase_memory()
        #self.firstGuess()
    
    
        
    def dic(self):
        d={'method':self.method,'Parameters':self.parameters,'Efficiency':self.efficiency,'Velocity':self.velocity}
        return d
    
    def erase_memory(self):
        self.method= []
        self.parameters=[]
        self.efficiency=[]
        self.velocity=[]
        
    def firstGuess(self):
        first_heuristic=np.linalg.norm(self.monty_opt.V,axis=1).mean() * self.monty_opt.atom.k
        self.monty_opt.evolve(N=self.monty_opt.N,ts=self.monty_opt.timesteps,dL=np.full(self.monty_opt.timesteps,first_heuristic))
        
        self.method= ['constant']
        self.parameters=[first_heuristic]
        self.efficiency=[self.monty_opt.efficiency()[-1]]
        self.velocity=[self.monty_opt.Velocity()[-1]]
    
    def reset(self):
        self.erase_memory()
        self.firstGuess()
        
    
    @staticmethod
    def linear_dL(x,dLi,dLf,T):
        b=x<T
        slope=(dLf-dLi)/T
        return b * (dLi+slope*x) + (1 - b) * dLf
    @staticmethod
    def twostep_dL(x,dLi,dLp,dLf,T1,T2,T3):
        a=x<=T1
        b=np.logical_and( x>T1, x<=T2)
        c=np.logical_and( x>T2, x<=T3)
        d=x>T3
        slope1=(dLp-dLi)/T1
        slope2=(dLf-dLp)/(T3-T2)
        
        return a*(dLi+x*slope1)+ b*dLp + c*(dLp+(x-T2)*slope2) + d*dLf
    
    def guess(self,function):
        T1,T2,T3=80,115,150
        dLi,dLp,dLf=self.monty_opt.heuristic_dL()[0],self.monty_opt.heuristic_dL()[T1],self.monty_opt.heuristic_dL()[-1]
        
        if function=='twostep_dL':
            return dLi,dLp,dLf,T1,T2,T3
        else:
            return dLi,dLf,T1
    
    def find_parameters(self,measure_params_V,measure_params_X,function):
        heuristic_dL= self.monty_opt.heuristic_dL(measure_params_V=measure_params_V,measure_params_X=measure_params_X)
        times=np.arange(0,heuristic_dL.shape[0],1)
        f=getattr(self,function)
        return scipy.optimize.curve_fit(f,times,heuristic_dL,p0=self.guess(function))[0]
    
    def plotapproximations(self,measure_params_V=None,measure_params_X=None):
        full=self.monty_opt.heuristic_dL(measure_params_V,measure_params_X)
        times=np.arange(0,full.shape[0],1)
        dLi,dLf,T = self.find_parameters(measure_params_V,measure_params_X,'linear_dL')
        one=self.linear_dL(times,dLi,dLf,T)
        
        dLi,dLp,dLf,T1,T2,T3 = self.find_parameters(measure_params_V,measure_params_X,'twostep_dL')
        two=self.twostep_dL(times,dLi,dLp,dLf,T1,T2,T3)
        
        errorone=np.abs(one-full).sum()/times[-1]
        errortwo=np.abs(two-full).sum()/times[-1]
        print('The two approximations have the corresponding errors',errorone,errortwo)
        
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=times,y=one ,name='linear_dL'))
        fig.add_trace(go.Scatter(x=times,y=two ,name='twostep_dL'))
        fig.add_trace(go.Scatter(x=times,y=full,name='heuristic_dL'))
        fig.write_image("plots/approximations.svg")
        fig.show()
        
    
    def optimize_dL(self,iterations,measure_params_V,measure_params_X,function='twostep_dL'):
        times=np.arange(0,self.monty_opt.timesteps,1)
        f=getattr(self,function)
        for i in range(iterations):
            
            if function=='twostep_dL':
                dLi,dLp,dLf,T1,T2,T3 = self.find_parameters(measure_params_V,measure_params_X,function)
                params=np.array([dLi,dLp,dLf,T1,T2,T3])
                new_dL= f(times,dLi,dLp,dLf,T1,T2,T3)
            else:
                dLi,dLf,T =self.find_parameters(measure_params_V,measure_params_X,function)
                params=np.array([dLi,dLf,T])
                new_dL= f(times,dLi,dLf,T)
            
            
            self.monty_opt.evolve(self.monty_opt.N,self.monty_opt.timesteps,dL=new_dL)
            self.method += [function]
            self.parameters+=[params]
            self.efficiency+=[self.monty_opt.efficiency()[-1]]
            self.velocity+=[self.monty_opt.Velocity()[-1]]   
            
        return max(self.efficiency)
    
    
    def exhaustive_tries(self, list_percentiles, list_deviations,approx_function='twostep_dL'):
        self.results = np.zeros((len(list_deviations),len(list_percentiles),3))
        for d in range(len(list_deviations)):
            self.monty_opt.sigmav=list_deviations[d]
            for p in range(len(list_percentiles)):
              print(list_deviations[d],list_percentiles[p]) 
                
              measure=[('module',None),('quantile',list_percentiles[p])]
              self.reset()
              self.optimize_dL(7,measure,measure,approx_function)
              
              best_iteration=self.efficiency.index(max(self.efficiency))
              
              self.results[d,p,0]=self.efficiency[best_iteration]
              self.results[d,p,1]=best_iteration
              self.results[d,p,2]=self.velocity[best_iteration]
             
              
        return 

    
if __name__=="__main__":
    print('MAIN')
    start_time = time.time()
    Magnesium=Atom(sct=2e4,lmbds=[457e-9,462e-9,634e-9,285e-9],m_uma=24.305)
    N,ts=1000,1001
    sigmas=np.linspace(0.8,1.5,num=20)
    percentiles=np.linspace(0.5,0.8,num=10)
    
    monty=MonteCarlo(atom=Magnesium,N=N,timesteps=ts,sigmav=1,Magnetic_gradient=np.array([5,5,2.5]))
    optimizer = Optimizer(monty) 
    
    optimizer.exhaustive_tries(percentiles, sigmas)
    
    eff_vs_sigmas = optimizer.results[:,:,0].max(axis=1)
    
    plt.plot(sigmas,eff_vs_sigmas)
    plt.show()
    

    
    
    
    # for sigmav in sigmas:
    #     print(f"Sigmav {sigmav} running... t={time.time()-start_time}")
    #     results=[]
    #     for p in percentiles:
    #         print(f"Percentile {p} running... t={time.time()-start_time}")
    #         measure=[('module',None),('quantile',p)]
    #         monty=MonteCarlo(atom=Magnesium,N=N,timesteps=ts,sigmav=sigmav,Magnetic_gradient=np.array([5,5,2.5]))
    #         optimizer = Optimizer(monty) 
    #         result=optimizer.optimize_dL(10,measure,measure,'twostep_dL')
    #         results += [result]
            
    #     plt.plot(percentiles,results)
    #     plt.title('sigmav' + str(sigmav))
    #     plt.show()
    
    
    print(f"Total time = {int((time.time()-start_time)//60)} min {(time.time()-start_time)%60}")
    # t = timeit.Timer(functools.partial(monty.evolve,N,ts,dL)) 
    # print(t.timeit(10))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
