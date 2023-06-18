#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:19:16 2020

@author: alex
------------------------------------


Fichier d'amorce pour les livrables de la problématique GRO640'


"""

import numpy as np
import scipy as sp
from mpl_toolkits import mplot3d # for plotting and debug only
import matplotlib.pyplot as plt

from pyro.control  import robotcontrollers
from pyro.control.robotcontrollers import EndEffectorPD
from pyro.control.robotcontrollers import EndEffectorKinematicController


###################
# Part 1
###################

def dh2T( r , d , theta, alpha ):
    """

    Parameters
    ----------
    r     : float 1x1
    d     : float 1x1
    theta : float 1x1
    alpha : float 1x1
    
    4 paramètres de DH

    Returns
    -------
    T     : float 4x4 (numpy array)
            Matrice de transformation

    """

    
    T = np.zeros((4,4))

    T[0,0] = np.cos(theta)
    T[0,1] = -np.sin(theta)*np.cos(alpha)
    T[0,2] = np.sin(theta)*np.sin(alpha)
    T[0,3] = r*np.cos(theta)

    T[1,0] = np.sin(theta)
    T[1,1] = np.cos(theta)*np.cos(alpha)
    T[1,2] = -np.cos(theta)*np.sin(alpha)
    T[1,3] = r*np.sin(theta)

    T[2,1] = np.sin(alpha)
    T[2,2] = np.cos(alpha)
    T[2,3] = d

    T[3,3] = 1

    return T



def dhs2T( r , d , theta, alpha ):
    """

    Parameters
    ----------
    r     : float nx1
    d     : float nx1
    theta : float nx1
    alpha : float nx1
    
    Colonnes de paramètre de DH

    Returns
    -------
    WTT     : float 4x4 (numpy array)
              Matrice de transformation totale de l'outil

    """
    WTT = np.zeros((4, 4))

    for i in range(np.size(r)):
        T = dh2T(r[i], d[i], theta[i], alpha[i])
        if i == 0:
            WTT = T
        else:
            WTT = WTT@T
    
    
    return WTT


def f(q):
    """
    

    Parameters
    ----------
    q : float 6x1
        Joint space coordinates

    Returns
    -------
    r : float 3x1 
        Effector (x,y,z) position

    """
    r = np.zeros((3,1))

    DDL = len(q)
    DH  = np.zeros((DDL,4))

    # Defining DH parameters
    DH[0,:] = [0.033, 0.147, q[0], -np.pi/2]
    DH[1,:] = [0.155, 0, q[1]- (np.pi/2), 0]
    DH[2,:] = [0.135, 0, q[2], 0]
    DH[3,:] = [0, 0, q[3] + (np.pi/2), np.pi/2]
    DH[4,:] = [0, 0.217, q[4], 0]

    # Computing the transformation matrix from the base to the tool
    T_tool_world = dhs2T(DH[:,0], DH[:,1], DH[:,2], DH[:,3])

    r = T_tool_world[0:3, 3]

    

    
    return r


###################
# Part 2
###################
    
class CustomPositionController( EndEffectorKinematicController ) :
    
    ############################
    def __init__(self, manipulator ):
        """ """
        
        EndEffectorKinematicController.__init__( self, manipulator, 1)
        
        ###################################################
        # Vos paramètres de loi de commande ici !!
        ###################################################
        self.penalty = 1.0
        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback law: u = c(y,r,t)
        
        INPUTS
        y = q   : sensor signal vector  = joint angular positions      dof x 1
        r = r_d : reference signal vector  = desired effector position   e x 1
        t       : time                                                   1 x 1
        
        OUPUTS
        u = dq  : control inputs vector =  joint velocities             dof x 1
        
        """
        
        # Feedback from sensors
        q = y
        
        # Jacobian computation
        J = self.J( q )
        
        
        # Ref and objectives
        r_desired   = r
        r_actual    = self.fwd_kin( q )
        

        # Error
        e  = r_desired - r_actual

        # Effector space speed
        dr_r = e * self.gains # Given at gains before function call.
        
        ################
        dq = np.zeros( self.m )  # place-holder de bonne dimension
        
        ##################################
        # Votre loi de commande ici !!!
        ##################################
        dq = np.linalg.inv(J.T @ J + (self.penalty**2)*np.identity(3)) @ (J.T @ dr_r)

        return dq
    
    
###################
# Part 3
###################
        

        
class CustomDrillingController( robotcontrollers.RobotController ) :
    """ 

    """
    
    ############################
    def __init__(self, robot_model ):
        """ """
        
        super().__init__( dof = 3 )
        
        self.robot_model = robot_model
        
        # Label
        self.name = 'Custom Drilling Controller'
        self.reference_xyz = np.array([0.25, 0.25, 0.39])
        
    def isInBounds(self, val: float, min: float, max: float):
        if val >= min and val <= max:
            return True
        
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUPUTS
        tau  : control inputs vector    m x 1, ici torque au moteur
        
        """
        
        # Ref
        f_e = r
        
        # Feedback from sensors
        x = y
        [ q , dq ] = self.x2q( x )
        
        # Robot model
        r = self.robot_model.forward_kinematic_effector( q ) # End-effector actual position
        J = self.robot_model.J( q )      # Jacobian matrix
        g = self.robot_model.g( q )      # Gravity vector
        H = self.robot_model.H( q )      # Inertia matrix
        C = self.robot_model.C( q , dq ) # Coriolis matrix
            
        ##################################
        # Votre loi de commande ici !!!
        ##################################
        tau = np.zeros(self.m)  # place-holder de bonne dimension
                
        # CMD en force
        if not self.isInBounds(r[2], 0.2, 0.40) or not self.isInBounds(r[0], 0.24, 0.26) or not self.isInBounds(r[1], 0.24, 0.26):
            f_e = np.array([100, 100, -10])
        elif self.isInBounds(r[2], 0.21, 0.40) and self.isInBounds(r[0], 0.24, 0.26) and self.isInBounds(r[1], 0.24, 0.26):
            f_e = np.array([0,0, -200])
        else:
            f_e = np.array([0,0,0])
        tau = J.T @ f_e + g
        
        # # CMD en impedance
        # if not self.isInBounds(r[2], 0.2, 0.40) or not self.isInBounds(r[0], 0.24, 0.26) or not self.isInBounds(r[1], 0.24, 0.26):
        #     Kp = np.diag([50,50,2]) # 50 N/m in x and y, 2 N/m in z
        #     Kd = np.diag([10,10,1]) # 10 N/m/s in x and y, 1 N/m/s in z
        #     f_e = Kp @ (self.reference_xyz - r) + Kd @ (- J @ dq)
        # elif self.isInBounds(r[2], 0.21, 0.40) and self.isInBounds(r[0], 0.24, 0.26) and self.isInBounds(r[1], 0.24, 0.26):
        #     # self.reference_xyz = np.array([0.25, 0.25, 0.2]) cette ligne peut etre omise
        #     Kp = np.diag([2,2 ,0.0]) # 100 N/m in x and y, 0 N/m in z
        #     Kd = np.diag([1,1, 0.0]) # 10 N/m/s in x and y, 0 N/m/s in z
        #     f_e = Kp @ (self.reference_xyz - r) + Kd @ (- J @ dq)
        #     f_e[2] = -200 # -200 a pas lair de marcher, jsp pourquoi? Pt pas dans le bon frame?
        # else:
        #     f_e = np.array([0,0, 0])
            
        tau = (J.T @ f_e) + g
        return tau
        
    
###################
# Part 4
###################

# def discretize_time(final_time, num_steps):
#     timestep = final_time / num_steps  # Calculate the timestep
#     time_values = [i * timestep/final_time for i in range(num_steps + 1)]  # Generate the time values
#     return time_values
    
def goal2r( r_0 , r_f , t_f ):
    """
    
    Parameters
    ----------
    r_0 : numpy array float 3 x 1
        effector initial position
    r_f : numpy array float 3 x 1
        effector final position
    t_f : float
        time 

    Returns
    -------
    r   : numpy array float 3 x l
    dr  : numpy array float 3 x l
    ddr : numpy array float 3 x l

    """
    # Time discretization
    l = 1000 # nb of time steps
    
    # Number of DoF for the effector only
    m = 3
    
    r = np.zeros((m,l))
    dr = np.zeros((m,l))
    ddr = np.zeros((m,l))
    
    #################################
    # Votre code ici !!!
    #################################
    
    # Calculate the time vector
    t = np.linspace(0, t_f, l)

    # Calculate the trajectory
    s_r = np.zeros(l) # For 0 to 1
    s_dr = np.zeros(l)
    s_ddr = np.zeros(l)
    
    #  Vecteur erreur
    v_r = r_f - r_0
    
    for i, t_i in enumerate(t): # Calcul les profils temporels et chemin polynomiaux de degre 3
        # s(t)
        s_r[i] = (3/t_f**2)*t_i**2 -(2/t_f**3)*t_i**3
        s_dr[i] = (6/t_f**2)*t_i - (6/t_f**3)*t_i**2
        s_ddr[i] = (6/t_f**2) - (12/t_f**3)*t_i
        
    # Dot product 3x1 with 1x1000 = 3x1000
    # r(s)
    r = np.outer(v_r, s_r) # dot product 
    for i in range(l):
        r[:, i] += r_0
    dr = np.outer(v_r, s_dr)
    ddr = np.outer(v_r, s_ddr)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(r[0,:], r[1,:], r[2,:], 'gray')
    print(r[:,0])
    print(r[:,l-1])

    return r, dr, ddr

def systeme_equations_non_lineaires( q : np.array, r : np.array, manipulator):
    e_x = ((manipulator.l2*np.cos(q[1]) + manipulator.l3*np.cos(q[1]+q[2]))*np.cos(q[0])) - r[0]
    e_y = ((manipulator.l2*np.cos(q[1]) + manipulator.l3*np.cos(q[1]+q[2]))*np.sin(q[0])) -r[1]
    e_z = (manipulator.l1 + manipulator.l2*np.sin(q[1]) + manipulator.l3*np.sin(q[1]+q[2])) - r[2]
    
    return np.array([e_x, e_y, e_z])
    
def r2q( r, dr, ddr , manipulator ):
    """
    Parameters
    ----------
    r   : numpy array float 3 x l
    dr  : numpy array float 3 x l
    ddr : numpy array float 3 x l
    
    manipulator : pyro object 

    Returns
    -------
    q   : numpy array float 3 x l
    dq  : numpy array float 3 x l
    ddq : numpy array float 3 x l

    """
    
    # Time discretization
    l = r.shape[1]
    
    # Number of DoF
    n = 3
    
    # Output dimensions
    q = np.zeros((n,l))
    dq = np.zeros((n,l))
    ddq = np.zeros((n,l))
    
    #################################
    # Votre code ici !!!
    ##################################
    
    
    #Initial conditions
    q0_guess = np.array([0, 1.2, -0.6]) # "Starting Guess for non lin equation solver"
    dq[:,0] = np.array([0, 0, 0])
    
    # Solve systeme d'equations non-lineaires pour chaque instant
    for i in range(l):
        q[:,i] = sp.optimize.fsolve(systeme_equations_non_lineaires, q0_guess, args = (r[:,i], manipulator))
        q0_guess = q[:,i]
        dq[:,i] = np.linalg.inv(manipulator.J(q[:,i])) @ dr[:,i]
        ddq[:,i] = np.linalg.inv(manipulator.J(q[:,i])) @ ddr[:,i] - manipulator.Jd(q[:,i], dq[:,i]) @ dq[:,i]
    return q, dq, ddq

def q2torque( q, dq, ddq , manipulator ):
    """
    Parameters
    ----------
    q   : numpy array float 3 x l
    dq  : numpy array float 3 x l
    ddq : numpy array float 3 x l
    
    manipulator : pyro object 

    Returns
    -------
    tau   : numpy array float 3 x l

    """
    
    # Time discretization
    l = q.shape[1]
    
    # Number of DoF
    n = 3
    
    # Output dimensions
    tau = np.zeros((n,l))
    tau2 = np.zeros((n,l))
    
    #################################
    # Votre code ici !!!
    ##################################
    for i in range(l):
        tau[:, i] = np.linalg.inv(manipulator.B(q[:,i]))@(manipulator.H(q[:,i]) @ ddq[:,i] + 
                     manipulator.C(q[:,i], dq[:,i]) @ dq[:,i] + 
                     manipulator.d(q[:,i], dq[:,i]) + 
                     manipulator.g(q[:,i]))
        
        tau2[:, i] = manipulator.actuator_forces(q[:,i], dq[:,i], ddq[:,i])
    
    return tau, tau2