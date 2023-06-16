#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:19:16 2020

@author: alex
------------------------------------


Fichier d'amorce pour les livrables de la problématique GRO640'


"""

import numpy as np

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
    DH[0,:] = [0.033, 0.08, q[0], -np.pi/2]
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
        J_obj2 = 
        
        # Ref and objectives
        # Primary objectif
        r_desired   = r
        r_actual    = self.fwd_kin( q )

        # Error
        e  = r_desired - r_actual
        q_e = self.q_d - q

        # Effector space speed
        dr_r = e * self.gains
        dq_null = q_e * self.gains_null
        
        ################
        dq = np.zeros( self.m )  # place-holder de bonne dimension
        
        ##################################
        # Votre loi de commande ici !!!
        ##################################
        J_pinv = np.linalg.pinv( J )
        Null_p = np.identity( self.dof ) - np.dot( J_pinv , J ) # Null space projection matrix
        dq_r = np.dot( J_pinv , dr_r ) + np.dot( Null_p, dq_null ) # Joint space speed

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
        
        
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
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
        
        u = np.zeros(self.m)  # place-holder de bonne dimension
        
        return u
        
    
###################
# Part 4
###################
        
    
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
    ##################################
    
    
    return r, dr, ddr


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
    
    #################################
    # Votre code ici !!!
    ##################################
    
    
    return tau