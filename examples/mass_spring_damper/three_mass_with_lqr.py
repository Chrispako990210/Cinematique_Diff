# -*- coding: utf-8 -*-"""Created on Jun 2 2021@author: Alex"""import numpy as npfrom pyro.dynamic               import massspringdamperfrom pyro.analysis.costfunction import QuadraticCostFunctionfrom pyro.control.lqr           import synthesize_lqr_controller# Plantsys = massspringdamper.ThreeMass()sys.m1 = 2sys.m2 = 3sys.k1 = 0.01sys.k2 = 2sys.b1 = 0.1sys.b2 = 0.2sys.compute_ABCD() # Update Matrices based on updated parameters#Full state feedback (default of class is x2 output only)sys.C = np.diag([1,1,1,1,1,1])sys.p = 6 # dim of output vectorsys.output_label = sys.state_labelsys.output_units = sys.state_units# Cost functioncf  = QuadraticCostFunction.from_sys( sys )cf.Q[0,0] = 0cf.Q[1,1] = 0cf.Q[2,2] = 1cf.Q[3,3] = 0cf.Q[4,4] = 0cf.Q[5,5] = 0.01cf.R[0,0] = 0.0001sys.cost_function = cf# LQR controllerctl = synthesize_lqr_controller( sys , cf )ctl.ref_label = ['x1d','x2d','x3d','dx1d','dx2d','dx3d']ctl.rbar[2] = 1# Simulation Closed-Loop with LQR controllercl_sys = ctl + syscl_sys.x0[2] = 0cl_sys.compute_trajectory( 300 )cl_sys.plot_trajectory('xu')cl_sys.plot_linearized_pz_map(2,2)cl_sys.plot_linearized_bode(2,2)cl_sys.animate_simulation()