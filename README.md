# Heat Equation Solver

## Overview
A standard solver of the 1D Heat Equation utilising Numpy. This project demonstrates the effectiveness of FDM and RK-2.

**Key Features**
- **FDM_RK2 Solver:** Solves the given initial conditions problem with order-2 accuracy.
- **Error_analysis:** Contains the function to calculate L_infinity norm and the order of convergence of errors.
- **Plotter:** Contains a function that utilises matplotlib to visually show the change in temperature and order of the error.

## Mathematical Overview
The solver addresses the heat equation

$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$

## Technical Implementation
- **Language:** Python 3.11.4 (Used because my pre-existing environment uses it)
- **Math Logic:** - Implements the standard Fite Difference Method with Central Difference for the second order and forward difference for the first order. And standard RK2, Heun's method, to minimise error.
  

