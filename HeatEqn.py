import numpy as np
import matplotlib.pyplot as plt
import time


#Calculate the error and order of convergence
def error_analysis(u, u_exact, M, prev_error) -> tuple[float, float]:
    error = np.zeros(M + 1)
    for i in range(M + 1):
        error[i] = u[i] - u_exact[i]

    max_error = np.max(np.abs(error))

    order = 0.0
    if prev_error > 1e-15:
        order = np.log(prev_error / max_error) / np.log(2)

    return max_error, order

def timer(func):
    def new_func(*args,**kwargs):
        start = time.perf_counter() # Switched to perf_counter for precision
        result = func(*args,**kwargs)
        end = time.perf_counter()
        return result, (end-start)
    return new_func

def error_analysis(u, u_exact, M, prev_error) -> tuple[float, float]:
    max_error = np.max(np.abs(u - u_exact)) # Vectorized for speed
    order = 0.0
    if prev_error > 1e-15:
        order = np.log(prev_error / max_error) / np.log(2)
    return max_error, order

def plot_results(L, x, u, u_exact, ms, errors):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Solution Comparison (Final M)
    ax1.plot(x, u, 'yo-', label='Numerical (FDM-RK2)', markersize=2)
    ax1.plot(x, u_exact, 'r--', label='Exact Solution', markersize=10)
    ax1.set_title(f"Solution at T_final (M={len(x)-1})")
    ax1.set_xlabel("x")
    ax1.set_ylabel("u(x, t)")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Log-Log Error Convergence
    ax2.loglog(ms, errors, 'o-', label='Measured Error')
    # Reference line for 2nd order convergence (Slope = 2)
    ax2.loglog(ms, [errors[0]*(ms[0]/m)**2 for m in ms], 'k--', label='2nd Order Slope')
    ax2.set_title("Error Convergence")
    ax2.set_xlabel("Grid Size (M)")
    ax2.set_ylabel("Max Error")
    ax2.legend()
    ax2.grid(True, which="both", ls="-")
    
    plt.tight_layout()
    plt.show()

@timer
def FDM_RK2(L,T,M, alpha, u): 
    # Spatial and Temporal Discretization

    dx = L / M
    dt = 0.5 * dx * dx
    time_steps = T / dt

    # Initialize u arrays
    u_new = np.zeros(M + 1)
    u_star = np.zeros(M + 1)

    #Solver loop
    for t in range(int(time_steps)):


        # RK2 Method (Heun's Method)

        #1. initialise the intermediate step for all spatial oints

        u_star[0] = u[0]
        u_star[M] = u[M]
        u_new[0] = u[0]
        u_new[M] = u[M]

        for j in range(1,M):
            u_star[j] = u[j] + alpha * dt * (u[j+1]- 2*u[j] + u [j-1])/dx**2
        
        for i in range(1,M):

            #2. Now calculate the RK slopes

            k1 = alpha*(u[i+1]-2*u[i]+u[i-1])/dx**2

            k2 = alpha* (u_star[i+1] - 2*u_star[i] + u_star[i-1])/dx**2

            #3. now calculate the new u values
            u_new[i] = u[i] + 0.5 * dt * (k1 + k2)
        
        #update all the values for the next time step
        u, u_new = u_new, u
    
    return u


if __name__ == "__main__":
    L = 1.0
    T_final = 0.1
    alpha = 1.0

    ms, errors = [], []

    print("   M    |  Max Error (L_inf) |  Order  |  CPU Time (s) ")
    print("--------|--------------------|---------|---------------")

    prev_error = 0.0

    #So that the variables are not "possibly unbound" when we try to print them later
    u = None
    u_exact = None
    u_final = None
    x = None
    comp_time = 0.0

    for M in [10, 20, 40, 80, 160, 320]:

        x = np.linspace(0, L, M + 1)
        
        # Initial Condition
        u = np.sin(np.pi * np.linspace(0, L, M + 1))

        u, comp_time = FDM_RK2(L, T_final, M, alpha, u)

        # Exact Solution
        u_exact = np.sin(np.pi * np.linspace(0, L, M + 1)) * np.exp(-alpha * (np.pi ** 2) * T_final)

        #calculate error and order
        max_error, order = error_analysis(u, u_exact, M, prev_error)

        ms.append(M)
        errors.append(max_error)

        if (M == 10):
             print("  {0:4d}  |     {1:1.4e}     |    -    |   {2:1.4f}".format(M, max_error, comp_time))

        else:
             print("  {0:4d}  |     {1:1.4e}     |  {2:.3f}  |   {3:1.4f}".format(M, max_error, order, comp_time))
        prev_error = max_error
    
    # Plot results for the finest grid
    plot_results(L, x, u, u_exact, ms, errors)