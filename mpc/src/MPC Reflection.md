# MPC Reflection
#selfdriving
## MPC Model

The MPC model includes the following state variables:

**State variables**
- x - the x position of the vehicle in the vehicle's grid map.
- y - the y position of the vehicle in the vehicle's grid map.
- psi - the relative vehicles orientation (vehicles coordinate system)
- velocity - the vehicle's velocity in the direction of psi.
- cross-track error
- psi error

**Actuations**
- steering - in range of (-0.43 : 0.43) radians (-25 : 25 degrees).
- throttle - in range of (-1.0 : 1.0).

**Constraints**
The cost function tries to minimize these constrains:
- cross track error
- psi error
- error in velocity vs. target speed
- changes in steering actuator.
- changes in throttle actuator.

## Timestep length and Elapsed duration between timesteps (N & dt)
I tried different combinations of values of N and dt, to provide the best output manuslly.

The values for N and dt that I tried N from 5 to 20 and dt from 0.05 to 0.2. Finally I think the combination with N = 10 and dt = 0.1 works best for me

## Polynomial Fitting and MPC Preprocessing

The trajectory waypoints are converted from the global/map coordinate system to the vehicles's coordinate system. 

The transformation:
```
          // calculate the cross track error by evaluating the polynomial at x = px
          // and subtracting y = 0.
          double cte = polyeval(coeffs, px);// error in track

          //psi is the angle with relation to the track
          double epsi = psi - atan(coeffs[1] + coeffs[2]*px + coeffs[3]*px*px);//
```
The transformation actually simplifies the calculations for cross track error and psi error because initial values for x and y relative to the vehicle are always zero.


## Model Predictive Control with Latency
The main idea to deal with the latency is to project the vehicle’s state by one latency period ahead. This is done by the basic kinetic model calculation.