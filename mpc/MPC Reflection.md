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

**State Prediction**
Use the kinematic constraint model for the state prediction.
```
          double cte = polyeval(coeffs, 0);// error in track
          double epsi = -atan(coeffs[1]);

          // Estimate the car's future position after latency time
          double delta = -steer_angle;
          px = v*cos(psi) * latency;
          py = v*sin(psi) * latency;
          psi = (-v* steer_angle * latency)/2.67;
          epsi += v*delta*latency/2.67;
          cte += v*sin(epsi)*latency;
          v = v + acceleration * latency;
```

## Timestep length and Elapsed duration between timesteps (N & dt)
I tried different combinations of values of N and dt, to provide the best output manuslly.

The values for N and dt that I tried N from 5 to 20 and dt from 0.05 to 0.2. Finally I think the combination with N = 10 and dt = 0.1 works best for me.

Small dt means short latency, which means the values changes are smaller as well, which will leads to more reliable path. If N became too big, the program would be slower, because it will need more data to do the prediction, the computation time increased. The N*dt will affect the path, if N*dt is too big, and when the car speed is high, it’s very easily to go out of path.

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