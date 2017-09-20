#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

size_t N = 10;
double dt = 0.1;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

double ref_cte = 0;
double ref_epsi = 0;
double ref_v = 50;

size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N - 1;

class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    fg[0] = 0;

    // State Cost
    for (int i=0; i<N; i++)
    {
      fg[0] += 2 * CppAD::pow(vars[cte_start+i], 2); // 2000
      fg[0] += 1500 * CppAD::pow(vars[epsi_start+i], 2); //2000
      fg[0] += 1 * CppAD::pow(vars[v_start+i]-ref_v, 2);
    }

    for (int i=0; i<N-1; i++)
    {
      fg[0] += 5 * CppAD::pow(vars[delta_start+i], 2);
      fg[0] += 5 * CppAD::pow(vars[a_start+i], 2);
    }

    for (int i=0; i<N-2; i++)
    {
      fg[0] += 500 * CppAD::pow(vars[delta_start+i+1]-vars[delta_start+i], 2);
      fg[0] += 500 * CppAD::pow(vars[a_start+i+1]-vars[a_start+i], 2);
    }

    // Initial constraints. All at 1 due to the cost being index 0;
    fg[1+x_start] = vars[x_start];
    fg[1+y_start] = vars[y_start];
    fg[1+psi_start] = vars[psi_start];
    fg[1+v_start]  = vars[v_start];
    fg[1+cte_start]  = vars[cte_start];
    fg[1+epsi_start] = vars[epsi_start];


      for (int i = 1; i < N; i++)
      {
        // i+1
        AD<double> x1 = vars[x_start+i];
        AD<double> y1 = vars[y_start+i];
        AD<double> psi1 = vars[psi_start+i];
        AD<double> v1 = vars[v_start+i];
        AD<double> cte1 = vars[cte_start+i];
        AD<double> epsi1 = vars[epsi_start+i];

        // i+0
        AD<double> x0 = vars[x_start+i-1];
        AD<double> y0 = vars[y_start+i-1];
        AD<double> psi0 = vars[psi_start+i-1];
        AD<double> v0 = vars[v_start+i-1];
        AD<double> cte0 = vars[cte_start+i-1];
        AD<double> epsi0 = vars[epsi_start+i-1];

        AD<double> delta0 = vars[delta_start+i-1];
        AD<double> a0 = vars[a_start+i-1];

        AD<double> f0 = coeffs[0]+coeffs[1]*x0+coeffs[2]*x0*x0+coeffs[3]*x0*x0*x0;
        AD<double> psides0 = CppAD::atan(3*coeffs[3]*x0*x0+2*coeffs[2]*x0+coeffs[1]);

        // Constrain x io be 0;
        fg[1+x_start+i] = x1-(x0+v0*CppAD::cos(psi0)*dt);  // Value at i1 has io be x1-ihe modified x0
        fg[1+y_start+i] = y1-(y0+v0*CppAD::sin(psi0)*dt);
        fg[1+psi_start+i] = psi1-(psi0+v0*delta0 / Lf*dt);
        fg[1+v_start+i] = v1-(v0+a0*dt);
        fg[1+cte_start+i] = cte1-((f0-y0)+(v0*CppAD::sin(epsi0)*dt));
        fg[1+epsi_start+i] = epsi1-((psi0-psides0)+v0*delta0 / Lf*dt);
      }
  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  double x = state[0];
  double y = state[1];
  double psi = state[2];
  double v = state[3];
  double cte = state[4];
  double epsi = state[5];
  // Set the number of model variables (includes both states and inputs).
  size_t n_vars = N*6 + (N-1)*2;
  // Set the number of constraints
  size_t n_constraints = N*6;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }

  vars[x_start] = x;
  vars[y_start] = y;
  vars[psi_start] = psi;
  vars[v_start] = v;
  vars[cte_start] = cte;
  vars[epsi_start] = epsi;

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);

  // Set all non-actuator upper and lower limits
  // Between the min and max possible values
  for (int i = 0; i < delta_start; i++)
  {
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] = 1.0e19;
  }

  // The upper and lower limits of delta
  // Between -25 and + 25. degrees (values in radians).
  for (int i = delta_start; i < a_start; i++)
  {
    vars_lowerbound[i] = -0.436332;
    vars_upperbound[i] = 0.436332;
  }

  // Acceleration / deceleration upper and lower limits
  // between -1 and 1
  for (int i = a_start; i < n_vars; i++)
  {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] = 1.0;
  }

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (int i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

  constraints_lowerbound[x_start] = x;
  constraints_lowerbound[y_start] = y;
  constraints_lowerbound[psi_start] = psi;
  constraints_lowerbound[v_start] = v;
  constraints_lowerbound[cte_start] = cte;
  constraints_lowerbound[epsi_start] = epsi;

  constraints_upperbound[x_start] = x;
  constraints_upperbound[y_start] = y;
  constraints_upperbound[psi_start] = psi;
  constraints_upperbound[v_start] = v;
  constraints_upperbound[cte_start] = cte;
  constraints_upperbound[epsi_start] = epsi;

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  vector<double> result;
  result.push_back(solution.x[delta_start]);
  result.push_back(solution.x[a_start]);

  for (unsigned int i = 0; i < N; i++)
  {
    result.push_back(solution.x[x_start+i]);
    result.push_back(solution.x[y_start+i]);
  }

  return result;
}