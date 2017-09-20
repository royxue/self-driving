#include "PID.h"

using namespace std;

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_in, double Kd_in, double Ki_in) {
  Kp = Kp_in;
  Ki = Ki_in;
  Kd = Kd_in;

  p_error = 0;
  d_error = 0;
  i_error = 0;
}

void PID::UpdateError(double cte) {
  d_error = cte - p_error;
  p_error = cte;
  i_error += cte;
}

double PID::TotalError() {
  double cte = -(p_error*Kp + d_error*Kd + i_error*Ki);
  return cte;
}
