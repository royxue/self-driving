#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state dimention
  n_x_ = 5;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.15;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // Set augmented state dimentsion
  n_aug_ = 7;

  // Set lambda
  lambda_ = 3 - n_aug_;

  // Set is_initialized to False
  is_initialized_ = false;

  // Set the initial weight
  weights_ = VectorXd(2*n_aug_+1);
  weights_(0) = lambda_/(lambda_+n_aug_);
  for(int i=0; i< 2*n_aug_; i++) {
    weights_(i+1) = 1/(2*(lambda_+n_aug_));
  }
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} mp The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage mp) {
  if (is_initialized_) {
    double delta_t = double(mp.timestamp_ - time_us_)/1000000;
    time_us_ = mp.timestamp_;

    // Predict
    Prediction(delta_t);

    // Update
    if (mp.sensor_type_ == MeasurementPackage::LASER && use_laser_){
      UpdateLidar(mp);
    }

    if (mp.sensor_type_ == MeasurementPackage::RADAR && use_radar_){
      UpdateRadar(mp);
    }

  } else {
    time_us_ = mp.timestamp_;

    // Intialize state vector
    x_ = VectorXd(n_x_);

    if (mp.sensor_type_ == MeasurementPackage::LASER && use_laser_){
      x_ << mp.raw_measurements_[0], mp.raw_measurements_[1], 0, 0, 0;
    }

    if (mp.sensor_type_ == MeasurementPackage::RADAR && use_radar_){
       double rho = mp.raw_measurements_[0];
       double phi = mp.raw_measurements_[1];
       double rho_dot = mp.raw_measurements_[2];
       x_ << rho * cos(phi), rho * sin(phi), rho_dot, 0, 0;
    }

    // Initialize convariance matrix
    P_ = MatrixXd(5,5);
    P_.fill(0.0);
    for (int i=0;i<5;i++){
      P_(i, i) = 1;
    }

    is_initialized_ = true;
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // Augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // Augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner( n_x_, n_x_ ) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;

  // Sigma Points Matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  MatrixXd A = P_aug.llt().matrixL();
  MatrixXd root =  sqrt(lambda_ + n_aug_) * A;
  Xsig_aug.col(0) = x_aug;
  for(int i = 0; i < n_aug_; i++){
    Xsig_aug.col(1 + i) = x_aug + root.col(i);
    Xsig_aug.col(1 + n_aug_ + i) = x_aug - root.col(i);
  }

  // Predicted Sigma Points Matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    double px_p, py_p;

    // Check for invalid division number
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * ( sin (yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * ( cos(yaw) - cos(yaw + yawd * delta_t) );
    } else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // Add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p  + nu_a * delta_t;

    yaw_p  = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  x_ = VectorXd(5);
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x_ = x_ + weights_(i) * Xsig_pred_ . col(i);
  }

  P_ = MatrixXd(5,5);
  P_.fill(0.0);
  for (int i = 0; i < 2*n_aug_ + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = atan2(sin(x_diff(3)), cos(x_diff(3)));
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} mp
 */
void UKF::UpdateLidar(MeasurementPackage mp) {
  MatrixXd R_ = MatrixXd(2, 2);
  R_ << std_laspx_ * std_laspx_, 0,
        0, std_laspy_ * std_laspy_;

  MatrixXd H_ = MatrixXd(2, 5);
  H_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0;

  VectorXd z = VectorXd(2);
  z(0) = mp.raw_measurements_[0];
  z(1) = mp.raw_measurements_[1];

  VectorXd y = z - H_ * x_;
  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();

  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
  P_ = (I - K * H_) * P_;

  NIS_L_ = y.transpose() * S.inverse() * y;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} mp
 */
void UKF::UpdateRadar(MeasurementPackage mp) {
  int n_z = 3;

  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  S(0,0) = std_radr_   * std_radr_;
  S(1,1) = std_radphi_ * std_radphi_;
  S(2,2) = std_radrd_  * std_radrd_;

  for (int i=0; i< 2 * n_aug_ + 1; i++) {
    double px = Xsig_pred_(0,i);
    double py = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double rho = sqrt(px * px + py * py);
    double phi = atan2(py, px);
    double rho_dot = (px * cos(yaw) * v + py * sin(yaw) * v) / sqrt(px * px + py * py);

    if (fabs(sqrt(px * px + py * py)) < 0.0001) {
      rho_dot = (px * cos(yaw) * v + py * sin(yaw) * v) / 0.0001;
    }

    Zsig(0,i) = rho;
    Zsig(1,i) = phi;
    Zsig(2,i) = rho_dot;

    z_pred += weights_(i) * Zsig.col(i);
  }

  for (int i=0; i<2*n_aug_+1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = atan2(sin(z_diff(1)), cos(z_diff(1)));
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  MatrixXd Tc = MatrixXd(n_x_, n_z);

  Tc.fill(0.0);
  for (int i=0; i < 2 * n_aug_ + 1; i++) {

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = atan2(sin(x_diff(3)), cos(x_diff(3)));

    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = atan2(sin(z_diff(1)), cos(z_diff(1)));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = Tc * S.inverse();

  VectorXd z = VectorXd(3);
  for (int i=0;i<3; i++) {
    z(i) = mp.raw_measurements_(i);
  }

  VectorXd z_diff = z - z_pred;
  z_diff(1) = atan2(sin(z_diff(1)), cos(z_diff(1)));

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  NIS_R_ = z_diff.transpose() * S.inverse() * z_diff;
}
