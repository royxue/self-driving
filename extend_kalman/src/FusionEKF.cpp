#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1000, 0,
        0, 0, 0, 1000;

  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
        0, 1, 0, 0;

  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;

  ekf_.PI_ = atan2(1, 1)*4;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

/*****************************************************************************
 *  Initialization
 ****************************************************************************/
void FusionEKF::Init(const MeasurementPackage &measurement_pack) {
  cout << "EKF: " << endl;
  ekf_.x_ = VectorXd(4);
  ekf_.x_ << 1, 1, 1, 1;

  float rho = measurement_pack.raw_measurements_[0]; 
  float theta = measurement_pack.raw_measurements_[1];

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    ekf_.x_ << rho * cos(theta), rho * sin(theta), 0, 0;
  }
  else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    ekf_.x_ << rho, theta, 0, 0;
  }
}

/*****************************************************************************
 *  Prediction
 ****************************************************************************/
void FusionEKF::Predict(const MeasurementPackage &measurement_pack) {
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0; //dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  float noise_ax = 9;
  float noise_ay = 9;
  float dt2 = dt * dt;
  float dt3 = pow(dt,3);
  float dt4 = pow(dt,4);

  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;
  
  ekf_.Q_ = MatrixXd(4,4);
  ekf_.Q_ << dt4 * noise_ax / 4, 0, dt3 * noise_ax / 2, 0,
             0, dt4 * noise_ay / 4, 0, dt3 * noise_ay /2,
             dt3 * noise_ax / 2, 0, dt2 * noise_ax, 0,
             0, dt3 * noise_ay / 2, 0, dt2 * noise_ay;

  ekf_.Predict();
}

/*****************************************************************************
 *  Update
 ****************************************************************************/
void FusionEKF::Update(const MeasurementPackage &measurement_pack) {
  VectorXd mp = measurement_pack.raw_measurements_;
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(mp);
  } else {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(mp);
  }
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  if (!is_initialized_) {
    Init(measurement_pack);
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }

  Predict(measurement_pack);
  Update(measurement_pack);

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}