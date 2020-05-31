#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

// Normalized Innovation Squared(NIS)
float UKF::CalculateNIS(const VectorXd &z_measured, const VectorXd &z_pred, const MatrixXd &covariance)
{
    VectorXd error = z_measured - z_pred;
    return error.transpose() * covariance.inverse() * error;
}

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
    n_x_ = 5;
    n_aug_ = 7;
    lambda_ = 3 - n_aug_;

    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(n_x_);

    // initial covariance matrix
    P_ = MatrixXd(n_x_, n_x_);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 30;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 30;

    /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

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

    /**
   * End DO NOT MODIFY section for measurement noise values 
   */

    /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

    // // Process noise standard deviation longitudinal acceleration in m/s^2
    // std_a_ = 0.2;
    // // Process noise standard deviation yaw acceleration in rad/s^2
    // std_yawdd_ = 0.2;

    std_a_ = 3.0; // Process noise standard deviation longitudinal acceleration in m/s^2
    std_yawdd_ = 1.0;

    is_initialized_ = false;
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
    weights_ = VectorXd(2 * n_aug_ + 1);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
    /**
    * TODO: Complete this function! Make sure you switch between lidar and radar
    * measurements.
    */

    if (!is_initialized_)
    {
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
        {

            // initialize using radar
            double rho = meas_package.raw_measurements_[0];
            double phi = meas_package.raw_measurements_[1];
            double rho_dot = meas_package.raw_measurements_[2];

            x_ << rho * cos(phi),
                rho * sin(phi),
                0,
                0,
                0;

            P_ << pow(std_radr_, 2), 0, 0, 0, 0,
                0, pow(std_radphi_, 2), 0, 0, 0,
                0, 0, pow(std_radrd_, 2), 0, 0,
                0, 0, 0, 1, 0,
                0, 0, 0, 0, 1;
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::LASER)

        {
            x_ << meas_package.raw_measurements_[0],
                meas_package.raw_measurements_[1],
                0,
                0,
                0;

            P_ << pow(std_laspx_, 2), 0, 0, 0, 0,
                0, pow(std_laspy_, 2), 0, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 0, 1, 0,
                0, 0, 0, 0, 1;
        }

        time_us_ = meas_package.timestamp_;

        for (int i = 0; i < Xsig_pred_.cols(); i++)
        {
            // set weights
            if (i == 0)
                weights_(i) = lambda_ / (lambda_ + n_aug_);
            else
                weights_(i) = 0.5 * 1 / (lambda_ + n_aug_);
        }

        is_initialized_ = true;
        return;
    }

    double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;

    time_us_ = meas_package.timestamp_;
    Prediction(dt);

    if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
    {
        UpdateLidar(meas_package);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
    {
        UpdateRadar(meas_package);
    }
}

void UKF::Prediction(double delta_t)
{
    /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    GenerateAugmentedSigmaPoints(Xsig_aug);
    SigmaPointPrediction(Xsig_aug, delta_t);
    PredictMeanAndCovariance();
}

void UKF::UpdateLidar(MeasurementPackage meas_package)
{
    /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

    int n_z = 2;

    MatrixXd Zsig{MatrixXd(n_z, 2 * n_aug_ + 1)};
    VectorXd z_pred{VectorXd(n_z)};
    MatrixXd covariance{MatrixXd(n_z, n_z)};

    PredictRadarMeasurement(n_z, meas_package.sensor_type_, Zsig, z_pred, covariance);

    UpdateState(n_z, Zsig, covariance, z_pred, meas_package.raw_measurements_);

    float nis = CalculateNIS(meas_package.raw_measurements_, z_pred, covariance);
    std::cout << "Radar NIS: " << nis << std::endl;
}

void UKF::UpdateRadar(MeasurementPackage meas_package)
{
    /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

    int n_z = 3;

    MatrixXd Zsig{MatrixXd(n_z, 2 * n_aug_ + 1)};
    VectorXd z_pred{VectorXd(n_z)};
    MatrixXd covariance{MatrixXd(n_z, n_z)};

    PredictRadarMeasurement(n_z, meas_package.sensor_type_, Zsig, z_pred, covariance);

    UpdateState(n_z, Zsig, covariance, z_pred, meas_package.raw_measurements_);

    float nis = CalculateNIS(meas_package.raw_measurements_, z_pred, covariance);
    std::cout << "Radar NIS: " << nis << std::endl;
}

void UKF::GenerateAugmentedSigmaPoints(MatrixXd &Xsig_aug)
{
    // create augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);

    // create augmented state covariance
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

    // create augmented mean state
    x_aug.fill(0);
    x_aug.head(n_x_) = x_;

    P_aug.fill(0.0);
    // create augmented covariance matrix
    P_aug.topLeftCorner(n_x_, n_x_) = P_;

    MatrixXd Q = MatrixXd(n_aug_ - n_x_, n_aug_ - n_x_);
    Q << std_a_, 0,
        0, std_yawdd_;

    P_aug.bottomRightCorner(n_aug_ - n_x_, n_aug_ - n_x_) = Q * Q;

    // create square root matrix
    MatrixXd A = P_aug.llt().matrixL();

    // create augmented sigma points
    float sigma_factor = sqrt(lambda_ + n_aug_);

    MatrixXd variance_offset_left = x_aug.rowwise().replicate(n_aug_) + sigma_factor * A;
    MatrixXd variance_offset_right = x_aug.rowwise().replicate(n_aug_) - sigma_factor * A;

    Xsig_aug.block<7, 1>(0, 0) = x_aug;
    Xsig_aug.block<7, 7>(0, 1) = variance_offset_left;
    Xsig_aug.block<7, 7>(0, 8) = variance_offset_right;
}

void UKF::SigmaPointPrediction(const MatrixXd &Xsig_points, const double &delta_t)
{
    // predict sigma points
    for (int i = 0; i < Xsig_points.cols(); i++)
    {
        VectorXd x_k = Xsig_points.block<5, 1>(0, i);
        float v_k = x_k[2];
        float phi_k = x_k[3];
        float phi_k_dot = x_k[4];

        float nu_a = Xsig_points(5, i);
        float nu_phi_dot_dot = Xsig_points(6, i);

        VectorXd velocity_term(n_x_);
        VectorXd acceleration_term(n_x_);

        // avoid division by zero
        if (fabs(phi_k_dot) > 1e-3)
        {
            velocity_term[0] = (v_k / phi_k_dot) * (sin(phi_k + phi_k_dot * delta_t) - sin(phi_k));
            velocity_term[1] = (v_k / phi_k_dot) * (-cos(phi_k + phi_k_dot * delta_t) + cos(phi_k));
        }
        else
        {
            velocity_term[0] = v_k * cos(phi_k) * delta_t;
            velocity_term[1] = v_k * sin(phi_k) * delta_t;
        }

        velocity_term[2] = 0;
        velocity_term[3] = phi_k_dot * delta_t;
        velocity_term[4] = 0;

        acceleration_term[0] = 0.5 * pow(delta_t, 2) * cos(phi_k) * nu_a;
        acceleration_term[1] = 0.5 * pow(delta_t, 2) * sin(phi_k) * nu_a;
        acceleration_term[2] = delta_t * nu_a;
        acceleration_term[3] = 0.5 * pow(delta_t, 2) * nu_phi_dot_dot;
        acceleration_term[4] = delta_t * nu_phi_dot_dot;

        // write predicted sigma points into right column
        Xsig_pred_.block<5, 1>(0, i) = x_k + (velocity_term + acceleration_term);
    }
}

/**
 * Programming assignment functions: 
 */

void UKF::PredictMeanAndCovariance()
{

    x_.fill(0);
    P_.fill(0);

    for (int i = 0; i < Xsig_pred_.cols(); i++)
    {
        // predict state mean
        x_ += weights_(i) * Xsig_pred_.col(i);
    }

    for (int i = 0; i < Xsig_pred_.cols(); i++)
    {

        VectorXd delta_x = Xsig_pred_.col(i) - x_;

        // angle normalization
        while (delta_x(3) > M_PI)
            delta_x(3) -= 2. * M_PI;
        while (delta_x(3) < -M_PI)
            delta_x(3) += 2. * M_PI;

        P_ += weights_(i) * delta_x * delta_x.transpose();
    }
}

void UKF::PredictRadarMeasurement(const int &n_z, MeasurementPackage::SensorType sensor_type, MatrixXd &Zsig, VectorXd &z_pred, MatrixXd &S)
{
    Zsig.fill(0);
    z_pred.fill(0);
    S.fill(0);

    for (int i = 0; i < Xsig_pred_.cols(); i++)
    {

        VectorXd x = Xsig_pred_.col(i);
        float px = x[0];
        float py = x[1];
        float v = x[2];
        float phi = x[3];
        float phi_dot = x[4];

        if (sensor_type == MeasurementPackage::RADAR)
        {
            Zsig(0, i) = sqrt(pow(px, 2) + pow(py, 2));
            Zsig(1, i) = atan2(py, px);
            Zsig(2, i) = ((px * cos(phi) * v) + (py * sin(phi) * v)) / sqrt(pow(px, 2) + pow(py, 2));
        }
        else if (sensor_type == MeasurementPackage::LASER)
        {

            Zsig(0, i) = px;
            Zsig(1, i) = py;
        }
    }

    for (int i = 0; i < Zsig.cols(); i++)
    {

        z_pred += weights_(i) * Zsig.col(i);
    }

    // calculate innovation covariance matrix S

    for (int i = 0; i < Zsig.cols(); i++)
    {
        VectorXd delta_z = Zsig.col(i) - z_pred;

        // angle normalization
        while (delta_z(1) > M_PI)
            delta_z(1) -= 2. * M_PI;
        while (delta_z(1) < -M_PI)
            delta_z(1) += 2. * M_PI;

        S += weights_(i) * delta_z * delta_z.transpose();
    }

    MatrixXd R = MatrixXd(n_z, n_z);

    if (sensor_type == MeasurementPackage::RADAR)
    {
        R << pow(std_radr_, 2), 0, 0,
            0, pow(std_radphi_, 2), 0,
            0, 0, pow(std_radrd_, 2);
    }
    else if (sensor_type == MeasurementPackage::LASER)
    {
        R << pow(std_laspx_, 2), 0,
            0, pow(std_laspy_, 2);
    }

    S += R;
}

void UKF::UpdateState(const int &n_z, const MatrixXd &Zsig, const MatrixXd &S, VectorXd &z_pred, const VectorXd &z)
{

    // create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0);

    // calculate cross correlation matrix

    for (int i = 0; i < Zsig.cols(); i++)
    {
        VectorXd delta_z = Zsig.col(i) - z_pred;

        // angle normalization
        while (delta_z(1) > M_PI)
            delta_z(1) -= 2. * M_PI;
        while (delta_z(1) < -M_PI)
            delta_z(1) += 2. * M_PI;

        VectorXd delta_x = Xsig_pred_.col(i) - x_;

        // angle normalization
        while (delta_x(3) > M_PI)
            delta_x(3) -= 2. * M_PI;
        while (delta_x(3) < -M_PI)
            delta_x(3) += 2. * M_PI;

        Tc += weights_(i) * delta_x * delta_z.transpose();
    }

    // calculate Kalman gain K;

    MatrixXd K = Tc * S.inverse();

    // update state mean and covariance matrix

    VectorXd delta_z = z - z_pred;

    // angle normalization
    while (delta_z(1) > M_PI)
        delta_z(1) -= 2. * M_PI;
    while (delta_z(1) < -M_PI)
        delta_z(1) += 2. * M_PI;

    x_ += K * delta_z;

    P_ -= K * S * K.transpose();
}