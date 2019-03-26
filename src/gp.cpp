#include "gp.h"

namespace gp{
    GaussianProcess::GaussianProcess(std::vector<Eigen::VectorXd> X, std::vector<double> Y, double prior_var, std::unique_ptr<Kernel> kernel)
        : X(X), Y(Y), prior_var(prior_var), kernel(std::move(kernel)){
        // initialize inverse covariance matrix
        compute_precision_matrix();
    }

    void GaussianProcess::add_point(Eigen::VectorXd x, double y){
        X.push_back(x);
        Y.push_back(y);
    }

    void GaussianProcess::compute_precision_matrix(){
        precision_matrix = kernel->compute_K(X, X);
        // adding prior precision on diagnal
        for(int row = 0; row < precision_matrix.rows(); row++){
            precision_matrix(row, row) += prior_var;
        }
        precision_matrix = precision_matrix.inverse();
    }

    std::pair<double, double> GaussianProcess::inference(Eigen::VectorXd& new_x){
        // convert std::vector to Eigen::VectorXd
        Eigen::VectorXd _Y = Eigen::Map<Eigen::MatrixXd>(Y.data(), Y.size(), 1);

        // compute vector k = [k(x_1, x_{N+1}, k(x_2, x_{N+1}...))]
        auto k = kernel->compute_k(X, new_x);
        auto alpha = precision_matrix * (_Y);
        double mean = alpha.transpose().dot(k);

        double c = kernel->compute_cov(new_x, new_x);
        double variance = c - k.transpose() * precision_matrix * k;

        //std::cout<<"x "<<new_x<<" k "<<k.transpose()<<std::endl;
        //std::cout<<"precision "<<precision_matrix<<std::endl;

        return std::make_pair(mean, variance);
    }

}
