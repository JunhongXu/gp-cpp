#include "rbf.h"
#include <math.h>

namespace gp{
    RBF::RBF(int inpt_dim, double variance, double len_scale): Kernel(inpt_dim, 2){
        name = "RBF";
        // initialize parameters
        params["variance"] = variance;
        params["len_scale"] = len_scale;
    }


    double RBF::compute_cov(Eigen::VectorXd& x1, Eigen::VectorXd& x2){
        return rbf_function(x1, x2);
    }

    Eigen::MatrixXd RBF::compute_K(std::vector<Eigen::VectorXd>& x1, std::vector<Eigen::VectorXd>& x2){
        int num_x1_samples = x1.size();
        int num_x2_samples = x2.size();

        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(num_x1_samples, num_x2_samples);

        for(int x1_idx = 0; x1_idx < num_x1_samples; x1_idx++){
            for(int x2_idx = 0; x2_idx < num_x2_samples; x2_idx++){
                auto& x1_sample = x1[x1_idx];
                auto& x2_sample = x2[x2_idx];
                double rbf_output = rbf_function(x1_sample, x2_sample);
                result(x1_idx, x2_idx) = rbf_output;
            }
        }
        return result;
    }

    Eigen::VectorXd RBF::compute_k(std::vector<Eigen::VectorXd>& X, Eigen::VectorXd& x){
        int num_x1_samples = X.size();

        Eigen::VectorXd result = Eigen::VectorXd::Zero(num_x1_samples, 1);

        for(int idx = 0; idx < num_x1_samples; idx++){
            auto& x1_sample = X[idx];
            double rbf_output = rbf_function(x1_sample, x);
            result(idx) = rbf_output;
        }
        return result;
    }

    double RBF::rbf_function(Eigen::VectorXd& x, Eigen::VectorXd& x_prime){
        // squared distance between x and x_prime
        double r = (x - x_prime).array().square().sum();
        // r / l^2
        double z = r / pow(params["len_scale"], 2);

        z = exp(-0.5 * z);
        return params["variance"] * z;
    }
}
