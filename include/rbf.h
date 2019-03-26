#pragma once
#include "kernel.h"

namespace gp{
    /** Squared Exponential Kernel or Radial Basis Function: k(x, x') = \sigma^2 * exp(-\frac{(x-x'){2l^2}})*/
    class RBF: public Kernel{
        public:
            /** Constructor for RBF kernel. RBF kernel has two parameters
             *  @param inpt_dim input dimension of the kernel function
             *  @param variance controls the variance of the RBS function
             *  @param len_scale the length scale parameter of the RBF function*/
            RBF(int inpt_dim, double variance, double len_scale);

            ~RBF(){}

            double compute_cov(Eigen::VectorXd& x1, Eigen::VectorXd& x2) override;

            Eigen::MatrixXd compute_K(std::vector<Eigen::VectorXd>& x1, std::vector<Eigen::VectorXd>& x2) override;

            Eigen::VectorXd compute_k(std::vector<Eigen::VectorXd>& X, Eigen::VectorXd& x) override;

        private:
            /** Calculate the output of RBF*/
            double rbf_function(Eigen::VectorXd& x, Eigen::VectorXd& x_prime);
    };
}
