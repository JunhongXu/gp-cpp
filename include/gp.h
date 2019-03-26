#pragma once
#include <memory>
#include <utility>
#include "kernel.h"

namespace gp{
    /** Gaussian Process class.*/
    class GaussianProcess{
        public:
            GaussianProcess(std::vector<Eigen::VectorXd>& X, std::vector<Eigen::VectorXd>& Y, std::unique_ptr<Kernel> kernel);

            void compute_cov();

            /** Make inference on a new data point given the observed ones,
             *  i.e., p(t_{N+1}|t_{1...N}, x_{N+1}).
             *  @param new_x the new data point which we do inference on
             *  @return mean and variance pair of the conditional distribution.*/
            std::pair<double, double> inference(Eigen::VectorXd& new_x);

        protected:
            // kernel's ownership will be moved from the outside of the class to this class
            std::unique_ptr<Kernel> kernel;
    };
}
