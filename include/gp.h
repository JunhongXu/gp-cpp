#pragma once
#include <memory>
#include <utility>
#include "kernel.h"

namespace gp{
    /** Gaussian Process class.*/
    class GaussianProcess{
        public:
            /** Constructor for GP with one-dimensional target value.
             *  @param X the input feature vector
             *  @param Y the target value
             *  @param prior_var hyperparameter representing the variance of the noise
             *  @param kernel the kernel to be used */
            GaussianProcess(std::vector<Eigen::VectorXd> X, std::vector<double> Y, double prior_var, std::unique_ptr<Kernel> kernel);

            void compute_precision_matrix();

            /** Add traning data point into the training set
             *  @param x input point feature vector
             *  @param y target value */
            void add_point(Eigen::VectorXd x, double y);

            /** Make inference on a new data point given the observed ones,
             *  i.e., p(t_{N+1}|t_{1...N}, x_{N+1}).
             *  @param new_x the new data point which we do inference on
             *  @return mean and variance pair of the conditional distribution.*/
            std::pair<double, double> inference(Eigen::VectorXd& new_x);

            int num_training_samples(){ return X.size(); }

            /** Get precision matrix*/
            Eigen::MatrixXd& get_precision_matrix(){ return precision_matrix; }

        protected:
            // kernel's ownership will be moved from the outside of the class to this class
            std::unique_ptr<Kernel> kernel;

            // prior variance
            double prior_var;

            // training data
            std::vector<Eigen::VectorXd> X;
            std::vector<double> Y;

            // inverse covariance matrix
            // TODO: we actually only need upper or lower triangle
            Eigen::MatrixXd precision_matrix;
    };
}
