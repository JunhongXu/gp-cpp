#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <string>
#include <map>

namespace gp{
    /**Kernel function base class. It represents a positive-definite function of two inputs x, x'.*/
    class Kernel{
        public:
            /** Constructor for parent class kernel.
             *  @param inpt_dim input dimension (the dimension of x)
             *  @param param_dim dimensionality of the parameters
             **/
            Kernel(int inpt_dim, int param_dim): inpt_dim(inpt_dim), param_dim(param_dim){
                name = "Parent";
            }

            // default deconstructor
            ~Kernel(){}

            /** Compute the covariance matrix between two sets of input values based on the kernel function k:
             *  x1 and x2, K_{nm} = k(x_{1n}, x_{1m}).
             *  @param x1 first input data matrix with dimension of num_example_1 by inpt_dim
             *  @param x2 second input data matrix with dimension of num_example_2 by inpt_dim
             *  @return the computed covariance matrix. */
            virtual Eigen::MatrixXd compute_K(std::vector<Eigen::VectorXd>& x1, std::vector<Eigen::VectorXd>& x2){
                std::cout<<"Not implemented!"<<std::endl;
                exit(1);
            }

            /** Overaloading << operator to be used for std::cout*/
            friend std::ostream& operator<< (std::ostream& os, const Kernel& kernel){
                std::string s = "-------------\n";
                s += "Kernel type: "+kernel.name + '\n';
                s += "Input dim: " + std::to_string(kernel.get_input_dim()) +  '\n';
                s += "Parameter dim: " + std::to_string(kernel.get_param_dim()) +  '\n';
                s += "Parameters: ";
                for(auto& param: kernel.params){
                    s += param.first + " " + std::to_string(param.second) + ", ";
                }
                s += "\n-------------";
                os << s;
                return os;
            }

            /** Get the dimensionality of input*/
            int get_input_dim() const { return inpt_dim; }
            /** Get the dimensionality of the parameter*/
            int get_param_dim() const { return param_dim; }


        protected:
            // input dimension of this kernel
            int inpt_dim;
            // parameter dimension
            int param_dim;
            // name of this kernel
            std::string name;
            // parameter is a map from string (key) to parameter value
            std::map<std::string, double> params;
    };
}

