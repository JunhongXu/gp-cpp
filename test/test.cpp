#define CATCH_CONFIG_MAIN
#include <random>
#include <catch.hpp>
#include <math.h>
#include "gp.h"
#include "rbf.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

TEST_CASE("RBF kernel test"){
    // one dimensional RBF
    gp::RBF rbf(1, 1, 1);
    std::cout<<rbf<<std::endl;
    SECTION("compute variance between 1 and 2, 1 and 3, 1 and 2 should be larger than 1 and 3"){
        std::vector<Eigen::VectorXd> x1 = {Eigen::Matrix<double, 1, 1>(1.0)};
        std::vector<Eigen::VectorXd> x2 = {Eigen::Matrix<double, 1, 1>(2.0), Eigen::Matrix<double, 1, 1>(3.0)};
        auto result = rbf.compute_K(x1, x2);
        REQUIRE(result(0) > result(1));
    }
}

TEST_CASE("GP test on sin data"){
    double noise_variance = 0.1;
    std::default_random_engine generator;
    std::normal_distribution<double> dist(0.0, noise_variance);

    std::vector<Eigen::VectorXd> X;
    std::vector<double> Y;
    for(int i = 0; i < 10; i++){
        double x = i * 0.1;
        double y = sin(x) + dist(generator);
        Y.push_back(y);
        X.push_back(Eigen::Matrix<double, 1, 1>(x));
        std::cout<<"Data "<<x<<","<<y<<std::endl;
    }

    gp::GaussianProcess gaussian_process(X, Y, noise_variance, std::unique_ptr<gp::Kernel>(new gp::RBF(1, 1, 1)));


    SECTION("plot the graph"){

        std::vector<double> predictions, upper_y, lower_y, test_x;

        for(int i = 0; i < 10; i++){
            std::cout<<"Data "<<i * 0.2<<std::endl;
            Eigen::VectorXd new_x = Eigen::Matrix<double, 1, 1>(i * 0.2);
            auto mean_var = gaussian_process.inference(new_x);
            predictions.push_back(mean_var.first);
            upper_y.push_back(mean_var.first + sqrt(mean_var.second) * 2);
            lower_y.push_back(mean_var.first - sqrt(mean_var.second) * 2);
            test_x.push_back(i * 0.2);
            std::cout<<"Mean is "<<mean_var.first<<std::endl;
            std::cout<<"Standard deviation is "<<sqrt(mean_var.second)<<std::endl;
        }

        // draw training data
        std::vector<double> _X;
        for(auto& _x: X){
            _X.push_back(_x(0));
        }
        std::map<std::string, std::string> keywords;
        keywords["alpha"] = "0.4";
        keywords["color"] = "grey";
        plt::plot(_X, Y, "o");
        plt::fill_between(test_x, lower_y, upper_y, keywords);
        plt::plot(test_x, predictions, "o");
        plt::legend();
        plt::show();
    }
}
