#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include "rbf.h"

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
