#ifndef PATH_SMOOTHER_HPP
#define PATH_SMOOTHER_HPP

#include "cubic_spline.hpp"
#include "lbfgs.hpp"

#include <Eigen/Eigen>

#include <cmath>
#include <cfloat>
#include <iostream>
#include <vector>
#include <iomanip>

namespace path_smoother
{

    class PathSmoother
    {
    private:
        cubic_spline::CubicSpline cubSpline;

        int pieceN;
        Eigen::Matrix3Xd diskObstacles;
        double penaltyWeight;
        Eigen::Vector2d headP;
        Eigen::Vector2d tailP;
        Eigen::Matrix2Xd points;
        Eigen::Matrix2Xd gradByPoints;

        lbfgs::lbfgs_parameter_t lbfgs_params;

    private:
        static inline double costFunction(void *ptr,
                                          const Eigen::VectorXd &x,
                                          Eigen::VectorXd &g)
        {
            // Hints:This is a function which define the costfunction, you can use
            // the content which *ptr point to initialize a quote of Class:
            // PathSmoother, the you can use the quote to visit the cubSpline
            // TODO
            auto instance = reinterpret_cast<path_smoother::PathSmoother *>(ptr);
            const int points_nums = instance->pieceN - 1;
            Eigen::Matrix2Xd grad;

            grad.resize(2, points_nums);
            grad.setZero();

            double cost = 0.0;

            // Step 1: 送入目标点，计算曲线系数
            Eigen::Matrix2Xd inPs;
            inPs.resize(2, points_nums);
            inPs.row(0) = x.head(points_nums);
            inPs.row(1) = x.tail(points_nums);
            instance->cubSpline.setInnerPoints(inPs);

            // Step 2: 计算能量函数值和梯度
            double energy = 0.0;
            Eigen::Matrix2Xd energy_grad;
            energy_grad.resize(2, points_nums);
            energy_grad.setZero();
            instance->cubSpline.getStretchEnergy(energy);
            instance->cubSpline.getGrad(energy_grad);

            cost += energy;
            grad += energy_grad;

            // Step3: 计算碰撞函数和代价
            double obstacles = 0.0;
            Eigen::Matrix2Xd potential_grad;
            potential_grad.resize(2, points_nums);
            potential_grad.setZero();

            for (int i = 0; i < points_nums; ++i)
            {
                for (int j = 0; j < instance->diskObstacles.cols(); ++j)
                {
                    Eigen::Vector2d diff = inPs.col(i) - instance->diskObstacles.col(j).head(2);
                    double distance = diff.norm();
                    double delta = instance->diskObstacles(2, j) - distance;

                    if (delta > 0.0)
                    {
                        obstacles += instance->penaltyWeight * delta;
                        potential_grad.col(i) += instance->penaltyWeight * ( -diff / distance);
                    }
                }
            }
            cost += obstacles;
            grad += potential_grad;

            // Step4: 得到结果
            g.setZero();
            g.head(points_nums) = grad.row(0).transpose();
            g.tail(points_nums) = grad.row(1).transpose();


            // std::cout << std::setprecision(10)
            std::cout << "------------------------" << "\n";
            std::cout << "Function Value: " << cost << "\n";
            std::cout << "Gradient Inf Norm: " << g.cwiseAbs().maxCoeff() << "\n";
            std::cout << "------------------------" << "\n";

            return cost;
        }

    public:
        inline bool setup(const Eigen::Vector2d &initialP,
                          const Eigen::Vector2d &terminalP,
                          const int &pieceNum,
                          const Eigen::Matrix3Xd &diskObs,
                          const double penaWeight)
        {
            pieceN = pieceNum;
            diskObstacles = diskObs;
            penaltyWeight = penaWeight;
            headP = initialP;
            tailP = terminalP;

            cubSpline.setConditions(headP, tailP, pieceN);

            points.resize(2, pieceN - 1);
            gradByPoints.resize(2, pieceN - 1);

            return true;
        }


        inline double optimize(CubicCurve &curve,
                               const Eigen::Matrix2Xd &iniInPs,
                               const double &relCostTol)
        {
            // NOT TODO
            Eigen::VectorXd x(pieceN * 2 - 2);
            Eigen::Map<Eigen::Matrix2Xd> innerP(x.data(), 2, pieceN - 1);
            innerP = iniInPs;

            double minCost = 0.0;
            lbfgs_params.mem_size = 64;
            lbfgs_params.past = 3;
            lbfgs_params.min_step = 1.0e-32;
            lbfgs_params.g_epsilon = 1.0e-6;
            lbfgs_params.delta = relCostTol;

            int ret = lbfgs::lbfgs_optimize(x,
                                            minCost,
                                            &PathSmoother::costFunction,
                                            nullptr,
                                            this,
                                            lbfgs_params);

            if (ret >= 0)
            {
                // cubSpline.setInnerPoints(innerP);
                cubSpline.getCurve(curve);
            }
            else
            {
                curve.clear();
                minCost = INFINITY;
                std::cout << "Optimization Failed: "
                          << lbfgs::lbfgs_strerror(ret)
                          << std::endl;
            }

            return minCost;
        }
    };

}

#endif