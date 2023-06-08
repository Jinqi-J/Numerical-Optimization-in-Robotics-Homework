#ifndef CUBIC_SPLINE_HPP
#define CUBIC_SPLINE_HPP

#include "cubic_curve.hpp"

#include <Eigen/Eigen>

#include <cmath>
#include <vector>

namespace cubic_spline
{

    // The banded system class is used for solving
    // banded linear system Ax=b efficiently.
    // A is an N*N band matrix with lower band width lowerBw
    // and upper band width upperBw.
    // Banded LU factorization has O(N) time complexity.
    class BandedSystem
    {
    public:
        // The size of A, as well as the lower/upper
        // banded width p/q are needed
        inline void create(const int &n, const int &p, const int &q)
        {
            // In case of re-creating before destroying
            destroy();
            N = n;
            lowerBw = p;
            upperBw = q;
            int actualSize = N * (lowerBw + upperBw + 1);
            ptrData = new double[actualSize];
            std::fill_n(ptrData, actualSize, 0.0);
            return;
        }

        inline void destroy()
        {
            if (ptrData != nullptr)
            {
                delete[] ptrData;
                ptrData = nullptr;
            }
            return;
        }

    private:
        int N;
        int lowerBw;
        int upperBw;
        // Compulsory nullptr initialization here
        double *ptrData = nullptr;

    public:
        // Reset the matrix to zero
        inline void reset(void)
        {
            std::fill_n(ptrData, N * (lowerBw + upperBw + 1), 0.0);
            return;
        }

        // The band matrix is stored as suggested in "Matrix Computation"
        inline const double &operator()(const int &i, const int &j) const
        {
            return ptrData[(i - j + upperBw) * N + j];
        }

        inline double &operator()(const int &i, const int &j)
        {
            return ptrData[(i - j + upperBw) * N + j];
        }

        // This function conducts banded LU factorization in place
        // Note that NO PIVOT is applied on the matrix "A" for efficiency!!!
        inline void factorizeLU()
        {
            int iM, jM;
            double cVl;
            for (int k = 0; k <= N - 2; ++k)
            {
                iM = std::min(k + lowerBw, N - 1);
                cVl = operator()(k, k);
                for (int i = k + 1; i <= iM; ++i)
                {
                    if (operator()(i, k) != 0.0)
                    {
                        operator()(i, k) /= cVl;
                    }
                }
                jM = std::min(k + upperBw, N - 1);
                for (int j = k + 1; j <= jM; ++j)
                {
                    cVl = operator()(k, j);
                    if (cVl != 0.0)
                    {
                        for (int i = k + 1; i <= iM; ++i)
                        {
                            if (operator()(i, k) != 0.0)
                            {
                                operator()(i, j) -= operator()(i, k) * cVl;
                            }
                        }
                    }
                }
            }
            return;
        }

        // This function solves Ax=b, then stores x in b
        // The input b is required to be N*m, i.e.,
        // m vectors to be solved.
        template <typename EIGENMAT>
        inline void solve(EIGENMAT &b) const
        {
            int iM;
            for (int j = 0; j <= N - 1; ++j)
            {
                iM = std::min(j + lowerBw, N - 1);
                for (int i = j + 1; i <= iM; ++i)
                {
                    if (operator()(i, j) != 0.0)
                    {
                        b.row(i) -= operator()(i, j) * b.row(j);
                    }
                }
            }
            for (int j = N - 1; j >= 0; --j)
            {
                b.row(j) /= operator()(j, j);
                iM = std::max(0, j - upperBw);
                for (int i = iM; i <= j - 1; ++i)
                {
                    if (operator()(i, j) != 0.0)
                    {
                        b.row(i) -= operator()(i, j) * b.row(j);
                    }
                }
            }
            return;
        }

        // This function solves ATx=b, then stores x in b
        // The input b is required to be N*m, i.e.,
        // m vectors to be solved.
        template <typename EIGENMAT>
        inline void solveAdj(EIGENMAT &b) const
        {
            int iM;
            for (int j = 0; j <= N - 1; ++j)
            {
                b.row(j) /= operator()(j, j);
                iM = std::min(j + upperBw, N - 1);
                for (int i = j + 1; i <= iM; ++i)
                {
                    if (operator()(j, i) != 0.0)
                    {
                        b.row(i) -= operator()(j, i) * b.row(j);
                    }
                }
            }
            for (int j = N - 1; j >= 0; --j)
            {
                iM = std::max(0, j - lowerBw);
                for (int i = iM; i <= j - 1; ++i)
                {
                    if (operator()(j, i) != 0.0)
                    {
                        b.row(i) -= operator()(j, i) * b.row(j);
                    }
                }
            }
        }
    };

    class CubicSpline
    {
    public:
        CubicSpline() = default;
        ~CubicSpline() { A.destroy(); }

    private:
        int N;
        Eigen::Vector2d headP;
        Eigen::Vector2d tailP;
        BandedSystem A;
        Eigen::MatrixX2d b;
        Eigen::Matrix2Xd coeff;

    public:
        inline void setConditions(const Eigen::Vector2d &headPos,
                                  const Eigen::Vector2d &tailPos,
                                  const int &pieceNum)
        {
            // This is a function to set the coefficients matrix A (AD=x represent the curve)
            // which has no 'return'.
            N = pieceNum;
            headP = headPos;
            tailP = tailPos;

            // TODO

            // 建立矩阵
            A.create(N - 1, 3, 3);
            b.resize(N - 1, 2);

            return;
        }

        inline void setInnerPoints(const Eigen::Ref<const Eigen::Matrix2Xd> &inPs)
        {
            // This is a function to set the inner point x (AD=x) of the curve which
            // has no return.
            // TODO
            A.reset();

            for (int i = 1; i < N - 2; i++)
            {
                A(i, i - 1) = 1;
                A(i, i)     = 4;
                A(i, i + 1) = 1;                
            }

            A(0, 0) = 4;
            A(0, 1) = 1;
            A(N - 2, N - 3) = 1;
            A(N - 2, N - 2) = 4;

            b.setZero();
            
            for (int i = 1; i < N - 2; ++i)
            {
                b.row(i) = 3 * (inPs.col(i + 1).transpose() - inPs.col(i - 1).transpose());
            }

            b.row(N - 2) = 3 * (tailP.transpose() - inPs.col(N - 3).transpose());
            b.row(0) = 3 * (inPs.col(1).transpose() - headP.transpose());

            A.factorizeLU();
            A.solve(b);

            // D 矩阵的 1 ～ N-1 行被存放在了 b 矩阵中
            Eigen::MatrixXd D;
            D.resize(N, 2);
            D.setZero();

            for (int i = 0; i < N - 1; ++i)
            {
                D.row(i + 1) = b.row(i);
            }

            // 系数矩阵 coeff = [ai, bi, ci, di];
            coeff.resize(2, 4 * N);
            coeff.setZero();
            
            for (int i = 1; i < N - 1; ++i)
            {
                coeff.col(4 * i + 0) = inPs.col(i - 1);
                coeff.col(4 * i + 1) = D.row(i).transpose();
                coeff.col(4 * i + 2) = 3 * (inPs.col(i) - inPs.col(i - 1)) - 2 *  D.row(i).transpose() -  D.row(i + 1).transpose();
                coeff.col(4 * i + 3) = 2 * (inPs.col(i - 1) - inPs.col(i)) + 1 *  D.row(i).transpose() +  D.row(i + 1).transpose();
            }

            coeff.col(0) = headP;
            coeff.col(1) = Eigen::Vector2d::Zero();
            coeff.col(2) = 3 * (inPs.col(0) - headP) - D.row(1).transpose();
            coeff.col(3) = 2 * (headP - inPs.col(0)) + D.row(1).transpose();

            coeff.col(4 * (N - 1) + 0) = inPs.col(N - 2);
            coeff.col(4 * (N - 1) + 1) = D.row(N - 1).transpose();
            coeff.col(4 * (N - 1) + 2) = 3 * (tailP - inPs.col(N - 2)) - 2 * D.row(N - 1).transpose();
            coeff.col(4 * (N - 1) + 3) = 2 * (inPs.col(N - 2) - tailP) + 1 * D.row(N - 1).transpose();
            
            b.resize(4 * N, 2);
            b = coeff.transpose();

            return;
        }

        inline void getCurve(CubicCurve &curve) const
        {
            // Not TODO
            curve.clear();
            curve.reserve(N);
            for (int i = 0; i < N; ++i)
            {
                curve.emplace_back(1.0,
                                   b.block<4, 2>(4 * i, 0)
                                       .transpose()
                                       .rowwise()
                                       .reverse());
            }
            return;
        }

        inline void getStretchEnergy(double &energy) const
        {
            // An example for you to finish the other function
            energy = 0.0;
            for (int i = 0; i < N; ++i)
            {
                energy +=  4.0 * b.row(4 * i + 2).squaredNorm() +
                          12.0 * b.row(4 * i + 2).dot(b.row(4 * i + 3)) +
                          12.0 * b.row(4 * i + 3).squaredNorm();
            }
            return;
        }

        inline const Eigen::MatrixX2d &getCoeffs(void) const
        {
            return b;
        }

        inline void getGrad(Eigen::Ref<Eigen::Matrix2Xd> gradByPoints) const
        {
            // This is a function to get the Grad
            // TODO

            // Step 1 计算 dD/dx
            Eigen::MatrixXd A_tmp;
            Eigen::MatrixXd partial_B;

            A_tmp.resize(N - 1, N - 1);
            A_tmp.setZero();

            partial_B.resize(N - 1, N - 1);
            partial_B.setZero();
            
            for (int i = 1; i < N - 2; i++)
            {
                A_tmp(i, i - 1) = 1;
                A_tmp(i, i) = 4;
                A_tmp(i, i + 1) = 1;
                partial_B(i, i - 1) = -3;
                partial_B(i, i + 1) = 3;                
            }

            A_tmp(0, 0) = 4;
            A_tmp(0, 1) = 1;
            partial_B(0, 1) = 3;            

            A_tmp(N - 2, N - 3) = 1;
            A_tmp(N - 2, N - 2) = 4;
            partial_B(N - 2, N - 3) = -3;

            //Eigen::MatrixXd A_inv = A_tmp.inverse();
            Eigen::MatrixXd partial_D = A_tmp.inverse() * partial_B;

            // * step2：获取 d(xi - xi+1)/dx
            Eigen::MatrixXd partial_diff_x;
            partial_diff_x.resize(N, N - 1);
            partial_diff_x.setZero();
            partial_diff_x(0, 0) = -1;
            partial_diff_x(N - 1, N - 2) = 1;

            for (int i = 1; i < N - 1; ++i)
            {
                partial_diff_x(i, i) = -1;
                partial_diff_x(i, i - 1) = 1;
            }

            // Step 3: 获取 dc/dx & dd/dx
            Eigen::MatrixXd partial_c;
            partial_c.resize(N, N - 1);
            partial_c.setZero();

            Eigen::MatrixXd partial_d;
            partial_d.resize(N, N - 1);
            partial_d.setZero();

            partial_c.row(0) = -3 * partial_diff_x.row(0) - partial_D.row(0);
            partial_d.row(0) = 2 * partial_diff_x.row(0) + partial_D.row(0);

            for (int i = 1; i < N - 1; ++i)
            {
                partial_c.row(i) = -3 * partial_diff_x.row(i) - 2 * partial_D.row(i - 1) - partial_D.row(i);
                partial_d.row(i) = 2 * partial_diff_x.row(i) + partial_D.row(i - 1) + partial_D.row(i);
            }

            partial_c.row(N - 1) = -3 * partial_diff_x.row(N - 1) - 2 * partial_D.row(N - 2);
            partial_d.row(N - 1) = 2 * partial_diff_x.row(N - 1) + partial_D.row(N - 2);

            // Step 4: 填入 gradByPoints
            gradByPoints.setZero();

            for (int i = 0; i < N; ++i)
            {
                Eigen::Vector2d c_i = coeff.col(4 * i + 2);
                Eigen::Vector2d d_i = coeff.col(4 * i + 3);
                
                gradByPoints += (24 * d_i + 12 * c_i) * partial_d.row(i) + (12 * d_i + 8 * c_i) * partial_c.row(i);
            }
        }
    };
}

#endif