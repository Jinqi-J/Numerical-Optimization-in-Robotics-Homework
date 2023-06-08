#include <iostream>

#include "sdqp/sdqp.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char **argv)
{
    int m = 5;
    Eigen::Matrix<double, 3, 3> Q;
    Eigen::Matrix<double, 3, 1> c;
    Eigen::Matrix<double, 3, 1> x;        // decision variables
    Eigen::Matrix<double, -1, 3> A(m, 3); // constraint matrix
    Eigen::VectorXd b(m);                 // constraint bound

    Q << 8.0, -6, 2.0, -6, 6, -3, 2, -3, 2;
    c << 1.0, 3.0, -2.0;

    A << 0.0, -1.0, -2.0,
        -1.0, 1.0, -3.0,
        1.0, -2.0, 0.0,
        -1.0, -2.0, -1.0,
        3.0, 5.0, 1.0;
    b << -1, 2, 7, 2, -1;

    Eigen::Matrix<double, 3, 1> y;        // decision variables
    y << -103.0/97, -93.0/97, 95.0/97;

    double coeff = 1;
    Vector3d xi = Vector3d::Zero();
    while(true)
    {
            Eigen::Matrix<double, 3, 3> P= Q + 2 * Eigen::Matrix3d::Identity();
            sdqp::sdqp<3>(P, c - 2 * xi, A, b, x);
            cout << "x = " << x << endl;
            if( (x - xi).norm() <= atof(argv[1]))
            {
                break;
            }
            xi = x;
    }

    std::cout << "coeff :" << coeff << std::endl;
    std::cout << "optimal sol: " << x.transpose() << std::endl;
    std::cout << "optimal obj: " << 0.5 * x.transpose() * Q * x + c.transpose() * x<< std::endl;

    return 0;
}
