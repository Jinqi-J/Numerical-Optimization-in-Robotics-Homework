# SDQP

SDQP: Small-Dimensional Strictly Convex Quadratic Programming in Linear Time

# About

1. This solver is super efficient for small-dimensional strictly convex QP with any constraint number, mostly encountered in computational geometry. It enjoys __linear complexity about the constraint number__.

2. The speed is faster than most numerical solvers in small-dimensional LP (<10) with a large constraints number (>100).

3. This solver computes __exact solutions or report infeasibility__.

4. This solver generalizes [Seidel's algorithm](https://link.springer.com/article/10.1007/BF02574699) __from LP to strictly convex QP__.

4. This solver is very elegant thus only [a header file](https://github.com/ZJU-FAST-Lab/SDQP/blob/main/include/sdqp/sdqp.hpp) with less than 400 lines is all you need.

If our lib helps your research, please cite us 
```
@misc{WANG2022SDQP,
    title={{SDQP: Small-Dimensional Strictly Convex Quadratic Programming in Linear Time}}, 
    author={Wang, Zhepei and Gao, Fei}, 
    year={2022},
    url={https://github.com/ZJU-FAST-Lab/SDQP}
}
```

# Interface

To solve a linear programming:

        min 0.5 x' Q x + c' x,
        s.t. A x <= b,

where x and c are d-dimensional vectors, Q an dxd positive definite matrix, b an m-dimensional vector, A an mxd matrix. It is assumed that d is small (<10) while m can be arbitrary value (1<= m <= 1e+8).

Only one function is all you need:

    double sdqp(const Eigen::Matrix<double, d, d> &Q,
                const Eigen::Matrix<double, d, 1> &c,
                const Eigen::Matrix<double, -1, d> &A,
                const Eigen::Matrix<double, -1, 1> &b,
                Eigen::Matrix<double, d, 1> &x)

Input:

        Q: positive definite matrix
        c: linear coefficient vector
        A: constraint matrix
        b: constraint bound vector

Output:

        x: optimal solution if solved
        return: finite value if solved
                infinity if infeasible

# Reference

1. Seidel, R., 1991. Small-dimensional linear programming and convex hulls made easy. Discrete & Computational Geometry, 6(3), pp.423-434.

# Maintaince and Ack

Thank Zijie Chen for fixing the conversion from QP to minimum norm.

If any bug, please contact [Zhepei Wang](https://zhepeiwang.github.io/) (<wangzhepei@live.com>).
