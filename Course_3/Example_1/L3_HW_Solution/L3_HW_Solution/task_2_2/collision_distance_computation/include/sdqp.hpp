/*
    MIT License

    Copyright (c) 2022 Zhepei Wang (wangzhepei@live.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef SDQP_HPP
#define SDQP_HPP

#include <Eigen/Eigen>
#include <cmath>
#include <random>

namespace sdqp
{
    constexpr double eps = 1.0e-12;

    enum
    {
        MINIMUM = 0,
        INFEASIBLE,
    };

    template <int d>
    inline void set_zero(double *x)
    {
        for (int i = 0; i < d; ++i)
        {
            x[i] = 0.0;
        }
        return;
    }

    /**
     * @brief x中的前d个元素与y中前d个元素做点乘
     * 
     * @tparam d 维度
     * @param x 
     * @param y 
     * @return double 
     */
    template <int d>
    inline double dot(const double *x,
                      const double *y)
    {
        double s = 0.0;
        for (int i = 0; i < d; ++i)
        {
            s += x[i] * y[i];
        }
        return s;
    }

    /**
     * @brief 计算x前d个元素的平方和
     * 
     * @tparam d 维度
     * @param x 
     * @return double 
     */
    template <int d>
    inline double sqr_norm(const double *x)
    {
        double s = 0.0;
        for (int i = 0; i < d; ++i)
        {
            s += x[i] * x[i];
        }
        return s;
    }

    /**
     * @brief 将原点投影到约束平面x上
     * 
     * @tparam d 维度
     * @param x 约束平面 法向量ABC+负的D
     * @param s PPT中的f/g^T*g
     * @param y 原点在约束平面上的投影
     */
    template <int d>
    inline void mul(const double *x,
                    const double s,
                    double *y)
    {
        for (int i = 0; i < d; ++i)
        {
            y[i] = x[i] * s;
        }
        return;
    }

    template <int d>
    inline int max_abs(const double *x)
    {
        int id = 0;
        double mag = std::fabs(x[0]);
        for (int i = 1; i < d; ++i)
        {
            const double s = std::fabs(x[i]);
            if (s > mag)
            {
                id = i;
                mag = s;
            }
        }
        return id;
    }

    template <int d>
    inline void cpy(const double *x,
                    double *y)
    {
        for (int i = 0; i < d; ++i)
        {
            y[i] = x[i];
        }
        return;
    }

    inline int move_to_front(const int i,
                             int *next,
                             int *prev)
    {
        if (i == 0 || i == next[0])
        {
            return i;
        }
        const int previ = prev[i];
        next[prev[i]] = next[i];
        prev[next[i]] = prev[i];
        next[i] = next[0];
        prev[i] = 0;
        prev[next[i]] = i;
        next[0] = i;
        return previ;
    }

    template <int d>
    inline int min_norm(const double *halves, // 以列为存储顺序的约束矩阵
                        const int n, // 不等式约束个数
                        const int m,
                        double *opt,
                        double *work,
                        int *next,
                        int *prev)
    {
        int status = MINIMUM;
        set_zero<d>(opt);
        if (m <= 0)
        {
            return status;
        }

        double *reflx = work;
        double *new_opt = reflx + d;
        double *new_halves = new_opt + (d - 1);
        double *new_work = new_halves + n * d;
        double new_origin[d] = {0.0};

        for (int i = 0; i != m; i = next[i])
        {
            // 取出约束矩阵中的第i列
            const double *plane_i = halves + (d + 1) * i;
            // 状态不满足约束 这里的plane_i[d]在传进来的时候已经取负号
            if (dot<d>(opt, plane_i) + plane_i[d] > (d + 1) * eps)
            {
                // 法向量模长的平方
                const double s = sqr_norm<d>(plane_i);
                // 防止分母太小 s会在下面做分母用
                if (s < (d + 1) * eps * eps)
                {
                    return INFEASIBLE;
                }
                // 原点在约束平面上的投影
                // 加负号是因为plane_i[d]传进来的时候已经取了负号
                // 同时这里的new_origin相当于ppt中的v和g
                mul<d>(plane_i, -plane_i[d] / s, new_origin);

                // 第一次不进行投影
                if (i == 0)
                {
                    continue;
                }

                // 选择绝对值最大的元素 其index作为投影的方向
                const int id = max_abs<d>(new_origin);
                const double g_norm = std::sqrt(sqr_norm<d>(new_origin));
                // u = g + sgn(g_i)*||g||*e_i
                cpy<d>(new_origin, reflx);
                if(new_origin[id] < 0.0){
                    reflx[id] += -g_norm; // u = reflx
                } else{
                    reflx[id] += g_norm;
                }

                       
                //////////////////////////////// HOMEWORK START ////////////////////////////////
                //
                // MISSION TO BE ACCOMPLISHED:
                //
                // now we know the best solution "opt" violates the i-th halfspace. Therefore,
                // please project all previous i halfspaces (from the 0-th to the (i-1)-th one)
                // onto the boundary of the i-th halfspace, then store all projected halfspaces
                // in the double-type-c-array "new_halves".
                // If you successfully complete the mission, the sdqp_example should prints
                //     optimal sol: 4.11111 9.15556 4.50022
                //     optimal obj: 201.14
                //     cons precision: *.********e-16
                // This means you obtained the correct exact solution (precision near DBL_EPSILON)
                //
                // VARIABLES YOU NEED TO ACCESS:
                //
                // opt is a d-dimensional double-type-c-array
                // opt contains an optimal solution that meets all linear constraints from the 0-th
                // to the (i-1)-th one, but it is known to violate the i-th halfspace here
                //
                // new_origin is also a d-dimensional double-type-c-array
                // new_origin contains the minimum norm point on the boundary of the i-th plane
                //
                // you should calculate the vector 'u' of Householder (you can review this concept in
                // the course) with the i_d th natural normal Orthogonal basis and store it in the
                // Array reflx

                // you can read all previous halfspaces via the iteration below
                //
                //     for (int j = 0; j != i; j = next[j])
                //     {
                //         const double *halfspace = halves + (d + 1) * j;
                //         // thus the j-th halfspace is the inequality below
                //         // halfspace[0] * x1 + halfspace[1] * x2 + ... + halfspace[d-1] * xd + halfspace[d] <= 0
                //     }
                //
                // you can write or store all your projected halfspaces via the iteration below
                //
                //     for (int j = 0; j != i; j = next[j])
                //     {
                //         double *proj_halfspace = new_halves + d * j;
                //         // thus the j-th projected halfspace is the inequality below
                //         // proj_halfspace[0] * y1 + proj_halfspace[1] * y2 + ... + proj_halfspace[d-2] * y(d-1) + proj_halfspace[d-1] <= 0
                //         // y1 to y(d-1) is the new coordinate constructed on the boundary of the i-th halfspace
                //     }
                //
                // TODO
                const double c = -2.0 / sqr_norm<d>(reflx);
                // 将先前的约束都投影到约束i平面上 所以遍历到i平面马上退出
                for(int j=0; j!=i; j=next[j]){
                    double* new_plane = new_halves + j*d;
                    const double* old_plane = halves + j*(d+1);

                    // 按ppt上的思路 投影后的约束变成AM和b-Av
                    // 其中 M的列向量为H的(d-1)个行向量 为了加速 这里直接算出AM的结果
                    const double cAiu = c * dot<d>(old_plane,reflx);
                    for(int k=0; k<d; k++){
                        if(k<id){
                            new_plane[k] = old_plane[k] + reflx[k]*cAiu;
                        }
                        else if(k>id){
                            new_plane[k-1] = old_plane[k] + reflx[k]*cAiu;
                        }
                    } 
                    // 正常是b`=b-Av 注意这里传入的是-b` old_plane[d]=-b
                    new_plane[d-1] = dot<d>(new_origin,old_plane) + old_plane[d];
                }        

                status = min_norm<d - 1>(new_halves, n, i, new_opt, new_work, next, prev);

                if (status == INFEASIBLE)
                {
                    return INFEASIBLE;
                }

                double coeff = 0.0;
                for (int j = 0; j < d; ++j)
                {
                    const int k = j < id ? j : j - 1;
                    coeff += j != id ? reflx[j] * new_opt[k] : 0.0;
                }
                coeff *= -2.0 / sqr_norm<d>(reflx);
                for (int j = 0; j < d; ++j)
                {
                    const int k = j < id ? j : j - 1;
                    opt[j] = new_origin[j] += j != id ? new_opt[k] + reflx[j] * coeff : reflx[j] * coeff;
                }

                // std::cout<<"**************"<<std::endl;
                // std::cout<<"i: "<<i<<std::endl;

                i = move_to_front(i, next, prev);
                
                // std::cout<<"move_to_front i: "<<i<<std::endl;
                // Eigen::VectorXi next_v = Eigen::Map<Eigen::VectorXi>(next,n);
                // std::cout<<"next: "<<std::endl<<next_v.transpose()<<std::endl;
                // Eigen::VectorXi prev_v = Eigen::Map<Eigen::VectorXi>(prev,n+1);
                // std::cout<<"prev: "<<std::endl<<prev_v.transpose()<<std::endl;
                // Eigen::VectorXd opt_v = Eigen::Map<Eigen::VectorXd>(opt,3);
                // std::cout<<"opt_v: "<<opt_v.transpose()<<std::endl;
            }
        }

        return status;
    }

    template <>
    inline int min_norm<1>(const double *halves,
                           const int n, // 一共有n个约束
                           const int m, // 在第m个约束上的投影
                           double *opt,
                           double *work,
                           int *next,
                           int *prev)
    {
        opt[0] = 0.0;
        bool l = false;
        bool r = false;
        // 遍历除第m个约束外所有约束
        for (int i = 0; i != m; i = next[i])
        {
            // 当递归到约束只有1维时 halves的矩阵形式为(按列顺序存储)
            //  a0  a1 ...  an
            // -b0 -b1 ... -bn
            const double a = halves[2 * i];
            const double b = halves[2 * i + 1];
            if (a * opt[0] + b > 2.0 * eps)
            {
                // 防止分母太小
                if (std::fabs(a) < 2.0 * eps)
                {
                    return INFEASIBLE;
                }

                l = l || a < 0.0;
                r = r || a > 0.0;
                // 由于是最小化二范数 最小值要接近于0 所以不可能出现大于右边又小于左边的情况
                // 出现这种情况就是没有可行解
                if (l && r)
                {
                    return INFEASIBLE;
                }

                opt[0] = -b / a;
            }
        }

        return MINIMUM;
    }

    /**
     * @brief 随机置换算法 Fisher-Yates洗牌算法
     * 目标是给定n个数 随机生成一个1~n的排列
     * 其中 这种随机是等概率的 即可以生成n!个排列 每个排列发生的概率是1/n!
     * 
     * @param n 
     * @param p 
     */
    inline void rand_permutation(const int n,
                                 int *p)
    {
        typedef std::uniform_int_distribution<int> rand_int;
        typedef rand_int::param_type rand_range;
        // BUG: 漏了个随机数种子
        std::random_device rd;
        static std::mt19937_64 gen(rd());
        // origin
        // static std::mt19937_64 gen();
        static rand_int rdi(0, 1);
        int j, k;
        for (int i = 0; i < n; ++i)
        {
            p[i] = i;
        }
        // 从0开始做置换 与ppt中不同 ppt是从n开始换
        for (int i = 0; i < n; ++i)
        {
            rdi.param(rand_range(0, n - i - 1));
            // 所以这里要加个偏置i
            j = rdi(gen) + i;
            k = p[j];
            p[j] = p[i];
            p[i] = k;
        }
    }

    /**
     * @brief solve 0.5*xT*x  s.t. Ax<=b
     * 
     * @tparam d 
     * @param A 
     * @param b 
     * @param x 
     * @return double return the cost of the objective function
     */
    template <int d>
    inline double sdmn(const Eigen::Matrix<double, -1, d> &A,
                       const Eigen::Matrix<double, -1, 1> &b,
                       Eigen::Matrix<double, d, 1> &x)
    {
        x.setZero();
        const int n = b.size();
        // 没有约束 最优解x为全零 cost为0
        if (n < 1)
        {
            return 0.0;
        }

        Eigen::VectorXi perm(n - 1);
        Eigen::VectorXi next(n);
        Eigen::VectorXi prev(n + 1);
        if (n > 1)
        {
            rand_permutation(n - 1, perm.data());
            prev(0) = 0;
            next(0) = perm(0) + 1;
            prev(perm(0) + 1) = 0;
            for (int i = 0; i < n - 2; ++i)
            {
                next(perm(i) + 1) = perm(i + 1) + 1;
                prev(perm(i + 1) + 1) = perm(i) + 1;
            }
            next(perm(n - 2) + 1) = n;
        }
        else
        {
            prev(0) = 0;
            next(0) = 1;
            next(1) = 1;
        }

        Eigen::Matrix<double, d + 1, -1, Eigen::ColMajor> halves(d + 1, n);
        Eigen::VectorXd work((n + 2) * (d + 2) * (d - 1) / 2 + 1 - d);
        
        // 对A的每一行元素算二范数
        const Eigen::VectorXd scale = A.rowwise().norm();
        // 对A的每一行进行归一化
        // halves = | A^T |
        //        = | -b^T |
        halves.template topRows<d>() = (A.array().colwise() / scale.array()).transpose();
        halves.template bottomRows<1>() = (-b.array() / scale.array()).transpose();

        // std::cout<<"A: "<<std::endl<<A<<std::endl;
        // std::cout<<"scale: "<<scale.transpose()<<std::endl;
        // std::cout<<"halves:"<<std::endl<<halves<<std::endl;
        // std::cout<<"begin: "<<std::endl;
        // std::cout<<"perm:"<<std::endl<<perm.transpose()<<std::endl;
        // std::cout<<"next"<<std::endl<<next.transpose()<<std::endl;
        // std::cout<<"prev"<<std::endl<<prev.transpose()<<std::endl;

        const int status = min_norm<d>(halves.data(), n, n,
                                       x.data(), work.data(),
                                       next.data(), prev.data());
        
        // std::cout<<"finshed"<<std::endl;
        // std::cout<<"next"<<std::endl<<next.transpose()<<std::endl;
        // std::cout<<"prev"<<std::endl<<prev.transpose()<<std::endl;

        double minimum = INFINITY;
        if (status != INFEASIBLE)
        {
            minimum = x.norm();
        }

        return minimum;
    }

    /**
     * minimize     0.5 x' Q x + c' x
     * subject to       A x <= b
     * Q must be positive definite
     **/
    
    template <int d>
    inline double sdqp(const Eigen::Matrix<double, d, d> &Q,
                       const Eigen::Matrix<double, d, 1> &c,
                       const Eigen::Matrix<double, -1, d> &A,
                       const Eigen::Matrix<double, -1, 1> &b,
                       Eigen::Matrix<double, d, 1> &x)
    {
        Eigen::LLT<Eigen::Matrix<double, d, d>> llt(Q);
        if (llt.info() != Eigen::Success)
        {
            return INFINITY;
        }
        // // BUG: L矩阵用错了
        // const Eigen::Matrix<double, -1, d> As = llt.matrixLLT()
        //                                             .template triangularView<Eigen::Upper>()
        //                                             .template solve<Eigen::OnTheRight>(A);
        // // 相当于ppt中 L^{-1}*L^{-T}*c
        // const Eigen::Matrix<double, d, 1> v = llt.solve(c);
        // // 相当于ppt中的f矩阵
        // const Eigen::Matrix<double, -1, 1> bs = A * v + b;

        // std::cout<<"A: "<<std::endl<<A<<std::endl;
        // std::cout<<"LLT: "<<std::endl<<llt.matrixLLT()<<std::endl;
        // Eigen::MatrixXd LLT_upper = llt.matrixLLT().template triangularView<Eigen::Upper>();
        // std::cout<<"LLT_upper:"<<std::endl<<LLT_upper<<std::endl;
        // std::cout<<"As:"<<std::endl<<As<<std::endl;
        // Eigen::MatrixXd L = llt.matrixL();
        // std::cout<<"L:"<<std::endl<<L<<std::endl;
        // std::cout<<"AL:"<<std::endl<<A*L.inverse().transpose()<<std::endl;


        // double minimum = sdmn<d>(As, bs, x);
        // if (!std::isinf(minimum))
        // {
        //     llt.matrixLLT()
        //         .template triangularView<Eigen::Upper>()
        //         .template solveInPlace<Eigen::OnTheLeft>(x);
        //     x -= v;
        //     minimum = 0.5 * (Q * x).dot(x) + c.dot(x);
        // }

        // my method
        const Eigen::Matrix<double , -1, d> As = llt.matrixL()
                                                    .transpose()
                                                    .template solve<Eigen::OnTheRight>(A);
        const Eigen::Matrix<double, d, 1> v = llt.solve(c);
        const Eigen::Matrix<double, -1, 1> bs = A * v + b;

        double minimum = sdmn<d>(As, bs, x);
        if (!std::isinf(minimum)){
            llt.matrixL()
                .transpose()
                .template solveInPlace<Eigen::OnTheLeft>(x);
            x -= v;
            minimum = 0.5 * (Q * x).dot(x) + c.dot(x);
        }
        

        return minimum;
    }

} // namespace sdqp

#endif
