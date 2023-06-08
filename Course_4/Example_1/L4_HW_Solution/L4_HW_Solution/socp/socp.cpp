#include <Eigen/Eigen>
#include <cmath>
#include <vector>
#include <iostream>
using namespace Eigen;
using namespace std;


class SocpSolver
{
public:
   SocpSolver(const VectorXd& c, const vector<MatrixXd>& As, const vector<VectorXd>& bs, const MatrixXd& G, const VectorXd& h)
   {
       c_ = c;
       As_ = As;
       bs_ = bs;
       G_ = G;
       h_ = h;
   }

   bool is_all_projection_valid(const VectorXd& x, double p, const vector<VectorXd>& us)
   {
       for(unsigned int i = 0; i< As_.size(); i++)
       {
            if((us[i]/p - get_projection(us[i]/p -  (As_[i]*x + bs_[i]))).lpNorm<Infinity>() >= 1e-4)
            {
                return false;
            }
       }
       return true;
   }

   VectorXd solve()
   {
        double p = 1;
    VectorXd lamda = VectorXd::Zero(h_.size());
    VectorXd x(c_.size());
    x.setZero();
    vector<VectorXd> us;
    for(unsigned int i = 0; i < As_.size(); i++)
    {
        VectorXd u(bs_[i].size());
        u.setZero();
        us.push_back(u);
    }
    while(true)
    {
            get_min_x(x, p, lamda, us);
            if((G_ * x - h_).lpNorm<Infinity>() < 1e-4 and is_all_projection_valid(x, p, us) and get_gradient(x, p, lamda, us).norm()/x.norm() < 1e-4)
            {
                break;
            }
            lamda = lamda + p * ((G_*x) - h_);
            for(unsigned j = 0; j < As_.size(); j++)
            {
                us[j] = get_projection(us[j] - p * (As_[j]*x + bs_[j]));
            }
            p = 2 * p;
            if(p > 1000)
            {
                p = 1000;
            }
    }
    return x;
 
   }

MatrixXd do_get_hession(const VectorXd& x, double p, const VectorXd& lamda, const MatrixXd& A, const VectorXd& b, const VectorXd& u)
{
   VectorXd inner = u - p * (A * x + b);
    double x0 = inner[0];
    VectorXd x1(u.size() - 1);
    x1[0] = inner[1];
    x1[1] = inner[2];
    x1[2] = inner[3];
    x1[3] = inner[4];
    x1[4] = inner[5];
    x1[5] = inner[6];
    x1[6] = inner[7];
    MatrixXd hessian(u.size(), u.size());
    if(x1.norm() <= x0)
    {
        hessian.setIdentity() ;
    }
    else if(x1.norm() <= -1 * x0)
    {
        hessian.setZero();
    }
    else
    {
       MatrixXd sub_hessian = (x0 + x1.norm())/(2 * x1.norm()) * MatrixXd::Identity(x1.size(), x1.size()) - x0 * x1 * x1.transpose()/(2*x1.norm() * x1.norm() *x1.norm());
       hessian(0, 0) = 1.0/2;
       for(int i = 1; i < 8; i++)
       {
          hessian(0, i) = x1[i - 1]/(2*x1.norm());
          hessian(i, 0) = x1[i - 1]/(2*x1.norm());
       }
       for(int i = 1; i < 8; i++)
       {
           for(int j = 1; j < 8; j++)
           {
               hessian(i, j) = sub_hessian(i -1, j -1);
           }
       }

    }
    return p * A.transpose() * hessian * A;


  
}

MatrixXd get_hession(const VectorXd& x, double p, const VectorXd& lamda, const vector<VectorXd>& us)
{
     MatrixXd hession = p * G_.transpose() * G_;
     for(unsigned int i = 0; i < As_.size(); i++)
     {
         hession += do_get_hession(x, p, lamda, As_[i], bs_[i], us[i]);
     }
     return hession;
   
}

VectorXd get_projection(const VectorXd& inner)
{
    double x0 = inner[0];
    VectorXd x1(inner.size() - 1);
    x1[0] = inner[1];
    x1[1] = inner[2];
    x1[2] = inner[3];
    x1[3] = inner[4];
    x1[4] = inner[5];
    x1[5] = inner[6];
    x1[6] = inner[7];
    VectorXd g(inner.size());
    g.setZero();
    if(x0 <= -1 * x1.norm())
    {
       g.setZero() ;
    }
    else if(abs(x0) < x1.norm())
    {
       g[0] = x1.norm();
       for(int i = 0; i < 7; i++)
       {
          g[i+1] = x1[i];
       }

       g = (x0 + x1.norm())/(2 * x1.norm()) * g;
    }
    else
    {
        g = inner;
    }
    return g;

}
VectorXd get_gradient(const VectorXd& x, double p, const VectorXd& lamda, const vector<VectorXd>& us)
{
    VectorXd gradient = c_ +  G_.transpose() * (lamda + p * ((G_ * x) - h_));
    for(unsigned int i = 0; i < As_.size(); i++)
    {
        gradient -= As_[i].transpose() * get_projection(us[i] - p * (As_[i]*x + bs_[i]));
    }
    return gradient;
}

double get_value(const VectorXd& x, double p, const VectorXd& lamda, const vector<VectorXd>& us)
{
    double value =  c_.transpose() * x + 0.5 * p * (pow(((G_*x) -h_ + lamda/p).norm(), 2));
    for(unsigned int i = 0; i < As_.size(); i++)
    {
       value += 0.5 * p * pow(get_projection(us[i]/p - (As_[i]*x + bs_[i])).norm(), 2);
    }
    return value;
}

double get_min_x(VectorXd &x, double p, const VectorXd& lamda, const vector<VectorXd>& us)
{
   while(true)
   {
           double value = get_value(x, p, lamda, us);
           auto gradient = get_gradient(x, p, lamda, us);
           if(gradient.norm() / x.norm() < 1e-4)
           {
               return value;
           }
           auto hession = get_hession(x, p, lamda, us);
           auto d = -1 * hession.inverse() * gradient;
           double t = 1;
           while(true)
           {
               if(value - get_value(x + t * d, p, lamda, us) >= -1e-4 * t * d.transpose() * gradient)
               {
                    break;
               }
               t = t/2;
           }
           x = x + t * d;
   }
   return 0;
}
private:
     VectorXd  c_;

     MatrixXd G_;


     VectorXd h_;

     vector<MatrixXd> As_ ;

     vector<VectorXd> bs_ ;
};

int main()
{
        VectorXd  c(7);
        c[0] = 1.0;
        c[1] = 2.0;
        c[2] = 3.0;
        c[3] = 4.0;
        c[4] = 5.0;
        c[5] = 6.0;
        c[6] = 7.0;

        MatrixXd G(1, 7);
        G(0, 0) = 0;
        G(0, 1) = 0;
        G(0, 2) = 0;
        G(0, 3) = 0;
        G(0, 4) = 0;
        G(0, 5) = 0;
        G(0, 6) = 0;


        VectorXd h = VectorXd(1);
        h[0] = 0;

        MatrixXd A = Matrix<double, 8, 7> ::Zero();
        A(0, 0) = 1;
        A(1, 0) = 7;
        A(2, 1) = 6;
        A(3, 2) = 5;
        A(4, 3) = 4;
        A(5, 4) = 3;
        A(6, 5) = 2;
        A(7, 6) = 1;

        VectorXd b = Matrix<double, 8 , 1>::Zero();
        b[0] = 1;
        b[1] = 1;
        b[2] = 3;
        b[3] = 5;
        b[4] = 7;
        b[5] = 9;
        b[6] = 11;
        b[7] = 13;


        SocpSolver socpSolver(c, {A}, {b}, G, h);
        auto x = socpSolver.solve();
        double value = c.transpose() * x;
        cout << "x = " << x << endl;
        cout << "value = " << value << endl;
 //  cout << "x = " << x <<endl;
     return 0;
}
