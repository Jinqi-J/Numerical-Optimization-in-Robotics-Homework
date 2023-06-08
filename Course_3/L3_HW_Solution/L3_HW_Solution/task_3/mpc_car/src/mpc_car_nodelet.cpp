#include <car_msgs/CarCmd.h>
#include <nav_msgs/Odometry.h>
#include <nodelet/nodelet.h>
#include <ros/ros.h>

#include <Eigen/Geometry>
#include <mpc_car/mpc_car.hpp>

#include <fstream>

namespace mpc_car {
class Nodelet : public nodelet::Nodelet {
 private:
  std::shared_ptr<MpcCar> mpcPtr_;
  ros::Timer plan_timer_;
  ros::Subscriber odom_sub_;
  ros::Publisher cmd_pub_;
  VectorX state_;
  bool init = false;
  double delay_ = 0.0;
  bool nmpc_ = false;

  void plan_timer_callback(const ros::TimerEvent& event) {
    if (init) {
      int ret = 0;
      if (nmpc_) {
        ros::Time t1 = ros::Time::now();
        // ret = mpcPtr_->solveNMPC(state_);
        ret = mpcPtr_->SolvePHR(state_);
        ros::Time t2 = ros::Time::now();
        double solve_time = (t2 - t1).toSec();
        std::cout << "solve nmpc costs: " << 1e3 * solve_time << "ms" << std::endl;

        std::ofstream foutC("/media/kris/Workspace/numerical_optimization/section_3/task_3/time.txt", std::ios::app);
        foutC.setf(std::ios::scientific, std::ios::floatfield);
        foutC.precision(15);
        foutC << 1e3 * solve_time << std::endl;
        foutC.close();
      } else {
        ros::Time t1 = ros::Time::now();
        ret = mpcPtr_->solveQP(state_);
        ros::Time t2 = ros::Time::now();
        double solve_time = (t2 - t1).toSec();
        std::cout << "solve qp costs: " << 1e3 * solve_time << "ms" << std::endl;

        std::ofstream foutC("/media/kris/Workspace/numerical_optimization/section_3/task_3/time.txt", std::ios::app);
        foutC.setf(std::ios::scientific, std::ios::floatfield);
        foutC.precision(15);
        foutC << 1e3 * solve_time << std::endl;
        foutC.close();
      }
      // assert(ret == 1);
      // TODO
      car_msgs::CarCmd msg;
      msg.header.frame_id = "world";
      msg.header.stamp = ros::Time::now();

      VectorX x;
      VectorU u;
      mpcPtr_->getPredictXU(0, x, u);
      std::cout << "u: " << u.transpose() << std::endl;
      std::cout << "x: " << x.transpose() << std::endl;
      std::cout << std::endl;
      // 发送控制
      msg.a = u(0);
      msg.delta = u(1);
      cmd_pub_.publish(msg);
      
      mpcPtr_->visualization();
    }
    return;
  }

  // 机器人状态估计的信息 x y yaw v
  void odom_call_back(const nav_msgs::Odometry::ConstPtr& msg) {
    double x = msg->pose.pose.position.x;
    double y = msg->pose.pose.position.y;
    Eigen::Quaterniond q(msg->pose.pose.orientation.w,
                         msg->pose.pose.orientation.x,
                         msg->pose.pose.orientation.y,
                         msg->pose.pose.orientation.z);
    Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(0, 1, 2);
    Eigen::Vector2d v(msg->twist.twist.linear.x, msg->twist.twist.linear.y);
    state_ << x, y, euler.z(), v.norm();
    init = true;
  }

 public:
  // 类似普通节点中的main()
  void onInit(void) {
    ros::NodeHandle nh(getMTPrivateNodeHandle());
    mpcPtr_ = std::make_shared<MpcCar>(nh);
    double dt = 0;
    nh.getParam("dt", dt);
    nh.getParam("delay", delay_);
    nh.getParam("nmpc", nmpc_);

    plan_timer_ = nh.createTimer(ros::Duration(dt), &Nodelet::plan_timer_callback, this);
    odom_sub_ = nh.subscribe<nav_msgs::Odometry>("odom", 1, &Nodelet::odom_call_back, this);
    cmd_pub_ = nh.advertise<car_msgs::CarCmd>("car_cmd", 1);
  }
};
}  // namespace mpc_car

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(mpc_car::Nodelet, nodelet::Nodelet);