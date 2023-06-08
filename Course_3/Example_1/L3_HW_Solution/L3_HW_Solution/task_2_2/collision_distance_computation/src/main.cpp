#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>

#include <Eigen/Core>

#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "../include/sdqp.hpp"

std::vector<Eigen::Vector2d> obs_lists;
Eigen::Vector2d obs_center;
Eigen::Vector2d robot_center;

ros::Publisher obstacle_pub;
ros::Publisher robot_pub;
ros::Publisher arrow_pub;

ros::Subscriber robot_sub;

void ObstacleGeneration(){
    std::random_device rd;
    std::mt19937 gen(rd());
    obs_center = Eigen::Vector2d(1.0,1.0);
    std::normal_distribution<double> random_x(obs_center.x(), 1.0);
    std::normal_distribution<double> random_y(obs_center.y(), 1.0);

    obs_lists.push_back(obs_center);
    for(size_t i=0; i<10; i++){
        double x = random_x(gen);
        double y = random_y(gen);
        obs_lists.push_back(Eigen::Vector2d(x,y));
    }
    for(auto& i : obs_lists){
        std::cout<<i.transpose()<<std::endl;
    }

    // visualization
    visualization_msgs::MarkerArray obstacle_array;
    visualization_msgs::Marker obstacle;
    obstacle.header.frame_id = "map";
    obstacle.header.stamp = ros::Time::now();
    for(size_t i=0; i<obs_lists.size(); i++){
        obstacle.id = i;
        obstacle.type = visualization_msgs::Marker::SPHERE;
        obstacle.action = visualization_msgs::Marker::ADD;
        obstacle.ns = "obstacle";
        obstacle.pose.position.x = obs_lists.at(i).x();
        obstacle.pose.position.y = obs_lists.at(i).y();
        obstacle.pose.position.z = 0.0;
        obstacle.pose.orientation.x = 0.0;
        obstacle.pose.orientation.y = 0.0;
        obstacle.pose.orientation.z = 0.0;
        obstacle.pose.orientation.w = 1.0;
        obstacle.scale.x = 0.2;
        obstacle.scale.y = 0.2;
        obstacle.scale.z = 0.2;
        obstacle.color.a = 1.0;
        obstacle.color.b = 0.0;
        obstacle.color.g = 0.0;
        obstacle.color.r = 1.0;

        obstacle_array.markers.push_back(obstacle);
    }
    obstacle_pub.publish(obstacle_array);
}

void CallBack(const geometry_msgs::PoseStamped::ConstPtr& msg){

    robot_center = Eigen::Vector2d(
        msg->pose.position.x,
        msg->pose.position.y);
    
    // 2-Dimensional QP Problem
    int m = 2;
    Eigen::Vector2d z = Eigen::Vector2d::Zero();
    Eigen::Matrix2d Q = Eigen::Matrix2d::Identity();
    Eigen::Vector2d c = Eigen::Vector2d::Zero();
    Eigen::MatrixXd A;
    A.resize(obs_lists.size(), 2);
    for(size_t i=0; i<obs_lists.size(); i++){
        // obs_lists是随机生成的
        // robot_center是用户指定的
        A.row(i) = (robot_center - obs_lists.at(i)).transpose();
    }
    Eigen::VectorXd b = -1.0 * Eigen::VectorXd::Ones(obs_lists.size());
    // std::cout<<"A: "<<std::endl<<A<<std::endl;
    // std::cout<<"b: "<<std::endl<<b.transpose()<<std::endl;

    double cost = sdqp::sdqp<2>(Q, c, A, b, z);

    Eigen::Vector2d x;
    std::cout<<"======= result ======="<<std::endl;
    if(cost != INFINITY){
        x = z / z.squaredNorm() + robot_center;
        std::cout<<"optimal z: "<<z.transpose()<<std::endl;
        std::cout<<"final cost: "<<cost<<std::endl;
        std::cout<<"collision vector: "<<x.transpose()<<std::endl;
        std::cout<<"collision distance: "<<x.norm()<<std::endl;
    } else {
        ROS_WARN("SDQP has not feasiable solution. Robot collided with an obstacle !");
    }
   
    
    // robot center
    visualization_msgs::Marker robot;
    robot.header.frame_id = "map";
    robot.header.stamp = msg->header.stamp;
    robot.id = 0;
    robot.type = visualization_msgs::Marker::SPHERE;
    robot.action = visualization_msgs::Marker::ADD;
    robot.ns = "robot_center";
    robot.pose.position.x = robot_center.x();
    robot.pose.position.y = robot_center.y();
    robot.pose.position.z = 0.0;
    robot.pose.orientation.x = 0.0;
    robot.pose.orientation.y = 0.0;
    robot.pose.orientation.z = 0.0;
    robot.pose.orientation.w = 1.0;
    robot.scale.x = 0.2;
    robot.scale.y = 0.2;
    robot.scale.z = 0.2;
    robot.color.a = 1.0;
    robot.color.b = 0.0;
    robot.color.g = 1.0;
    robot.color.r = 0.0;
    robot_pub.publish(robot);

    // collision arrow
    if(cost != INFINITY){
        visualization_msgs::Marker arrow;
        arrow.header.frame_id = "map";
        arrow.header.stamp = msg->header.stamp;
        arrow.id = 0;
        arrow.action = visualization_msgs::Marker::ADD;
        arrow.ns = "collisin_arrow";
        arrow.scale.x = 0.1;
        arrow.scale.y = 0.1;
        arrow.scale.z = 0.5;
        arrow.color.a = 1.0;
        arrow.color.b = 1.0;
        arrow.color.g = 0.0;
        arrow.color.r = 0.0;
        geometry_msgs::Point p1, p2;
        p1.x = robot_center.x();
        p1.y = robot_center.y();
        p1.z = 0.0;
        p2.x = x.x();
        p2.y = x.y();
        p2.z = 0.0;
        arrow.points.push_back(p1);
        arrow.points.push_back(p2);
        arrow_pub.publish(arrow);
    } else {
        visualization_msgs::Marker arrow;
        arrow.header.frame_id = "map";
        arrow.header.stamp = msg->header.stamp;
        arrow.id = 0;
        arrow.ns = "collisin_arrow";
        arrow.action = visualization_msgs::Marker::DELETE;
        arrow.color.a = 0.0;
        arrow_pub.publish(arrow);
    }
   
}

int main(int argc, char** argv){
    ros::init(argc, argv,"collision_distance_computation_node");
    ros::NodeHandle nh;

    obstacle_pub = nh.advertise<visualization_msgs::MarkerArray>(
        "obstacle_array", 1, true);
    robot_pub = nh.advertise<visualization_msgs::Marker>(
        "robot_center", 1, true);
    arrow_pub = nh.advertise<visualization_msgs::Marker>(
        "collision_arrow", 1, true);

    robot_sub = nh.subscribe("/move_base_simple/goal", 1, CallBack);

    ObstacleGeneration();

    ros::spin();
}