"""Publish /initialpose until AMCL reports a pose (replaces clicking '2D Pose Estimate' in RViz)."""
import math
import sys

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.qos import QoSDurabilityPolicy, QoSProfile


def main():
    rclpy.init(args=sys.argv)
    node = rclpy.create_node('set_initial_pose')
    x = node.declare_parameter('x', -2.0).value
    y = node.declare_parameter('y', -0.5).value
    yaw = node.declare_parameter('yaw', 0.0).value
    pub = node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
    done = {'v': False}
    amcl_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(PoseWithCovarianceStamped, '/amcl_pose',
                             lambda _: done.__setitem__('v', True), amcl_qos)

    def tick():
        m = PoseWithCovarianceStamped()
        m.header.frame_id = 'map'
        m.header.stamp = node.get_clock().now().to_msg()
        m.pose.pose.position.x, m.pose.pose.position.y = float(x), float(y)
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        m.pose.covariance[0] = m.pose.covariance[7] = 0.25
        m.pose.covariance[35] = 0.068
        pub.publish(m)

    node.create_timer(2.0, tick)
    while rclpy.ok() and not done['v']:
        rclpy.spin_once(node, timeout_sec=0.5)
    node.get_logger().info(f'AMCL initialised at ({x}, {y}, {yaw})')
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
