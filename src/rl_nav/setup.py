from glob import glob
from setuptools import setup

package_name = 'rl_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='deiva',
    description='PPO LiDAR navigation trained in a 2D sim and benchmarked against Nav2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rl_policy_node = rl_nav.rl_policy_node:main',
            'benchmark = rl_nav.benchmark:main',
            'train_ppo = rl_nav.train_ppo:main',
            'eval_2d = rl_nav.eval_2d:main',
            'set_initial_pose = rl_nav.set_initial_pose:main',
        ],
    },
)
