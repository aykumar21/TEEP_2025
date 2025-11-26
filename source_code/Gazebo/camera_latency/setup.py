from setuptools import setup

package_name = 'camera_latency'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ayushkumar',
    maintainer_email='your_email@example.com',
    description='Camera latency measurement node',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'latency_node = camera_latency.latency_node:main',
        ],
    },
)
