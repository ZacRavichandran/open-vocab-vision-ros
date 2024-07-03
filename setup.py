from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=["open_vocab_vision_ros"],
    package_dir={"": "src"},
)
setup(**setup_args)
