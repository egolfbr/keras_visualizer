from distutils.core import setup 

setup(name="keras_dot_visualizer",
      version="0.2.0",
      description="Front-end Keras Visualizer",
      author="Brian Egolf",
      author_email="egolfbr@miamioh.edu",
      url="https://github.com/egolfbr/keras_visualizer",
      package_dir = {'':'keras-dot-visualizer'},
      py_modules = ["writedot"]
  
)
