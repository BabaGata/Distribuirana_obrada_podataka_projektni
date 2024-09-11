import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

mod = SourceModule(open("hello.cu").read())

hello = mod.get_function("hello")

hello(block=(1,1,1), grid=(1,1))

