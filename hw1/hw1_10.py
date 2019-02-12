# This block checks for proper cython setup arguments
# you can provide your own if you want to test out different
# options and this block will be mainly ignored
import sys
print("======================================================================")
py_name = sys.argv[0]
module_name = sys.argv[0][:-3] # removes ".py"
cy_name = module_name + ".pyx"
if len(sys.argv) == 1: # no command line args
    print(cy_name + " compiling with default args")
    print("\tadding \"build_ext\" and \"--inplace\" to sys.argv")
    sys.argv.extend( ["build_ext", "--inplace"] )


# This block compiles/sets up the hw10 module
# from the  hw10.pyx cython file
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
print("   Cython may give a depricated NumPy API warning.")
print("   This warning is safe to ignore.\n")
setup(
    ext_modules = cythonize(
        Extension(
            module_name,
            [cy_name],
            define_macros=[("NPY_NO_DEPRECATED_API",None)]
        )
    )
)
print(module_name + " setup complete!")
print("======================================================================\n")


# This is where I import the pre-compiled 
# module and enter the Cython layer
import hw1_10 #change module name
if __name__ == "__main__":
    hw1_10.main()
