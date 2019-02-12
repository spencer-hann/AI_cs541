import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from os import popen as get_console_size


#========================================================================#
#      REPLACE module_name WITH THE NAME OF YOUR CYTHON MODULE           #
#========================================================================#
module_name = "tile_puzzle_solver" #    should not include .pyx'         #
#========================================================================#


## If module_name not hard coded, get from command line ##
 # module_name can be hard coded in for simplicity's sake, this way it
 # is easier to include with specific .pyx file when being shared,
 # or if setup scrip is left generic, module_name can be passed in
 # from command line, so that it is more versitile (don't have to 
 # hard code module_name in if you don't want to)
if module_name == "__module_name_here__":
    module_name = sys.argv[1]
    del sys.argv[1] # done with name


# correct for ".pyx" at end of module name
if module_name[-4:] == ".pyx":
    cy_name = module_name
    module_name = module_name[:-4]
else:
    cy_name = module_name + ".pyx"


# displays bar across console
_, console_width = get_console_size('stty size', 'r').read().split()
console_width = int(console_width)
print('=' * console_width)


print("Setup script for", module_name)


## Default compile options ##
 # This block checks for Cython setup arguments
 # you can provide your own if you want to test out different
 # options and this block will be mainly ignored
if len(sys.argv) == 1: # no command line args
    print(cy_name + " compiling with default args")
    print("\tadding \"build_ext\" and \"--inplace\" to sys.argv")
    sys.argv.extend( ["build_ext", "--inplace"] )


# This block compiles/sets up the module from the .pyx cython file
# this includes both compiling Cython to C and then C to Assembly
print("Cython may give a depricated NumPy API warning. This warning is safe to ignore.\n")
print('-' * console_width)
setup(
    ext_modules = cythonize(
        Extension(
            module_name,
            [cy_name],
            define_macros=[("NPY_NO_DEPRECATED_API",None)]
        )
    )
)
print('-' * console_width)


print('\n' + module_name + " setup complete!")

# displays bar across console
print('=' * console_width)

