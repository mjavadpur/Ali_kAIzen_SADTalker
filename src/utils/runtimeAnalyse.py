'''
@mjavadpur
This is a sample code to analysing code executation time.

'''

import pstats
import cProfile

cProfile.run("my_function()", "my_func_stats")

p = pstats.Stats("my_func_stats")
p.sort_stats("cumulative").print_stats()