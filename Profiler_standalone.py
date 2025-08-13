import cProfile, pstats
def profile_script(entry_function):
    profiler = cProfile.Profile()
    profiler.runcall(entry_function)
    profiler.dump_stats('profile_stats.prof')

def is_user_defined(filename):
    return 'python' not in filename and 'lib' not in filename and 'Game'
    #return True

def custom_filter():
    p = pstats.Stats('profile_stats.prof')
    p.strip_dirs()
    p.sort_stats('cumulative')
    for filename in p.stats:
        if is_user_defined(filename[0]):
            p.print_stats(filename)

if __name__ == "__main__":
    custom_filter()
