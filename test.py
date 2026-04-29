import cProfile
import pstats
import moead

MAX_GENERATIONS = 300

cProfile.run('moead.main(MAX_GENERATIONS, num_bots=30)', 'moead_stats')
p = pstats.Stats('moead_stats')
p.strip_dirs().sort_stats('cumulative').print_stats(10)
