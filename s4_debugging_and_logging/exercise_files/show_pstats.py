import pstats
p = pstats.Stats('exercise_files/profile.txt')
p.sort_stats('cumulative').print_stats(10)