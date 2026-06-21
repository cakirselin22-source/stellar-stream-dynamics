import matplotlib

# Headless backend for the whole test session -- must happen before any
# test module imports matplotlib.pyplot (streamtools.viz.plotting does, at
# import time).
matplotlib.use("Agg")
