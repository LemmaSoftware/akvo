from USGSPlots import USGSPlots
import plotyaml
import sys


if len(sys.argv) != 4 and len(sys.argv) != 2:
    print('Usage:')
    print('whatever.py filename channel option')
    print('Use whatever.py -h for detailed help')
    sys.exit()

if len(sys.argv) == 2:
    print('help!')
    sys.exit()


fname = str(sys.argv[1])
chan = sys.argv[2]

myplot = USGSPlots()

myplot.getData(fname,chan)


