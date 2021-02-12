from lib.MyStats import *
import sys

if __name__ == "__main__":
    if len(sys.argv) == 1:
        path = "datasets/data.csv"
    else:
        path = sys.argv[1]
    describe(path, header=False)
    input("~~Enter To Continue~~")

