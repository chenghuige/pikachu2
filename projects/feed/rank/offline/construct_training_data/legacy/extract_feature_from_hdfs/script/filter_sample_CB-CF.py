import sys
import os

cf_p = 0
cf_n = 0
cb_p = 0
cb_n = 0

def main():
    for line in sys.stdin:
        global cf_p,cf_n,cb_p,cb_n
        tuple = line.split("\t")
        if "MID_" not in line and  "CB" in line:
            if tuple[0] == "1":
                cb_p += 1
            else:
                cb_n += 1
        if "CF" in line:
            if tuple[0] == "1":
                cf_p += 1
            else:
                cf_n += 1
    filepath = os.environ.get('mapreduce_map_input_file')
    filename = os.path.split(filepath)[-2]
    print filename + "\t" + str(cf_p) + "\t"+str(cf_n) + "\t" + str(cb_p) + "\t" + str(cb_n)
if __name__ == "__main__":
    main()
