#coding=gbk

import sys

def main():
    last_key = ""
    sample_num = 0
    for line in sys.stdin:
        line = line.strip()
        if line == "":
            continue
        if line != last_key:
            if last_key != "":
                print >> sys.stdout, last_key + "\t" + str(sample_num)
            last_key = line
            sample_num = 0
        sample_num += 1
    if last_key != "":
        print >> sys.stdout, last_key + "\t" + str(sample_num)

if __name__ == "__main__":
    main()
