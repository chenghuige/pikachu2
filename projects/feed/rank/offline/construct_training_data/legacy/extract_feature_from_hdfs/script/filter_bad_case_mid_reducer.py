#coding=gbk

import sys

def main():
    last_mid = ""
    sample_num = 0
    for line in sys.stdin:
        line = line.strip().decode("gbk", "ignore")
        if line == "":
            continue
        if line != last_mid:
            if last_mid != "" and (sample_num > 1000 or sample_num < 5):
                print >> sys.stdout, last_mid + "\t" + str(sample_num)
            last_mid = line
            sample_num = 0
        sample_num += 1
    if last_mid != "" and (sample_num > 1000 or sample_num < 5):
        print >> sys.stdout, last_mid + "\t" + str(sample_num)

if __name__ == "__main__":
    main()
