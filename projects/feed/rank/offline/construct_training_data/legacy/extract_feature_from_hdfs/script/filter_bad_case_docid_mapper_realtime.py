#coding=gbk

import sys

def main():
    for line in sys.stdin:
        line = line.strip().decode("gbk", "ignore")
        if line == "":
            continue
        line_tuple = line.split("\t")
        if len(line_tuple) > 4:
            try:
                print >> sys.stdout, line_tuple[2]+"\t"+line_tuple[0]
            except:
                continue
        
if __name__ == "__main__":
    main()
