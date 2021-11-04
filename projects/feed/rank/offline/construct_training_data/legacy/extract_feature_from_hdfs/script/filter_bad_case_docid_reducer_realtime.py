#coding=gbk

import sys

def main():
    last_mid = ""
    sample_num = 0
    pos_num = 0
    neg_num = 0
    for line in sys.stdin:
        line = line.strip().decode("gbk", "ignore")
        if line == "":
            continue
	line_tuple = line.split("\t")
	if len(line_tuple) < 2:
	    continue
	(mid, click) = line_tuple

        if mid != last_mid:
            #if last_mid != "" and ((sample_num > 4 * 10000 or sample_num < 10) or pos_num == 0 or neg_num == 0):
            if last_mid != "" and sample_num > 4 * 10000:
                print >> sys.stdout, last_mid + "\t" + str(sample_num) + "\t" + str(pos_num) + "\t" + str(neg_num)
            last_mid = mid
            sample_num = 0
	    pos_num = 0
	    neg_num = 0
        sample_num += 1
	if click == "1":
	    pos_num +=1
	elif click == "0":
	    neg_num +=1
	else:
	    print "wrong",line
    #if last_mid != "" and ((sample_num > 4 * 10000 or sample_num < 10) or pos_num == 0 or neg_num == 0):
    if last_mid != "" and (sample_num > 4 * 10000):
        print >> sys.stdout, last_mid + "\t" + str(sample_num) + "\t" + str(pos_num) + "\t" + str(neg_num)

if __name__ == "__main__":
    main()
