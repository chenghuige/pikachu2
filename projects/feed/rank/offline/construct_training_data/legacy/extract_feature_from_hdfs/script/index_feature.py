#coding=gbk

import sys

def main():
    if len(sys.argv) < 3:
        print >> sys.stderr, "Usage python " + sys.argv[0] + " feature_infogain index_file" 
        sys.exit(-1)

    feature_infogain_file = sys.argv[1]
    feature_indx_file=sys.argv[2]
    feature_list = []
    model_name = ""
    
    # for din model
    feature_meta_file = ""
    if len(sys.argv) >= 5:
	model_name = sys.argv[3]
	feature_meta_file = sys.argv[4]

    with open(feature_infogain_file, "r") as fp:
        for line in fp:
            line = line.strip().decode("gbk", "ignore")
            if line == "":
                continue
            line_tuple = line.split("\t")
            if len(line_tuple) < 3:
                continue
            (feature, freq, info_gain) = line_tuple[:3]
            if int(freq) < 10:
                continue
            if float(info_gain) < 0.0000000000000001:
                continue
	    if "din" in model_name and "EM_MEM" in line: # for din model
		continue
	    feature_list.append(feature)
    
    feature_list.sort()
    index_fd = open(feature_indx_file, "w")
    index = 0
    last_category=""
    last_index=0
   
    base_num = 0
    doc_s = "z1"
    doc_num = 0
    acc_s = "z2"
    acc_num = 0
    topic_s = "z3"
    topic_num = 0
    
    for feature in feature_list:
        cur_category=feature.split('\a')[0]
        index += 1
        s = feature + "\t" + str(index)
        index_fd.write("%s\n" % s.encode("gbk", "ignore"))
       
	if "din" in model_name:
       	    if feature[0:2] == doc_s:
                doc_num+=1
            elif feature[0:2] == acc_s:
                acc_num+=1
            elif feature[0:2] == topic_s:
                topic_num+=1
            else:
                base_num+=1

    index_fd.close()

    if "din" in model_name and feature_meta_file != "":
	meta_fd = open(feature_meta_file, "w") 
	meta_fd.write("%d-%d-%d-%d\n" % (base_num,doc_num,acc_num,topic_num))
 	meta_fd.close()

if __name__ == "__main__":
    main()    
