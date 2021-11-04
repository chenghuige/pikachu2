#coding=gbk

import sys

def main():
    if len(sys.argv) < 3:
        print >> sys.stderr, "Usage python " + sys.argv[0] + " feature_infogain index_file" 
        sys.exit(-1)

    feature_infogain_file = sys.argv[1]
    feature_index_file=sys.argv[2]
    feature_info_out=sys.argv[3]
    feature_index_dict =dict()
    feature_ori_dict = dict()
    model_name = ""
    with open(feature_infogain_file, "r") as fp:
        for line in fp:
            line = line.strip().decode("gbk", "ignore")
            if line == "":
                continue
            line_tuple = line.split("\t")
            if len(line_tuple) < 3:
                continue
            (feature, freq, info_gain) = line_tuple[:3]
	    feature_ori_dict[feature] = freq+"\t"+info_gain
    with open(feature_index_file, "r") as fp:
        for line in fp:
            line = line.strip().decode("gbk", "ignore")
            if line == "":
                continue
            line_tuple = line.split("\t")
            if len(line_tuple) < 2:
                continue
            (feature, index ) = line_tuple[:2]
	    feature_index_dict[feature] = index+"\t"+feature_ori_dict[feature]
    
    index_fd = open(feature_info_out, "w")
   
    for feature in feature_index_dict:
        cur_category=feature.split('\a')[0]
        s =  feature_index_dict[feature] + "\t" + feature
        index_fd.write("%s\n" % s.encode("gbk", "ignore"))
       
    index_fd.close()

if __name__ == "__main__":
    main()    
