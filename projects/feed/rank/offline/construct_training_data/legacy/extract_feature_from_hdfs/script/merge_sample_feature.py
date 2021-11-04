import sys
import os
import datetime

def main():
    day = sys.argv[1]
    date_day = datetime.datetime.strptime(day, '%Y%m%d%H')
    inter2 = datetime.timedelta(hours=20)
    inter1 = datetime.timedelta(hours=15)
    end = date_day - inter2
    end1 = date_day - inter1
    for line in sys.stdin:
	line = line.strip("\n")
        filepath = os.environ.get('mapreduce_map_input_file')
        filename = os.path.split(filepath)[-2]
        cur_hour = filename.split("/")
        date_cur = datetime.datetime.strptime(cur_hour[-2],'%Y%m%d%H')
        if date_cur > end1:
            print line
	elif date_cur > end and date_cur < end1:
	    tuple = line.split("\t")
            if tuple[0] == "0" and ("MITATKW" in line or "MITATAC" in line):
                continue
        else:
            tuple = line.split("\t")
            if tuple[0] == "1" and ("MITATKW" in line or "MITATAC" in line):
                print line


if __name__ == "__main__":
    main()
