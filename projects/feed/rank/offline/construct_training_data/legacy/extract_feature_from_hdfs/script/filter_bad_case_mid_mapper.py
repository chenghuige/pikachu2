# coding=gbk

import sys


def main():
    model_name = ""
    product_need = "shida"

    if len(sys.argv) >= 2:
        model_name = sys.argv[1]
    if model_name == "sgsapp_nfm" or model_name == "sgsapp_wd" or "sgsapp" in model_name:
        product_need = "sgsapp"
    print product_need
    for line in sys.stdin:
        line = line.strip().decode("gbk", "ignore")
        if line == "":
            continue
        line_tuple = line.split("\t")
        if len(line_tuple) < 9:
            continue
        if len(line_tuple) > 9:
            (mid, doc_id, ts, feedback_info, interest, hot_info,
             article_info, account_info, click, product) = line_tuple[0:10]
        if product_need not in product:
            continue
        (mid, docid) = line_tuple[:2]
        try:
            print >> sys.stdout, mid
        except:
            continue


if __name__ == "__main__":
    main()
