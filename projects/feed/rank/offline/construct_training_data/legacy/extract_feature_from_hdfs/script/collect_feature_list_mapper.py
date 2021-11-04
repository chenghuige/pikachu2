# coding=gbk

import sys
import json


def main():
    model_name = ""
    if len(sys.argv) >= 2:
        model_name = sys.argv[1]
    split_idx = 8

    feature_freq_dict = {}
    split_idx_feature = 0
    for line in sys.stdin:
        try:
            line = line.strip().decode("gbk", "ignore")
            if line == "":
                continue
            line_tuple = line.split("\t")

            if len(line_tuple) < (split_idx + 1):
                continue
            split_idx_feature = split_idx+1

            if len(line_tuple) < (split_idx_feature + 1):
                continue
            click = line_tuple[0]
            if "TOTAL_SAMPLES" not in feature_freq_dict:
                feature_freq_dict["TOTAL_SAMPLES"] = {}
            if click not in feature_freq_dict["TOTAL_SAMPLES"]:
                feature_freq_dict["TOTAL_SAMPLES"][click] = 0
            feature_freq_dict["TOTAL_SAMPLES"][click] += 1
            for feature in line_tuple[split_idx_feature:]:
                if "cycle_profile_click" in feature or "cycle_profile_show" in feature or "cycle_profile_dur" in feature:
                    continue
                feature_tuple = feature.split(":\b")
                feature = feature_tuple[0]
                if feature not in feature_freq_dict:
                    feature_freq_dict[feature] = {}
                if click not in feature_freq_dict[feature]:
                    feature_freq_dict[feature][click] = 0
                feature_freq_dict[feature][click] += 1
        except:
            continue

    for feature in feature_freq_dict:
        for click in feature_freq_dict[feature]:
            s = feature + "\t" + click + "\t" + \
                str(feature_freq_dict[feature][click])
            print >> sys.stdout, s.encode("gbk", "ignore")


if __name__ == "__main__":
    main()
