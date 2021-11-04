import json
from zipfile import ZipFile, ZIP_DEFLATED

from util import test_spearmanr

if __name__ == '__main__':
    result_zip = 'result.zip'
    result_json = 'result.json'
    annotation_file = './data/pairwise/label.tsv'
    with ZipFile(result_zip, 'r', compression=ZIP_DEFLATED) as zip_file:
        with zip_file.open(result_json) as f:
            vid_embedding = json.load(f)

    spearmanr = test_spearmanr(vid_embedding, annotation_file)
    print(f'Spearmanr of test_a is: {spearmanr:.4f}')
