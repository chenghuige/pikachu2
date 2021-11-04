import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import tensorflow as tf

from config import parser
from data_helper import FeatureParser
from model import MultiModal


def main():
    args = parser.parse_args()
    feature_parser = FeatureParser(args)
    files = args.test_a_file
    dataset = feature_parser.create_dataset(files, training=False, batch_size=args.test_batch_size)
    model = MultiModal(args)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(args.ckpt_file)
    print(f"Restored from {args.ckpt_file}")

    vid_embedding = {}
    for batch in dataset:
        _, embeddings = model(batch, training=False)
        vids = batch['vid'].numpy().astype(str)
        embeddings = embeddings.numpy().astype(np.float16)
        for vid, embedding in zip(vids, embeddings):
            vid_embedding[vid] = embedding.tolist()
    with open(args.output_json, 'w') as f:
        json.dump(vid_embedding, f)
    with ZipFile(args.output_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(args.output_json)


if __name__ == '__main__':
    main()
