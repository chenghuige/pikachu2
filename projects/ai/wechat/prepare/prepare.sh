pushd .
cd ../jupyter
python actions.py
python vocab-corpus.py
popd

sh gen-vocabs.sh 
python ./dump-feed-emb.py
python pca.py ../input/feed_embeddings.npy ../input/feed_pca_embeddings.npy

pushd .
cd ../jupyter
python doc-info.py
python doc-static-stats.py
python history.py 
python history-days.py
python glove-corpus.py
popd

sh glove-all.sh

days="13 14.5 14 15"
for day in $days; do
  sh dump-glove.sh
done
