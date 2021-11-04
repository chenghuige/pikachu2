day=$1

python dump-glove-emb.py ../input/$day/glove_doc_pos.vectors.txt ../input/$day/doc_pos_emb.npy doc nonorm
python dump-glove-emb.py ../input/$day/glove_author_pos.vectors.txt ../input/$day/author_pos_emb.npy author nonorm
python dump-glove-emb.py ../input/$day/glove_singer_pos.vectors.txt ../input/$day/singer_pos_emb.npy singer nonorm
python dump-glove-emb.py ../input/$day/glove_song_pos.vectors.txt ../input/$day/song_pos_emb.npy song nonorm
python dump-glove-emb.py ../input/$day/glove_doc_finish.vectors.txt ../input/$day/doc_finish_emb.npy doc nonorm
python dump-glove-emb.py ../input/$day/glove_author_finish.vectors.txt ../input/$day/author_finish_emb.npy author nonorm
python dump-glove-emb.py ../input/$day/glove_singer_finish.vectors.txt ../input/$day/singer_finish_emb.npy singer nonorm
python dump-glove-emb.py ../input/$day/glove_song_finish.vectors.txt ../input/$day/song_finish_emb.npy song nonorm
