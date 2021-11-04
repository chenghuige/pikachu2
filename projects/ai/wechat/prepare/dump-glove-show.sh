day=$1

python dump-glove-emb.py ../input/$day/glove_doc.vectors.txt ../input/$day/doc_norm_emb.npy doc norm
python dump-glove-emb.py ../input/$day/glove_user.vectors.txt ../input/$day/user_norm_emb.npy user norm
python dump-glove-emb.py ../input/$day/glove_author.vectors.txt ../input/$day/author_norm_emb.npy author norm
python dump-glove-emb.py ../input/$day/glove_singer.vectors.txt ../input/$day/singer_norm_emb.npy singer norm
python dump-glove-emb.py ../input/$day/glove_song.vectors.txt ../input/$day/song_norm_emb.npy song norm

python dump-glove-emb.py ../input/$day/glove_doc.vectors.txt ../input/$day/doc_emb.npy doc nonorm
python dump-glove-emb.py ../input/$day/glove_user.vectors.txt ../input/$day/user_emb.npy user nonorm
python dump-glove-emb.py ../input/$day/glove_author.vectors.txt ../input/$day/author_emb.npy author nonorm
python dump-glove-emb.py ../input/$day/glove_singer.vectors.txt ../input/$day/singer_emb.npy singer nonorm
python dump-glove-emb.py ../input/$day/glove_song.vectors.txt ../input/$day/song_emb.npy song nonorm
