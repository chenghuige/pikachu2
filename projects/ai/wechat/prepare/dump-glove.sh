day=$1
#python dump-glove-emb.py ../input/glove_word.vectors.txt ../input/word_norm_emb.npy word norm
#python dump-glove-emb.py ../input/glove_word.vectors.txt ../input/word_emb.npy word nonorm

#python dump-glove-emb.py ../input/$day/glove_doc.vectors.txt ../input/$day/doc_norm_emb.npy doc norm
#python dump-glove-emb.py ../input/$day/glove_user.vectors.txt ../input/$day/user_norm_emb.npy user norm
#python dump-glove-emb.py ../input/$day/glove_author.vectors.txt ../input/$day/author_norm_emb.npy author norm
#python dump-glove-emb.py ../input/$day/glove_singer.vectors.txt ../input/$day/singer_norm_emb.npy singer norm
#python dump-glove-emb.py ../input/$day/glove_song.vectors.txt ../input/$day/song_norm_emb.npy song norm

python dump-glove-emb.py ../input/$day/glove_doc.vectors.txt ../input/$day/doc_emb.npy doc nonorm
python dump-glove-emb.py ../input/$day/glove_user.vectors.txt ../input/$day/user_emb.npy user nonorm
python dump-glove-emb.py ../input/$day/glove_author.vectors.txt ../input/$day/author_emb.npy author nonorm
python dump-glove-emb.py ../input/$day/glove_singer.vectors.txt ../input/$day/singer_emb.npy singer nonorm
python dump-glove-emb.py ../input/$day/glove_song.vectors.txt ../input/$day/song_emb.npy song nonorm

#python dump-glove-emb.py ../input/glove_tag_valid.vectors.txt ../input/tag_valid_norm_emb.npy tag norm
#python dump-glove-emb.py ../input/glove_key_valid.vectors.txt ../input/key_valid_norm_emb.npy key norm
