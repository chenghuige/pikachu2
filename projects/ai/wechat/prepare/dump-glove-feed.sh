python dump-glove-emb.py ../input/glove_word.vectors.txt ../input/word_norm_emb.npy word norm
python dump-glove-emb.py ../input/glove_tag.vectors.txt ../input/tag_norm_emb.npy tag norm
python dump-glove-emb.py ../input/glove_key.vectors.txt ../input/key_norm_emb.npy key norm

python dump-glove-emb.py ../input/glove_word.vectors.txt ../input/word_emb.npy word nonorm
python dump-glove-emb.py ../input/glove_tag.vectors.txt ../input/tag_emb.npy tag nonorm
python dump-glove-emb.py ../input/glove_key.vectors.txt ../input/key_emb.npy key nonorm

