# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

CORPUS=../input/key_corpus.txt
VOCAB_FILE=../input/glove_key.vocab
COOCCURRENCE_FILE=../input/key.cooccurrence.bin
COOCCURRENCE_SHUF_FILE=../input/key.cooccurrence.shuf.bin
BUILDDIR=build
SAVE_FILE=../input/glove_key.vectors
VERBOSE=2
MEMORY=100.0
VOCAB_MIN_COUNT=1
VECTOR_SIZE=128
MAX_ITER=10
WINDOW_SIZE=15
BINARY=2
NUM_THREADS=8
X_MAX=10

$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE

$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE

$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE

$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE

rm $COOCCURRENCE_FILE
rm $COOCCURRENCE_SHUF_FILE
rm $SAVE_FILE.bin

python dump-glove-emb.py ../input/glove_key.vectors.txt ../input/key_glove_emb.npy key nonorm

