TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

i=$1

o=${i/.cc/.so}

echo $TF_INC
echo $TF_LIB
#g++ -std=c++11 -shared $i -o $o -I $TF_INC -l tensorflow_framework -L $TF_LIB -fPIC -Wl,-rpath $TF_LIB

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared time.cc -o time.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3

#g++ -std=c++11 -shared time.cc -o time.so -fPIC -O3 -I /usr/include/eigen3 -I /usr/include/google/tensorflow -I /usr/include/google/tensorflow/tensorflow/contrib/makefile/downloads/absl -I /usr/include/google/tensorflow/tensorflow/contrib/makefile/downloads/protobuf/src -I /usr/include/google/tensorflow/tensorflow/contrib/makefile/downloads/googletest/googletest/include/ -ltensorflow_cc -ltensorflow_framework
