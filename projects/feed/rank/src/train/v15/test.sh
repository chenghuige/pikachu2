SHELL_FOLDER=$(dirname "$0")
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
echo $SHELL_FOLDER
echo ${SHELL_FOLDER##*/}
x=`basename "$PWD"`
echo $x

#. ../../../../tools/bin/shflags
#. ../../../../../../tools/bin/shflags

#DEFINE_string 'mark' 'video' 'video or tuwen' 'n'

# Parse the command-line.
#FLAGS "$@" || exit 1
#eval set -- "${FLAGS_ARGV}"

##cho 'haha' ${FLAGS_mark}

#echo $*

x="${mark:-video2}"
echo $x
