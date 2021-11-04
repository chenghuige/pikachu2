cat $1 | wk -F ',' '{if ($4 == "4") {print $1","$4","$5}}'  | python eval.py
