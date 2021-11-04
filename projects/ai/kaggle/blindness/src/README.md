TODO FIXME batchnorm problem to be fixed as if K.set_learning_phrase(1) when training will not convergent, valid loss boom.. train loss decrease ok  

but set learning phrase is a must for like dropout ops, without it easily overfit you can veryfiy using examples/tf/melt imdb example 
python ./melt-train.py --eager=0 --dropout=1 =0 and compare  

# utils/melt/app/train.py  
  K.set_learning_phase(1)  
  loss = melt.tower(lambda i: train_fn(x[i], y[i]), num_gpus)
  K.set_learning_phase(0)


