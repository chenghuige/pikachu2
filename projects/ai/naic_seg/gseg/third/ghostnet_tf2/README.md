# ghostnet_tf2
An implementation of GhostNet for Tensorflow 2.0+ (From the paper "GhostNet: More Features from Cheap Operations")

Link to paper: https://arxiv.org/pdf/1911.11907.pdf

## Using Ghostnet

This implementation is a normal Keras Model object.
You initialize it, build or compile it and it is ready to fit!

Dummy example:
```
from ghost_model import GhostNet

# Initialize model with 10 classes
model = GhostNet(10)

# Compile and fit
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy']) 
model.fit(data)

```

Check out the Jupyter notebook "mnist_example.ipynb" in this repository for an example of using this implementation on a real dataset.
