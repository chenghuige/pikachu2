# Image segmentation loss functions implemented in Keras
Binary and multiclass loss function for image segmentation with one-hot encoded masks of 
shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>). Implemented in Keras.

## Loss functions
All loss functions are implemented using Keras callback structure:

```python
def example_loss() -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        pass

    return loss
```

### Binary loss functions
| Loss function                    | Implementation                                                                                                  |
|----------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Dice's coefficient loss          | https://github.com/maxvfischer/keras-image-segmentation-loss-functions/blob/master/losses/binary_losses.py#L50  |
| Weighted Dice cross entropy loss | https://github.com/maxvfischer/keras-image-segmentation-loss-functions/blob/master/losses/binary_losses.py#L74  |
| Tversky loss                     | https://github.com/maxvfischer/keras-image-segmentation-loss-functions/blob/master/losses/binary_losses.py#L114 |
| Weighted cross entropy           | https://github.com/maxvfischer/keras-image-segmentation-loss-functions/blob/master/losses/binary_losses.py#L137 |
| Balanced cross entropy           | https://github.com/maxvfischer/keras-image-segmentation-loss-functions/blob/master/losses/binary_losses.py#L177 |
| Focal loss                       | https://github.com/maxvfischer/keras-image-segmentation-loss-functions/blob/master/losses/binary_losses.py#L223 |

### Multiclass loss functions
| Loss function                            | Implementation                                                                                                      |
|------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Weighted Tanimoto loss                   | https://github.com/maxvfischer/keras-image-segmentation-loss-functions/blob/master/losses/multiclass_losses.py#L8   |
| Weighted Dice's coefficient loss         | https://github.com/maxvfischer/keras-image-segmentation-loss-functions/blob/master/losses/multiclass_losses.py#L42  |
| Weighted squared Dice's coefficient loss | https://github.com/maxvfischer/keras-image-segmentation-loss-functions/blob/master/losses/multiclass_losses.py#L74  |
| Weighted cross entropy                   | https://github.com/maxvfischer/keras-image-segmentation-loss-functions/blob/master/losses/multiclass_losses.py#L107 |
| Focal loss                               | https://github.com/maxvfischer/keras-image-segmentation-loss-functions/blob/master/losses/multiclass_losses.py#L150 |
