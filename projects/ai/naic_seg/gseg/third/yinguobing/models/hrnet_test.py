from tensorflow import keras
from hrnet import HRNetBody, hrnet_body

if __name__ == "__main__":
    inputs = keras.Input((256, 256, 64))

    # Functional API
    body_func = hrnet_body()
    outputs_func = body_func(inputs)
    model_func = keras.Model(inputs, outputs_func, name="hrn_func")
    model_func.summary()
    model_func.save("./saved_model/hrnet_body_func")

    # Subclassed model
    body_subc = HRNetBody(name="HRNetBody")
    outputs_subc = body_subc(inputs)
    model_subc = keras.Model(inputs, outputs_subc, name="hrn_subc")
    model_subc.summary()
    model_subc.save("./saved_model/hrnet_body_subc")
