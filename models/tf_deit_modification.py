"""
Attempt at modifying the tf model graph to replace gelu with its tanh approximation

DOES NOT WORK AS INTENDED
"""


def validate(layer):
    # print(layer_type)
    if hasattr(layer, 'activation'):
        print(layer.activation)

    if hasattr(layer, 'layers'):
        for sub_layer in layer.layers:
            validate(sub_layer)


def modify_layers(layer):
    layer_type = type(layer).__name__
    print(layer_type)
    if hasattr(layer, 'activation'):
        # print(layer_type, layer.activation.__name__)

        if layer.activation.__name__ == 'gelu':
            print(layer.activation)
            print("replaced activation")
            layer.activation = tf.keras.layers.Activation('relu')
            return layer
    return layer


def flatten_model(model):
    if not any(hasattr(layer, 'layers') for layer in model.layers):
        return model  # No sub-model defined within this model

    flat_model = keras.Sequential()

    def recursive_flatten(submodel):
        for layer in submodel.layers:
            if hasattr(layer, 'layers'):
                recursive_flatten(layer)
            elif isinstance(layer, keras.Sequential):
                flat_model.add(layer)
            else:
                flat_model.add(layer)

    recursive_flatten(model)

    return flat_model


def gelu_approximation(x):
    return tf.keras.activations.gelu(x, approximate=True)


class GeluApproxLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GeluApproxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GeluApproxLayer, self).build(input_shape)

    def call(self, inputs):
        return 0.5 * inputs * (1.0 + tf.tanh(np.sqrt(2 / np.pi) * (inputs + 0.044715 * tf.pow(inputs, 3))))

    def compute_output_shape(self, input_shape):
        return input_shape


def SpatialExtractorDeit2():
    model = deit.deit_tiny_distilled_patch16_224(pretrained=True)
    return model


def SpatialExtractorDeiT() -> keras.Model:
    # vit = hub.KerasLayer(model_gcs_path, trainable=False)

    vit = keras.models.load_model("deit_tiny_distilled_patch16_224_fe")

    dummy_inputs = tf.ones((1, 224, 224, 3))
    _ = vit(dummy_inputs)

    vit.build(input_shape=(1, 224, 224, 3))
    vit.summary()
    ''''''
    copy_layers = []
    for index, layer in enumerate(vit.layers):

        if not hasattr(layer, 'layers'):
            copy_layers.append(layer)
            print(layer.get_weights())
            print(layer)
            continue

        for sublayer in vit.layers[index].layers:
            print(sublayer)
            if hasattr(sublayer, 'activation') and sublayer.activation.__name__ == 'gelu':
                print(sublayer.activation)
                print("replaced activation")
                # replace with approx
                sublayer.activation = gelu_approximation
                print(sublayer.activation)
            if isinstance(sublayer, keras.layers.Dense):
                print(sublayer.get_weights())
            # insert layer

        layer.save("deittmp/deit_tiny_distilled_patch16_224_fe_GELUapprox" + str(index))
        vit.layers[index] = keras.models.load_model(
            "deittmp/deit_tiny_distilled_patch16_224_fe_GELUapprox" + str(index))

    # return model
    # reconstruct the model as functional (keras)
    inputs = keras.layers.Input(shape=(224, 224, 3))
    x, outputs = None, None
    dist = None

    subfolder_list = [subfolder for subfolder in os.listdir("deittmp") if
                      os.path.isdir(os.path.join("deittmp", subfolder))]

    subfolder_list.sort(key=lambda a: int(a.split("GELUapprox")[-1]))
    for index, subfolder in enumerate(subfolder_list):
        subfolder_path = os.path.join("deittmp", subfolder)

        if os.path.exists(subfolder_path):
            model = keras.models.load_model(subfolder_path)
            model.summary()
            print(model)
            if index == 0:
                x = model(inputs)
                x = keras.layers.ZeroPadding1D(padding=(1, 1))(x)
            else:
                x = model(x)[0]

            # Or perform some other operations on the model
        else:
            print(f"Model in subfolder {subfolder} does not exist in the specified path: {subfolder_path}")

    '''
    for index, layer in enumerate(vit.layers):

        if hasattr(layer, 'layers'):
            model = keras.models.load_model("deittmp/deit_tiny_distilled_patch16_224_fe_GELUapprox" + str(index))

            model.summary()
            print(model)
            if index == 0:
                x = model(inputs)
                x = keras.layers.ZeroPadding1D(padding=(1, 1))(x)
            else:
                x = model(x)[0]
            # model = model_surgery.prepare_for_tflite(model)

        else:

            print(layer)

            if layer.name == "distillation_head":

                continue

            if layer.name == "classification_head":
                #insert stride slice
                # Define the indices for strided slice
                begin_indices = [0, 0, 0]  # Start from the beginning of the first and third dimensions
                end_indices = [1, 1, 768]  # Include the entire first and third dimensions
                strides = [1, 1, 1]  # Stride of 2 for the second dimension

                # Create a Lambda layer with the strided slice operation
                x = keras.layers.Lambda(lambda a: tf.strided_slice(
                    a,
                    begin_indices,
                    end_indices,
                    strides=strides
                ), name="strided_slice")(x)

            config = layer.get_config()
            weights = layer.get_weights()
            cloned_layer = type(layer).from_config(config)

            try:
                cloned_layer.build(cloned_layer.input_shape)
                cloned_layer.set_weights(weights)
                print("loading weights")
            except:
                pass
            x = cloned_layer(x)
            if isinstance(layer, keras.layers.Dense):
                # print number of units
                print(layer.units)
            print(cloned_layer.input.shape)
            print(cloned_layer.output.shape)
            print("failed to load model")

        print(model.inputs)
        print(model.outputs)

        '''
    for l in copy_layers:
        if l.name == "distillation_head":
            continue
        if l.name == "classification_head":

            x1 = keras.layers.Lambda(lambda a: tf.strided_slice(
                a,
                [0, 1, 0],
                [1, 2, 768],
                strides=[1, 1, 1]
            ), name="strided_slice")(x)
            x2 = keras.layers.Lambda(lambda a: tf.strided_slice(
                a,
                [0, 0, 0],
                [1, 1, 768],
                strides=[1, 1, 1]
            ), name="strided_slice1")(x)
            # get layer in l for which l.name == "distillation_head"
            distillation_head_layer = None
            for layer in copy_layers:
                if layer.name == "distillation_head":
                    distillation_head_layer = layer
                    break
            x1 = distillation_head_layer(x1)
            x2 = l(x2)
            # add
            x = keras.layers.Add()([x1, x2])
            continue

        x = l(x)

    outputs = x
    vit = keras.models.Model(inputs=inputs, outputs=outputs)

    vit.build(input_shape=(None, 224, 224, 3))
    vit.summary()

    inp = keras.layers.Input(shape=(224, 224, 3))
    # x = keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=input_shape)(inp)
    # x = PreprocessTFLayer()(x)
    x = keras.layers.Normalization(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                   variance=[(0.229 * 255) ** 2, (0.224 * 255) ** 2, (0.225 * 255) ** 2], )(inp)
    vec = vit(x)
    # vec = dist(vec)
    # expand dim
    out = keras.layers.GlobalAveragePooling1D()(vec)
    model = keras.models.Model(inputs=inp, outputs=out)
    model.summary()
    # save

    return model
