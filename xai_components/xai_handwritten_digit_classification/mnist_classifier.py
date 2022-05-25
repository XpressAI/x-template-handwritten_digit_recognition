from xai_components.base import InArg, OutArg, InCompArg, Component, xai_component

#------------------------------------------------------------------------------
#                    Xircuits Component : DownloadDataset
#------------------------------------------------------------------------------
@xai_component
class DownloadDataset(Component):
    dataset_name: InArg[str]
    
    def __init__(self):
        self.done = False
        self.dataset_name = InArg(None)
    
    def execute(self, ctx):
        from tensorflow import keras
        import numpy as np
        dataset_name = self.dataset_name.value
        if dataset_name == 'mnist':
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        elif dataset_name == 'fashion_mnist':
            (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

        print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
        print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        y_train = keras.utils.to_categorical(y_train)
        y_test = keras.utils.to_categorical(y_test)
        
        ctx.update({'x_train': x_train,
                    'y_train': y_train,
                    'x_test' : x_test,
                    'y_test' : y_test})
        
        self.done = True
        
#------------------------------------------------------------------------------
#                    Xircuits Component : VisualizeData
#------------------------------------------------------------------------------
@xai_component
class VisualizeData(Component):
    
    def __init__(self):
        self.done = False
    
    def execute(self, ctx):
        import matplotlib.pyplot as plt
        train_data = ctx['x_train']
        
        for i in range(9):
            ax = plt.subplot(3,3,i+1)
            plt.imshow(train_data[i], cmap='gray')
            plt.axis('off')
        plt.show()
        
        self.done = True
        
#------------------------------------------------------------------------------
#                    Xircuits Component : CreateModel
#------------------------------------------------------------------------------
@xai_component
class CreateModel(Component):
    loss: InArg[str]
    optimizer: InArg[str]
    
    compiled_model: OutArg[any]
    
    def __init__(self):
        self.done = False
        self.loss = InArg('categorical_crossentropy')
        self.optimizer = InArg('adam')
        
        self.compiled_model = OutArg(None)
        
    def execute(self, ctx):
        from tensorflow import keras
        train_data = ctx['x_train']
        train_labels = ctx['y_train']
        
        input_shape = train_data[0].shape
        num_class = int(train_labels[0].shape[0])
        
        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(num_class, activation="softmax"),
            ]
        )
        
        model.compile(
            loss=self.loss.value,
            optimizer=self.optimizer.value,
            metrics=['accuracy']
        )
        
        model.summary()
        
        self.compiled_model.value = model
        
        self.done = True
        
#------------------------------------------------------------------------------
#                    Xircuits Component : TrainModel
#------------------------------------------------------------------------------
@xai_component
class TrainModel(Component):
    compiled_model: InArg[any]
    epochs: InArg[int]
    batch_size: InArg[int]
    validation_split: InArg[float]
    
    model: OutArg[any]
    
    def __init__(self):
        self.done = False
        self.compiled_model = InArg(None)
        self.epochs = InArg(None)
        self.batch_size = InArg(None)
        self.validation_split = InArg(0.1)
        
        self.model = OutArg(None)
    
    def execute(self, ctx):
        model = self.compiled_model.value
        x_train = ctx['x_train']
        y_train = ctx['y_train']
        epochs = self.epochs.value
        batch_size = self.batch_size.value
        validation_split = self.validation_split.value
        
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        
        self.model.value = model
        ctx.update({'train_history': history.history})
        self.done = True

#------------------------------------------------------------------------------
#                    Xircuits Component : EvaluateModel
#------------------------------------------------------------------------------
@xai_component
class EvaluateModel(Component):
    model: InArg[any]
    
    def __init__(self):
        self.done = False
        self.model = InArg(None)
    
    def execute(self, ctx):
        from sklearn.metrics import classification_report
        import numpy as np
        model = self.model.value
        x_test = ctx['x_test']
        y_test = ctx['y_test']
        
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("Loss: {:.2f}".format(loss))
        print("Accuracy: {:.2f}".format(accuracy))
        
        y_pred = model.predict(x_test)
        y_pred = [np.argmax(i) for i in y_pred]
        y_test = [np.argmax(i) for i in y_test]
        print(classification_report(y_test, y_pred, digits=10))
        
        self.done = True
 
#------------------------------------------------------------------------------
#                    Xircuits Component : PlotTrainingMetrics
#------------------------------------------------------------------------------
@xai_component
class PlotTrainingMetrics(Component):

    def __init__(self):
        self.done = False
    
    def execute(self, ctx) -> None:
        import matplotlib.pyplot as plt
        history = ctx['train_history']

        acc = history['accuracy']
        val_acc = history['val_accuracy']

        loss = history['loss']
        val_loss = history['val_loss']
        
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.ylim([0, 1.0])
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()

        self.done = True
        
#------------------------------------------------------------------------------
#                    Xircuits Component : SaveModel
#------------------------------------------------------------------------------
@xai_component
class SaveModel(Component):
    model: InArg[any]
    save_model_path: InArg[str]
    keras_format: InArg[bool]
    
    def __init__(self):
        self.done = False
        self.model = InArg(None)
        self.save_model_path = InArg(None)
        self.keras_format = InArg(False)
    
    def execute(self, ctx):
        import os
        model = self.model.value
        model_name = self.save_model_path.value
        
        dirname = os.path.dirname(model_name)
        
        if len(dirname):
            os.makedirs(dirname, exist_ok=True)
        
        if self.keras_format.value:
            model_name = model_name + '.h5'
        else:
            model_name = model_name
        model.save(model_name)
        print(f"Saving model at: {model_name}")
        ctx.update({'saved_model_path': model_name})
        self.done = True

#------------------------------------------------------------------------------
#                    Xircuits Component : ConvertTFModelToOnnx
#------------------------------------------------------------------------------
@xai_component
class ConvertTFModelToOnnx(Component):
    output_onnx_path: InArg[str]
    
    def __init__(self):
        self.done = False
        self.output_onnx_path = InArg(None)
        
    def execute(self, ctx):
        import os
        saved_model = ctx['saved_model_path']
        onnx_path = self.output_onnx_path.value
        dirname = os.path.dirname(onnx_path)
        if len(dirname):
            os.makedirs(dirname, exist_ok=True)
            
        os.system(f"python -m tf2onnx.convert --saved-model {saved_model} --opset 11 --output {onnx_path}.onnx")
        print(f'Converted {saved_model} TF model to {onnx_path}.onnx')
        
        self.done = True
