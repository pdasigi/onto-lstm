import sys
import pickle
import numpy

from keras.models import Model, load_model

from index_data import DataProcessor


class PrepositionModel(object):
    def __init__(self, **kwargs):
        self.data_processor = DataProcessor()
        self.embed_dim = kwargs.pop("embed_dim", 50)
        self.numpy_rng = numpy.random.RandomState(12345)
        self.model = None
        self.best_epoch = 0  # index of the best epoch
        self.model_name_prefix = None
        self.model_name = None
        self.encoder = None  # subclasses need to define their encoders
        self.custom_objects = {}
        self.validation_split = 0.1

    def _get_input_layers(self, train_inputs):
       raise NotImplementedError

    def _get_output_layers(self, inputs, dropout, embedding_file, num_mlp_layers):
        raise NotImplementedError

    def train(self, train_inputs, train_labels, num_epochs=20, embedding_file=None, num_mlp_layers=0,
              dropout=None, patience=6):
        '''
        train_inputs (numpy_array): Indexed Head + preposition + child
        train_labels (numpy_array): One-hot matrix indicating labels
        num_epochs (int): Maximum number of epochs to run
        embedding (numpy): Optional pretrained embedding
        num_mlp_layers (int): Number of layers between the encoded inputs and the final softmax
        dropout (dict): Dict containing dropout values after "embedding" and "encoder"
        patience (int): Early stopping patience
        '''
        inputs = self._get_input_layers(train_inputs)
        outputs = self._get_output_layers(inputs, dropout, embedding_file, num_mlp_layers)
        model = Model(input=inputs, output=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        print >>sys.stderr, "%s model summary:" % self.model_name
        model.summary()
        best_accuracy = 0.0
        num_worse_epochs = 0
        for epoch_id in range(num_epochs):
            print >>sys.stderr, "Epoch: %d" % epoch_id
            history = model.fit(train_inputs, train_labels, validation_split=self.validation_split, nb_epoch=1)
            validation_accuracy = history.history['val_acc'][0]  # history['val_acc'] is a list of size nb_epoch
            if validation_accuracy > best_accuracy:
                self.save_model(epoch_id)
                self.best_epoch = epoch_id
                num_worse_epochs = 0
                best_accuracy = validation_accuracy
            elif validation_accuracy < best_accuracy:
                num_worse_epochs += 1
                if num_worse_epochs >= patience:
                    print >>sys.stderr, "Stopping training."
                    break
        self.save_best_model()

    def test(self, inputs, targets):
        if not self.model:
            raise RuntimeError("Model not trained!")
        metrics = self.model.evaluate(inputs, targets)
        print >>sys.stderr, "Test accuracy: %.4f" % (metrics[1])  # The first metric is loss.
        self.write_predictions(inputs)

    def process_data(self, input_file, onto_aware, for_test=False):
        # Subclasses define this method based on the task.
        raise NotImplementedError

    def write_predictions(self, inputs):
        # Subclasses define their own ways of writing predictions depending on the task.
        raise NotImplementedError

    def save_model(self, epoch):
        '''
        Saves the current model using the epoch id to identify the file.
        '''
        self.model.save("%s_%d.model" % (self.model_name_prefix, epoch))
        pickle.dump(self.data_processor, open("%s.dataproc" % self.model_name_prefix, "wb"))

    def save_best_model(self):
        '''
        Copies the model corresponding to the best epoch as the final model file.
        '''
        from shutil import copyfile
        best_model_file = "%s_%d.model" % (self.model_name_prefix, self.best_epoch)
        final_model_file = "%s.model" % self.model_name_prefix
        copyfile(best_model_file, final_model_file)

    def load_model(self, epoch=None):
        '''
        Loads a saved model. If epoch id is provided, will load the corresponding model. Or else,
        will load the best model.
        '''
        if not epoch:
            self.model = load_model("%s.model" % self.model_name_prefix,
                                    custom_objects=self.custom_objects)
        else:
            self.model = load_model("%s_%d.model" % (self.model_name_prefix, epoch),
                                    custom_objects=self.custom_objects)
        self.model.summary()
        self.data_processor = pickle.load(open("%s.dataproc" % self.model_name_prefix, "rb"))
