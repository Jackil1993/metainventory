import pandas as pd
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import keras_metrics
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.model_selection import cross_val_score
from keras.utils import plot_model


class Data:
    def __init__(self, simulation, scaling=True):
        self.simulation = simulation
        self.scaling = scaling

    def load(self):
        # z-score standartization
        def scaler(data):
            return scale(data)
        # load dataset
        if self.simulation == 1:
            df = pd.read_excel('10_class.xlsx').values
            # split into input (X) and output (Y) variables
            X = df[:, :75]
            Y = df[:, 75]
            if self.scaling == True:
                X = scaler(X)

        else:
            df = pd.read_excel('simulation2_trainingset_class.xlsx').values
            # split into input (X) and output (Y) variables
            X = df[:, :17]
            Y = df[:, 17]
            if self.scaling == True:
                X = scaler(X)

        return X, Y


class Model:
    def __init__(self, plot=False, summary=False):
        self.plot = plot
        self.summary = summary

    def baseline_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim=75, kernel_initializer='normal', activation='elu'))
        # model.add(ActivityRegularization(l1=0.05, l2=0.05))
        model.add(Dropout(0.3, noise_shape=None, seed=None))
        model.add(Dense(20, kernel_initializer='normal', activation='elu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adamax',
                      metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])

        if self.summary == True:
            print(model.summary())

        if self.plot == True:
            plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False)

        return model


class Training:
    def __init__(self, cv=False):
        model = Model()
        data = Data(1)
        self.X, self.Y = data.load()

        if cv == False:
            self.estimator = KerasClassifier(build_fn=model.baseline_model, epochs=60, batch_size=200, verbose=0, validation_split=0.35)
        else:
            self.estimator = KerasClassifier(build_fn=model.baseline_model, epochs=60, batch_size=200, verbose=0)

    def train_model(self, to_plot=True):
        history = self.estimator.fit(self.X, self.Y)
        results = self.estimator.predict(self.X)

        cm = confusion_matrix(self.Y, results)
        kappa = cohen_kappa_score(self.Y, results)
        print(cm)
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
        loss = history.history['val_loss'][-1]
        print('acc ', acc)
        print('prec ', prec)
        print('rec ', rec)
        print('f1 ', f1)
        print('kappa', kappa)
        print('loss ', loss)

        if to_plot == True:
            plt.title('Accuracy')
            plt.plot(history.history['acc'], label='Train')
            plt.plot(history.history['val_acc'], label='Test')
            plt.xlabel('Epoch')
            plt.legend()
            plt.show()

            plt.plot(history.history['val_precision'], label='Precision')
            plt.plot(history.history['val_recall'], label='Recall')
            plt.xlabel('Epoch')
            plt.legend()
            plt.show()

            plt.title('Loss')
            plt.plot(history.history['loss'], label='Train')
            plt.plot(history.history['val_loss'], label='Test')
            plt.xlabel('Epoch')
            plt.legend()
            plt.show()

    def cv_score(self, cv=10):
        cv_score = cross_val_score(self.estimator, self.X, self.Y, cv=cv, scoring='f1')
        print(cv_score)
        ax = sns.boxplot(cv_score)
        plt.show()

if __name__== "__main__":
    t = Training(cv=True)
    t.train_model()
