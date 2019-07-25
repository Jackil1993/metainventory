import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense, Dropout, ActivityRegularization
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils import plot_model
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.diagnostic import het_breuschpagan


class Data:
    def __init__(self, simulation, scaling=False):
        self.simulation = simulation
        self.scaling = scaling

    def load(self):
        # z-score standartization
        def scaler(data):
            return scale(data.values)
        # load dataset
        if self.simulation == 1:
            df = pd.read_excel('new_10_trainingset_full.xlsx').values
            if self.scaling == True:
                df = scaler(df)
            # split into input (X) and output (Y) variables
            X = df[:, :75]
            Y = df[:, 75]

        else:
            df = pd.read_excel('simulation2_trainingset_full.xlsx')
            if self.scaling == True:
                df = scaler(df)
            # split into input (X) and output (Y) variables
            X = df[:, :25]
            Y = df[:, 25]

        return X, Y


class Model:
    def __init__(self, plot=False, summary=False):
        self.plot = plot
        self.summary = summary

    def baseline_model(self):
        model = Sequential()
        model.add(Dense(200, input_dim=75,  kernel_initializer='normal', activation='elu'))
        model.add(Dropout(0.1, noise_shape=None, seed=None))
        model.add(ActivityRegularization(l1=300, l2=300))
        model.add(Dense(100, kernel_initializer='normal', activation='elu'))
        model.add(Dropout(0.1, noise_shape=None, seed=None))
        model.add(ActivityRegularization(l1=200, l2=200))
        model.add(Dense(100, kernel_initializer='normal', activation='elu'))
        model.add(Dropout(0.1, noise_shape=None, seed=None))
        model.add(ActivityRegularization(l1=100, l2=100))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mse', optimizer='adamax')

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
            self.estimator = KerasRegressor(build_fn=model.baseline_model, epochs=200, batch_size=2, verbose=0, validation_split=0.25)
        else:
            self.estimator = KerasRegressor(build_fn=model.baseline_model, epochs=100, batch_size=2, verbose=0)

    def train_model(self, plot_distribution=False, residual_analysis=False, learning_path=False, ols=False, save=False):
        history = self.estimator.fit(self.X, self.Y)
        results = self.estimator.predict(self.X)
        r2 = r2_score(self.Y, results)
        adjusted_r2 = 1 - (1-r2)*(1000-1)/(1000-75-1)
        see = sqrt(sum((self.Y-results)**2)/(1000-75))
        mse = mean_squared_error(self.Y, results)
        print('explained variance ', explained_variance_score(self.Y, results))
        print('r2 ', r2)
        print('adjusted ', adjusted_r2)
        print('mse ', mse)
        print('Standard error of the estimate ', see)

        if plot_distribution == True:
            ax = sns.boxplot(x=['data', 'prediction'], y=[self.Y, results])
            plt.show()
            ax = sns.violinplot(data=[self.Y, results])
            plt.show()

        if residual_analysis == True:
            residuals = [i - j for i, j in zip(self.Y, results)]
            print(stats.anderson(residuals, dist='norm'))
            print('mean ', sum(residuals) / len(residuals))
            sns.distplot(residuals, bins=20, kde=True,
                         kde_kws={"color": "r", "lw": 3, "label": "Kernel density estimation"})
            plt.legend()
            plt.xlabel('residuals')
            plt.show()
            res = stats.probplot(residuals, plot=plt)
            plt.show()

        if learning_path == True:
            # Plot training & validation loss values
            plt.plot([i for i in history.history['loss']], label='Train')
            plt.plot([i for i in history.history['val_loss']], label='Test')
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.show()

        if ols == True:
            model = sm.OLS(self.Y, self.X)
            results = model.fit()
            resids = results.resid
            exog = results.model.exog
            print(results.summary())
            print(het_breuschpagan(resids, exog))

    def cv_score(self, cv=10):
        cv_score = cross_val_score(self.estimator, self.X, self.Y, cv=cv, scoring='r2')
        print(cv_score)
        ax = sns.boxplot(cv_score)
        plt.show()

if __name__== "__main__":
    t = Training()
    t.train_model(plot_distribution=True, learning_path=True)