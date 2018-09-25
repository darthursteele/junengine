import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

class TSCrossValidationHandler():
    def __init__(self, model, name, n_splits=3):
        """Class for handling time-series model cross-validation explicitly.
        
        This is needed in order to view the model fit results for each of the folds
        and to extract the model predictions for use as training features for an 
        ensemble model.        
        """
        self.model = model
        self.name = name
        self.n_splits = n_splits
        self.test_X = []
        self.test_resid_Y = []
        self.test_true_Y = []
        self.test_pred_Y = []
        self.train_X = []
        self.train_true_Y = []
        self.train_resid_Y = []
        self.train_pred_Y = []
        self.mses = []
        self.mapes = []
        self.figsize = (14, 9)

    def do_cross_val(self, X, Y):
        ## Do a log transform on Y
        tscv = TimeSeriesSplit(n_splits=self.n_splits) 
        i = 1
        for i_train, i_test in tscv.split(X):
            self.model.fit(X[i_train], Y[i_train])
            
            self.train_resid_Y.append([y-y_true for y_true,y in zip( Y[i_train], self.model.predict(X[i_train]))])
            self.train_true_Y.append(Y[i_train]) 
            self.train_X.append(i_train)
            self.train_pred_Y.append([y for y in self.model.predict(X[i_train])])
            self.test_resid_Y.append([y-y_true for y,y_true in zip(self.model.predict(X[i_test]), Y[i_test])])
            self.test_true_Y.append(Y[i_test])
            self.test_X.append(i_test)
            self.test_pred_Y.append([y for y in self.model.predict(X[i_test])])
                                      
            self.mses.append(mean_squared_error(Y[i_test], self.model.predict(X[i_test])))
           
            self.mapes.append( 100*np.mean( [ (y-Y[i])/Y[i] for i, y in zip(i_test, self.model.predict(X[i_test]))  ] ) )
                
            print(f"Fold {i} complete with: {len(i_train)} training samples and {len(i_test)} test samples.")
            i += 1

    def fit_and_evaluate(self, X_train, Y_train, X_test, Y_test):
        """Do a single fit to a training dataset and evaluation on a test dataset."""
        self.model.fit(X_train, Y_train)
        y_train_pred = self.model.predict(X_train)
        y_train_errs = [y-y_true for y_true, y in zip(Y_train, y_train_pred)]
        y_test_pred = self.model.predict(X_test)
        y_test_errs = [y-y_true for y_true, y in zip(Y_test, y_test_pred)]
        mse = mean_squared_error(Y_test, y_test_pred)
        mape = 100*np.mean([(y-Y)/Y for Y,y in zip(Y_test,y_test_pred)])
        pstddev = 100*np.std([(y-Y)/Y for Y,y in zip(Y_test,y_test_pred)])
        return { "Model":self.name, 
                 "MSE":mse, 
                 "MAPE":mape, 
                 "STDDEV":pstddev, 
                 "Y_errs":np.array(y_test_errs), 
                 "Y_predicted":y_test_pred }            
            
    def report_metrics(self):
        mse_mean = round(np.sqrt(np.mean(self.mses)),3)
        mse_std = round(np.sqrt(np.std(self.mses)),3)
        mape_mean = round(np.mean(self.mapes),3)
        mape_std = round(np.std(self.mapes),3)
        print(f"Cross-validation results for {self.name}:")
        print(f"   RMSE:   {mse_mean} \xb1 {mse_std}")
        print(f"   MAPE: ({mape_mean} +/- {mape_std})%")
        print(f"   N Splits: {self.n_splits}\n")
        
    def get_metrics(self):
        mse_mean = round(np.sqrt(np.mean(self.mses)),3)
        mse_std = round(np.sqrt(np.std(self.mses)),3)
        mape_mean = round(np.mean(self.mapes),3)
        mape_std = round(np.std(self.mapes),3)    
        return { 'mse_mean':mse_mean, 'mse_std':mse_std, 'mape_mean':mape_mean, 'mape_std':mape_std }
    
    def plot_fits(self):
        """Plot the results of cross validation.
        
        Returns:
        ax -- An axis object containing the resultant plot
        
        """
        X_all_test = np.array(self.test_X).flatten()
        Y_all_test = np.array(self.test_true_Y).flatten()
        Y_all_test_pred = np.array(self.test_pred_Y).flatten()
        Y_resid_all_test = np.array(self.test_resid_Y).flatten()
        X_all_train = np.array(self.train_X).flatten()
        Y_all_train = np.array(self.train_true_Y).flatten()
        Y_all_fit = np.array(self.train_pred_Y).flatten()
        Y_all_fit_resid = np.array(self.train_resid_Y).flatten()
            
        # Plot the real data vs. the model 
        fig = plt.figure(figsize=self.figsize)
        fig.suptitle(f"Cross-validation Results for {self.name}", fontsize=16)
        plt.subplot(311)
        plt.title("Model Predicted Price vs. Actual Price (all folds combined)")
        ax1 = plt.plot(X_all_test, Y_all_test, marker='o', color='black', label="True Y")
        ax1 = plt.plot(X_all_test, Y_all_test_pred, marker='o', color='blue', label="Predicted Y" )
        
        plt.subplot(312)
        plt.title("Model Prediction Errors")
        ax2 = plt.plot(X_all_test, Y_resid_all_test, marker='o', color='blue')
        sqrtmse = np.sqrt(np.mean(self.mses))
        ax2 = plt.plot([ X_all_test[0], X_all_test[-1]], [-sqrtmse, -sqrtmse], 'r--')
        ax2 = plt.plot([ X_all_test[0], X_all_test[-1]], [ sqrtmse,  sqrtmse], 'r--')
        
        
        plt.subplot(313)
        plt.title("Model Fit residuals (Final Training Split)")
        ax3 = plt.plot(X_all_train[-1], Y_all_fit_resid[-1], marker='o', color='green')
        
        plt.show()
        
        return fig
