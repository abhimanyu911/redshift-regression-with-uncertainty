from astropy.table import Table
from scipy.stats import zscore
from livelossplot import PlotLossesKeras
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import sys
import os
from scipy.stats import gaussian_kde
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import zscore, boxcox
from tqdm import tqdm
pd.options.mode.chained_assignment = None
tfpl = tfp.layers 
tfd = tfp.distributions
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base


def set_seed(seed: int = 0) -> None:
  np.random.seed(seed)
  tf.random.set_seed(seed)
  # When running on the CuDNN backend, two further options must be set
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  # Set a fixed value for the hash seed
  os.environ["PYTHONHASHSEED"] = str(seed)
  print(f"Random seed set as {seed}")


def get_indeterminate_column_info(df = None):
    if df is None:
        raise ValueError("Dataframe must be provided.")
    else:
        flag = False
        for column in list(df.columns.values):
            if df[column].isnull().values.any():
                print(f'{column} has {df[column].isnull().sum()} null values')
                flag = True
            elif df[column].isin([-np.inf]).values.any():
                print(f'{column} has {df[column].isin([np.inf, -np.inf]).values.sum()} indeterminate values')
                flag = True
        if not(flag):
            print('All null/indeterminate values handled')


def frequentist_evaluation(X_test = None, y_test = None, model = None, model_name = None, data_split = None):
    subdirname = f'{model_name}_{data_split}'
    fileloc = f'./predictions/{subdirname}'
    os.makedirs(fileloc, exist_ok=True)
    filename = {
        'csv' : fileloc+'/predictions.csv',
        'npy' : fileloc+'/prediction_array.npy'
    }
    y_pred = model.predict(X_test)
    corr_coeff = np.corrcoef(np.array(y_test).reshape(-1), y_pred.reshape(-1))[0,1]
    r2_value = r2_score(y_test, y_pred) 
    rmse_value = mean_squared_error(y_test, y_pred, squared = False)


    df = pd.read_csv(f'./{data_split}_samples.csv')
    df['Redshift_preds'] = list(np.squeeze(y_pred))
    df.to_csv(fileloc+'/preds_with_features.csv', index=False)
    generate_csv(
        y_test_list= list(y_test),
        mean_predictions=list(np.squeeze(y_pred)),
        std_predictions=list(np.squeeze(np.zeros(y_pred.shape))),
        filename=filename['csv']
    )
    np.save(filename['npy'], y_pred)

    print(f'RMSE = {rmse_value}')
    print(f'R^2 = {r2_value}')
    print(f'Correlation = {corr_coeff}')


def ensembled_evaluation(X_test = None, y_test = None, model = None, ensemble_size = None, 
                         thresholds = None, pred_mode = None, model_name = None, data_split = None):
    
    subdirname = f'{model_name}_{data_split}'
    fileloc = f'./predictions/{subdirname}'
    os.makedirs(fileloc, exist_ok=True)
    filename = {
        'csv' : fileloc+'/predictions.csv',
        'npy' : fileloc+'/prediction_array.npy'
    }
    successful_pred_list = []
    corresponding_label_list = []
    y_test_list = list(y_test)
    predictions = np.zeros((X_test.shape[0], ensemble_size))
    for i in tqdm(range(ensemble_size)):
        predictions[:,i] = np.squeeze(model.predict(X_test, verbose=0))
    mean_predictions = np.mean(predictions, axis = 1)
    std_predictions = np.std(predictions, axis = 1)
    
    np.save(filename['npy'],predictions)
    generate_csv(
        y_test_list=y_test_list,
        mean_predictions= mean_predictions,
        std_predictions= std_predictions,
        filename = filename['csv']
    )
    df = pd.read_csv(f'./{data_split}_samples.csv')
    df['Redshift_preds'] = mean_predictions
    df.to_csv(fileloc+'/preds_with_features.csv', index=False)

    for threshold in thresholds:
        successful_pred_list = []
        corresponding_label_list = []
        for i,std in enumerate(std_predictions):
            if 3*std < threshold:
                if pred_mode == 'mean':
                    successful_pred_list.append(mean_predictions[i])
                else:
                    successful_pred_list.append(np.max(predictions[i,:]))
                corresponding_label_list.append(y_test_list[i])

        classified_samples = len(successful_pred_list)/X_test.shape[0]*100
        corr_coeff = np.corrcoef(corresponding_label_list,successful_pred_list)[0,1]
        r2 = r2_score(corresponding_label_list, successful_pred_list)
        rmse = mean_squared_error(corresponding_label_list, successful_pred_list, squared=False)
        print(f'\nFor {threshold}:')
        print(f'Samples predicted = {classified_samples}')
        print(f'RMSE = {rmse}')
        print(f'R^2 = {r2}')
        print(f'Correlation = {corr_coeff}')  

def train(model = None, mode = None, X_train= None, y_train= None, X_val = None, y_val = None, epochs = None):
    model.compile(optimizer='adam', loss='mae',  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        f'./checkpoints_{mode}/best',
        monitor='val_root_mean_squared_error',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        save_freq='epoch'
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 100
    )

    # Train the model
    model.fit(
        X_train,
        y_train, 
        epochs=epochs, 
        batch_size=256, 
        verbose=1, 
        validation_data=(X_val, y_val), 
        callbacks = [PlotLossesKeras(), ckpt, early_stop]
    )


def generate_csv(y_test_list, mean_predictions, std_predictions, filename):
    data = {'true_value': y_test_list, 'mean': mean_predictions, 'std': std_predictions}
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def plot_scatter_plot(model_name = None, data_split = None):
    # Read the data from the CSV file
    dirname = 'predictions'
    subdir = f'{model_name}_{data_split}'
    file = f'./{dirname}/{subdir}/predictions.csv'

    data = pd.read_csv(file)

    # Extract the 'true_value' and 'mean' columns
    true_value = data['true_value']
    mean = data['mean']

    # Calculate the correlation coefficient
    correlation = np.corrcoef(true_value, mean)[0, 1]

    # Calculate the R^2 score
    r2 = r2_score(true_value, mean)

    # Calculate the RMSE
    rmse = mean_squared_error(true_value, mean, squared=False)

    # Plot the scatter plot with diagonal line
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=mean, y=true_value)
    plt.plot([0, true_value.max()], [0, true_value.max()], color='red', linestyle='--')
    plt.xlabel('Mean')
    plt.ylabel('True Value')
    plt.title('Scatter Plot with diagonal line (represents a perfect prediction)')

    # Set the position of the box and adjust y-axis limit
    plt.ylim(0, max(true_value) + 1)
    plt.xlim(0, max(mean)+0.5)

    # Display the RMSE, R^2, and correlation coefficient values in a single box
    text_box = f'RMSE = {rmse:.4f}\nR^2 = {r2:.4f}\nCorrelation = {correlation:.4f}'
    plt.text(0.05, 0.80, text_box, transform=plt.gca().transAxes,fontsize=14,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.5'))

    plt.show()


from scipy import stats
from astropy.modeling import models, fitting
from scipy.optimize import curve_fit

def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp( - (x - mean)**2 / (2*standard_deviation ** 2))

def plot_sample_distribution(model_name = None, sample_idx=0, y_test = None, data_split = None, no_of_bins = 60, fresh_ensemble = False,
                             fresh_predictions = None, save_plot = True):
    
    dirname = 'predictions'
    subdir = f'{model_name}_{data_split}'

    if not(fresh_ensemble):
        file = f'./{dirname}/{subdir}/prediction_array.npy'
        predictions = np.load(file)
        predictions = predictions[sample_idx, :]
    else:
        predictions = fresh_predictions
    
    sample_value = y_test[sample_idx]
    mean = np.mean(predictions)
    std = np.std(predictions)
    predictions = list(predictions)
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))
    #plt.ylim((0, 0.10))

    # Normalize the histogram to ensure the area under the curve sums up to 1
    hist, bin_edges = np.histogram(predictions, bins=no_of_bins, density=True)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + bin_width / 2  # Compute the bin centers
    hist_norm = hist/np.sum(hist)


    # Create a masked array to exclude bars beyond 3 sigma
    mask = np.logical_or(bin_centers < mean - 3 * std, bin_centers > mean + 3 * std)
    masked_hist = np.ma.masked_array(hist, mask=mask)
    plt.ylim((0, np.max(hist/np.sum(hist))+np.std(hist/np.sum(hist))))
    

    popt, _ = curve_fit(gaussian, bin_centers, hist_norm, p0=[1., 0., 1.])
    x_interval_for_fit = np.linspace(bin_edges[0], bin_edges[-1], hist.shape[0])

    #skewness = stats.skew(predictions)
    #kurtosis = stats.kurtosis(predictions)
    chi_square, p = stats.chisquare(hist_norm, gaussian(x_interval_for_fit, *popt)*1/np.sum(gaussian(x_interval_for_fit, *popt)))

    # Set the bar colors based on sigma ranges
    bar_colors = []
    for x in bin_centers:
        if mean - 1 * std <= x <= mean + 1 * std:
            bar_colors.append('lightblue')
        elif mean - 2 * std <= x <= mean + 2 * std and (x> mean + 1*std or x< mean - 1*std):
            bar_colors.append('green')
        else:
            bar_colors.append('red')


    # Plot the bars with colors and exclude bars beyond 3 sigma
    plt.bar(bin_centers, masked_hist * bin_width, width=bin_width, color=bar_colors)
    plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt)*1/np.sum(gaussian(x_interval_for_fit, *popt)), c='black')
    plt.legend([
    plt.Rectangle((0, 0), 1, 1, color='lightblue'),
    plt.Rectangle((0, 0), 1, 1, color='green'),
    plt.Rectangle((0, 0), 1, 1, color='red')
], ['\u03BC - \u03C3 \u2264  z \u2264  \u03BC + \u03C3', '\u03BC - 2\u03C3 \u2264  z < \u03BC - \u03C3 and \u03BC + \u03C3 < z \u2264  \u03BC + 2\u03C3', '\u03BC - 3\u03C3 \u2264  z < \u03BC - 2\u03C3 and \u03BC + 2\u03C3 < z \u2264  \u03BC + 3\u03C3'], loc='upper right')
    chi_squared_text = f"\u03C7\u00B2 = {chi_square:.4f}"
    plt.text(0.05, 0.95, chi_squared_text, transform=plt.gca().transAxes, ha='left', va='top', bbox=dict(facecolor='white', edgecolor='black'))
    plt.xlabel('Redshift (z)')
    plt.ylabel('Density')
    plt.title(f'Distribution of Redshift Predictions (True value = {sample_value:.4f})')
    fileloc = f'./{dirname}/{subdir}/{subdir}_distribution_{sample_idx}.png'
    if save_plot:
        plt.savefig(fileloc, dpi = 120)
    plt.show()
    
def unknown_predictions(data = None, model = None, ensemble_size = 1000, model_name = None):
    subdirname = f'{model_name}_unknown_redshift'
    fileloc = f'./predictions/{subdirname}'
    os.makedirs(fileloc, exist_ok=True)
    filename = {
        'csv' : fileloc+'/predictions.csv',
        'npy' : fileloc+'/prediction_array.npy'
    }

    predictions = np.zeros((data.shape[0], ensemble_size))
    for i in range(ensemble_size):
        predictions[:,i] = np.squeeze(model.predict(data, verbose=0))
    mean_predictions = np.mean(predictions, axis = 1)
    std_predictions = np.std(predictions, axis = 1)

    np.save(filename['npy'],predictions)
    df = pd.read_csv('./unknown_redshift_samples.csv')
    df['Redshift_mean'] = mean_predictions.reshape(-1, 1)
    df['Redshift_std'] = std_predictions.reshape(-1, 1)

    df.to_csv(filename['csv'], index=False)

def plot_redshift_distribution(mode):

    title_text = {
        'bayesian' : mode.capitalize(),
        'mcdropout' : 'MC Dropout'
    }

    train_file = f"{mode}_train.csv"
    val_file = f"{mode}_val.csv"
    test_file = f"{mode}_test.csv"
    preds_file = f"unknown_redshift_{mode}_preds.csv"

    # Load the data from CSV files
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    test_data = pd.read_csv(test_file)
    preds_data = pd.read_csv(preds_file)

    # Extract mean and std values from the train, val, and test datasets
    train_mean = train_data['mean'].values
    val_mean = val_data['mean'].values
    test_mean = test_data['mean'].values

    # Compute the histogram for the known data
    known_mean_std = np.concatenate([train_mean, val_mean, test_mean])
    hist_known, bin_edges_known = np.histogram(known_mean_std, bins=30, density=True)
    bin_width_known = bin_edges_known[1] - bin_edges_known[0]
    bin_centers_known = bin_edges_known[:-1] + bin_width_known / 2

    # Compute the histogram for the unknown data
    unknown_mean = preds_data['Redshift_mean'].values
    hist_unknown, bin_edges_unknown = np.histogram(unknown_mean, bins=30, density=True)
    bin_width_unknown = bin_edges_unknown[1] - bin_edges_unknown[0]
    bin_centers_unknown = bin_edges_unknown[:-1] + bin_width_unknown / 2

    # Set the plot style and size
    sns.set(style="darkgrid")
    plt.figure(figsize=(12, 6))

    # Plot the known data histogram
    plt.subplot(1, 2, 1)
    plt.ylim((0, 0.2))
    bar_colors_known = []
    for x in bin_centers_known:
        if np.abs(x) <= np.mean(known_mean_std) + 1 * np.std(known_mean_std):
            bar_colors_known.append('lightblue')
        elif np.abs(x) > np.mean(known_mean_std) + 1 * np.std(known_mean_std) and np.abs(x) <= np.mean(known_mean_std) + 2 * np.std(known_mean_std):
            bar_colors_known.append('green')
        elif np.abs(x) > np.mean(known_mean_std) + 2 * np.std(known_mean_std) and np.abs(x) <= np.mean(known_mean_std) + 3 * np.std(known_mean_std):
            bar_colors_known.append('red')
    mask_known = (np.abs(bin_centers_known) > np.mean(known_mean_std) + 3 * np.std(known_mean_std))
    masked_hist_known = np.ma.masked_array(hist_known, mask=mask_known)
    plt.bar(bin_centers_known, masked_hist_known * bin_width_known, width=bin_width_known, color=bar_colors_known)
    #plt.bar(bin_centers_known, hist_known * bin_width_known, width=bin_width_known, color=bar_colors_known)
    plt.xlabel('Redshift (z)')
    plt.ylabel('Density')
    plt.title(f'{title_text[mode]} prediction - Known Redshift samples')

    plt.legend([
        plt.Rectangle((0, 0), 1, 1, color='lightblue'),
        plt.Rectangle((0, 0), 1, 1, color='green'),
        plt.Rectangle((0, 0), 1, 1, color='red')
        ], ['|z|<1'+''.join(r'$\sigma$'), '|z|<2'+''.join(r'$\sigma$'), '|z|<3'+''.join(r'$\sigma$')], 
        loc='upper right'
    )

    # Plot the unknown data histogram
    plt.subplot(1, 2, 2)
    plt.ylim((0, 0.2))
    bar_colors_unknown = []
    mean_unknown = np.mean(unknown_mean)
    std_unknown = np.std(unknown_mean)
    for x in bin_centers_unknown:
        if np.abs(x) <= mean_unknown + 1 * std_unknown:
            bar_colors_unknown.append('lightblue')
        elif np.abs(x) <= mean_unknown + 2 * std_unknown and np.abs(x) > mean_unknown + 1 * std_unknown:
            bar_colors_unknown.append('green')
        elif x > mean_unknown + 2 * std_unknown and np.abs(x) <= mean_unknown + 3 * std_unknown:
            bar_colors_unknown.append('red')
    mask_unknown = (np.abs(bin_centers_unknown) > mean_unknown + 3 * std_unknown)
    masked_hist_unknown = np.ma.masked_array(hist_unknown, mask=mask_unknown)
    plt.bar(bin_centers_unknown, masked_hist_unknown * bin_width_unknown, width=bin_width_unknown, color=bar_colors_unknown)
    plt.xlabel('Redshift (z)')
    plt.ylabel('Density')
    plt.title(f'{title_text[mode]} prediction - Unknown Redshift samples')

    # Display legend
    plt.legend([
        plt.Rectangle((0, 0), 1, 1, color='lightblue'),
        plt.Rectangle((0, 0), 1, 1, color='green'),
        plt.Rectangle((0, 0), 1, 1, color='red')
        ], ['|z|<=1'+''.join(r'$\sigma$'), '|z|<2'+''.join(r'$\sigma$'), '|z|<3'+''.join(r'$\sigma$')],
        loc = 'upper right'
    )

    # Display the plot
    plt.tight_layout()
    plt.show()

def compare_real_and_predicted_redshifts(model_name=None, no_of_bins=None):
    dirname = 'predictions'
    train_file = f'./{dirname}/{model_name}_train/predictions.csv'
    val_file = f'./{dirname}/{model_name}_val/predictions.csv'
    test_file = f'./{dirname}/{model_name}_test/predictions.csv'

    # Load predictions from train, validation, and test files
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    df_test = pd.read_csv(test_file)

    # Concatenate the dataframes into one
    df = pd.concat([df_train, df_val, df_test], ignore_index=True)

    real_redshifts = df['true_value']
    predictions = df['mean']

    # Set seaborn darkgrid style
    sns.set(style="darkgrid")

    # Calculate histograms
    hist_predictions, bin_edges = np.histogram(predictions, bins=no_of_bins, density=True)
    hist_real_redshifts, bin_edges = np.histogram(real_redshifts, bins=no_of_bins, density=True)

    # Normalize histograms
    hist_predictions_norm = hist_predictions / np.sum(hist_predictions)
    hist_real_redshifts_norm = hist_real_redshifts / np.sum(hist_real_redshifts)

    # Set larger figure size
    plt.figure(figsize=(12, 6))

    # Plot normalized histograms using plt.bar
    bar_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + bar_width / 2

    plt.bar(bin_centers, hist_predictions_norm, width=bar_width, alpha=0.6, color='blue', label='Predictions')
    plt.bar(bin_centers, hist_real_redshifts_norm, width=bar_width, alpha=0.6, color='red', label='Real Redshifts')
    plt.xticks(np.arange(0, 4.25, 0.25))  # Set x-axis tick values
    plt.xlabel('Redshift')
    plt.ylabel('Normalized Frequency')
    plt.title('Distributions of Real Redshifts and Predictions')
    plt.legend()

    plt.show()




def plot_uncertainty_calibration(no_of_bins, model_name, data_split):
    csv_path = f'./predictions/{model_name}_{data_split}/predictions.csv'
    df = pd.read_csv(csv_path)

    # Calculate the number of examples and bins
    T = len(df)
    N = no_of_bins

    # Sort the DataFrame by standard deviation
    df_sorted = df.sort_values('std')

    # Assign each example to a bin
    df_sorted['bin'] = pd.cut(df_sorted.index, bins=N, labels=False)

    # Calculate the root mean variance per bin (RMV)
    uncert_Bm = df_sorted.groupby('bin')['std'].apply(lambda x: (x**2).mean())

    # Calculate the root mean squared error per bin (RMSE)
    err_Bm = df_sorted.groupby('bin').apply(lambda x: ((x['true_value'] - x['mean'])**2).mean())

    # Calculate the uncertainty calibration error (UCE)
    uce = np.sum(df_sorted.groupby('bin').size() / T * np.abs(err_Bm - uncert_Bm))

    print(f"Uncertainty Calibration Error (UCE): {uce}")

    # Perform min-max normalization
    normalized_err_Bm = (err_Bm - err_Bm.min()) / (err_Bm.max() - err_Bm.min())
    normalized_uncert_Bm = (uncert_Bm - uncert_Bm.min()) / (uncert_Bm.max() - uncert_Bm.min())

    # Sort x and y arrays
    x = normalized_uncert_Bm.values
    y = normalized_err_Bm.values
    sort_indices = np.argsort(x)
    x_sorted = x[sort_indices]
    y_sorted = y[sort_indices]

    # Plot scatter plot with smooth curve
    sns.set(style="darkgrid")
    plt.scatter(x_sorted, y_sorted)
    plt.plot(x_sorted, y_sorted, color='blue', linestyle='--', alpha=0.5)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Perfect Calibration')
    plt.xlabel('Normalized uncertainty per bin')
    plt.ylabel('Normalized Mean Squared Error per Bin (MSE)')
    plt.title('Scatter Plot: Normalized Uncertainty vs Normalized MSE')
    plt.legend(loc='upper left')

    # Add UCE text box
    text_box = f'UCE: {uce:.4f}'
    plt.text(0.65, 0.05, text_box, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4'), transform=plt.gca().transAxes)

    # Display the plot
    plt.show()