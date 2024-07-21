import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Function to read raster data and extract bands
def read_raster(file_path):
    with rasterio.open(file_path) as src:
        data = src.read()
    return data

# Normalize the raster based on provided percentiles with logarithmic transformation
def normalize_raster(raster, norm_percentiles):
    for i in range(raster.shape[-1]):
        # Extract the band data
        band = raster[:, :, i]
        
        # Check for zero or negative values before log transformation
        if np.any(band <= 0):
            print(f"Zero or negative values found in band {i+1} before log transformation")
            band[band <= 0] = np.nan  # Assign NaN to invalid values to avoid log issues
        
        # Apply logarithmic transformation
        band = np.log(band * 0.005 + 1)
        
        # Perform normalization
        lower, upper = norm_percentiles[i]
        band = (band - lower) / upper
        
        # Check for NaN values after normalization
        if np.any(np.isnan(band)):
            print(f"NaN values detected in band {i+1} after normalization")
        
        # Replace the band data with the normalized data
        raster[:, :, i] = band
    
    return raster

# Load and preprocess all raster data
def load_and_preprocess_data(file_paths, norm_percentiles):
    X_list = []
    y_list = []
    
    for file in file_paths:
        data = read_raster(file)
        data = np.moveaxis(data, 0, -1)  # Move the band axis to the last dimension
        
        # Print raw data statistics before normalization
        print(f"Raw data statistics for file {file}:")
        for i in range(9):
            print(f"Band {i+1} - Mean: {np.nanmean(data[:, :, i])}, Std Dev: {np.nanstd(data[:, :, i])}, Min: {np.nanmin(data[:, :, i])}, Max: {np.nanmax(data[:, :, i])}")
        
        # Normalize the input bands (1-9)
        data[:, :, :9] = normalize_raster(data[:, :, :9], norm_percentiles)
        
        # Print normalized data statistics
        print(f"Normalized data statistics for file {file}:")
        for i in range(9):
            print(f"Band {i+1} - Mean: {np.nanmean(data[:, :, i])}, Std Dev: {np.nanstd(data[:, :, i])}, Min: {np.nanmin(data[:, :, i])}, Max: {np.nanmax(data[:, :, i])}")

        # Extract input (bands 1-9) and output (bands 10-12)
        X = data[:, :, :9].reshape(-1, 9)
        y = data[:, :, 9:12].reshape(-1, 3)
        
        # Remove any rows with NaN values
        nan_mask = ~np.isnan(X).any(axis=1)
        X = X[nan_mask]
        y = y[nan_mask]
        
        X_list.append(X)
        y_list.append(y)
    
    X_combined = np.concatenate(X_list, axis=0)
    y_combined = np.concatenate(y_list, axis=0)
    
    # Find indices where any of the categories is above 0.7 or below 0.3
    woody_indices = np.where((y_combined[:, 0] > 0.7) | (y_combined[:, 0] < 0.3))[0]
    herbaceous_indices = np.where((y_combined[:, 1] > 0.7) | (y_combined[:, 1] < 0.3))[0]
    bare_indices = np.where((y_combined[:, 2] > 0.7) | (y_combined[:, 2] < 0.3))[0]

    # Determine the minimum number of samples available for any category
    min_samples = min(len(woody_indices), len(herbaceous_indices), len(bare_indices))

    # Randomly sample min_samples indices from each category
    woody_indices = np.random.choice(woody_indices, min_samples, replace=False)
    herbaceous_indices = np.random.choice(herbaceous_indices, min_samples, replace=False)
    bare_indices = np.random.choice(bare_indices, min_samples, replace=False)

    # Combine the indices and select the corresponding data points
    selected_indices = np.concatenate([woody_indices, herbaceous_indices, bare_indices])
    X_selected = X_combined[selected_indices]
    y_selected = y_combined[selected_indices]
    
    # Check for NaN values in the combined data
    if np.isnan(X_selected).any() or np.isnan(y_selected).any():
        print("NaN values found in the data. Please check the normalization process and input data.")
    
    return X_selected, y_selected

# Define the file paths
training_files = [
    r"C:\0_Documents\10_ETH\Thesis\Python\DL model_files\training\2_DL model\fullrast2_tile1_sept2022.tif",
    r"C:\0_Documents\10_ETH\Thesis\Python\DL model_files\training\2_DL model\fullrast2_tile1_sept2023.tif",
    r"C:\0_Documents\10_ETH\Thesis\Python\DL model_files\training\2_DL model\fullrast2_tile2_sept2022.tif",
    r"C:\0_Documents\10_ETH\Thesis\Python\DL model_files\training\2_DL model\fullrast2_tile2_sept2023.tif",
    r"C:\0_Documents\10_ETH\Thesis\Python\DL model_files\training\2_DL model\fullrast2_tile3_sept2022.tif",
    r"C:\0_Documents\10_ETH\Thesis\Python\DL model_files\training\2_DL model\fullrast2_tile3_sept2023.tif",
    r"C:\0_Documents\10_ETH\Thesis\Python\DL model_files\training\2_DL model\fullrast2_tile4_apr2020.tif",
    r"C:\0_Documents\10_ETH\Thesis\Python\DL model_files\training\2_DL model\fullrast2_tile5_apr2020.tif",
    r"C:\0_Documents\10_ETH\Thesis\Python\DL model_files\training\2_DL model\fullrast2_tile6_sept2023.tif"
]

# Define normalization percentiles
NORM_PERCENTILES = np.array([
    [1.7417268007636313, 2.023298706048351],
    [1.7261204997060209, 2.038905204308012],
    [1.6798346251414997, 2.179592821212937],
    [1.7734969472909623, 2.2890068333026603],
    [2.289154079164943, 2.6171674549378166],
    [2.382939712192371, 2.773418590375327],
    [2.3828939530384052, 2.7578332604178284],
    [2.1952484264967844, 2.789092484314204],
    [1.554812948247501, 2.4140534947492487]])

# Load and preprocess data
X, y = load_and_preprocess_data(training_files, NORM_PERCENTILES)

import numpy as np

# Function to remove rows with NaN values from X and corresponding values in y
def remove_nan_rows(X, y):
    nan_mask = ~np.isnan(X).any(axis=1)
    X_clean = X[nan_mask]
    y_clean = y[nan_mask]
    return X_clean, y_clean

# Use the function to clean X and y
X_clean, y_clean = remove_nan_rows(X, y)

# Verify the cleaned data statistics
print("Number of training data points after removing NaNs:", X_clean.shape[0])
print("Input data statistics (X):")
print("Mean:", np.mean(X_clean, axis=0))
print("Std Dev:", np.std(X_clean, axis=0))
print("Min:", np.min(X_clean, axis=0))
print("Max:", np.max(X_clean, axis=0))

print("Target data statistics (y):")
print("Mean:", np.mean(y_clean, axis=0))
print("Std Dev:", np.std(y_clean, axis=0))
print("Min:", np.min(y_clean, axis=0))
print("Max:", np.max(y_clean, axis=0))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simpler model
model = Sequential([
    Dense(64, activation='relu', input_shape=(9,)),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # Use softmax to ensure the outputs sum to 1
])

# Compile the model with a slightly higher learning rate
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_absolute_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=128, validation_split=0.2)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("Mean Absolute Error for each band:", mae)

# Print MAE for each band separately
for i, label in enumerate(['Woody Vegetation', 'Herbaceous Vegetation', 'Bare Ground']):
    print(f"Mean Absolute Error for {label}: {mean_absolute_error(y_test[:, i], y_pred[:, i])}")

# Save the trained model
model_save_path = "C:/0_Documents/10_ETH/Thesis/DLmodel.h5"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")


# Load and normalize the test raster
input_raster_path = r"C:/0_Documents/10_ETH/Thesis/Analysis2/Sentinel-2_2023_7_Onguma.tif"

with rasterio.open(input_raster_path) as src:
    raster_array = src.read()
    raster_array = np.moveaxis(raster_array, 0, -1)  # Move the band axis to the last dimension

# Normalize the raster
raster_array[:, :, :9] = normalize_raster(raster_array[:, :, :9], NORM_PERCENTILES)

# Prepare data for prediction
raster_shape = raster_array.shape
raster_array_flattened = raster_array[:, :, :9].reshape(-1, raster_shape[-1])

# Predict cover fractions
predictions_flattened = model.predict(raster_array_flattened)

# Reshape predictions to the original raster shape
predictions = predictions_flattened.reshape(raster_shape[0], raster_shape[1], 3)

# Calculate mean and standard deviation for each band of the prediction raster
mean_predictions = np.mean(predictions, axis=(0, 1))
std_predictions = np.std(predictions, axis=(0, 1))

# Save the predictions to a new raster file
output_raster_path = r"C:/0_Documents/10_ETH/Thesis/Analysis2/Predicted_Cover_Fractions_final.tif"

with rasterio.open(
    output_raster_path,
    'w',
    driver='GTiff',
    height=predictions.shape[0],
    width=predictions.shape[1],
    count=3,
    dtype=predictions.dtype,
    crs=src.crs,
    transform=src.transform
) as dst:
    dst.write(np.moveaxis(predictions, -1, 0))

print("Mean predictions for each band:")
for i, label in enumerate(['Woody Vegetation', 'Herbaceous Vegetation', 'Bare Ground']):
    print(f"Mean {label}: {mean_predictions[i]}")

print("Standard deviation of predictions for each band:")
for i, label in enumerate(['Woody Vegetation', 'Herbaceous Vegetation', 'Bare Ground']):
    print(f"Standard deviation {label}: {std_predictions[i]}")
