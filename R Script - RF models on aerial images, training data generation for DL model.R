### ------------------------------------------------------------------------------------------------------------------
### Task 1: Growth form fractions from aerial photograph ------------------------------------
### ------------------------------------------------------------------------------------------------------------------

rm(list = ls())

# Load required libraries
library(terra)
library(ranger)

# Working directory
setwd("C:/0_Documents/10_ETH/Thesis/Python/DL model_files/training/1_RF model")

# Reading in the training data (manully obtained point samples from QGIS)
plots1 <- read.csv("rf_trainingset_tile1.csv")
orthophoto1 <- rast("pleiades_maptile1_georeferenced.tif") # aerial image

plots2 <- read.csv("rf_trainingset_tile2.csv")
orthophoto2 <- rast("pleiades_maptile2_georeferenced.tif") # aerial image

plots3 <- read.csv("rf_trainingset_tile3.csv")
orthophoto3 <- rast("pleiades_maptile3_georeferenced.tif") # aerial image

plots4 <- read.csv("rf_trainingset_rainfall1.csv")
orthophoto4 <- rast("pleiades_maptile4_georeferenced_ongumarainfall1.tif") # aerial image

plots5 <- read.csv("rf_trainingset_rainfall2.csv")
orthophoto5 <- rast("pleiades_maptile5_georeferenced_ongumarainfall2.tif") # aerial image

plots6 <- read.csv("rf_trainingset_ongava.csv")
orthophoto6 <- rast("pleiades_maptile6_georeferenced_ongava.tif") # aerial image


### The following section has been run and all changes have been saved.
### Add 3 additional predictors
names(orthophoto1)
plot(orthophoto1)
orthophoto1 <- orthophoto1[[c("maptile1_georeferenced_1", "maptile1_georeferenced_2", "maptile1_georeferenced_3")]] # extract rgb channels
orthophoto2 <- orthophoto2[[c("maptile2_georeferenced_1", "maptile2_georeferenced_2", "maptile2_georeferenced_3")]] # extract rgb channels
orthophoto3 <- orthophoto3[[c("maptile3_georeferenced_1", "maptile3_georeferenced_2", "maptile3_georeferenced_3")]] # extract rgb channels
orthophoto4 <- orthophoto4[[c("pleiades_maptile4_georeferenced_ongumarainfall1_1", 
                              "pleiades_maptile4_georeferenced_ongumarainfall1_2", 
                              "pleiades_maptile4_georeferenced_ongumarainfall1_3")]] 
orthophoto5 <- orthophoto5[[c("pleiades_maptile5_georeferenced_ongumarainfall2_1", 
                              "pleiades_maptile5_georeferenced_ongumarainfall2_2", 
                              "pleiades_maptile5_georeferenced_ongumarainfall2_3")]] 
orthophoto6 <- orthophoto6[[c("pleiades_maptile6_georeferenced_ongava_1", 
                              "pleiades_maptile6_georeferenced_ongava_2", 
                              "pleiades_maptile6_georeferenced_ongava_3")]] 
orthophotovalues1 <- values(orthophoto1)
orthophotovalues2 <- values(orthophoto2)
orthophotovalues3 <- values(orthophoto3)
orthophotovalues4 <- values(orthophoto4)
orthophotovalues5 <- values(orthophoto5)
orthophotovalues6 <- values(orthophoto6)
orthophotovalues1 <- as.data.frame(orthophotovalues1)
orthophotovalues2 <- as.data.frame(orthophotovalues2)
orthophotovalues3 <- as.data.frame(orthophotovalues3)
orthophotovalues4 <- as.data.frame(orthophotovalues4)
orthophotovalues5 <- as.data.frame(orthophotovalues5)
orthophotovalues6 <- as.data.frame(orthophotovalues6)
names(orthophotovalues1) <- c("R","G","B")
names(orthophotovalues2) <- c("R","G","B")
names(orthophotovalues3) <- c("R","G","B")
names(orthophotovalues4) <- c("R", "G", "B")
names(orthophotovalues5) <- c("R", "G", "B")
names(orthophotovalues6) <- c("R", "G", "B")


# 1. Add the visible vegetation index as predictor (Higginbottom et al. 2018)
# R0, G0, B0 are constants used to reference green color (Joseph and Devadas 2015)
calculate_VVI <- function(df, R0 = 30, G0 = 50, B0 = 1) {
  VVI <- ((1 - df$R - R0) * (1 - df$G - G0) * (1 - df$B - B0)) / ((df$R + R0) * (df$G + G0) * (df$B + B0))
  return(VVI)
}

VVI1 <- calculate_VVI(orthophotovalues1)
VVI2 <- calculate_VVI(orthophotovalues2)
VVI3 <- calculate_VVI(orthophotovalues3)
VVI4 <- calculate_VVI(orthophotovalues4)
VVI5 <- calculate_VVI(orthophotovalues5)
VVI6 <- calculate_VVI(orthophotovalues6)

# 2. Add the values of the two first principle components of a PCA run on the images (Higginbottom et al. 2018)
# Perform PCA on the matrix
perform_PCA <- function(matrix_data) {
  # Center the data
  centered_data <- scale(matrix_data, center = TRUE, scale = FALSE)

  # Compute covariance matrix
  cov_matrix <- cov(centered_data)

  # Perform PCA
  pca_result <- prcomp(centered_data, scale. = FALSE)

  return(pca_result)
}

# Call the function to perform PCA
pca_result1 <- perform_PCA(matrix_data = orthophotovalues1)
pca_result2 <- perform_PCA(matrix_data = orthophotovalues2)
pca_result3 <- perform_PCA(matrix_data = orthophotovalues3)
pca_result4 <- perform_PCA(matrix_data = orthophotovalues4)
pca_result5 <- perform_PCA(matrix_data = orthophotovalues5)
pca_result6 <- perform_PCA(matrix_data = orthophotovalues6)


# Extract the loadings (eigenvectors) of the first two principal components
loadings_first_two_pc1 <- pca_result1$rotation[, 1:2]
loadings_first_two_pc2 <- pca_result2$rotation[, 1:2]
loadings_first_two_pc3 <- pca_result3$rotation[, 1:2]
loadings_first_two_pc4 <- pca_result4$rotation[, 1:2]
loadings_first_two_pc5 <- pca_result5$rotation[, 1:2]
loadings_first_two_pc6 <- pca_result6$rotation[, 1:2]


# Project the original dataframe onto the first two principal components
pcdf1 <- as.data.frame(predict(object = pca_result1, orthophotovalues1)[, 1:2])
pcdf2 <- as.data.frame(predict(object = pca_result2, orthophotovalues2)[, 1:2])
pcdf3 <- as.data.frame(predict(object = pca_result3, orthophotovalues3)[, 1:2])
pcdf4 <- as.data.frame(predict(object = pca_result4, orthophotovalues4)[, 1:2])
pcdf5 <- as.data.frame(predict(object = pca_result5, orthophotovalues5)[, 1:2])
pcdf6 <- as.data.frame(predict(object = pca_result6, orthophotovalues6)[, 1:2])
orthophotovalues1$PC1 <- pcdf1$PC1
orthophotovalues2$PC1 <- pcdf2$PC1
orthophotovalues3$PC1 <- pcdf3$PC1
orthophotovalues4$PC1 <- pcdf4$PC1
orthophotovalues5$PC1 <- pcdf5$PC1
orthophotovalues6$PC1 <- pcdf6$PC1
orthophotovalues1$PC2 <- pcdf1$PC2
orthophotovalues2$PC2 <- pcdf2$PC2
orthophotovalues3$PC2 <- pcdf3$PC2
orthophotovalues4$PC2 <- pcdf4$PC2
orthophotovalues5$PC2 <- pcdf5$PC2
orthophotovalues6$PC2 <- pcdf6$PC2
orthophotovalues1$VVI <- VVI1
orthophotovalues2$VVI <- VVI2
orthophotovalues3$VVI <- VVI3
orthophotovalues4$VVI <- VVI4
orthophotovalues5$VVI <- VVI5
orthophotovalues6$VVI <- VVI6


# Add the predictors to the 3 rasters
# Add three bands
orthophoto1b <- orthophoto1
orthophoto2b <- orthophoto2
orthophoto3b <- orthophoto3
orthophoto4b <- orthophoto4
orthophoto5b <- orthophoto5
orthophoto6b <- orthophoto6
values(orthophoto1) <- orthophotovalues1[,1:3]
values(orthophoto2) <- orthophotovalues2[,1:3]
values(orthophoto3) <- orthophotovalues3[,1:3]
values(orthophoto4) <- orthophotovalues4[,1:3]
values(orthophoto5) <- orthophotovalues5[,1:3]
values(orthophoto6) <- orthophotovalues6[,1:3]
values(orthophoto1b) <- orthophotovalues1[,4:6]
values(orthophoto2b) <- orthophotovalues2[,4:6]
values(orthophoto3b) <- orthophotovalues3[,4:6]
values(orthophoto4b) <- orthophotovalues4[,4:6]
values(orthophoto5b) <- orthophotovalues5[,4:6]
values(orthophoto6b) <- orthophotovalues6[,4:6]
orthophoto1 <- c(orthophoto1,orthophoto1b)
orthophoto2 <- c(orthophoto2,orthophoto2b)
orthophoto3 <- c(orthophoto3,orthophoto3b)
orthophoto4 <- c(orthophoto4, orthophoto4b)
orthophoto5 <- c(orthophoto5, orthophoto5b)
orthophoto6 <- c(orthophoto6, orthophoto6b)
names(orthophoto1) <- c("R","G","B","PC1","PC2","VVI")
names(orthophoto2) <- c("R","G","B","PC1","PC2","VVI")
names(orthophoto3) <- c("R","G","B","PC1","PC2","VVI")
names(orthophoto4) <- c("R", "G", "B", "PC1", "PC2", "VVI")
names(orthophoto5) <- c("R", "G", "B", "PC1", "PC2", "VVI")
names(orthophoto6) <- c("R", "G", "B", "PC1", "PC2", "VVI")

writeRaster(orthophoto1, "maptile1_georeferenced.tif", overwrite = T)
writeRaster(orthophoto2, "maptile2_georeferenced.tif", overwrite = T)
writeRaster(orthophoto3, "maptile3_georeferenced.tif", overwrite = T)
writeRaster(orthophoto4, "maptile4_georeferenced.tif", overwrite = T)
writeRaster(orthophoto5, "maptile5_georeferenced.tif", overwrite = T)
writeRaster(orthophoto6, "maptile6_georeferenced.tif", overwrite = T)



# Nomenclature
# 1: Woody
# 2: Herbaceous
# 3: Bare

# Step 1: Prepare the data

# Growth form types
habitat_points1 <- plots1$class
habitat_points2 <- plots2$class
habitat_points3 <- plots3$class
habitat_points4 <- plots4$class
habitat_points5 <- plots5$class
habitat_points6 <- plots6$class


# Extract the pixel values of the satellite bands for the chosen locations
chosen_locations1 <- data.frame(x = plots1$Longitude, y = plots1$Latitude)
chosen_locations2 <- data.frame(x = plots2$Longitude, y = plots2$Latitude)
chosen_locations3 <- data.frame(x = plots3$Longitude, y = plots3$Latitude)
chosen_locations4 <- data.frame(x = plots4$Longitude, y = plots4$Latitude)
chosen_locations5 <- data.frame(x = plots5$Longitude, y = plots5$Latitude)
chosen_locations6 <- data.frame(x = plots6$Longitude, y = plots6$Latitude)
data1 <- extract(orthophoto1, chosen_locations1, ID = F)
data2 <- extract(orthophoto2, chosen_locations2, ID = F)
data3 <- extract(orthophoto3, chosen_locations3, ID = F)
data4 <- extract(orthophoto4, chosen_locations4, ID = F)
data5 <- extract(orthophoto5, chosen_locations5, ID = F)
data6 <- extract(orthophoto6, chosen_locations6, ID = F)

# Combine the growth form classes and the channels of the orthophoto into one dataframe
df1 <- data.frame(habitat_points1, data1)
df2 <- data.frame(habitat_points2, data2)
df3 <- data.frame(habitat_points3, data3)
df4 <- data.frame(habitat_points4, data4)
df5 <- data.frame(habitat_points5, data5)
df6 <- data.frame(habitat_points6, data6)
names(df1) <- c("habitat_points", "R", "G", "B", "PC1", "PC2", "VVI")
names(df2) <- c("habitat_points", "R", "G", "B", "PC1", "PC2", "VVI")
names(df3) <- c("habitat_points", "R", "G", "B", "PC1", "PC2", "VVI")
names(df4) <- c("habitat_points", "R", "G", "B", "PC1", "PC2", "VVI")
names(df5) <- c("habitat_points", "R", "G", "B", "PC1", "PC2", "VVI")
names(df6) <- c("habitat_points", "R", "G", "B", "PC1", "PC2", "VVI")

# Step 2: Train the random forest model

# Split the data into training and testing sets
train_data <- rbind(df2,df3,df4,df5,df6)
train_data$habitat_points <- as.factor(train_data$habitat_points)
train_data <- na.omit(train_data)

test_data <- df1
test_data$habitat_points <- as.factor(test_data$habitat_points)
test_data <- na.omit(test_data)

# Train the random forest model
rf_model <- ranger(habitat_points ~ ., data = train_data, num.trees = 500)

# Validation
# Extract the test data and actual tree cover values
test_input <- test_data[, -1]  # Remove the habitat_points column
actual_habitat_points <- test_data$habitat_points

# Predict tree cover for the test set using the trained model
predicted_habitat_points <- predict(rf_model, data = test_input)$predictions

# Evaluate the performance of the model
accuracy <- mean(predicted_habitat_points == actual_habitat_points)
print(paste("Accuracy:", accuracy)) # 94.8% accuracy

# Predictions
orthophoto <- orthophoto6 # run the following section for all 6 images

# Prediction
# Define the size of the chunks in degrees based on memory constraints and raster resolution
chunk_size_x <- 0.01
chunk_size_y <- 0.01

# Calculate the number of chunks in each dimension
num_chunks_x <- ceiling((xmax(orthophoto)-xmin(orthophoto)) / chunk_size_x)
num_chunks_y <- ceiling((ymax(orthophoto)-ymin(orthophoto)) / chunk_size_y)

# Initialize an empty list to store the predicted chunks
predicted_chunks <- list()

# Define helper function
pfun <- \(...) {
  predict(...)$predictions
}

# Loop through each chunk
for (i in 1:num_chunks_x) {
  for (j in 1:num_chunks_y) {
    # Define the extent of the current chunk in geographic units
    xmin <- xmin(orthophoto) + (i - 1) * chunk_size_x
    xmax <- min(xmax(orthophoto), xmin + chunk_size_x)
    ymin <- ymin(orthophoto) + (j - 1) * chunk_size_y
    ymax <- min(ymax(orthophoto), ymin + chunk_size_y)
    
    # Create an extent object for the chunk
    chunk_extent <- ext(c(xmin, xmax, ymin, ymax))
    
    # Crop the raster to extract the chunk
    chunk <- crop(orthophoto, chunk_extent)
    
    # Make predictions on the chunk
    predicted_chunk <- predict(chunk,rf_model, fun = pfun, na.rm = T)
    
    # Store the predicted chunk
    predicted_chunks[[length(predicted_chunks) + 1]] <- predicted_chunk
  }
}

# Create a SpatRasterCollection
s <- sprc(predicted_chunks)

# Merge the rasters
m <- merge(s)

# Store the output
writeRaster(m,  "predicted_growthforms_tile6.tif", overwrite = T)

### ------------------------------------------------------------------------------------------------------------------
### Task 2: Aggregating growth form fractions to sentinel2 raster template and training data extraction --------------
### ------------------------------------------------------------------------------------------------------------------

rm(list = ls())
setwd("C:/0_Documents/10_ETH/Thesis/Python/DL model_files/training/1_RF model")
library(terra)
#library(sf)
#onguma <- read_sf("C:/0_Documents/10_ETH/Thesis/GIS/Onguma_sf/Onguma_EPSG_4326.shp")[1]

## Reading in the data
raster_cat1 <- rast("predicted_growthforms_tile1.tif")
raster_cat2 <- rast("predicted_growthforms_tile2.tif")
raster_cat3 <- rast("predicted_growthforms_tile3.tif")
raster_cat4 <- rast("predicted_growthforms_tile4.tif")
raster_cat5 <- rast("predicted_growthforms_tile5.tif")
raster_cat6 <- rast("predicted_growthforms_tile6.tif")
setwd("C:/0_Documents/10_ETH/Thesis/Python/DL model_files/training/2_DL model")
list.files()
sentinel1 <- rast("sentinel-2 onguma sept 2023.tif") # september 2023 (when aerial images were taken)
sentinel2 <- rast("sentinel-2 onguma sept 2022.tif") # september 2022 (one year before aerial images were taken)
sentinel4 <- rast("sentinel-2 onguma april 2020.tif") # april 2020 (when aerial images were taken)
sentinel5 <- rast("sentinel-2 ongava.tif")


dl_training_generation <- function(raster_cat, sentinelraw, outputname){
  
 
  sentinel <- crop(sentinelraw, raster_cat)
  
  # Convert SpatRaster to an array
  image_data_array <- as.array(sentinel)
  
  values(sentinel) <- image_data_array
  
  ### Aerial image preprocessing
  #1. transform input to 3 binary bands
  band1 <- raster_cat # woody
  band1[band1 == 1] <- 1
  band1[band1 == 2] <- 0
  band1[band1 == 3] <- 0
  band2 <- raster_cat # herbaceous
  band2[band2 == 1] <- 0
  band2[band2 == 2] <- 1
  band2[band2 == 3] <- 0
  band3 <- raster_cat # bare
  band3[band3 == 1] <- 0
  band3[band3 == 2] <- 0
  band3[band3 == 3] <- 1
  
  #2. reproject it using nearest neighbor on multiple of resolution of sentinel band
  new_raster <- rast(nrow = nrow(sentinel)*10, ncol = ncol(sentinel)*10, 
                     xmin = xmin(sentinel), ymin = ymin(sentinel), 
                     xmax = xmax(sentinel), ymax = ymax(sentinel), 
                     crs = crs(sentinel))
  
  band1 <- resample(band1, new_raster, method = "near")
  band2 <- resample(band2, new_raster, method = "near")
  band3 <- resample(band3, new_raster, method = "near")
  
  binary_raster <- rast(list(band1,band2,band3))
  names(binary_raster) <- c("Woody", "Herbaceous", "Bare")
  #plot(binary_raster)
  
  #3. aggregate the raster up to sentinel resolution and normalize to 0-1
  growthforms <- aggregate(binary_raster, fact = 10, fun = sum, na.rm=T)/100
  #plot(growthforms)
  output <- rast(list(sentinel,growthforms))
  summary(output)
  
  writeRaster(output, paste0("fullrast2_", outputname, ".tif"))
  
  #4. create training data for deep learning model
  randomsample <- spatSample(output, 5000, method = "regular", replace = F, na.rm = T, as.df = T)
  ??spatSample
  x <- randomsample[,1:9]
  y <- randomsample[,10:12]
  
  write.csv(x, paste0("x2_", outputname, ".csv"))
  write.csv(y, paste0("y2_", outputname, ".csv")) 
}


dl_training_generation(raster_cat1, sentinel1, "tile1_sept2023") # Onguma dry season
dl_training_generation(raster_cat1, sentinel2, "tile1_sept2022") # Onguma dry season
dl_training_generation(raster_cat2, sentinel1, "tile2_sept2023") # Onguma dry season
dl_training_generation(raster_cat2, sentinel2, "tile2_sept2022") # Onguma dry season
dl_training_generation(raster_cat3, sentinel1, "tile3_sept2023") # Onguma dry season
dl_training_generation(raster_cat3, sentinel2, "tile3_sept2022") # Onguma dry season
dl_training_generation(raster_cat4, sentinel4, "tile4_apr2020") # Onguma rainy season
dl_training_generation(raster_cat5, sentinel4, "tile5_apr2020") # Onguma rainy season
dl_training_generation(raster_cat6, sentinel5, "tile6_sept2023") # Ongava dry season