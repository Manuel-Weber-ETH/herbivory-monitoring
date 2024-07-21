### ------------------------------------------------------------------------------------------------------------------
### Result plots ------------------------------------
### ------------------------------------------------------------------------------------------------------------------

### Ongava: ADC, OGL, OTC, TIE, MAR

# Onguma ____________________________________________________________________________________________

library(ggplot2)
library(viridis)  # For a perceptually uniform color palette
library(scales)   # For better scaling options
library(terra)    # For raster data
library(sf)       # For spatial data
library(dplyr)    # For data manipulation
library(RColorBrewer)

# Set working directory
setwd("C:/0_Documents/10_ETH/Thesis/Analysis/")

# # Initialize an empty list to store dataframes
# combined_df_list <- list()
# 
# # Loop through the numbers 26 to 28 and 31 to 49
# for (week_num in c(26:28, 31:49)) {
#   # Read in the CSV file for each week and add it to the list
#   filepath <- paste0("Onguma/week_", week_num, "/results.csv")
#   if (file.exists(filepath)) {
#     df <- read.csv(filepath)
#     # Add a column for the week number
#     df$Week <- week_num
#     combined_df_list[[length(combined_df_list) + 1]] <- df
#   }
# }
# 
# # Combine all dataframes into a single dataframe
# combined_df <- do.call(rbind, combined_df_list)
# combined_df <- na.omit(combined_df)
# combined_df$grazing <- combined_df$Grazing_units.ha_herbaceous_vegetation_selective + combined_df$Grazing_units.ha_herbaceous_vegetation_bulk
# combined_df$browsing <- combined_df$Browsing_units.ha_woody_vegetation_selective + combined_df$Browsing_units.ha_woody_vegetation_bulk
# 
# write.csv(combined_df, "results3_Onguma.csv")

# Read and preprocess the raster data
vegetation <- rast("C:/0_Documents/10_ETH/Thesis/Analysis/Onguma/week_26/vegetation_categories_raster.tif")
vegetation_band1 <- vegetation[[2]]  # herbaceous
vegetation_band2 <- vegetation[[1]]  # woody

# Read the Voronoi polygons
voronoi <- read_sf("C:/0_Documents/10_ETH/Thesis/Analysis/Onguma/week_26/voronoi_polygons.shp")
voronoi <- st_transform(voronoi, crs = 4326)

# Mask the raster with Voronoi polygons
vegetation1_masked <- mask(vegetation_band1, voronoi)
vegetation2_masked <- mask(vegetation_band2, voronoi)

# Convert the masked raster to a data frame
vegetation_df1 <- as.data.frame(vegetation1_masked, xy = TRUE, na.rm = TRUE)
colnames(vegetation_df1) <- c("x", "y", "value")
vegetation_df2 <- as.data.frame(vegetation2_masked, xy = TRUE, na.rm = TRUE)
colnames(vegetation_df2) <- c("x", "y", "value")

# Load the data
combined_df <- read.csv("results_Onguma.csv")
combined_df$Waterpoint <- as.factor(combined_df$Waterpoint)

# Read the waterpoints
waterpoints <- read.csv("C:/0_Documents/10_ETH/Thesis/Analysis/Onguma/waterholes_onguma.csv")
names(waterpoints) <- c("Name", "Waterpoint", "Longitude", "Latitude", "Area")

# Calculate average browsing and grazing values for each Waterpoint
average_browsing <- combined_df %>%
  group_by(Waterpoint) %>%
  summarize(avg_browsing = mean(Browsing_units.ha_woody_vegetation_selective, na.rm = TRUE))

average_grazing <- combined_df %>%
  group_by(Waterpoint) %>%
  summarize(avg_grazing = mean(Grazing_units.ha_herbaceous_vegetation_selective, na.rm = TRUE))

# Specify the Waterpoints for separate color palettes
special_sites <- c("B10", "B11", "B14")

# Calculate ranks and assign colors for special sites
special_browsing <- average_browsing %>%
  filter(Waterpoint %in% special_sites) %>%
  mutate(rank = rank(avg_browsing),
         color = brewer.pal(9, "Blues")[3:7][rank])

special_grazing <- average_grazing %>%
  filter(Waterpoint %in% special_sites) %>%
  mutate(rank = rank(avg_grazing),
         color = brewer.pal(9, "Blues")[3:7][rank])

# Calculate ranks and assign colors for other sites
other_browsing <- average_browsing %>%
  filter(!Waterpoint %in% special_sites) %>%
  mutate(rank = rank(avg_browsing),
         color = brewer.pal(9, "Reds")[2:9][rank])

other_grazing <- average_grazing %>%
  filter(!Waterpoint %in% special_sites) %>%
  mutate(rank = rank(avg_grazing),
         color = brewer.pal(9, "Reds")[2:9][rank])

# Combine the data back together
average_browsing <- bind_rows(special_browsing, other_browsing)
average_grazing <- bind_rows(special_grazing, other_grazing)

# Create named vectors for the colors to use in scale_color_manual
browsing_colors <- setNames(average_browsing$color, average_browsing$Waterpoint)
grazing_colors <- setNames(average_grazing$color, average_grazing$Waterpoint)

# Merge the colors back into the combined_df
combined_df <- combined_df %>%
  left_join(average_browsing %>% select(Waterpoint, browsing_color = color), by = "Waterpoint") %>%
  left_join(average_grazing %>% select(Waterpoint, grazing_color = color), by = "Waterpoint")

# Handle any missing values in the dataset before plotting
combined_df <- combined_df %>%
  filter(!is.na(Browsing_units.ha_woody_vegetation_selective) & !is.na(Grazing_units.ha_herbaceous_vegetation_selective))

# Plot browsing curve
browsingcurve <- ggplot(combined_df, aes(x = Week, y = Browsing_units.ha_woody_vegetation_selective, group = Waterpoint, color = browsing_color)) +
  geom_line(linewidth = 1) +
  scale_color_identity() +  # Use the colors directly
  scale_y_log10(labels = scales::comma) +
  labs(x = "Week", y = "Browsing pressure", color = "Waterpoint") +
  theme_minimal(base_size = 15) +
  theme(
    legend.position = "none",
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12)
  )

print(browsingcurve)

# Plot grazing curve
grazingcurve <- ggplot(combined_df, aes(x = Week, y = Grazing_units.ha_herbaceous_vegetation_selective, group = Waterpoint, color = grazing_color)) +
  geom_line(linewidth = 1) +
  scale_color_identity() +  # Use the colors directly
  scale_y_log10(labels = scales::comma) +
  labs(x = "Week", y = "Grazing pressure", color = "Waterpoint") +
  theme_minimal(base_size = 15) +
  theme(
    legend.position = "none",
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12)
  )

print(grazingcurve)


### Tile 3: Browsing Map

# Filter waterpoints to include only those that appear in combined_df
filtered_waterpoints <- waterpoints %>% filter(Waterpoint %in% combined_df$Waterpoint)

# Calculate quantiles for the vegetation values
quantiles <- quantile(vegetation_df2$value, probs = seq(0, 1, by = 0.1))


# Define a color palette from white to dark green
colors <- c("white", "darkgreen")

browsingmap <- ggplot() +
  geom_raster(data = vegetation_df2, aes(x = x, y = y, fill = value)) +  # Add raster layer
  scale_fill_gradientn(colors = colors, values = scales::rescale(quantiles), na.value = "transparent", name = "Woody cover") +
  geom_sf(data = voronoi, fill = NA, color = "black", linewidth = 0.8) +  # Add Voronoi polygons
  geom_point(data = filtered_waterpoints, aes(x = Longitude, y = Latitude, color = Waterpoint), size = 5) +  # Add waterpoints
  scale_color_manual(values = browsing_colors) +
  theme_minimal(base_size = 15) +
  theme(
    legend.position = "none",
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12)
  ) +
  labs(x = "Longitude", y = "Latitude", color = "Waterpoint", fill = "Woody cover")

browsingmap

### Tile 4: Grazing Map

# Filter waterpoints to include only those that appear in combined_df
filtered_waterpoints <- waterpoints %>% filter(Waterpoint %in% combined_df$Waterpoint)

# Calculate quantiles for the vegetation values
quantiles <- quantile(vegetation_df1$value, probs = seq(0, 1, by = 0.1))

# Define a color palette from white to dark green
colors <- c("white", "darkgreen")

grazingmap <- ggplot() +
  geom_raster(data = vegetation_df1, aes(x = x, y = y, fill = value)) +  # Add raster layer
  scale_fill_gradientn(colors = colors, values = scales::rescale(quantiles), na.value = "transparent", name = "Herbaceous cover") +
  geom_sf(data = voronoi, fill = NA, color = "black", linewidth = 0.8) +  # Add Voronoi polygons
  geom_point(data = filtered_waterpoints, aes(x = Longitude, y = Latitude, color = Waterpoint), size = 5) +  # Add filtered waterpoints
  scale_color_manual(values = grazing_colors, na.translate = FALSE) +
  theme_minimal(base_size = 15) +
  theme(
    legend.position = "none",
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12),
    axis.title.y = element_text(size = 14, color = "black"),  # Ensure y-axis title is visible
    axis.text.y = element_text(size = 12, color = "black")   # Ensure y-axis labels are visible
  ) +
  labs(x = "Longitude", y = "Latitude", fill = "Herbaceous cover", color = "Waterpoint")

grazingmap




# Ongava ____________________________________________________________________________________________

# Read and preprocess the raster data
vegetation <- rast("C:/0_Documents/10_ETH/Thesis/Analysis/Ongava/2022/vegetation_categories_raster.tif")
vegetation_band1 <- vegetation[[2]]  # herbaceous
vegetation_band2 <- vegetation[[1]]  # woody

# Read the Voronoi polygons
voronoi <- read_sf("C:/0_Documents/10_ETH/Thesis/Analysis/Ongava/2022/voronoi_polygons.shp")
voronoi <- st_transform(voronoi, crs = 4326)

# Mask the raster with Voronoi polygons
vegetation1_masked <- mask(vegetation_band1, voronoi)
vegetation2_masked <- mask(vegetation_band2, voronoi)

# Convert the masked raster to a data frame
vegetation_df1 <- as.data.frame(vegetation1_masked, xy = TRUE, na.rm = TRUE)
colnames(vegetation_df1) <- c("x", "y", "value")
vegetation_df2 <- as.data.frame(vegetation2_masked, xy = TRUE, na.rm = TRUE)
colnames(vegetation_df2) <- c("x", "y", "value")

# Read the waterpoints
waterpoints <- read.csv("C:/0_Documents/10_ETH/Thesis/Analysis/Ongava/waterholes_ongava.csv")
names(waterpoints) <- c("Waterpoint", "Longitude", "Latitude")

# # Initialize an empty list to store dataframes
# combined_df_list <- list()
# 
# df <- read.csv("Ongava/2021/results.csv")
# df$Year <- 2021
# combined_df_list[[1]] <- df
# df <- read.csv("Ongava/2022/results.csv")
# df$Year <- 2022
# combined_df_list[[2]] <- df
# 
# 
# # Combine all dataframes into a single dataframe
# combined_df <- do.call(rbind, combined_df_list)
# combined_df <- na.omit(combined_df)
# combined_df$grazing <- combined_df$Grazing_units.ha_herbaceous_vegetation_selective + combined_df$Grazing_units.ha_herbaceous_vegetation_bulk
# combined_df$browsing <- combined_df$Browsing_units.ha_woody_vegetation_selective + combined_df$Browsing_units.ha_woody_vegetation_bulk
# 
# write.csv(combined_df, "results3_Ongava.csv")

# Load the data
combined_df <- read.csv("results_Ongava.csv")
combined_df$Waterpoint <- as.factor(combined_df$Waterpoint)

# Calculate average browsing and grazing values for each Waterpoint
average_browsing <- combined_df %>%
  group_by(Waterpoint) %>%
  summarize(avg_browsing = mean(browsing, na.rm = TRUE))

average_grazing <- combined_df %>%
  group_by(Waterpoint) %>%
  summarize(avg_grazing = mean(grazing, na.rm = TRUE))

# Specify the Waterpoints for separate color palettes
special_sites <- c("ADC", "OGL", "OTC", "TIE", "MAR")

# Calculate ranks and assign colors for special sites
special_browsing <- average_browsing %>%
  filter(Waterpoint %in% special_sites) %>%
  mutate(rank = rank(avg_browsing),
         color = brewer.pal(9, "Blues")[3:7][rank])

special_grazing <- average_grazing %>%
  filter(Waterpoint %in% special_sites) %>%
  mutate(rank = rank(avg_grazing),
         color = brewer.pal(9, "Blues")[3:7][rank])

# Calculate ranks and assign colors for other sites
other_browsing <- average_browsing %>%
  filter(!Waterpoint %in% special_sites) %>%
  mutate(rank = rank(avg_browsing),
         color = brewer.pal(9, "Reds")[3:9][rank])

other_grazing <- average_grazing %>%
  filter(!Waterpoint %in% special_sites) %>%
  mutate(rank = rank(avg_grazing),
         color = brewer.pal(9, "Reds")[3:9][rank])

# Combine the data back together
average_browsing <- bind_rows(special_browsing, other_browsing)
average_grazing <- bind_rows(special_grazing, other_grazing)

# Check for NAs in color assignments
print(average_browsing)
print(average_grazing)

# Create named vectors for the colors to use in scale_color_manual
browsing_colors <- setNames(average_browsing$color, average_browsing$Waterpoint)
grazing_colors <- setNames(average_grazing$color, average_grazing$Waterpoint)

# Merge the colors back into the combined_df
combined_df <- combined_df %>%
  left_join(average_browsing %>% select(Waterpoint, browsing_color = color), by = "Waterpoint") %>%
  left_join(average_grazing %>% select(Waterpoint, grazing_color = color), by = "Waterpoint")

# Verify the merged data for completeness and ensure no NAs in color columns
print(combined_df)

# Handle any missing values in the dataset before plotting
combined_df <- combined_df %>%
  filter(!is.na(browsing) & !is.na(grazing))

# Plot browsing curve
browsingcurve2 <- ggplot(combined_df, aes(x = Year, y = browsing, group = Waterpoint, color = browsing_color)) +
  geom_point(size = 5) +  # Points at each year
  geom_line(size = 0.5) +  # Thin lines connecting points
  scale_color_identity() +  # Use the colors directly
  scale_y_log10(labels = scales::comma) +
  scale_x_continuous(breaks = unique(combined_df$Year), limits = c(min(combined_df$Year) - 0.5, max(combined_df$Year) + 0.5)) +  # Ensure only the actual values on x-axis with some padding
  labs(x = "Year", y = "Browsing pressure", color = "Waterpoint") +
  theme_minimal(base_size = 15) +
  theme(
    legend.position = "none",
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12)
  )

print(browsingcurve2)

# Plot grazing curve
grazingcurve2 <- ggplot(combined_df, aes(x = Year, y = grazing, group = Waterpoint, color = grazing_color)) +
  geom_point(size = 5) +  # Points at each year
  geom_line(size = 0.5) +  # Thin lines connecting points
  scale_color_identity() +  # Use the colors directly
  scale_y_log10(labels = scales::comma) +
  scale_x_continuous(breaks = unique(combined_df$Year), limits = c(min(combined_df$Year) - 0.5, max(combined_df$Year) + 0.5)) +  # Ensure only the actual values on x-axis with some padding
  labs(x = "Year", y = "Grazing pressure", color = "Waterpoint") +
  theme_minimal(base_size = 15) +
  theme(
    legend.position = "none",
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12)
  )

print(grazingcurve2)




### Tile 3: Browsing Map
# Filter waterpoints to include only those that appear in combined_df
filtered_waterpoints <- waterpoints %>% filter(Waterpoint %in% combined_df$Waterpoint)

# Calculate quantiles for the vegetation values
quantiles <- quantile(vegetation_df2$value, probs = seq(0, 1, by = 0.1))

# Define a color palette from white to dark green
colors <- c("white", "darkgreen")

browsingmap2 <- ggplot() +
  geom_raster(data = vegetation_df2, aes(x = x, y = y, fill = value)) +  # Add raster layer
  scale_fill_gradientn(colors = colors, values = scales::rescale(quantiles), na.value = "transparent", name = "Woody cover") +
  geom_sf(data = voronoi, fill = NA, color = "black", linewidth = 0.8) +  # Add Voronoi polygons
  geom_point(data = filtered_waterpoints, aes(x = Longitude, y = Latitude, color = Waterpoint), size = 5) +  # Add waterpoints
  scale_color_manual(values = browsing_colors) +
  theme_minimal(base_size = 15) +
  theme(
    legend.position = "none",
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12)
  ) +
  labs(x = "Longitude", y = "Latitude", color = "Waterpoint", fill = "Woody cover")

browsingmap2

### Tile 4: Grazing Map
# Filter waterpoints to include only those that appear in combined_df
filtered_waterpoints <- waterpoints %>% filter(Waterpoint %in% combined_df$Waterpoint)

# Calculate quantiles for the vegetation values
quantiles <- quantile(vegetation_df1$value, probs = seq(0, 1, by = 0.1))

grazingmap2 <- ggplot() +
  geom_raster(data = vegetation_df1, aes(x = x, y = y, fill = value)) +  # Add raster layer
  scale_fill_gradientn(colors = colors, values = scales::rescale(quantiles), na.value = "transparent", name = "Herbaceous cover") +
  geom_sf(data = voronoi, fill = NA, color = "black", linewidth = 0.8) +  # Add Voronoi polygons
  geom_point(data = filtered_waterpoints, aes(x = Longitude, y = Latitude, color = Waterpoint), size = 5) +  # Add waterpoints
  scale_color_manual(values = grazing_colors) +
  theme_minimal(base_size = 15) +
  theme(
    legend.position = "none",
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12)
  ) +
  labs(x = "Longitude", y = "Latitude", color = "Waterpoint", fill = "Herbaceous cover")

grazingmap2


# Species-specific results Onguma _______________________________________________________________
library(ggplot2)
library(viridis)  # For a perceptually uniform color palette
library(scales)   # For better scaling options
library(terra)    # For raster data
library(sf)       # For spatial data
library(dplyr)    # For data manipulation
library(RColorBrewer)

# Set working directory
setwd("C:/0_Documents/10_ETH/Thesis/Analysis/")

# Load the data
combined_df <- read.csv("results_Onguma.csv")
combined_df$Waterpoint <- as.factor(combined_df$Waterpoint)

# Read the waterpoints
waterpoints <- read.csv("C:/0_Documents/10_ETH/Thesis/Analysis/Onguma/waterholes_onguma.csv")
names(waterpoints) <- c("Name", "Waterpoint", "Longitude", "Latitude", "Area")

# Calculate average impala and elephant values for each Waterpoint
average_impala <- combined_df %>%
  group_by(Waterpoint) %>%
  summarize(avg_impala = mean(impala, na.rm = TRUE))

average_elephant <- combined_df %>%
  group_by(Waterpoint) %>%
  summarize(avg_elephant = mean(elephant, na.rm = TRUE))

# Specify the Waterpoints for separate color palettes
special_sites <- c("B10", "B11", "B14")

# Calculate ranks and assign colors for special sites
special_impala <- average_impala %>%
  filter(Waterpoint %in% special_sites) %>%
  mutate(rank = rank(avg_impala),
         color = brewer.pal(9, "Blues")[3:7][rank])

special_elephant <- average_elephant %>%
  filter(Waterpoint %in% special_sites) %>%
  mutate(rank = rank(avg_elephant),
         color = brewer.pal(9, "Blues")[3:7][rank])

# Calculate ranks and assign colors for other sites
other_impala <- average_impala %>%
  filter(!Waterpoint %in% special_sites) %>%
  mutate(rank = rank(avg_impala),
         color = brewer.pal(9, "Reds")[2:9][rank])

other_elephant <- average_elephant %>%
  filter(!Waterpoint %in% special_sites) %>%
  mutate(rank = rank(avg_elephant),
         color = brewer.pal(9, "Reds")[2:9][rank])

# Combine the data back together
average_impala <- bind_rows(special_impala, other_impala)
average_elephant <- bind_rows(special_elephant, other_elephant)

# Create named vectors for the colors to use in scale_color_manual
impala_colors <- setNames(average_impala$color, average_impala$Waterpoint)
elephant_colors <- setNames(average_elephant$color, average_elephant$Waterpoint)

# Merge the colors back into the combined_df
combined_df <- combined_df %>%
  left_join(average_impala %>% select(Waterpoint, impala_color = color), by = "Waterpoint") %>%
  left_join(average_elephant %>% select(Waterpoint, elephant_color = color), by = "Waterpoint")

# Handle any missing values in the dataset before plotting
combined_df <- combined_df %>%
  filter(!is.na(impala) & !is.na(elephant))

# Plot impala curve
impalacurve <- ggplot(combined_df, aes(x = Week, y = impala, group = Waterpoint, color = impala_color)) +
  geom_line(linewidth = 1) +
  scale_color_identity() +  # Use the colors directly
  scale_y_log10(labels = scales::comma) +
  labs(x = "Week", y = "Detected impala per day", color = "Waterpoint") +
  theme_minimal(base_size = 15) +
  theme(
    legend.position = "none",
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12)
  )

print(impalacurve)

# Plot elephant curve
elephantcurve <- ggplot(combined_df, aes(x = Week, y = elephant, group = Waterpoint, color = elephant_color)) +
  geom_line(linewidth = 1) +
  scale_color_identity() +  # Use the colors directly
  scale_y_log10(labels = scales::comma) +
  labs(x = "Week", y = "Detected elephants per day", color = "Waterpoint") +
  theme_minimal(base_size = 15) +
  theme(
    legend.position = "none",
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12)
  )

print(elephantcurve)
