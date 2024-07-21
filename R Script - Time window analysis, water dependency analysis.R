### ------------------------------------------------------------------------------------------------------------------
### Task 1: Plotting RAIs with different time windows ------------------------------------
### ------------------------------------------------------------------------------------------------------------------

rm(list = ls())
setwd("C:/0_Documents/10_ETH/Thesis/Test run week 36")
list.files(pattern = "RAI_")

min10 <- read.csv("week 36_RAI_10min.csv")
min30 <- read.csv("week 36_RAI_30min.csv")
min5 <- read.csv("week 36_RAI_5min.csv")
sites <- min5$Site

library(dplyr)
min10 <- select(min10, starts_with(match = c("RAI"), ignore.case = T))
min10$window <- 10
min30 <- select(min30, starts_with(match = c("RAI"), ignore.case = T))
min30$window <- 30
min5 <- select(min5, starts_with(match = c("RAI"), ignore.case = T))
min5$window <- 5

data <- rbind(min5, min10, min30)
windows <- data$window
data <- data[-ncol(data)]
library(stringr)
plotdata <- data.frame("Species" = "Test",
                       "Site" = "Test",
                       "Window" = "Test",
                       "RAI" = 0)

for(i in 1:ncol(data)){
  species <- substring(colnames(data)[i],5)
  species <- str_to_title(species)
  flattened <- data.frame("Species" = species,
                          "Site" = sites,
                          "Window" = windows,
                          "RAI" = data[,i])
  plotdata <- rbind(plotdata, flattened)
  }
plotdata <- plotdata[-1,]

plotdata$Window <- factor(plotdata$Window, levels = c("5", "10", "30"))

rhino <- filter(plotdata, Species == "Rhino" & Window == "30")
sum(rhino$RAI) # 37. if we consider group size of 1.5, 30 would work if the drink once a day (overestimate?)

library(gridExtra)
plots <- list()

for (i in 1:length(unique(plotdata$Site))) {
  plotdata1 <- filter(plotdata, Site == unique(plotdata$Site)[i])
  p <- ggplot(plotdata1, aes(fill = Window, y = RAI, x = Species))+
    geom_bar(position = "dodge", stat = "identity")+
    theme_bw()+
    theme(legend.position = "none")
    ggtitle(paste0("Site ", unique(plotdata$Species)[i]))
  plots[[i]] <- p
}

# Arrange plots in a grid layout
grid.arrange(grobs = plots, nrow = 3)


# Calculating slopes of standardized linear regressions (how many detection events are lost/gained for each minute added to the time window?)
output <- matrix(data = NA, nrow = length(unique(plotdata$Species)), ncol = length(unique(plotdata$Site)))

for (i in 1:length(unique(plotdata$Species))) {
  plotdata1 <- filter(plotdata, Species == unique(plotdata$Species)[i])
  for(j in 1:length(unique(plotdata$Site))) {
    plotdata2 <- filter(plotdata1, Site == unique(plotdata$Site)[j])
    plotdata2$RAI <- plotdata2$RAI/mean(plotdata2$RAI)
    if(sum(plotdata2$RAI !=0) & !is.na(plotdata2$RAI[1])){
      model <- lm(plotdata2$RAI ~ as.numeric(plotdata2$Window))
      output[i,j] <- coef(model)[2] 
    }
  }
}


# Convert matrix to dataframe and transpose it
data_df <- as.data.frame((output))
data_df$Species <- unique(plotdata$Species)
rownames(data_df) <- factor(unique(plotdata$Site))

data <- data.frame("Value" = 0, "Species"=0)
for(i in 1:11){
  spec <- as.vector(t(data_df[i,-ncol(data_df)]))
  spec <- data.frame("Value" = spec)
  spec$Species <- data_df[i, ncol(data_df)]
  data <- rbind(data, spec)
}
data <- data[-1,]

# Create scatterplot
ggplot(data, aes(x = Species, y = Value, color = Species)) +
  geom_point(position = position_jitter(width = 0.2), alpha = 0.6) +
  scale_color_brewer(palette = "Set1") +
  theme_minimal() +
  labs(x = "Species", y = "Site")


### ------------------------------------------------------------------------------------------------------------------
### Task 2: RAI rarefaction curves ------------------------------------
### ------------------------------------------------------------------------------------------------------------------

rm(list = ls())
setwd("C:/0_Documents/10_ETH/Thesis/Test run week 36")
df <- read.csv("week 36_full annotations.csv")

# Load required libraries
library(dplyr)
library(lubridate)
library(ggplot2)
library(tidyr)

# Step 1: Extract site information
df$Site <- sub('.*\\\\[^\\\\]+\\\\[^\\\\]+\\\\([^\\\\]+)\\\\.*', '\\1', df$Name)

# Reshape the data to long format
df_long <- df %>%
  pivot_longer(cols = -c(Name, Date_Time, Site), names_to = "Species", values_to = "Probability") %>%
  filter(Probability != 0)  # Filter out rows where Probability is 0

# Convert Date_Time to datetime object
df_long$Date_Time <- as.POSIXct(df_long$Date_Time, format = "%Y:%m:%d %H:%M:%S")

# Arrange the data by Site and Date_Time
df_long <- df_long %>%
  arrange(Site, Date_Time)

# Calculate time differences within each Site and Species combination
df_long <- df_long %>%
  group_by(Site, Species) %>%
  mutate(Time_Diff = difftime(Date_Time, lag(Date_Time), units = "mins"))


df_long$Time_Diff <- replace(df_long$Time_Diff, is.na(df_long$Time_Diff), 0)


# Create a sequence of time windows from 1 to 30
time_windows <- 1:60

# Create an empty dataframe to store results
test <- data.frame(Site = character(), Species = character())

df_long$Time_Diff <- as.numeric(df_long$Time_Diff)

# Loop through each time window
for (window in time_windows) {
  # Count the number of events within each time window
  result <- df_long %>%
    group_by(Site, Species) %>%
    summarise(Events = sum(Time_Diff >= window, na.rm = TRUE))
  
  # Append the result to the test dataframe
  test <- merge(test, result, by = c("Site", "Species"), all = TRUE)
}

# Rename columns
colnames(test)[-c(1,2)] <- paste0("Events_", time_windows)

# Fill NA values with 0
test[is.na(test)] <- 0

library(ggplot2)
library(dplyr)

# Reshape the data for plotting and convert Time_Window to numeric
normalized_data_long <- pivot_longer(test, starts_with("Events"), names_to = "Time_Window", values_to = "Normalized_Events") %>%
  mutate(Time_Window = as.numeric(gsub("Events_", "", Time_Window)))

# Group the data by Species and Site, and normalize the y-axis values within each group
normalized_data_long <- normalized_data_long %>%
  group_by(Species, Site) %>%
  mutate(Normalized_Events = (Normalized_Events - min(Normalized_Events, na.rm = TRUE)) / 
           (max(Normalized_Events, na.rm = TRUE) - min(Normalized_Events, na.rm = TRUE)))

# Plot
ggplot(normalized_data_long, aes(x = Time_Window, y = Normalized_Events, color = Site)) +
  geom_line() +
  facet_wrap(~Species) +
  labs(title = "Normalized Event Counts by Time Window",
       x = "Time Window (minutes)",
       y = "Normalized Event Count (0-1)") +
  theme_minimal()


# Calibrate on rhino
rhino <- filter(normalized_data_long, Species == "rhino")
absrhino <- filter(df, rhino !=0) # number of images labeled rhino
absrhino <- absrhino[,c("rhino", "Site")]
absrhino <- absrhino %>%
  group_by(Site) %>%
  summarize(Count = n())
(total <- 50/1.5/2*9) #number of rhinos/average group size/drinking frequency*days
# assumption: detectability = 1
# 117 detections over 7 days
test <- filter(rhino, Time_Window == 27)
test <- merge(test, absrhino, by = c("Site"), all = TRUE)
test$eventsfor5 <- test$Normalized_Events*test$Count
sum(test$eventsfor5) # 27 minutes closest



# Calibrate on elephant
elephant <- filter(normalized_data_long, Species == "elephant")
abselephant <- filter(df, elephant !=0) # number of images labeled elephant
abselephant <- abselephant[,c("elephant", "Site")]
abselephant <- abselephant %>%
  group_by(Site) %>%
  summarize(Count = n())
(total <- 25/1.5/1*7) #25 individuals, 2 average group size, drinking frequency 1
# 117 detections over 7 days
test <- filter(elephant, Time_Window == 28)
test <- merge(test, abselephant, by = c("Site"), all = TRUE)
test$eventsfor5 <- test$Normalized_Events*test$Count
sum(test$eventsfor5) # 28 minutes closest

# Anova, aggregation of sites

means <- aggregate(normalized_data_long$Normalized_Events, by = list(normalized_data_long$Species, normalized_data_long$Time_Window), FUN = mean, na.rm = T)
names(means) <- c("Species", "Window", "Events")
means$Species <- as.factor(means$Species)

library(stringr)

ggplot(means, aes(x = Window, y = Events, color = Species)) +
  geom_line() +
  labs(x = "Time window [min]", y = "Proportion of distinct events", color = "Species") +
  theme_minimal() +
  scale_color_discrete(labels = function(x) str_to_title(x))

# Fit the ANOVA model
model <- aov(Events ~ Species, data = means)

# Perform ANOVA
anova_result <- summary(model)

# Print the ANOVA table
print(anova_result)

#Df Sum Sq Mean Sq F value Pr(>F)
#Species       8  0.194 0.02427   0.555  0.814
#Residuals   261 11.420 0.04375 
# No statistically significant difference

### We are using a time window of 25 minutes


### ------------------------------------------------------------------------------------------------------------------
### Task 3: Proximity analysis to water using aerial game count 2023 ------------------------------------
### ------------------------------------------------------------------------------------------------------------------

library(geosphere)  # for distance calculation
library(ggplot2)
library(dplyr)
library(tidyr)

# Function to calculate distance between two points
get_distance <- function(lat1, lon1, lat2, lon2) {
  distHaversine(c(lon1, lat1), c(lon2, lat2))  # Haversine formula
}

# Function to calculate minimum distance to waterhole for each observation
get_min_distance <- function(lat, lon) {
  distances <- apply(water[, c("Latitude", "Longitude")], 1, 
                     function(x) get_distance(lat, lon, x[1], x[2]))
  min(distances)
}

# Read data
setwd("C:/0_Documents/10_ETH/Thesis/R")
count <- read.csv("game_count_2023.csv")
count[which(count$Species == "Gemsbok"),]$Species <- "Oryx"
water <- read.csv("waterholes_onguma.csv")

# Calculate minimum distances for all observations
count$MinDistance <- mapply(get_min_distance, count$Latitude, count$Longitude)

# Create a density plot (optional, just for visualization)
density_plot <- ggplot(count, aes(x = MinDistance, color = Species)) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribution of distances to nearest waterhole for each species",
       x = "Distance to nearest waterhole (m)", y = "Density") +
  scale_color_discrete(name = "Species") +
  scale_y_continuous(limits = c(0, 0.001)) +
  theme_minimal()

# Print the density plot
print(density_plot)

# Function to calculate the area under the density curve
calculate_areas <- function(species_data) {
  dens <- density(species_data$MinDistance, from = 0, to = max(species_data$MinDistance))
  left_area <- sum(dens$y[dens$x <= 2000]) * (dens$x[2] - dens$x[1])
  right_area <- sum(dens$y[dens$x > 2000]) * (dens$x[2] - dens$x[1])
  return(c(left_area, right_area))
}

# Calculate the areas for each species
areas_df <- count %>%
  nest_by(Species) %>%
  mutate(Areas = list(calculate_areas(data))) %>%
  unnest_wider(Areas, names_sep = "") %>%
  rename(LeftArea = Areas1, RightArea = Areas2) %>%
  ungroup()

# Print the resulting dataframe
print(areas_df)

areas_df_filtered <- areas_df[-8,]

# Perform the Wilcoxon signed-rank test to compare the areas
wilcox_test_result <- wilcox.test(areas_df_filtered$LeftArea, areas_df_filtered$RightArea, paired = TRUE)

# Print the Wilcoxon test results
print(wilcox_test_result)

### Relative distance to water
library(sf)
library(terra)
library(ggplot2)
library(dplyr)
library(tidyr)

# Load Onguma shapefile
Onguma <- read_sf("C:/0_Documents/10_ETH/Thesis/Running the app/Onguma_sf/Onguma EPSG 4326 without Ruimte.shp")[1]

# Convert Onguma to terra format
Onguma_terra <- vect(Onguma)

# Function to calculate minimum distance to waterhole for each point
get_min_distance <- function(point) {
  min_distance <- min(apply(water[, c("Latitude", "Longitude")], 1, 
                            function(x) get_distance(point[2], point[1], x[1], x[2])))
  return(min_distance)
}

# Calculate minimum distances for all points in the shapefile
dist <- apply(crds(Onguma_terra), 1, get_min_distance)
dist <- as.data.frame(dist)

# Create a density plot for Onguma distances
distance_plot <- ggplot(dist, aes(x = dist)) +
  geom_density(fill = "blue", alpha = 0.5) +
  labs(title = "Distribution of distances to nearest waterhole in Onguma",
       x = "Distance to nearest waterhole (m)", y = "Density") +
  theme_minimal() +
  scale_x_continuous(limits = c(0, max(dist$dist)))  # Setting x-axis limits from 0 to maximum value in the data

# Print the plot
print(distance_plot)

# Calculate density for each species and normalize
calculate_density <- function(species_data, x_values) {
  dens <- density(species_data$MinDistance, from = 0, to = max(species_data$MinDistance), n = 1000)
  approx(dens$x, dens$y, xout = x_values)$y
}

# Prepare the x values for density estimation
x_values <- seq(0, max(dist$dist), length.out = 1000)

# Calculate the density values for all species and for the overall data
overall_density <- density(dist$dist, from = 0, to = max(dist$dist), n = 1000)
overall_y_values <- approx(overall_density$x, overall_density$y, xout = x_values)$y

# Initialize the density_df dataframe
density_df <- data.frame(x = x_values, y_values = overall_y_values)

# Extract unique species names
species_names <- unique(count$Species)

# Calculate densities for each species and add to the density_df dataframe
for (i in seq_along(species_names)) {
  species_data <- count %>% filter(Species == species_names[i])
  density_df[[paste0("y_", i)]] <- calculate_density(species_data, x_values)
}

# Replace NAs with 0
density_df[is.na(density_df)] <- 0

# Normalize densities by the overall density
for (i in seq_along(species_names)) {
  density_df[[paste0("y_norm_", i)]] <- density_df[[paste0("y_", i)]] / density_df$y_values
}

# Replace NAs with 0 again after normalization
density_df[is.na(density_df)] <- 0

# Calculate the area under the density curve
calculate_areas <- function(y_values) {
  left_area <- sum(y_values[x_values <= 2000]) * (x_values[2] - x_values[1])
  right_area <- sum(y_values[x_values > 2000]) * (x_values[2] - x_values[1])
  return(c(left_area, right_area))
}

# Calculate areas for each species
areas <- data.frame(Species = species_names)
for (i in seq_along(species_names)) {
  areas[i, c("LeftArea", "RightArea")] <- calculate_areas(density_df[[paste0("y_norm_", i)]])
}

# Perform the Wilcoxon signed-rank test
wilcox_test_result <- wilcox.test(areas$LeftArea, areas$RightArea, paired = TRUE)

# Print the Wilcoxon test results
print(wilcox_test_result)

# Plotting
plot <- ggplot(density_df, aes(x = x)) +
  geom_line(aes(y = y_norm_1 / sum(y_norm_1), color = species_names[1]), size = 0.5) +
  geom_line(aes(y = y_norm_2 / sum(y_norm_2), color = species_names[2]), size = 0.5) +
  geom_line(aes(y = y_norm_3 / sum(y_norm_3), color = species_names[3]), size = 0.5) +
  geom_line(aes(y = y_norm_4 / sum(y_norm_4), color = species_names[4]), size = 0.5) +
  geom_line(aes(y = y_norm_5 / sum(y_norm_5), color = species_names[5]), size = 0.5) +
  geom_line(aes(y = y_norm_6 / sum(y_norm_6), color = species_names[6]), size = 0.5) +
  geom_line(aes(y = y_norm_7 / sum(y_norm_7), color = species_names[7]), size = 0.5) +
  geom_line(aes(y = y_norm_9 / sum(y_norm_9), color = species_names[9]), size = 0.5) +
  geom_line(aes(y = y_norm_10 / sum(y_norm_10), color = species_names[10]), size = 0.5) +
  labs(title = "Herbivore distributions relative to water from aerial count",
       x = "Distance to water [m]",
       y = "Normalized density") +
  scale_color_manual(name = "Species", 
                     values = c("red", "blue", "green", "purple", "orange", 
                                "cyan", "magenta", "darkgreen", 
                                "brown", "black"),
                     labels = species_names[-8]) +
  theme_minimal()

# Print the plot
print(plot)


areas$left <- areas$LeftArea/(areas$LeftArea+areas$RightArea)
areas$right <- areas$RightArea/(areas$LeftArea+areas$RightArea)

plot(areas$right ~ areas$left)

areas_filtered <- areas[-8,]

scatter_plot <- ggplot(areas_filtered, aes(x = left, y = right, label = Species)) +
  geom_point(size = 2, color = "black") +
  geom_text_repel() +
  labs(x = "Proportion of animals detected < 2000 m from water",
       y = "Proportion of animals detected > 2000 m from water") +
  theme_minimal()

# Print the scatter plot
print(scatter_plot)
