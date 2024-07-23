# Spatially and Temporally Explicit Herbivory Monitoring in Semi-arid Savannas
## Abstract
Effective management of protected areas is critical to mitigating the global biodiversity crisis. In water-limited trophic savannas, altered herbivory regimes lead to ecosystem degradation. Consequently, there is a need for monitoring tools that can track the impact of herbivores on vegetation in space and time to inform adaptive management and restoration efforts. Here, we present the Spatially and Temporally Explicit Herbivory Monitoring (STEHM) tool, an innovative methodology that couples an object detection model (YOLO v10) applied to waterpoint-based camera trap images to detect herbivores with a deep-learning model to estimate vegetation category fractions from high-resolution satellite imagery (Sentinel-2). This tool allows for monitoring of herbivory across finer spatial and temporal scales than previously possible, enabling assessments at resolutions down to a few square kilometers depending on the waterpoint density, and intervals as frequent as weekly. STEHM facilitates adaptive herbivory management by tying data collection on herbivore dynamics to surface water, a primary determinant of large herbivore distribution in semi-arid environments, and provides an approach to disentangle the ecological drivers influencing plant-herbivore interactions. This refined monitoring capability can enhance conservation strategies and promote the restoration of savannas, by providing detailed, real-time data on herbivore densities and vegetation changes, allowing for more targeted and adaptive management interventions.

## Content
- R Script - Time window analysis, water dependency analysis.R: Time window analysis.
- R Script - RF models on aerial images, training data generation for DL model.R: Vegetation model 1.
- R Script - Result plots.R: Visualization script.
- vegetation_model.py: Training script for vegetation model 2.
- YOLO_prediction.py: Inference script for YOLOv10 model.
- herbivory_monitoring_tool_v1.0.py: GUI for herbivory monitoring.
- camera_image_classifier_v1.0.py: GUI for camera trap image classification.

![methods](https://github.com/user-attachments/assets/4fb181d0-d730-4cb5-9c33-75a7adbdbcee)
