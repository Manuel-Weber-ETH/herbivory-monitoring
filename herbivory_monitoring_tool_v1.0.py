########################################################################################################################################################################################
### Spatially and Temporally Explicit Herbivory Monitoring in Water-limited Savannas
### Desktop Application
### Manuel Weber
### https://github.com/Manuel-Weber-ETH/cameratraps.git
########################################################################################################################################################################################

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk as NavigationToolbar2
import os
import shutil
import glob
import sys
from datetime import date, timedelta, datetime
import contextily as ctx
import numpy as np
import matplotlib.colors as mcolors
import rasterio
from PIL import Image, ExifTags
from rasterio.mask import geometry_mask, mask
import shapely
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from matplotlib.patches import Patch
from tensorflow.keras.models import load_model


output_base_path = ""
results_df = pd.DataFrame()
checkbox_vars = {}

try:
    script_dir = sys._MEIPASS  # Path where PyInstaller unpacks files
except AttributeError:
    script_dir = os.path.abspath(".")  # If not bundled, use the current directory

class HerbivoryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Herbivory Monitoring Tool")
        self.root.state('zoomed')  # Fullscreen mode
        self.site_names = []
        self.plot_vars = {}  # Dictionary to hold variables for checkboxes
        self.checkbox_vars = {}  # Dictionary to hold species checkbox variables
        self.setup_ui()

    def setup_ui(self):
        # Main layout frames
        control_frame = ttk.Frame(self.root, width=300)  # Set a fixed width for the control frame
        control_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        visualization_frame = ttk.Frame(self.root)
        visualization_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Step 1: Initialization frame
        init_frame = ttk.Labelframe(control_frame, text="Step 1: Initialization", padding=10)
        init_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5, columnspan=3)

        ttk.Button(init_frame, text="Select output directory", command=self.browse_and_create_output_directory).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(init_frame, text="Import boundaries", command=self.import_boundaries).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(init_frame, text="Import waterpoints", command=self.load_waterpoints_csv).grid(row=0, column=2, padx=5, pady=5)
        self.show_satellite_var = tk.BooleanVar(value=False)
        satellite_checkbox = ttk.Checkbutton(init_frame, text="Satellite backgroung", variable=self.show_satellite_var, command=self.update_voronoi_plot_based_on_checkboxes)
        satellite_checkbox.grid(row=0, column=3, padx=5, pady=5, sticky="w")


        # Step 2: Camera trap images frame
        cam_trap_frame = ttk.Labelframe(control_frame, text="Step 2: Camera trap images", padding=10)
        cam_trap_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5, columnspan=3)

        # Combine label and entry into one cell
        conf_frame = ttk.Frame(cam_trap_frame)
        conf_frame.grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(conf_frame, text="Confidence threshold:").pack(side=tk.LEFT)
        self.prob_thresh_entry = ttk.Entry(conf_frame, width=10)
        self.prob_thresh_entry.pack(side=tk.LEFT)
        self.prob_thresh_entry.insert(0, "0.75")  # Default value of 0.75

        ttk.Button(cam_trap_frame, text="Compute RAIs", command=self.compute_rais).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(cam_trap_frame, text="Copy images to folders for selected species", command=self.organize_images).grid(row=0, column=2, padx=5, pady=5)

        # Step 3: Vegetation frame
        veg_frame = ttk.Labelframe(control_frame, text="Step 3: Vegetation", padding=10)
        veg_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5, columnspan=3)

        # Create a single button to load raster and run vegetation model
        ttk.Button(veg_frame, text="Run model", command=self.load_and_run_vegetation_model).grid(row=0, column=1, columnspan=1, padx=5, pady=5)

        ttk.Button(veg_frame, text="Download GEE script", command=self.download_text_file).grid(row=0, column=0, padx=5, pady=5)

        progress_frame = ttk.Frame(control_frame)
        progress_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5, columnspan=3)
        self.progress_text = tk.Text(progress_frame, height=3, width=40)
        self.progress_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=visualization_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


        toolbar_frame = ttk.Frame(visualization_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar = NavigationToolbar2(self.canvas, toolbar_frame)
        toolbar.update()

        # Data display frame
        self.data_frame = ttk.Frame(control_frame)
        self.data_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')

    def browse_and_create_output_directory(self):
        global output_base_path
        base_dir = filedialog.askdirectory()
        if not base_dir:
            messagebox.showerror("Error", "No directory selected.")
            return
        
        today = date.today()
        session_dir_name = f"run_{today}"
        full_path = os.path.join(base_dir, session_dir_name)
        
        try:
            os.makedirs(full_path, exist_ok=True)
            output_base_path = full_path
            messagebox.showinfo("Success", f"Output will be saved in: {output_base_path}")
        except PermissionError:
            messagebox.showerror("Permission Denied", "You do not have permission to create a directory in the selected location.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create directory: {e}")

    def import_boundaries(self):
        file_path = filedialog.askopenfilename(filetypes=[("Shapefiles", "*.shp"), ("KML files", "*.kml")])
        if file_path:
            if file_path.endswith(".shp"):
                self.boundaries_gdf = gpd.read_file(file_path)
            elif file_path.endswith(".kml"):
                self.boundaries_gdf = gpd.read_file(file_path, driver='KML')

            # Plot the boundaries outline as a black line
            self.ax.clear()
            self.boundaries_gdf.boundary.plot(ax=self.ax, edgecolor='black')

            self.canvas.draw()

    def load_waterpoints_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.waterpoints_df = pd.read_csv(file_path)
            self.waterpoints_df['Site'] = self.waterpoints_df['Site'].astype(str)  # Ensure Site is treated as string
            self.site_names = self.waterpoints_df['Site'].tolist()

            if hasattr(self, 'site_names') and self.site_names:
                self.waterpoints_gdf = gpd.GeoDataFrame(
                    self.waterpoints_df, 
                    geometry=gpd.points_from_xy(self.waterpoints_df.Longitude, self.waterpoints_df.Latitude)
                )
                # Ensure the waterpoints GeoDataFrame has a CRS
                if self.waterpoints_gdf.crs is None:
                    self.waterpoints_gdf.set_crs(epsg=4326, inplace=True)

                self.waterpoints_gdf.plot(ax=self.ax, color='black', markersize=5)
                for x, y, label in zip(self.waterpoints_gdf.geometry.x, self.waterpoints_gdf.geometry.y, self.site_names):
                    self.ax.text(x, y, label, fontsize=8, color='black')

                # Voronoi tessellation
                self.voronoi_polygons_gdf = self.create_voronoi_polygons_old(self.waterpoints_gdf)

                # Plot Voronoi polygons
                self.voronoi_polygons_gdf.boundary.plot(ax=self.ax, edgecolor='black')

                self.canvas.draw()

                # Update data display with area information
                self.update_data_display(["Sites"] + self.site_names)
                self.update_voronoi_area_row()  # Ensure "Area (ha)" row appears immediately

                self.progress_text.insert(tk.END, f"Loaded waterpoints: {', '.join(self.site_names)}\n")


    def update_data_table_with_vegetation(self):
        if not hasattr(self, 'vegetation_raster_array'):
            messagebox.showerror("Error", "Vegetation raster not loaded.")
            return

        veg_categories = ['Woody fraction', 'Herbaceous fraction', 'Bare fraction']
        veg_data = {category: [] for category in veg_categories}

        for idx, row in self.voronoi_polygons_gdf.iterrows():
            mask = geometry_mask([row.geometry], transform=self.vegetation_raster.transform, invert=True, out_shape=self.vegetation_raster_array.shape[1:])
            for i, category in enumerate(veg_categories):
                category_band = self.vegetation_raster_array[i, :, :]
                veg_fraction = category_band[mask].mean()
                veg_data[category].append(veg_fraction)

        for category in veg_categories:
            self.add_vegetation_row(category, veg_data[category])


    def create_voronoi_polygons_old(self, waterpoints_gdf):
        if waterpoints_gdf.crs is None:
            waterpoints_gdf.set_crs(epsg=4326, inplace=True)

        if self.boundaries_gdf.crs is None:
            self.boundaries_gdf.set_crs(epsg=4326, inplace=True)

        coords = np.array(waterpoints_gdf[['Longitude', 'Latitude']])
        polygon = self.boundaries_gdf.geometry.iloc[0]

        bound = polygon.buffer(100).envelope.boundary

        boundarypoints = [bound.interpolate(distance=d) for d in range(0, int(np.ceil(bound.length)), 100)]
        boundarycoords = np.array([[p.x, p.y] for p in boundarypoints])

        all_coords = np.concatenate((boundarycoords, coords))

        vor = Voronoi(points=all_coords)

        lines = [shapely.geometry.LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]

        polys = shapely.ops.polygonize(lines)

        voronois = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys), crs=waterpoints_gdf.crs)

        if voronois.crs != self.boundaries_gdf.crs:
            voronois = voronois.to_crs(self.boundaries_gdf.crs)

        voronois = gpd.clip(voronois, self.boundaries_gdf)

        # Assign site names to the polygons
        coords = waterpoints_gdf.geometry.apply(lambda p: (p.x, p.y)).tolist()
        site_names = []
        for poly in voronois.geometry:
            centroid = poly.centroid
            nearest_site = min(coords, key=lambda p: centroid.distance(shapely.geometry.Point(p)))
            site_name = waterpoints_gdf[waterpoints_gdf.geometry == shapely.geometry.Point(nearest_site)]['Site'].values[0]
            site_names.append(site_name)
        voronois['Site'] = site_names

        voronois_3857 = voronois.to_crs(epsg=3857)

        voronois['Area (ha)'] = voronois_3857.geometry.area / 10000

        output_voronoi_path = os.path.join(output_base_path, 'voronoi_polygons.shp')
        voronois.to_file(output_voronoi_path)

        return voronois

    def update_voronoi_area_row(self):
        if hasattr(self, 'voronoi_polygons_gdf'):
            areas = self.voronoi_polygons_gdf['Area (ha)'].tolist()
            row_data = ['Area [ha]'] + [f"{area:.2f}" for area in areas]
            row_index = 1  # Assuming this is the first row after the header
            
            # Remove any existing checkboxes in this row to prevent duplicates
            for widget in self.data_frame.grid_slaves(row=row_index):
                widget.destroy()
            
            self.add_data_row(row_data, row_index)
            
            # Add the checkbox only if it doesn't already exist
            if 'Area [ha]' not in self.plot_vars:
                plot_var = tk.BooleanVar(value=True)
                self.plot_vars['Area [ha]'] = plot_var
                plot_checkbox = ttk.Checkbutton(self.data_frame, variable=plot_var, command=self.update_voronoi_plot)
                plot_checkbox.grid(row=row_index, column=len(self.site_names) + 1, sticky='nsew')



    def clip_raster_to_polygons(self, band):
        masked_band = np.zeros_like(band, dtype=float)
        for idx, row in self.voronoi_polygons_gdf.iterrows():
            mask = geometry_mask([row.geometry], transform=self.vegetation_raster.transform, invert=True, out_shape=band.shape)
            masked_band[mask] = band[mask]
        return masked_band

    def get_raster_extent(self):
        transform = self.vegetation_raster.transform
        width = self.vegetation_raster.width
        height = self.vegetation_raster.height
        return (transform[2], transform[2] + width * transform[0], transform[5] + height * transform[4], transform[5])

    def get_polygon_color(self, site, class_name):
        df = None
        if hasattr(self, 'rai_df') and class_name in self.rai_df.columns:
            df = self.rai_df
        elif hasattr(self, 'relative_feeding_units_df') and class_name in self.relative_feeding_units_df.columns:
            df = self.relative_feeding_units_df

        if df is not None:
            site_data = df[df['Waterpoint'] == site]
            if not site_data.empty:
                max_value = df[class_name].max()
                value = site_data[class_name].values[0]
                if not pd.isna(value) and max_value > 0:  # Ensure value and max_value are valid
                    normalized_value = value / max_value
                    rgba_color = plt.cm.Reds(normalized_value)
                    return rgba_color[:3] + (0.7,)
        return None  # No color if no values

    def update_data_display(self, columns):
        # Clear previous data
        for widget in self.data_frame.winfo_children():
            widget.destroy()

        # Create headers
        for col_index, col in enumerate(columns):
            if col_index == 0:
                width = 25  # Increased width for the first column
            else:
                width = 7
            label = ttk.Label(self.data_frame, text=col, borderwidth=1, relief="solid", width=width)
            label.grid(row=0, column=col_index, sticky='nsew')

        self.data_frame.update_idletasks()

    def update_plot(self):
        self.update_voronoi_plot_based_on_checkboxes()

    def browse_file(self, entry_field, message, file_types):
        messagebox.showinfo("Information", message)
        filepath = filedialog.askopenfilename(filetypes=file_types)
        if filepath:
            entry_field.delete(0, tk.END)
            entry_field.insert(0, filepath)
        else:
            messagebox.showerror("Error", "No file selected.")

    def browse_file_raster(self):
        messagebox.showinfo("Information", "Please provide a Sentinel-2 1C raster that corresponds to the period during which the images were taken. Cloud coverage can degrade algorithm performance.")
        filepath = filedialog.askopenfilename(filetypes=[("TIF files", "*.tif")])
        if filepath:
            self.progress_text.delete("1.0", tk.END)  # Clear the text widget
            self.progress_text.insert(tk.END, filepath)  # Insert the selected file path
        else:
            messagebox.showerror("Error", "No file selected.")

    def compute_relative_feeding_units(self):
        # Call the output function to ensure calculations and CSV saving are done once
        self.output()

        # Load the results CSV to update the DataFrame
        relative_feeding_units_file = f"{output_base_path}/results.csv"
        self.relative_feeding_units_df = pd.read_csv(relative_feeding_units_file)

        self.progress_text.insert(tk.END, f"Relative feeding units loaded from: {relative_feeding_units_file}\n")

    def save_feeding_units_plot(self):
        sorted_results = self.relative_feeding_units_df.sort_values(by='Waterpoint')

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {
            'Selective Grazing': '#2ca25f',
            'Bulk Grazing': '#99d8c9',
            'Selective Browsing': '#8c6c4f',
            'Bulk Browsing': '#c9ae91'
        }
        bar_width = 0.6
        y_positions = np.arange(len(sorted_results))

        for i, (index, row) in enumerate(sorted_results.iterrows()):
            ax.barh(i, row['Grazing_units/ha_herbaceous_vegetation_selective'], color=colors['Selective Grazing'], edgecolor='white', height=bar_width)
            ax.barh(i, row['Grazing_units/ha_herbaceous_vegetation_bulk'], left=row['Grazing_units/ha_herbaceous_vegetation_selective'], color=colors['Bulk Grazing'], edgecolor='white', height=bar_width)
            ax.barh(i, -row['Browsing_units/ha_woody_vegetation_selective'], color=colors['Selective Browsing'], edgecolor='white', height=bar_width)
            ax.barh(i, -row['Browsing_units/ha_woody_vegetation_bulk'], left=-row['Browsing_units/ha_woody_vegetation_selective'], color=colors['Bulk Browsing'], edgecolor='white', height=bar_width)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(sorted_results['Waterpoint'])
        ax.set_xlabel('Feeding units per camera trap day and ha of woody or herbaceous vegetation')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{abs(x):.2f}"))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.subplots_adjust(left=0.1, right=0.98, top=0.85, bottom=0.15)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

        legend_elements = [
            Patch(facecolor=colors['Bulk Browsing'], edgecolor='none', label='Bulk Browsing'),
            Patch(facecolor=colors['Selective Browsing'], edgecolor='none', label='Selective Browsing'),
            Patch(facecolor=colors['Selective Grazing'], edgecolor='none', label='Selective Grazing'),
            Patch(facecolor=colors['Bulk Grazing'], edgecolor='none', label='Bulk Grazing')
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize='small', handlelength=1, handleheight=1, markerscale=0.8)

        plot_path = os.path.join(output_base_path, 'relative_feeding_units_per_area_of_biomass.png')
        plt.savefig(plot_path)
        plt.close(fig)

    def load_and_run_vegetation_model(self):
        global output_base_path
        output_raster_path = f"{output_base_path}/vegetation_categories_raster.tif"

        if os.path.exists(output_raster_path):
            messagebox.showinfo("Information", "Vegetation categories raster already exists. Loading the existing raster.")
            self.load_existing_vegetation_raster(output_raster_path)
        else:
            self.browse_file_raster()
            self.run_vegetation_category_model()

        # Compute relative feeding units and add them to the table
        self.compute_relative_feeding_units()
        self.save_feeding_units_plot()

        # Ensure the table is updated only once after all calculations are done
        self.add_relative_feeding_units_to_table()

    def load_existing_vegetation_raster(self, raster_path):
        try:
            self.vegetation_raster = rasterio.open(raster_path)
            self.vegetation_raster_array = self.vegetation_raster.read()
            self.progress_text.insert(tk.END, f"Vegetation category raster loaded from: {raster_path}\n")
            self.update_data_table_with_vegetation()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the vegetation raster: {e}")

    def run_vegetation_category_model(self):
        input_raster_path = self.progress_text.get("1.0", tk.END).strip()
        if not os.path.exists(input_raster_path):
            messagebox.showerror("Error", "Input raster file not found.")
            return

        self.classify_raster(input_raster_path, self.progress_text, None)

    def classify_raster(self, input_raster_path, progress_text, progress_queue):
        global output_base_path
        if not output_base_path:
            messagebox.showerror("Error", "Output directory not set.")
            return

        self.progress_text.insert(tk.END, "Loading model...\n")

        model_path = os.path.join(script_dir, "DLmodelweights.h5")

        try:
            dl_model = load_model(model_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the model: {e}")
            return

        raster_array = self.raster_to_array(input_raster_path)
        if raster_array is None:
            return

        self.progress_text.insert(tk.END, "Raster normalization...\n")

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

        raster_array[:, :, :9] = self.normalize_raster(raster_array[:, :, :9], NORM_PERCENTILES)

        self.progress_text.insert(tk.END, "Applying model...\n")

        output_raster_array = self.apply_model_to_raster(dl_model, raster_array)
        if output_raster_array is None:
            return

        output_raster_array = np.clip(output_raster_array, 0, 1)
        sum_bands = np.sum(output_raster_array, axis=2, keepdims=True)
        sum_bands[sum_bands == 0] = 1  # Prevent division by zero
        output_raster_array = output_raster_array / sum_bands

        self.progress_text.insert(tk.END, "Success. Saving output raster...\n")

        output_raster_path = f"{output_base_path}/vegetation_categories_raster.tif"

        with rasterio.open(input_raster_path) as src:
            profile = src.profile

        profile.update(
            count=3,  # Update the number of bands to 3
            dtype=output_raster_array.dtype,
            crs='EPSG:4326'  # Ensure the saved raster is in EPSG:4326
        )

        try:
            with rasterio.open(output_raster_path, 'w', **profile) as dst:
                for band_index in range(output_raster_array.shape[2]):
                    dst.write(output_raster_array[:, :, band_index], band_index + 1)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save the output raster: {e}")
            return

        self.vegetation_raster = rasterio.open(output_raster_path)
        self.vegetation_raster_array = self.vegetation_raster.read()
        self.progress_text.insert(tk.END, f"Vegetation category raster saved at: {output_raster_path}\n")
        messagebox.showinfo("Success", "Vegetation category raster saved successfully.")
        self.update_data_table_with_vegetation()

    def raster_to_array(self, raster_path):
        try:
            with rasterio.open(raster_path) as src:
                raster_array = src.read()
                raster_array = np.moveaxis(raster_array, 0, -1)  # Move the band axis to the last dimension
            return raster_array
        except rasterio.errors.RasterioIOError:
            messagebox.showerror("Error", "Failed to read the raster file. Please check the file path and try again.")
            return None

    def apply_model_to_raster(self, model, raster_array):
        if raster_array is None:
            return None

        height, width, bands = raster_array.shape
        flattened_raster = raster_array.reshape(-1, bands)
        predictions = model.predict(flattened_raster)
        predictions_normalized = predictions / predictions.sum(axis=1, keepdims=True)
        output = predictions_normalized.reshape(height, width, 3)
        return output

    def normalize_raster(self, raster, norm_percentiles):
        for i in range(raster.shape[-1]):
            band = raster[:, :, i]
            band = np.log(band * 0.005 + 1)
            lower, upper = norm_percentiles[i]
            raster[:, :, i] = (band - lower) / upper
        return raster

    def update_data_table_with_vegetation(self):
        if not hasattr(self, 'vegetation_raster_array'):
            messagebox.showerror("Error", "Vegetation raster not loaded.")
            return

        veg_categories = ['Woody fraction', 'Herbaceous fraction', 'Bare fraction']
        veg_data = {category: [] for category in veg_categories}

        for idx, row in self.voronoi_polygons_gdf.iterrows():
            mask = geometry_mask([row.geometry], transform=self.vegetation_raster.transform, invert=True, out_shape=self.vegetation_raster_array.shape[1:])
            for i, category in enumerate(veg_categories):
                category_band = self.vegetation_raster_array[i, :, :]
                veg_fraction = category_band[mask].mean()
                veg_data[category].append(veg_fraction)

        for category in veg_categories:
            self.add_vegetation_row(category, veg_data[category])

    def add_vegetation_row(self, category, data):
        row_data = [category] + [f"{value:.2f}" for value in data]
        row_index = len(self.data_frame.winfo_children()) // (len(self.site_names) + 1)  # Calculate new row index
        self.add_data_row(row_data, row_index)
        self.add_class_checkbox(category, row_index)

    def download_text_file(self): 
        content = """
        //Go to https://code.earthengine.google.com/

        // Define the area of interest using a shapefile
        var geometry = ee.FeatureCollection('path/to/your/uploaded/shapefile');

        // Filter Sentinel-2 imagery for the specified date range and cloud cover
        var sentinel2 = ee.ImageCollection('COPERNICUS/S2')
        .filterBounds(geometry)
        .filterDate('YYYY-MM-DD', 'YYYY-MM-DD')
        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 1);

        // Select the latest image
        var image = sentinel2.limit(1, 'system:time_start', false).first();

        // Select a median image
        //var image = sentinel2.median(); // if the image is incomplete, substitute the above line with this line

        image = image.toFloat().resample('bilinear').reproject(image.select('B2').projection());

        // Clip the image to the specified geometry
        var clippedImage = image.clip(geometry);
        // if the above is not working try: var clippedImage = image.clip(geometry.geometry());

        // Select the bands of interest
        var bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12'];

        // Select the desired bands from the clipped image
        var selectedBands = clippedImage.select(bands);

        // Print the clipped image to the map
        // Map.addLayer(clippedImage, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'Clipped Image');

        // Center the map on the specified geometry
        Map.centerObject(geometry, 10);

        // Define export parameters
        var exportParams = {
        image: selectedBands,
        description: 'description', // Change the description as needed
        scale: 10,
        folder: 'GEE_exports' // Specify the folder in your Google Drive
        };

        // Export the image to Google Drive
        Export.image.toDrive(exportParams);
        """
        
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'w') as file:
                file.write(content)
            messagebox.showinfo("Success", f"File saved to: {file_path}")


    def compute_rais(self):
        messagebox.showinfo("Information", "The general output directory should contain camera trap imagery predictions in a file named full_image_annotations.csv.")
        self.progress_text.insert(tk.END, "Computing RAIs...\n")
        # Call the RAI calculation method
        self.RAI()

        # Read the RAI CSV file
        rai_file = f"{output_base_path}/RAI.csv"
        if not os.path.exists(rai_file):
            messagebox.showerror("Error", "RAI file not found.")
            return

        self.rai_df = pd.read_csv(rai_file)

        # Update the data display with RAIs
        self.update_data_with_rais()

        self.progress_text.insert(tk.END, "RAIs computation and table update completed.\n")

    def add_data_row(self, row_data, row_index):
        for col_index, data in enumerate(row_data):
            width = 25 if col_index == 0 else 7  # Increased width for the first column
            label = ttk.Label(self.data_frame, text=data, borderwidth=1, relief="solid", width=width)
            label.grid(row=row_index, column=col_index, sticky='nsew')

    def update_voronoi_plot(self):
        self.update_voronoi_plot_based_on_checkboxes()

    def get_polygon_color(self, site, class_name):
        df = None
        if hasattr(self, 'rai_df') and class_name in self.rai_df.columns:
            df = self.rai_df
        elif hasattr(self, 'relative_feeding_units_df') and class_name in self.relative_feeding_units_df.columns:
            df = self.relative_feeding_units_df

        if df is not None:
            site_data = df[df['Waterpoint'] == site]
            if not site_data.empty:
                max_value = df[class_name].max()
                value = site_data[class_name].values[0]
                if not pd.isna(value) and value != 0:
                    normalized_value = value / max_value
                    rgba_color = plt.cm.Reds(normalized_value) if normalized_value > 0 else (0.5, 0.5, 0.5, 0.7)
                    return rgba_color[:3] + (0.7,)
        return None  # No color if no values

    def update_voronoi_plot_based_on_checkboxes(self):
        # Clear existing plot
        self.ax.clear()

        # Plot satellite imagery as the background if checkbox is checked
        if self.show_satellite_var.get() and hasattr(self, 'boundaries_gdf'):
            xlim = self.boundaries_gdf.total_bounds[[0, 2]]  # minx, maxx
            ylim = self.boundaries_gdf.total_bounds[[1, 3]]  # miny, maxy
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            ctx.add_basemap(self.ax, crs=self.boundaries_gdf.crs.to_string(), source=ctx.providers.Esri.WorldImagery, alpha=0.9)
        
        # Always plot boundaries first
        if hasattr(self, 'boundaries_gdf'):
            self.boundaries_gdf.boundary.plot(ax=self.ax, edgecolor='black')

        # Always plot waterpoints
        if hasattr(self, 'waterpoints_gdf'):
            self.waterpoints_gdf.plot(ax=self.ax, color='black', markersize=5)
            for x, y, label in zip(self.waterpoints_gdf.geometry.x, self.waterpoints_gdf.geometry.y, self.site_names):
                self.ax.text(x, y, label, fontsize=8, color='black')

        # Plot vegetation layers first as background
        vegetation_layers = {'Woody fraction': 0, 'Herbaceous fraction': 1, 'Bare fraction': 2}
        colormaps = {'Woody fraction': 'Greens', 'Herbaceous fraction': 'YlGn', 'Bare fraction': 'YlOrBr'}

        for class_name, band_index in vegetation_layers.items():
            if self.plot_vars.get(class_name, tk.BooleanVar()).get() and hasattr(self, 'vegetation_raster_array'):
                band_data = self.vegetation_raster_array[band_index]
                extent = self.get_raster_extent()

                # Normalize the data based on quantiles
                quantiles = np.quantile(band_data, [0.02, 0.98])
                norm = mcolors.Normalize(vmin=quantiles[0], vmax=quantiles[1])

                self.ax.imshow(band_data, extent=extent, cmap=colormaps[class_name], norm=norm, interpolation='nearest', alpha=1.0)

        # Plot Voronoi polygons or other layers based on the checkbox values
        if hasattr(self, 'voronoi_polygons_gdf'):
            for idx, row in self.voronoi_polygons_gdf.iterrows():
                site = row['Site']
                outline = self.plot_vars.get('Area [ha]', tk.BooleanVar()).get()
                filled = False
                color = None

                for class_name, var in self.plot_vars.items():
                    if var.get() and class_name not in vegetation_layers:
                        color = self.get_polygon_color(site, class_name)
                        if color:
                            filled = True
                            break

                if outline and not filled:
                    self.voronoi_polygons_gdf.loc[[idx]].plot(ax=self.ax, edgecolor='black', facecolor='none', alpha=0.5)
                elif filled and color:
                    self.voronoi_polygons_gdf.loc[[idx]].plot(ax=self.ax, color=color, edgecolor='black', alpha=0.5)

        self.ax.set_aspect('auto')
        self.canvas.draw()

    def add_class_checkbox(self, class_name, row_index):
        if class_name not in self.plot_vars:
            self.plot_vars[class_name] = tk.BooleanVar()
        plot_checkbox = ttk.Checkbutton(self.data_frame, variable=self.plot_vars[class_name], command=self.update_voronoi_plot_based_on_checkboxes)
        plot_checkbox.grid(row=row_index, column=len(self.site_names) + 1, sticky='nsew')

    def update_data_with_rais(self):
        headers = ["Sites"] + self.site_names

        # Clear previous data
        for widget in self.data_frame.winfo_children():
            widget.destroy()

        # Create headers
        for col_index, col in enumerate(headers):
            width = 25 if col_index == 0 else 7  # Increased width for the first column
            label = ttk.Label(self.data_frame, text=col, borderwidth=1, relief="solid", width=width)
            label.grid(row=0, column=col_index, sticky='nsew')

        row_index = 1

        # Add area row
        if hasattr(self, 'voronoi_polygons_gdf'):
            areas = self.voronoi_polygons_gdf['Area (ha)'].tolist()
            self.add_data_row(['Area [ha]'] + [f"{area:.2f}" for area in areas], row_index)
            
            # Ensure only one checkbox for Area [ha]
            if 'Area [ha]' not in self.plot_vars:
                plot_var = tk.BooleanVar(value=True)
                self.plot_vars['Area [ha]'] = plot_var
            plot_checkbox = ttk.Checkbutton(self.data_frame, variable=self.plot_vars['Area [ha]'], command=self.update_voronoi_plot)
            plot_checkbox.grid(row=row_index, column=len(self.site_names) + 1, sticky='nsew')
            row_index += 1

        # Add RAI rows for relevant columns only (from time_range_days onwards)
        relevant_columns = [
            ('time_range_days', 'Coverage [days]'), 
            ('Grazing_units_absolute_selective', 'Abs. selective grazing'), 
            ('Browsing_units_absolute_selective', 'Abs. selective browsing'), 
            ('Grazing_units_absolute_bulk', 'Abs. bulk grazing'), 
            ('Browsing_units_absolute_bulk', 'Abs. bulk browsing')
        ] + [(col, f"{col.split('_', 1)[-1].capitalize()} [number/day]") for col in self.rai_df.columns if col.startswith('RAI_')]

        name_mapping = {
            'RAI_eland': 'Eland [number/day]',
            'RAI_elephant': 'Elephant [number/day]',
            'RAI_giraffe': 'Giraffe [number/day]',
            'RAI_impala': 'Impala [number/day]',
            'RAI_kudu': 'Kudu [number/day]',
            'RAI_oryx': 'Oryx [number/day]',
            'RAI_rhino': 'Rhino [number/day]',
            'RAI_wildebeest': 'Wildebeest [number/day]',
            'RAI_zebra': 'Zebra [number/day]',
            'RAI_springbok': 'Springbok [number/day]'
        }

        for class_name, display_name in relevant_columns:
            if class_name in name_mapping:
                display_name = name_mapping[class_name]
            rai_row = [display_name]
            for site in self.site_names:
                if site in self.rai_df['Waterpoint'].values:
                    rai_value = self.rai_df.loc[self.rai_df['Waterpoint'] == site, class_name].values[0]
                    rai_row.append(f"{rai_value:.2f}")
                else:
                    rai_row.append("")
            self.add_data_row(rai_row, row_index)
            self.add_class_checkbox(class_name, row_index)
            row_index += 1

    def get_image_metadata(self, image_path):
        try:
            img = Image.open(image_path)
            exif_data = img._getexif()
            for tag, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name == "DateTimeOriginal":
                    return value
        except Exception as e:
            self.progress_text.insert(tk.END, f"Error extracting metadata for {image_path}: {e}\n")
        return None

    def organize_images(self):
        global results_df

        if results_df.empty:
            csv_file_path = os.path.join(output_base_path, 'full_image_annotations.csv')
            if os.path.exists(csv_file_path):
                results_df = pd.read_csv(csv_file_path)
                self.progress_text.insert(tk.END, f"Loaded classification results from {csv_file_path}\n")
            else:
                messagebox.showerror("Error", "No classification results found. Perform classification first.")
                return

        threshold = float(self.prob_thresh_entry.get()) if self.prob_thresh_entry.get() else 0.75
        selected_species = [species for species, var in self.plot_vars.items() if var.get() and species.startswith('RAI_')]

        if not selected_species:
            messagebox.showerror("Error", "No species selected.")
            return

        for species in selected_species:
            species_name = species.split('_', 1)[-1]
            species_confidence = f"{species_name}_confidence"
            species_folder = os.path.join(output_base_path, species_name)
            if not os.path.exists(species_folder):
                os.makedirs(species_folder)

            filtered_df = results_df[results_df[species_confidence] >= threshold]

            for image_path in filtered_df['Name']:
                try:
                    shutil.copy(image_path, species_folder)
                except Exception as e:
                    self.progress_text.insert(tk.END, f"Error moving image {image_path}: {str(e)}\n")

        messagebox.showinfo("Success", "Images have been organized into species folders.")

    def RAI(self):
        def parse_datetime(date_str):
            formats = [
                "%Y:%m:%d %H:%M:%S",  # Original format
                "%Y/%m/%d %H:%M:%S",  # Common format
                "%Y-%m-%d %H:%M:%S",  # ISO format
                "%d/%m/%Y %H:%M",     # European format with no seconds
                "%Y:%m:%d %H:%M",     # Similar to original format but without seconds
                "%d-%m-%Y %H:%M:%S",  # European format with dashes
                "%m/%d/%Y %H:%M:%S",  # US format with slashes
                "%m-%d-%Y %H:%M:%S",  # US format with dashes
                "%d %b %Y %H:%M:%S",  # European format with abbreviated month
                "%d %B %Y %H:%M:%S",  # European format with full month
                "%d-%b-%Y %H:%M:%S",  # European format with abbreviated month and dashes
                "%d-%B-%Y %H:%M:%S",  # European format with full month and dashes
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            return None


        messagebox.showinfo("Warning", "All cameras at the same waterpoint need to have their time synchronized for duplicates to be effectively excluded.")
        self.progress_text.insert(tk.END, "Computing relative abundance indices (RAIs) from detections and converting to feeding units...\n")

        global output_base_path
        if not output_base_path:
            messagebox.showerror("Error", "Output directory not set.")
            return

        df = pd.read_csv(f"{output_base_path}/full_image_annotations.csv")
        df = df[df['Date_Time'] != "NA"]
        df['Date_Time'] = df['Date_Time'].astype(str).apply(parse_datetime)
        df = df.dropna(subset=['Date_Time'])
        df['Folder'] = df['Name'].apply(lambda x: os.path.basename(os.path.dirname(x)))

        species_names = list([col.split('_')[0] for col in df if '_count' in col])

        threshold = float(self.prob_thresh_entry.get()) if self.prob_thresh_entry.get() else 0.75

        # Initial Event Delimitation (site- and species-specific)
        df = df.sort_values(by=['Folder', 'Date_Time'])
        df['Event'] = None

        for folder_name, group in df.groupby('Folder'):
            for species in species_names:
                species_conf_col = f"{species}_confidence"
                species_count_col = f"{species}_count"
                species_group = group[group[species_conf_col] > threshold]
                event_id = 0
                last_time = None
                for idx, row in species_group.iterrows():
                    if last_time is None or (row['Date_Time'] - last_time).total_seconds() > 1500:
                        event_id += 1
                    df.at[idx, 'Event'] = event_id
                    last_time = row['Date_Time']

        # Additional Condition for "elephant" and "rhino"
        for species in ["elephant", "rhino"]:
            for folder_name, group in df.groupby('Folder'):
                species_events = group[group[f"{species}_confidence"] > threshold]
                all_events = species_events['Event'].unique()
                new_event_id = 0
                merged_events = set()
                for event in all_events:
                    if event in merged_events:
                        continue
                    event_rows = species_events[species_events['Event'] == event]
                    if len(event_rows) == 0:
                        continue
                    next_event = event + 1
                    while next_event in all_events:
                        next_event_rows = species_events[species_events['Event'] == next_event]
                        if len(next_event_rows) == 0:
                            next_event += 1
                            continue
                        if len(group[(group['Date_Time'] > event_rows['Date_Time'].max()) & 
                                    (group['Date_Time'] < next_event_rows['Date_Time'].min()) & 
                                    (group[f"{species}_confidence"] <= threshold)]) >= 4:
                            break
                        merged_events.add(next_event)
                        event_rows = pd.concat([event_rows, next_event_rows])
                        next_event += 1
                    new_event_id += 1
                    df.loc[(df['Folder'] == folder_name) & (df['Event'].isin(event_rows['Event'].unique())), 'Event'] = new_event_id

        # Counting Individuals
        final_counts = {}
        for folder_name, group in df.groupby('Folder'):
            final_counts[folder_name] = {}
            for species in species_names:
                species_count_col = f"{species}_count"
                species_conf_col = f"{species}_confidence"
                species_group = group[group[species_conf_col] > threshold]
                event_max_counts = species_group.groupby('Event')[species_count_col].max()
                final_counts[folder_name][species] = event_max_counts.sum()

        # Calculating Time Range
        results = pd.DataFrame.from_dict(final_counts, orient='index').reset_index().rename(columns={'index': 'Waterpoint'})

        for folder_name, group in df.groupby('Folder'):
            min_time = group['Date_Time'].min()
            max_time = group['Date_Time'].max()
            time_range_days = (max_time - min_time).total_seconds() / (60 * 60 * 24)
            results.loc[results['Waterpoint'] == folder_name, 'time_range_days'] = time_range_days
    

        # Calculating RAIs
        for species in species_names:
            results[f'RAI_{species}'] = results.apply(lambda row: row.get(species, 0) / row['time_range_days'] if row['time_range_days'] > 0 else 0, axis=1)

        herbivory_data = {
            "Species": ["zebra", "wildebeest", "oryx", "eland", "elephant", "impala", "rhino", "giraffe", "kudu", "springbok"],
            "Grazing unit": [1.32, 1, 1.1, 2, 9.8, 0.3, 3.1, 3.2, 0.8, 0.3],
            "Browsing unit": [1.59, 1.21, 1.36, 2.4, 11.78, 0.4, 3.76, 3.8, 1, 0.37],
            "Selectivity": [0, 0, 1, 1, 0, 1, 0, 1, 1, 1]
        }
        herbivory_df = pd.DataFrame(herbivory_data)

        results['Grazing_units_absolute_selective'] = 0.0
        results['Browsing_units_absolute_selective'] = 0.0
        results['Grazing_units_absolute_bulk'] = 0.0
        results['Browsing_units_absolute_bulk'] = 0.0

        for col in results.columns:
            if col.startswith('RAI_'):
                species = col.split('_')[-1]
                grazing_unit = herbivory_df.loc[herbivory_df['Species'] == species, 'Grazing unit'].values
                browsing_unit = herbivory_df.loc[herbivory_df['Species'] == species, 'Browsing unit'].values

                if grazing_unit.size > 0:
                    if herbivory_df.loc[herbivory_df['Species'] == species, 'Selectivity'].values == 1:
                        results['Grazing_units_absolute_selective'] += grazing_unit[0] * results[col].fillna(0)
                    else:
                        results['Grazing_units_absolute_bulk'] += grazing_unit[0] * results[col].fillna(0)
                if browsing_unit.size > 0:
                    if herbivory_df.loc[herbivory_df['Species'] == species, 'Selectivity'].values == 1:
                        results['Browsing_units_absolute_selective'] += browsing_unit[0] * results[col].fillna(0)
                    else:
                        results['Browsing_units_absolute_bulk'] += browsing_unit[0] * results[col].fillna(0)

        results.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

        csv_file = f"{output_base_path}/RAI.csv"
        results.to_csv(csv_file, index=False)
        self.progress_text.insert(tk.END, f"Relative abundance and feeding indices saved to: {csv_file}\n")
        self.save_absolute_feeding_units_plot(results, output_base_path)

    def save_absolute_feeding_units_plot(self, results, output_base_path):
        sorted_results = results.sort_values(by='Waterpoint')

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {
            'Selective Grazing': '#2ca25f',
            'Bulk Grazing': '#99d8c9',
            'Selective Browsing': '#8c6c4f',
            'Bulk Browsing': '#c9ae91'
        }
        bar_width = 0.6
        y_positions = np.arange(len(sorted_results))

        for i, (index, row) in enumerate(sorted_results.iterrows()):
            ax.barh(i, row['Grazing_units_absolute_selective'], color=colors['Selective Grazing'], edgecolor='white', height=bar_width)
            ax.barh(i, row['Grazing_units_absolute_bulk'], left=row['Grazing_units_absolute_selective'], color=colors['Bulk Grazing'], edgecolor='white', height=bar_width)
            ax.barh(i, -row['Browsing_units_absolute_selective'], color=colors['Selective Browsing'], edgecolor='white', height=bar_width)
            ax.barh(i, -row['Browsing_units_absolute_bulk'], left=-row['Browsing_units_absolute_selective'], color=colors['Bulk Browsing'], edgecolor='white', height=bar_width)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(sorted_results['Waterpoint'])
        ax.set_xlabel('Feeding units detected per camera trap day')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{abs(int(x))}"))  # Remove "-" and round the labels
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.subplots_adjust(left=0.1, right=0.98, top=0.85, bottom=0.15)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

        legend_elements = [
            Patch(facecolor=colors['Bulk Browsing'], edgecolor='none', label='Bulk Browsing'),
            Patch(facecolor=colors['Selective Browsing'], edgecolor='none', label='Selective Browsing'),
            Patch(facecolor=colors['Selective Grazing'], edgecolor='none', label='Selective Grazing'),
            Patch(facecolor=colors['Bulk Grazing'], edgecolor='none', label='Bulk Grazing')
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize='small', handlelength=1, handleheight=1, markerscale=0.8)

        plot_path = os.path.join(output_base_path, 'absolute_feeding_units.png')
        plt.savefig(plot_path)
        plt.close(fig)

    def relative_feeding_units_plot(self):
        results = pd.read_csv(f"{output_base_path}/results.csv")

        results = results.dropna(subset=[
            'Grazing_units/ha_herbaceous_vegetation_selective', 
            'Browsing_units/ha_woody_vegetation_selective', 
            'Grazing_units/ha_herbaceous_vegetation_bulk', 
            'Browsing_units/ha_woody_vegetation_bulk'
        ])

        # Sort results by waterpoint name alphabetically
        sorted_results = results.sort_values(by='Waterpoint')

        # Plot the data
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {
            'Selective Grazing': '#2ca25f',
            'Bulk Grazing': '#99d8c9',
            'Selective Browsing': '#8c6c4f',
            'Bulk Browsing': '#c9ae91'
        }
        bar_width = 0.6
        y_positions = np.arange(len(sorted_results))

        for i, (index, row) in enumerate(sorted_results.iterrows()):
            ax.barh(i, row['Grazing_units/ha_herbaceous_vegetation_selective'], color=colors['Selective Grazing'], edgecolor='white', height=bar_width)
            ax.barh(i, row['Grazing_units/ha_herbaceous_vegetation_bulk'], left=row['Grazing_units/ha_herbaceous_vegetation_selective'], color=colors['Bulk Grazing'], edgecolor='white', height=bar_width)
            ax.barh(i, -row['Browsing_units/ha_woody_vegetation_selective'], color=colors['Selective Browsing'], edgecolor='white', height=bar_width)
            ax.barh(i, -row['Browsing_units/ha_woody_vegetation_bulk'], left=-row['Browsing_units/ha_woody_vegetation_selective'], color=colors['Bulk Browsing'], edgecolor='white', height=bar_width)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(sorted_results['Waterpoint'])
        ax.set_xlabel('Feeding units per camera trap day and ha of woody or herbaceous vegetation')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{abs(x):.2f}"))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        plt.subplots_adjust(left=0.1, right=0.98, top=0.85, bottom=0.15)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

        legend_elements = [
            Patch(facecolor=colors['Bulk Browsing'], edgecolor='none', label='Bulk Browsing'),
            Patch(facecolor=colors['Selective Browsing'], edgecolor='none', label='Selective Browsing'),
            Patch(facecolor=colors['Selective Grazing'], edgecolor='none', label='Selective Grazing'),
            Patch(facecolor=colors['Bulk Grazing'], edgecolor='none', label='Bulk Grazing')
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize='small', handlelength=1, handleheight=1, markerscale=0.8)

        plot_path = os.path.join(output_base_path, 'relative_feeding_units_per_area_of_biomass.png')
        plt.savefig(plot_path)
        
    def output(self):
        self.progress_text.insert(tk.END, "Computing final results and visualizations...\n")

        global output_base_path
        if not output_base_path:
            messagebox.showerror("Error", "Output directory not set.")
            return
        
        # File paths
        rai_file = f"{output_base_path}/RAI.csv"
        
        if not os.path.exists(rai_file):
            messagebox.showerror("Error", "RAI file not found.")
            return

        rai_df = pd.read_csv(rai_file)

        # Ensure Waterpoint column is treated as string
        rai_df['Waterpoint'] = rai_df['Waterpoint'].astype(str)

        vegetation_categories = ['Woody fraction', 'Herbaceous fraction', 'Bare fraction']
        veg_means = {category: [] for category in vegetation_categories}

        for idx, row in self.voronoi_polygons_gdf.iterrows():
            mask = geometry_mask([row.geometry], transform=self.vegetation_raster.transform, invert=True, out_shape=self.vegetation_raster_array.shape[1:])
            for i, category in enumerate(vegetation_categories):
                category_band = self.vegetation_raster_array[i, :, :]
                veg_fraction = category_band[mask].mean()
                veg_means[category].append(veg_fraction)

        veg_means_df = pd.DataFrame(veg_means)
        veg_means_df.index = self.voronoi_polygons_gdf['Site'].astype(str)

        # Merge RAI and vegetation data
        relative_feeding_units_df = pd.merge(rai_df, veg_means_df, left_on='Waterpoint', right_index=True)

        # Calculate relative feeding units
        relative_feeding_units_df['Grazing_units/ha_herbaceous_vegetation_selective'] = (
            relative_feeding_units_df['Grazing_units_absolute_selective'] / 
            relative_feeding_units_df['Herbaceous fraction']
        )
        relative_feeding_units_df['Browsing_units/ha_woody_vegetation_selective'] = (
            relative_feeding_units_df['Browsing_units_absolute_selective'] / 
            relative_feeding_units_df['Woody fraction']
        )
        relative_feeding_units_df['Grazing_units/ha_herbaceous_vegetation_bulk'] = (
            relative_feeding_units_df['Grazing_units_absolute_bulk'] / 
            relative_feeding_units_df['Herbaceous fraction']
        )
        relative_feeding_units_df['Browsing_units/ha_woody_vegetation_bulk'] = (
            relative_feeding_units_df['Browsing_units_absolute_bulk'] / 
            relative_feeding_units_df['Woody fraction']
        )

        output_file = f"{output_base_path}/results.csv"
        relative_feeding_units_df.to_csv(output_file, index=False)
        self.progress_text.insert(tk.END, f"Relative feeding units and vegetation categories results table saved to: {output_file}\n")

        self.relative_feeding_units_df = relative_feeding_units_df

        self.relative_feeding_units_plot()

    def add_relative_feeding_units_to_table(self):
        metrics = [
            ('Grazing_units/ha_herbaceous_vegetation_selective', 'Rel. selective grazing'), 
            ('Browsing_units/ha_woody_vegetation_selective', 'Rel. selective browsing'), 
            ('Grazing_units/ha_herbaceous_vegetation_bulk', 'Rel. bulk grazing'), 
            ('Browsing_units/ha_woody_vegetation_bulk', 'Rel. bulk browsing')
        ]

        for metric, display_name in metrics:
            row_data = [display_name]
            for site in self.site_names:
                if site in self.relative_feeding_units_df['Waterpoint'].values:
                    value = self.relative_feeding_units_df.loc[self.relative_feeding_units_df['Waterpoint'] == site, metric].values[0]
                    row_data.append(f"{value:.2f}")
                else:
                    row_data.append("")
            row_index = len(self.data_frame.grid_slaves()) // (len(self.site_names) + 1)  # Correct calculation of new row index
            self.add_data_row(row_data, row_index)
            self.add_class_checkbox(metric, row_index)

if __name__ == "__main__":
    root = tk.Tk()
    app = HerbivoryApp(root)
    root.mainloop()
