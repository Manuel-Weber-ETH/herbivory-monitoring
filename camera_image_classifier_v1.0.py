########################################################################################################################################################################################
### Spatially and Temporally Explicit Herbivory Monitoring in Water-limited Savannas
### Camera Trap Imagery Classifier
### Manuel Weber
### https://github.com/Manuel-Weber-ETH/cameratraps.git
########################################################################################################################################################################################

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO
import pandas as pd
import os
import sys
from PIL import Image, ExifTags
from datetime import date
import geopandas as gpd


# Handle the base path for PyInstaller
try:
    base_path = sys._MEIPASS  # Path where PyInstaller unpacks files
except AttributeError:
    base_path = os.path.abspath(".")  # If not bundled, use the current directory

output_base_path = ""
results_df = pd.DataFrame()
checkbox_vars = {}

class HerbivoryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera trap imagery classifier")
        self.setup_ui()

    def setup_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Button(frame, text="Select Output Directory", command=self.browse_and_create_output_directory).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(frame, text="Select Image Folder", command=self.browse_folder).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Classify Images", command=self.classify_images).grid(row=0, column=3, padx=5, pady=5)

        self.progress_text = tk.Text(frame, height=6, width=80)
        self.progress_text.grid(row=1, column=0, columnspan=4, padx=5, pady=5)

        conf_frame = ttk.Frame(frame)
        conf_frame.grid(row=0, column=2, padx=5, pady=5)
        ttk.Label(conf_frame, text="Confidence threshold:").pack(side=tk.LEFT)
        self.prob_thresh_entry = ttk.Entry(conf_frame, width=10)
        self.prob_thresh_entry.pack(side=tk.LEFT)
        self.prob_thresh_entry.insert(0, "0.75")

    def browse_folder(self):
        messagebox.showinfo("Information", "Please provide the path to a directory with folders that correspond to waterpoints. Only images in waterpoint folders will be classified.")
        self.folder_path = filedialog.askdirectory()
        if not self.folder_path:
            messagebox.showerror("Error", "No folder selected.")
        else:
            self.progress_text.insert(tk.END, f"Selected folder: {self.folder_path}\n")

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

    def get_image_metadata(self, image_path):
        try:
            img = Image.open(image_path)
            exif_data = img._getexif()
            if not exif_data:
                return None
            for tag, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name == "DateTimeOriginal":
                    return value
        except Exception as e:
            self.progress_text.insert(tk.END, f"Error extracting metadata for {image_path}: {e}\n")
        return None

    def classify_images(self):
        if not hasattr(self, 'folder_path') or not self.folder_path:
            messagebox.showerror("Error", "No folder selected.")
            return

        if not output_base_path:
            messagebox.showerror("Error", "No output directory selected.")
            return

        model_path = os.path.join(base_path, "yoloweightsv10_lib.pt")
        
        try:
            yolo_model = YOLO(model_path)
            self.progress_text.insert(tk.END, "YOLO model loaded successfully\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {e}")
            return

        try:
            threshold = float(self.prob_thresh_entry.get())
            self.progress_text.insert(tk.END, "Classifying images...\n")
            image_path = self.folder_path
            results = yolo_model.predict(source=os.path.join(image_path, '*/*'), stream=True, save=False, conf=threshold)

            temp_results = []
            for r in results:
                max_confidences = {f"{species}_confidence": 0 for species in yolo_model.names.values()}
                individual_counts = {species: 0 for species in yolo_model.names.values()}
                date_time = self.get_image_metadata(r.path)
                if date_time is None:
                    date_time = "NA"
                    self.progress_text.insert(tk.END, f"Metadata not found for image: {r.path}\n")

                for idx, class_id in enumerate(r.boxes.cls.numpy()):
                    species = yolo_model.names[int(class_id)]
                    confidence = r.boxes.conf.numpy()[idx]
                    individual_counts[species] += 1  # Count the number of individuals detected
                    if confidence > max_confidences[f"{species}_confidence"]:
                        max_confidences[f"{species}_confidence"] = confidence

                max_confidences['Name'] = r.path
                max_confidences['Date_Time'] = date_time
                max_confidences.update({f"{species}_count": count for species, count in individual_counts.items()})
                temp_results.append(max_confidences)

            species_columns = [f"{species}_confidence" for species in yolo_model.names.values()]
            count_columns = [f"{species}_count" for species in yolo_model.names.values()]
            columns = ['Name', 'Date_Time'] + species_columns + count_columns

            global results_df
            results_df = pd.DataFrame(temp_results, columns=columns)

            self.progress_text.insert(tk.END, "All images classified.\n")

            csv_file = f"{output_base_path}/full_image_annotations.csv"
            results_df.to_csv(csv_file, index=False)

            self.progress_text.insert(tk.END, f"Success. Predictions saved under {output_base_path}/full_image_annotations.csv.\n")

            na_count = results_df['Date_Time'].isna().sum()
            total_count = len(results_df)
            messagebox.showinfo("Success", f"Predictions saved. Select species in table to sort images into folders. {na_count} out of {total_count} images lack metadata and cannot be used to compute RAIs.")
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = HerbivoryApp(root)
    root.mainloop()