# Tool_Bbox_Creator

Bounding box labeling tool for object detection dataset creation and management.

## Overview

- Manual Annotation: Intuitive GUI for creating, editing, and deleting bounding boxes.
- Auto-Labeling: Automatic inference and labeling using integrated Ultralytics YOLO models.
- Database Integration: Reliable storage and management of classes and annotations via MariaDB.
- Flexible Workflow: Supports loading individual images or entire image folders.
- Smart Synchronization: Automatic synchronization of YOLO model classes with the database.
- Standardized Export: Full support for importing and exporting data in the YOLO label format.

## Features

- Automated Workflow: One-click "Auto Label All" function to generate draft annotations.
- Data Management: Centralized management of image lists, bounding boxes, and object classes.
- Model Compatibility: Seamless integration with custom YOLO .pt models.
- Robust Persistence: All annotation data and class information are securely stored in a MariaDB database.

## Project Structure

- `main.py`: Application entry point.
- `auto_bbox_gui.py`: Main GUI application logic.
- `auto_bbox_mode.py`: Auto-labeling engine and image processing pipeline.
- `auto_bbox_db.py`: Database CRUD operations and management.
- `ui_theme.py`: Custom dark theme configuration for UI consistency.
- `config.py`: Global project configurations and settings.
- `workspace/`: Directory for exported labels, visualization outputs, and temporary data.

## Database

The application stores annotation information in a MariaDB database.

Main tables include:

- `db_class_info`: class information
- `db_bbox_info`: bounding box annotations

The class list is automatically synchronized when a new YOLO model containing additional classes is selected.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main packages used in this project:

- ultralytics
- torch
- opencv-python
- pillow
- pymysql (or mariadb)
- pyyaml
- tkinter

## Usage

Run the application:

```bash
python main.py
```

Typical workflow:

1. Launch: Execute the application by running python main.py.
2. Setup: Open the directory containing your target images.
3. Configure (Optional): Select a custom YOLO .pt model for inference, or proceed with the default model to initiate auto-labeling.
4. Annotate: Generate initial labels using **Auto Label All Images**, or manually create, refine, and edit bounding boxes at any time.
5. Sync & Export: Save your final annotations to the MariaDB database and export them to standard YOLO text files.

## Notes

- Only valid YOLO `.pt` model files can be selected for auto labeling engine.
- Loading a new model will automatically trigger a synchronization process, adding any new classes found in the model to the database.
-The application ensures data consistency by maintaining information in both the database and local YOLO label files.