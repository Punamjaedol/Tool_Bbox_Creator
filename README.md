# Tool_Bbox_Creator

Bounding box labeling tool for object detection dataset creation and management.

## Overview

- Manual bounding box annotation with an interactive GUI
- Automatic labeling using Ultralytics YOLO models
- Bounding box and class management with a MariaDB database
- Support for loading image folders or individual images
- Automatic synchronization of model classes with the class database
- Export and import in YOLO label format
- Designed for efficient object detection dataset creation

## Features

- Manual bounding box creation, editing, and deletion
- Auto labeling for all images using a YOLO `.pt` model
- Image list and bounding box management
- Class selection and management
- Automatic synchronization between YOLO model classes and the database
- Draft label generation and visualization
- Database-based storage of classes and annotations

## Project Structure

- `main.py`: application entry point
- `auto_bbox_gui.py`: main GUI application
- `auto_bbox_mode.py`: auto-labeling and image processing logic
- `auto_bbox_db.py`: database CRUD functions
- `ui_theme.py`: dark UI theme configuration
- `config.py`: project configuration
- `workspace/`: generated labels, visualization images, and temporary files
- `best.pt`: default auto-label model

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

1. Open an image folder.
2. Select or change the auto-label YOLO model.
3. Run **Auto Label All Images** (optional).
4. Review and edit bounding boxes.
5. Save annotations to the database and YOLO label files.

## Notes

- Only valid YOLO `.pt` model files can be selected for auto labeling.
- When a new model is loaded, any classes not already in the database are automatically added.
- Bounding boxes are stored in both the database and YOLO text format.
- The application uses a dark-themed interface optimized for annotation work.