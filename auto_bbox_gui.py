import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

from PIL import Image, ImageTk

from auto_bbox_utils import *
import auto_bbox_mode  as mode
from ui_theme import apply, paint, BG
from config import *

# App setup / UI
class BBoxReviewApp:
    """Simple GUI for reviewing and editing YOLO bbox labels."""
    HANDLE_SIZE = 4
    MIN_ZOOM_FACTOR = 0.5
    MAX_ZOOM_FACTOR = 4.0
    ZOOM_STEP = 1.15
    CLASSES_TXT = WORK_DIR / "classes.txt"
    VISUALIZED_DIR = DRAFT_VIS_DIR
    DEFAULT_CLASS = "None"
    
    # App setup / UI
    def __init__(
        self,
        root,
        image_files,
        label_dir,
        class_id=-1,
        max_canvas_w=1200,
        max_canvas_h=780,
    ):
        self.root = root
        self.image_files = list(image_files)
        self.label_dir = Path(label_dir)
        self.class_id = class_id
        self.max_canvas_w = max_canvas_w
        self.max_canvas_h = max_canvas_h

        self.db_connected = False

        self.image_idx = 0
        self.image_path = None
        self.original_img = None
        self.tk_img = None
        self.scale = 1.0
        self.zoom_factor = 1.0
        self.boxes = []
        self.selected_idx = None
        self.drag_mode = None
        self.drag_start = None
        self.original_box = None
        self.box_classes = []
        self.loaded_box_classes = []
        self.class_names = mode.sync_classes_from_db(self.CLASSES_TXT) or []
        self.selected_class_var = tk.StringVar()
        self.class_listbox = None
        self.image_dir_var = tk.StringVar(value=self.get_current_image_dir())
        self.resize_after_id = None
        self.class_popup = None
        self.auto_model_classes = None
        
        self.auto_label_model_path = Path(r"D:/KCH/workspace/tool_bbox_creator/yolov8n.pt")

        self.root.title("BBOX Creator")
        width = 1400
        height = 650
        self.root.geometry(
            f"{width}x{height}+"
            f"{self.root.winfo_screenwidth() // 2 - width // 2}+"
            f"{self.root.winfo_screenheight() // 2 - height // 2}"
        )
        
        self.build_ui()
        apply(self.root)
        paint(self.root)
        self.bind_events()
        self.load_current_image()

    def build_ui(self):
        # HEADER 
        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X)

        self.status_var = tk.StringVar()
        ttk.Label(top, textvariable=self.status_var, anchor="w").pack(side=tk.LEFT, padx=12)

        self.db_status_var = tk.StringVar()
        ttk.Label(top, textvariable=self.db_status_var, anchor="w").pack(side=tk.LEFT, padx=12)
        self.btn_connect = ttk.Button(top, text="DB Connect", command=self.open_db_connect_dialog)
        self.btn_connect.pack(side=tk.LEFT, padx=3, pady=3)
        self.btn_set_table = ttk.Button(top, text="Set DB Table", command=self.open_db_table_manager, state=tk.DISABLED)
        self.btn_set_table.pack(side=tk.LEFT, padx=3, pady=3)
        self.btn_disconnect = ttk.Button(top, text="Disconnect", command=self.disconnect_db_gui, state=tk.DISABLED)
        self.btn_disconnect.pack(side=tk.LEFT, padx=3, pady=3)
        self.update_db_status()

        # BODY
        body = tk.PanedWindow(self.root, bg=BG, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=6, showhandle=True)

        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # BODY - LEFT
        left = ttk.Frame(body, width=500)
        body.add(left, minsize=350, stretch="never")

        self.canvas_frame = ttk.Frame(body)
        self.canvas = tk.Canvas(self.canvas_frame, cursor="crosshair")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        body.add(self.canvas_frame, minsize=400, stretch="always")

        frame_controls = ttk.Frame(self.canvas_frame, style="Background.TFrame")
        frame_controls.pack(side=tk.BOTTOM, fill=tk.X, pady=0)
        self.status_bar = ttk.Label(frame_controls, text="", anchor="center", relief="flat", style="Background.TLabel")
        self.status_bar.pack(fill=tk.X)
        button_inner = ttk.Frame(frame_controls, style="Background.TFrame")
        button_inner.pack(anchor=tk.CENTER)

        ttk.Button(button_inner, text="Prev", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_inner, text="Save", command=self.save_current_labels).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_inner, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        side = ttk.Frame(body, width=360)
        body.add(side, minsize=280, stretch="never")

        # BODY - LEFT - OPEN DIRECTORY
        ttk.Label(left, text="File & Filter", anchor="w").pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 2))
        open_frame = ttk.Frame(left)
        open_frame.pack(side=tk.TOP, fill=tk.X, padx=6)
        ttk.Button(open_frame, text="Folder Open", command=self.browse_image_dir).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(open_frame, text="Single File Open", command=self.open_single_file).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0),)

        ttk.Label(left, text="Directory", anchor="w").pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 2))
        dir_frame = ttk.Frame(left)
        dir_frame.pack(side=tk.TOP, fill=tk.X, padx=6)
        ttk.Entry(dir_frame, textvariable=self.image_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dir_frame, text="Browse", command=self.browse_image_dir).pack(side=tk.LEFT, padx=(4, 0))
        
        
        # BODY - LEFT - AUTO LABEL MODEL
        model_frame = ttk.Frame(left)
        model_frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)
        self.model_status_var = tk.StringVar(value=f"Auto-Label Model: {self.auto_label_model_path.name}")
        ttk.Label(model_frame, textvariable=self.model_status_var).pack(side=tk.LEFT)
        ttk.Button(model_frame, text="Change Model", command=self.select_auto_label_model).pack(side=tk.RIGHT, padx=2, pady=(0, 4))
        
        # BODY - LEFT - IMAGE LIST
        image_list_header_frame = ttk.Frame(left)
        image_list_header_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(2, 2))
        ttk.Label(image_list_header_frame, text="Images", anchor="w").pack(side=tk.LEFT, fill=tk.X, padx=6, pady=(2, 2))
        ttk.Button(image_list_header_frame, text="Auto Label All Images", command=self.auto_label).pack(side=tk.RIGHT, padx=0, pady=(2, 0))
        image_list_frame = ttk.Frame(left)
        image_list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6)
        self.image_listbox = tk.Listbox(image_list_frame, borderwidth=0, exportselection=False, height=12)
        image_scroll = ttk.Scrollbar(image_list_frame, orient=tk.VERTICAL, command=self.image_listbox.yview)
        self.image_listbox.config(yscrollcommand=image_scroll.set)
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        image_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # BODY - RIGHT - BOX LIST
        box_list_header_frame = ttk.Frame(side)
        box_list_header_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(2, 2))
        ttk.Label(box_list_header_frame, text="Boxes", anchor="w").pack(side=tk.LEFT, fill=tk.X, padx=6, pady=(8, 2))
        ttk.Button(box_list_header_frame, text="Delete Box", command=self.delete_selected_box).pack(side=tk.RIGHT, padx=(3, 17))
        
        box_list_frame = ttk.Frame(side)
        box_list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6)
        columns = ("seq", "class", "x1", "y1", "x2", "y2", "w", "h")
        self.box_table = ttk.Treeview(box_list_frame, columns=columns, show="headings", height=10)
        for column in columns:
            self.box_table.heading(column, text=column)
        self.box_table.column("seq", width=45, anchor=tk.CENTER)
        self.box_table.column("class", width=90, anchor=tk.W)
        for column in ("x1", "y1", "x2", "y2", "w", "h"):
            self.box_table.column(column, width=40, anchor=tk.E)
        box_scroll = ttk.Scrollbar(box_list_frame, orient=tk.VERTICAL, command=self.box_table.yview)
        self.box_table.config(yscrollcommand=box_scroll.set)
        self.box_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        box_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # BODY - RIGHT - CLASS LIST
        class_frame = ttk.LabelFrame(side, text="Class Select")
        class_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=8)
        class_list_frame = ttk.Frame(class_frame)
        class_list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=(4, 0))
        
        self.class_listbox = tk.Listbox(class_list_frame, borderwidth=0, exportselection=False, height=7)
        class_scroll = ttk.Scrollbar(class_list_frame, orient=tk.VERTICAL, command=self.class_listbox.yview)
        self.class_listbox.config(yscrollcommand=class_scroll.set)
        self.class_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        class_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        button_frame = ttk.Frame(class_frame)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)        
        ttk.Button(button_frame, text="Class Manage", command=self.open_class_manager).pack(side=tk.RIGHT, anchor=tk.E, padx=4, pady=(0, 6,))
        ttk.Button(button_frame, text="Initialize Local Class", command=self.init_local_class).pack(side=tk.RIGHT, anchor=tk.E, padx=4, pady=(0, 6,))
        self.refresh_image_list()
        self.refresh_class_list()

    def bind_events(self):
        self.canvas_frame.bind("<Configure>", self.on_canvas_frame_resize)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<Leave>", self.clear_pointer_guides)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.canvas.bind("<B3-Motion>", self.on_canvas_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_canvas_right_release)

        self.image_listbox.bind("<<ListboxSelect>>", self.on_image_list_select)
        self.box_table.bind("<<TreeviewSelect>>", self.on_box_list_select)
        if self.class_listbox is not None:
            self.class_listbox.bind("<<ListboxSelect>>", self.on_class_list_select)
        
        self.canvas.config(takefocus=1)
        self.root.bind("<Delete>", lambda event: self.delete_selected_box()) # 딜리트 안 먹어서 아래의 버튼으로 사용
        self.root.bind("<Control-s>", lambda event: self.save_current_labels())
        self.root.bind("<F5>", lambda event: self.refresh_all())
        self.root.bind("<Left>", lambda event: self.prev_image())
        self.root.bind("<Right>", lambda event: self.next_image())
    
    def refresh_all(self):
        """Refresh the image list, box list, and class list."""
        self.refresh_image_list()
        self.refresh_box_list()
        self.refresh_class_list()

        # 현재 이미지가 있으면 화면도 다시 갱신
        if self.image_path is not None:
            self.load_current_image()    

    # Image file selection
    def get_current_image_dir(self):
        if not self.image_files:
            return ""
        try:
            return str(Path(self.image_files[0]).parent)
        except Exception:
            return ""

    def find_images_in_dir(self, image_dir):
        image_dir = Path(image_dir)
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if not image_dir.exists():
            debug_warn(f"Image directory does not exist: {image_dir}")
            return []
        return sorted([f for f in image_dir.rglob("*") if f.suffix.lower() in valid_exts])

    def browse_image_dir(self):
        selected_dir = filedialog.askdirectory(initialdir=self.image_dir_var.get() or ".")
        if not selected_dir:
            return
        self.image_dir_var.set(selected_dir)
        self.load_image_dir_from_entry()

    def open_single_file(self):
        selected_file = filedialog.askopenfilename(
            initialdir=self.image_dir_var.get() or ".",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not selected_file:
            return

        self.save_current_labels()
        selected_path = Path(selected_file)
        self.image_dir_var.set(str(selected_path.parent))
        next_files = self.find_images_in_dir(selected_path.parent)
        if selected_path not in next_files:
            next_files = sorted(next_files + [selected_path])

        self.image_files = next_files
        self.image_idx = next_files.index(selected_path)
        debug_info(f"Opened single file: {selected_path}")
        self.load_current_image()

    def load_image_dir_from_entry(self):
        next_files = self.find_images_in_dir(self.image_dir_var.get())
        if not next_files:
            messagebox.showwarning("No images", "No image files were found in this directory.")
            return

        self.save_current_labels()
        self.image_files = next_files
        self.image_idx = 0
        debug_info(f"Loaded image directory: {self.image_dir_var.get()}")
        debug_item(f"images={len(self.image_files)}")
        self.load_current_image()

    # Image loading / rendering
    def load_current_image(self):
        if not self.image_files:
            debug_warn("No images to review.")
            return

        self.image_path = self.image_files[self.image_idx]
        debug_info(f"Review image: {self.image_path}")

        pil_img = Image.open(self.image_path).convert("RGB")
        self.original_img = pil_img
        self.zoom_factor = 1.0
        self.boxes = self.load_boxes_for_image()
        if len(self.loaded_box_classes) == len(self.boxes):
            self.box_classes = list(self.loaded_box_classes)
        else:
            self.box_classes = [self.default_class_name() for _ in self.boxes]
        self.selected_idx = None
        self.render_image()
        self.draw_boxes()
        self.refresh_image_list()
        self.refresh_box_list()
        self.update_status()

    def load_boxes_for_image(self):
        label_path = self.label_dir / f"{self.image_path.stem}.txt"
        label_lines = read_yolo_labels(label_path)
        img_w, img_h = self.original_img.size

        boxes = []
        box_classes = []
        for line in label_lines:
            box = yolo_to_xyxy(line, img_w, img_h)
            if box is not None:
                boxes.append(box)
                box_classes.append(self.class_name_from_label(line))

        self.loaded_box_classes = box_classes
        debug_info(f"Loaded labels: {label_path} boxes={len(boxes)}")
        return boxes

    def get_canvas_area_size(self):
        area_w = max(1, self.canvas_frame.winfo_width())
        area_h = max(1, self.canvas_frame.winfo_height())

        if area_w <= 1 or area_h <= 1:
            area_w = self.max_canvas_w
            area_h = self.max_canvas_h

        return area_w, area_h

    def render_image(self):
        if self.original_img is None:
            return

        img_w, img_h = self.original_img.size
        area_w, area_h = self.get_canvas_area_size()
        fit_scale = min(area_w / img_w, area_h / img_h)
        self.scale = fit_scale * self.zoom_factor
        show_w = max(1, int(img_w * self.scale))
        show_h = max(1, int(img_h * self.scale))

        show_img = self.original_img.resize((show_w, show_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(show_img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.canvas.config(scrollregion=(0, 0, show_w, show_h))

    def on_canvas_frame_resize(self, event):
        if self.original_img is None:
            return
        if self.resize_after_id is not None:
            self.root.after_cancel(self.resize_after_id)
        self.resize_after_id = self.root.after(80, self.redraw_after_resize)

    def redraw_after_resize(self):
        self.resize_after_id = None
        self.render_image()
        self.draw_boxes()

    def on_mouse_wheel(self, event):
        if self.original_img is None:
            return

        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            zoom_delta = self.ZOOM_STEP
        else:
            zoom_delta = 1 / self.ZOOM_STEP

        next_zoom = max(self.MIN_ZOOM_FACTOR, min(self.MAX_ZOOM_FACTOR, self.zoom_factor * zoom_delta))
        if next_zoom == self.zoom_factor:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        img_x = canvas_x / self.scale
        img_y = canvas_y / self.scale

        self.zoom_factor = next_zoom
        self.render_image()
        self.draw_boxes()

        img_w, img_h = self.original_img.size
        show_w = max(1, int(img_w * self.scale))
        show_h = max(1, int(img_h * self.scale))
        next_canvas_x = img_x * self.scale
        next_canvas_y = img_y * self.scale
        view_w = max(1, self.canvas.winfo_width())
        view_h = max(1, self.canvas.winfo_height())
        self.canvas.xview_moveto(max(0, min(1, (next_canvas_x - event.x) / max(1, show_w - view_w))))
        self.canvas.yview_moveto(max(0, min(1, (next_canvas_y - event.y) / max(1, show_h - view_h))))
        self.draw_pointer_guides(event.x, event.y)

    def show_status_message(self, message, duration=1000):
        """상태 표시줄에 메시지를 띄우고 duration 뒤에 초기화합니다."""
        self.status_bar.config(text=message)
        # duration 이후에 텍스트를 "Ready"로 되돌림
        self.root.after(duration, lambda: self.status_bar.config(text=""))

    # Lists / selection state
    def refresh_image_list(self):
        self.image_listbox.delete(0, tk.END)
        for idx, img_path in enumerate(self.image_files):
            label_path = self.label_dir / f"{img_path.stem}.txt"
            has_boxes = len(read_yolo_labels(label_path)) > 0
            status = "OK" if has_boxes else "바운딩박스 미존재"
            self.image_listbox.insert(tk.END, f"{idx + 1:03d} [{status}] {img_path.name}")
            if not has_boxes:
                self.image_listbox.itemconfig(idx, fg="red")
        self.select_listbox_index(self.image_listbox, self.image_idx)

    def refresh_box_list(self):
        for item in self.box_table.get_children():
            self.box_table.delete(item)

        for idx, box in enumerate(self.boxes):
            x1, y1, x2, y2 = self.normalize_box(box)
            class_name = self.box_classes[idx] if idx < len(self.box_classes) else "Unknown"
            self.box_table.insert(
                "",
                tk.END,
                iid=str(idx),
                values=(idx + 1, class_name, x1, y1, x2, y2, x2 - x1, y2 - y1),
            )

        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.boxes):
            self.box_table.selection_set(str(self.selected_idx))
            self.box_table.see(str(self.selected_idx))
        self.update_class_field()

    def refresh_class_list(self):
        if self.class_listbox is None:
            return
        self.class_listbox.delete(0, tk.END)
        for class_name in self.class_names:
            self.class_listbox.insert(tk.END, class_name)
        

    def select_listbox_index(self, listbox, idx):
        listbox.selection_clear(0, tk.END)
        if idx is None:
            return
        if 0 <= idx < listbox.size():
            listbox.selection_set(idx)
            listbox.see(idx)

    def on_image_list_select(self, event):
        selection = self.image_listbox.curselection()
        if not selection:
            return
        next_idx = selection[0]
        if next_idx == self.image_idx:
            return
        self.save_current_labels()
        self.image_idx = next_idx
        self.load_current_image()

    def on_box_list_select(self, event):
        selection = self.box_table.selection()
        if not selection:
            return
        self.selected_idx = int(selection[0])
        debug_item(f"Selected box from list: {self.boxes[self.selected_idx]}")
        self.draw_boxes()
        self.update_class_field()
        self.update_status()

    def on_class_list_select(self, event):
        selection = self.class_listbox.curselection()
        if not selection:
            return
        self.selected_class_var.set(self.class_listbox.get(selection[0]))
        if self.selected_idx is not None:
            self.apply_selected_class()

    def update_status(self):
        self.status_var.set(
            f"{self.image_idx + 1}/{len(self.image_files)} | {self.image_path.name} | boxes={len(self.boxes)}"
        )

    # Class names / selection
    def init_local_class(self):
        if mode.is_db_connected():
            messagebox.showwarning("Initialize Local Class", "This action is only available when DB is not connected.")
            return

        confirm = messagebox.askyesno(
            "Initialize Local Class",
            "This will delete all locally saved classes (local_classes.json).\nContinue?"
        )
        if not confirm:
            return

        status, class_names = mode.init_local_classes(self.CLASSES_TXT)
        if status == "db_connected":
            messagebox.showwarning("Initialize Local Class", "This action is only available when DB is not connected.")
            return

        self.class_names = class_names
        self.refresh_class_list()
        self.refresh_box_list()
        self.draw_boxes()
        messagebox.showinfo("Initialize Local Class", "Local classes have been reset.")
        
    def load_class_names(self):
        return mode.load_class_names(self.CLASSES_TXT)

    def default_class_name(self):
        current = self.selected_class_var.get().strip()
        if current:
            return current
        if self.class_names:
            return self.class_names[0]
        return "Unknown"

    def class_name_from_label(self, line):
        parts = line.split()
        if not parts:
            return self.default_class_name()
        try:
            class_idx = int(float(parts[0]))
        except ValueError:
            return self.default_class_name()
        if 0 <= class_idx < len(self.class_names):
            return self.class_names[class_idx]
        return self.default_class_name()

    def class_id_from_name(self, class_name):
        try:
            return self.class_names.index(class_name)
        except ValueError:
            return self.class_id

    def update_class_field(self):
        if self.selected_idx is None or self.selected_idx >= len(self.boxes):
            self.selected_class_var.set("")
            return

        class_name = self.box_classes[self.selected_idx] if self.selected_idx < len(self.box_classes) else ""
        self.selected_class_var.set(class_name)

    def apply_selected_class(self):
        if self.selected_idx is None:
            debug_warn("No selected box to apply class.")
            return

        while len(self.box_classes) < len(self.boxes):
            self.box_classes.append(self.default_class_name())

        self.box_classes[self.selected_idx] = self.selected_class_var.get().strip() or "Unknown"
        debug_item(f"Applied class: box={self.selected_idx + 1}, class={self.box_classes[self.selected_idx]}")
        self.refresh_box_list()
        self.update_status()

    def set_box_class(self, class_name):
        if self.selected_idx is None:
            return
        self.selected_class_var.set(class_name)
        self.apply_selected_class()

    def show_class_menu(self, root_x, root_y):        
        if self.selected_idx is None or not self.class_names:
            return

        if self.class_popup is not None and self.class_popup.winfo_exists():
            self.class_popup.destroy()

        popup = tk.Toplevel(self.root)
        self.class_popup = popup
        # popup.transient(self.root)
        popup.attributes("-topmost", True)
        popup.geometry(f"220x{min(10, len(self.class_names)) * 24 + 8}+{root_x}+{root_y}")
        
        search_var = tk.StringVar()

        search_entry = ttk.Entry(popup, textvariable=search_var)
        search_entry.pack(fill=tk.X, padx=5, pady=5)

        frame = ttk.Frame(popup)
        frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        listbox = tk.Listbox(frame, 
                             bg = "#343A40",
                            fg = "#FFFFFF",
                            selectbackground = "#0E639C",
                            selectforeground = "#FFFFFF",
                            yscrollcommand=scrollbar.set)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=listbox.yview)

        def update_list(*args):
            keyword = search_var.get().lower()

            listbox.delete(0, tk.END)

            for cls in self.class_names:
                if keyword in cls.lower():
                    listbox.insert(tk.END, cls)

        def choose_class(event=None):
            selection = listbox.curselection()

            if not selection:
                return

            cls = listbox.get(selection[0])
            self.set_box_class(cls)
            popup.destroy()

        search_var.trace_add("write", update_list)

        listbox.bind("<Double-Button-1>", choose_class)
        listbox.bind("<Return>", choose_class)

        popup.bind("<Escape>", lambda e: popup.destroy())
        update_list()
        search_entry.focus_set()

    def on_canvas_right_click(self, event):
        idx, mode = self.hit_test(event.x, event.y)
        if idx is not None:
            self.selected_idx = idx
            self.drag_mode = None
            self.drag_start = None
            self.original_box = None
            self.draw_boxes()
            self.refresh_box_list()
            self.update_status()
            self.show_class_menu(event.x_root, event.y_root)
            self.right_dragging = False
            return
        # 2) 빈 공간 → pan 시작
        self.right_dragging = True
        self.canvas.scan_mark(event.x, event.y)

    def on_canvas_right_drag(self, event):
        if self.right_dragging:
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_canvas_right_release(self, event):
        self.right_dragging = False

    def save_class_names(self):
        mode.save_class_names(self.CLASSES_TXT, self.class_names)
        self.refresh_class_list()
           
    def sync_class_text_from_db(self):
        class_names = mode.sync_classes_from_db(self.CLASSES_TXT)
        if class_names is None:
            messagebox.showerror("DB Error", "Failed to load classes from db_class_info.")
            return False
        self.class_names = class_names
        self.refresh_class_list()
        return True
    
    def refresh_classes_from_text(self):
        self.class_names = self.load_class_names()
        self.refresh_class_list()

    # DB class management
    def add_db_class(self, class_name):
        status, class_names = mode.add_class(class_name, self.CLASSES_TXT)
        if status == "exists":
            messagebox.showwarning("Class exists", "Already active class name.")
            return False
        self.class_names = class_names
        self.refresh_class_list()
        return True
    
    def rename_db_class(self, old_name, new_name):
        status, class_names = mode.rename_class(old_name, new_name, self.CLASSES_TXT)
        if status == "same":
            return True
        if status == "name_taken":
            messagebox.showwarning("Class exists", "Already existing class name.")
            return False
        if status == "not_found":
            messagebox.showwarning("Class missing", "Selected class is not active in DB.")
            return False
        self.class_names = class_names
        self.refresh_class_list()
        self.box_classes = [new_name if v == old_name else v for v in self.box_classes]
        return True

    def hide_db_class(self, class_name):
        status, class_names = mode.hide_class(class_name, self.CLASSES_TXT)
        if status == "not_found":
            messagebox.showwarning("Class missing", "Selected class is not active in DB.")
            return False
        self.class_names = class_names
        self.refresh_class_list()
        fallback = self.class_names[0] if self.class_names else "Unknown"
        self.box_classes = [fallback if v == class_name else v for v in self.box_classes]
        return True

    def open_class_manager(self):
        self.refresh_classes_from_text()
        dialog = tk.Toplevel(self.root)

        dialog.title("Class Manage")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.geometry(f"420x300+{self.root.winfo_x() + self.root.winfo_width()//2 - 210}+{self.root.winfo_y() + self.root.winfo_height()//2 - 150}")
        
        dialog.focus_set()
        dialog.lift()
        dialog.bind("<Escape>", lambda e: dialog.destroy())

        list_frame = ttk.Frame(dialog)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        class_list = tk.Listbox(list_frame, borderwidth=0, exportselection=False)
        class_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=class_list.yview)
        class_list.config(yscrollcommand=class_scroll.set)
        class_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        class_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        edit_frame = ttk.Frame(dialog)
        edit_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)
        name_var = tk.StringVar()
        ttk.Entry(edit_frame, textvariable=name_var).pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        def reload_dialog_list():
            class_list.delete(0, tk.END)
            for name in self.class_names:
                class_list.insert(tk.END, name)

        def select_dialog_class(event=None):
            selection = class_list.curselection()
            if selection:
                name_var.set(class_list.get(selection[0]))

        def add_dialog_class():
            name = name_var.get().strip()
            if not name:
                return
            if self.add_db_class(name):
                self.refresh_classes_from_text()
                reload_dialog_list()

        def rename_dialog_class():
            selection = class_list.curselection()
            name = name_var.get().strip()
            if not selection or not name:
                return
            idx = selection[0]
            old_name = self.class_names[idx]
            if self.rename_db_class(old_name, name):
                self.refresh_classes_from_text()
                reload_dialog_list()
                if idx < class_list.size():
                    class_list.selection_set(idx)
                self.refresh_box_list()
                self.draw_boxes()

        def hide_dialog_class():
            selection = class_list.curselection()
            if not selection:
                return
            idx = selection[0]
            old_name = self.class_names[idx]
            if self.hide_db_class(old_name):
                self.refresh_classes_from_text()
                reload_dialog_list()
                self.refresh_box_list()
                self.draw_boxes()

        class_list.bind("<<ListboxSelect>>", select_dialog_class)
        ttk.Button(edit_frame, text="Add", command=add_dialog_class).pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Button(edit_frame, text="Rename", command=rename_dialog_class).pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Button(edit_frame, text="Delete(Hide)", command=hide_dialog_class).pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Button(edit_frame, text="Close", command=dialog.destroy).pack(side=tk.BOTTOM, fill=tk.X, pady=2)
        reload_dialog_list()

    def db_class_id_from_name(self, class_name):
        class_id = mode.get_db_class_id(class_name)
        if class_id is not None:
            return class_id
        debug_warn(f"Class not found in DB: {class_name}")
        return self.class_id_from_name(class_name)

    def update_db_status(self):
        if mode.is_db_connected():
            cfg = mode.get_db_config()
            self.db_status_var.set(f"DB: Connected ({cfg['host']}:{cfg['port']}/{cfg['dbname']})")
            self.db_connected = True
            self.btn_disconnect.config(state=tk.NORMAL)
            self.btn_set_table.config(state=tk.NORMAL)
        else:
            self.db_status_var.set("DB: Not Connected (local only)")
            self.db_connected = False
            self.btn_disconnect.config(state=tk.DISABLED)
            self.btn_set_table.config(state=tk.DISABLED)

    def open_db_connect_dialog(self):
        cfg = mode.get_db_config()
        dialog = tk.Toplevel(self.root)
        dialog.title("DB Connect")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.geometry(
            f"320x280+{self.root.winfo_x() + self.root.winfo_width()//2 - 160}+"
            f"{self.root.winfo_y() + self.root.winfo_height()//2 - 115}"
        )
        dialog.bind("<Escape>", lambda e: dialog.destroy())

        host_var = tk.StringVar(value=cfg["host"])
        user_var = tk.StringVar(value=cfg["user"])
        password_var = tk.StringVar(value=cfg["password"])
        port_var = tk.StringVar(value=str(cfg["port"]))
        dbname_var = tk.StringVar(value=cfg["dbname"])

        fields = [
            ("Host", host_var, {}),
            ("User", user_var, {}),
            ("Password", password_var, {"show": "*"}),
            ("Port", port_var, {}),
            ("DB Name", dbname_var, {}),
        ]
        for label_text, var, opts in fields:
            row = ttk.Frame(dialog)
            row.pack(side=tk.TOP, fill=tk.X, padx=10, pady=4)
            ttk.Label(row, text=label_text, width=10, anchor="w").pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, **opts).pack(side=tk.LEFT, fill=tk.X, expand=True)

        def try_connect():
            try:
                port = int(port_var.get().strip())
            except ValueError:
                messagebox.showerror("DB Connect", "Port must be a number.")
                return

            ok = mode.connect_db(
                host_var.get().strip(), user_var.get().strip(),
                password_var.get(), port, dbname_var.get().strip(),
            )
            if not ok:
                messagebox.showerror("DB Connect", "Failed to connect to the database.\nCheck the connection info and try again.")
                return
            else:
                messagebox.showinfo("DB Connect", "Successfully connected to the database!")

            self.update_db_status()
            self.sync_class_text_from_db()
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        ttk.Button(button_frame, text="Connect", command=try_connect).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        dialog.focus_set()
        dialog.lift()
    
    def disconnect_db_gui(self):
        mode.disconnect_db()
        self.db_status_var.set("DB: Not Connected (local only)")
        self.btn_connect.config(state=tk.NORMAL)
        self.btn_set_table.config(state=tk.DISABLED)
        self.btn_disconnect.config(state=tk.DISABLED)

        self.class_names = mode.sync_classes_from_db(self.CLASSES_TXT) or []
        self.refresh_box_list()
        self.refresh_class_list()

        self.root.focus_set()
        messagebox.showinfo("DB Disconnect", "Disconnected from DB. Now using Local mode.")

    def open_db_table_manager(self):
        if not mode.is_db_connected():
            return
            
        win = tk.Toplevel(self.root)
        win.title("Set DB Tables")
        win.geometry(f"400x200+{self.root.winfo_x() + self.root.winfo_width()//2 - 200}+{self.root.winfo_y() + self.root.winfo_height()//2 - 100}")
        
        # 테이블명 입력 필드
        ttk.Label(win, text="Class Info Table Name:").pack(pady=(10, 0))
        self.entry_class_table = ttk.Entry(win)
        self.entry_class_table.insert(0, mode.get_table_name('class_table')) # 기존값 불러오기
        self.entry_class_table.pack(fill="x", padx=20)
        
        ttk.Label(win, text="BBox Data Table Name:").pack(pady=(10, 0))
        self.entry_bbox_table = ttk.Entry(win)
        self.entry_bbox_table.insert(0, mode.get_table_name('bbox_table')) # 기존값 불러오기
        self.entry_bbox_table.pack(fill="x", padx=20)
        
        # 저장 버튼
        def save_table_names():
            mode.set_table_name("class_table", self.entry_class_table.get())
            mode.set_table_name("bbox_table", self.entry_bbox_table.get())
            messagebox.showinfo("Success", "Table names updated.")
            win.destroy()
            
        ttk.Button(win, text="Save", command=save_table_names).pack(pady=20)
        
    # Box drawing / coordinates
    def on_canvas_motion(self, event):
        self.draw_pointer_guides(event.x, event.y)
        self.update_canvas_cursor(event.x, event.y)
        if self.original_img is None:
            return

        x = self.canvas.canvasx(event.x) / self.scale
        y = self.canvas.canvasy(event.y) / self.scale
        over_box = False

        for (x1, y1, x2, y2) in self.boxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                over_box = True
                break

        if over_box:
            self.canvas.config(cursor="fleur")
        else:
            self.canvas.config(cursor="")

    def clear_pointer_guides(self, event=None):
        self.canvas.delete("guide")
        self.canvas.config(cursor="crosshair")

    def draw_pointer_guides(self, x, y):
        self.canvas.delete("guide")
        if self.original_img is None:
            return

        img_w, img_h = self.original_img.size
        show_w = max(1, int(img_w * self.scale))
        show_h = max(1, int(img_h * self.scale))
        guide_x = max(0, min(show_w, self.canvas.canvasx(x)))
        guide_y = max(0, min(show_h, self.canvas.canvasy(y)))

        self.canvas.create_line(guide_x, 0, guide_x, show_h, fill="cyan", dash=(4, 3), tags="guide")
        self.canvas.create_line(0, guide_y, show_w, guide_y, fill="cyan", dash=(4, 3), tags="guide")

    def update_canvas_cursor(self, x, y):
        if self.original_img is None or self.drag_mode is not None:
            return

        _, mode = self.hit_test(x, y)
        if mode == "move":
            self.canvas.config(cursor="fleur")
        else:
            self.canvas.config(cursor="crosshair")

    def draw_boxes(self):
        self.canvas.delete("box")
        self.canvas.delete("handle")

        for idx, box in enumerate(self.boxes):
            x1, y1, x2, y2 = self.to_canvas_box(box)
            color = "yellow" if idx == self.selected_idx else "lime"
            width = 3 if idx == self.selected_idx else 2
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width, tags="box")

            if idx == self.selected_idx:
                for hx, hy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                    hs = self.HANDLE_SIZE
                    self.canvas.create_rectangle(
                        hx - hs,
                        hy - hs,
                        hx + hs,
                        hy + hs,
                        fill="yellow",
                        outline="black",
                        tags="handle",
                    )

    def to_canvas_box(self, box):
        return [int(round(v * self.scale)) for v in box]

    def to_image_point(self, x, y):
        img_w, img_h = self.original_img.size
        ix = int(round(self.canvas.canvasx(x) / self.scale))
        iy = int(round(self.canvas.canvasy(y) / self.scale))
        ix = max(0, min(img_w - 1, ix))
        iy = max(0, min(img_h - 1, iy))
        return ix, iy

    def hit_test(self, x, y):
        ix, iy = self.to_image_point(x, y)
        handle_px = max(6, int(round(self.HANDLE_SIZE / self.scale)))

        for idx in reversed(range(len(self.boxes))):
            x1, y1, x2, y2 = self.boxes[idx]
            corners = {
                "nw": (x1, y1),
                "ne": (x2, y1),
                "sw": (x1, y2),
                "se": (x2, y2),
            }
            for mode, (hx, hy) in corners.items():
                if abs(ix - hx) <= handle_px and abs(iy - hy) <= handle_px:
                    return idx, mode

            if min(x1, x2) <= ix <= max(x1, x2) and min(y1, y2) <= iy <= max(y1, y2):
                return idx, "move"

        return None, "create"

    # Mouse box editing
    def on_mouse_down(self, event):
        self.canvas.focus_set()

        idx, mode = self.hit_test(event.x, event.y)
        ix, iy = self.to_image_point(event.x, event.y)
        self.selected_idx = idx
        self.drag_mode = mode
        self.drag_start = (ix, iy)
        self.original_box = list(self.boxes[idx]) if idx is not None else None

        if mode == "create":
            self.boxes.append([ix, iy, ix, iy])
            self.box_classes.append("Unknown")
            self.selected_idx = len(self.boxes) - 1
            self.original_box = list(self.boxes[self.selected_idx])
            debug_item(f"Start new box: ({ix}, {iy})")

        self.draw_boxes()
        self.draw_pointer_guides(event.x, event.y)
        self.refresh_box_list()
        self.update_status()

    def on_mouse_move(self, event):
        if self.selected_idx is None or self.drag_start is None:
            return

        ix, iy = self.to_image_point(event.x, event.y)
        sx, sy = self.drag_start
        box = list(self.original_box)

        if self.drag_mode == "move":
            dx = ix - sx
            dy = iy - sy
            box = [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]
            # box = self.clamp_box(box)
        elif self.drag_mode == "nw":
            box[0], box[1] = ix, iy
        elif self.drag_mode == "ne":
            box[2], box[1] = ix, iy
        elif self.drag_mode == "sw":
            box[0], box[3] = ix, iy
        elif self.drag_mode == "se":
            box[2], box[3] = ix, iy
        elif self.drag_mode == "create":
            box[2], box[3] = ix, iy

        self.boxes[self.selected_idx] = self.clamp_box(box)
        self.draw_boxes()
        self.draw_pointer_guides(event.x, event.y)
        self.update_class_field()

    def on_mouse_up(self, event):
        was_create = (self.drag_mode == "create")
        if self.selected_idx is not None:
            self.boxes[self.selected_idx] = self.normalize_box(self.boxes[self.selected_idx])
            debug_item(f"Box edited: {self.boxes[self.selected_idx]}")

        self.drag_mode = None
        self.drag_start = None
        self.original_box = None
        self.draw_boxes()
        self.draw_pointer_guides(event.x, event.y)
        self.refresh_box_list()
        self.update_status()
        if was_create:
            self.root.after(10, lambda: self.show_class_menu(event.x_root, event.y_root))
            # self.show_class_menu(event.x_root, event.y_root)

    def normalize_box(self, box):
        x1, y1, x2, y2 = box
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        return self.clamp_box([x1, y1, x2, y2])

    def clamp_box(self, box):
        img_w, img_h = self.original_img.size
        x1, y1, x2, y2 = box
        return [
            max(0, min(img_w - 1, x1)),
            max(0, min(img_h - 1, y1)),
            max(0, min(img_w - 1, x2)),
            max(0, min(img_h - 1, y2)),
        ]

    def delete_selected_box(self):
        if self.selected_idx is None:
            debug_warn("No selected box to delete.")
            return
        deleted = self.boxes.pop(self.selected_idx)
        if self.selected_idx < len(self.box_classes):
            self.box_classes.pop(self.selected_idx)
        debug_item(f"Deleted box: {deleted}")
        self.selected_idx = None
        self.draw_boxes()
        self.refresh_box_list()
        self.update_status()

    # Label save / image navigation
    def save_current_labels(self):
        if self.original_img is None or self.image_path is None:
            return
        class_name = None
        self.label_dir.mkdir(parents=True, exist_ok=True)
        label_path = self.label_dir / f"{self.image_path.stem}.txt"
        img_w, img_h = self.original_img.size
        boxes = [self.normalize_box(box) for box in self.boxes]
        with open(label_path, "w", encoding="utf-8") as f:
            for idx, (x1, y1, x2, y2) in enumerate(boxes):
                if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
                    continue
                class_name = self.box_classes[idx] if idx < len(self.box_classes) else self.default_class_name()
                class_id = self.class_id_from_name(class_name)
                x, y, w, h = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        debug_info(f"Saved reviewed labels: {label_path}")      
        self.save_current_bboxes_to_db(self.image_path, class_name, boxes)
        self.save_boxed_image(str(self.VISUALIZED_DIR / f"{self.image_path.stem}_draft_boxed{self.image_path.suffix}"), self.image_path, boxes)
        self.refresh_image_list()
        self.root.update_idletasks()

        self.show_status_message("Saved successfully!", duration=500)

    def save_boxed_image(self, save_path, image_path, boxes):        
        mode.save_boxed_image(save_path, image_path, boxes)

    def select_auto_label_model(self):
        confirm = messagebox.askyesno(
        "Class Synchronization",
        "Model classes will be synchronized with the database.\nContinue?"
    )
        if not confirm: return

        model_path = filedialog.askopenfilename(
            title="Select Auto Label Model",
            filetypes=[
                ("YOLO Model", "*.pt"),
                ("All Files", "*.*")
            ]
        )
        if not model_path: return

        if Path(model_path).suffix.lower() != ".pt":
            messagebox.showerror("Invalid Model", "Only YOLO model files (.pt) can be selected.")
            return

        status, model_class_names, class_names = mode.setup_auto_label_model(model_path, self.CLASSES_TXT)
        if status == "invalid_model":
            messagebox.showerror(
                "Invalid Model",
                "The selected file is not a valid YOLO model.\nThe current model will be kept."
            )
            return
        if status == "db_error":
            messagebox.showerror("DB Error", "Failed to sync classes with db_class_info.")
            return

        self.auto_model_classes = model_class_names
        self.class_names = class_names
        self.refresh_class_list()

        self.auto_label_model_path = model_path
        model_name = Path(model_path).name
        self.model_status_var.set(f"Auto-Label Model: {model_name}")
        return
    
    def auto_label(self):
        ok = messagebox.askyesno(
        "Auto Label",
        "Auto labeling will recreate draft labels for the current image list. Continue?",
        )
        if not ok:
            return
        
        results = mode.auto_label_images(self.get_current_image_dir(), self.auto_label_model_path)
        if not results:
            messagebox.showerror(
                "Auto Label",
                "No objects were detected in the selected images.\nThere is nothing to label."
            )
            return

        for item in results:
            model_cls_id = item["class_id"]
            class_name = self.auto_model_classes[model_cls_id]
            save_yolo_label(self.label_dir / f"{item['image_path'].stem}.txt", item["yolo_boxes"])
            self.save_current_bboxes_to_db(item["image_path"], class_name, item["boxes"])
            self.refresh_all()          
        messagebox.showinfo("Auto Label", "Auto labeling is complete.")
   
    def save_current_bboxes_to_db(self, image_path, class_name, boxes):
        class_id = self.db_class_id_from_name(class_name)
        mode.save_bboxes_to_db(image_path, class_id, boxes)

    def prev_image(self):
        self.save_current_labels()
        if self.image_idx > 0:
            self.image_idx -= 1
            self.load_current_image()

    def next_image(self):
        self.save_current_labels()
        if self.image_idx < len(self.image_files) - 1:
            self.image_idx += 1
            self.load_current_image()
        else:
            messagebox.showinfo("Review complete", "Last image reviewed.")

def open_gui(label_dir):
    """Open a GUI to adjust bbox labels and save edited YOLO txt files."""
    image_files = mode.load_imgs(TOOL_DIR / "input")
    if not image_files:
        messagebox.showinfo("BBox Creator", "No images to open in review GUI.")

    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    debug_info("Opening bbox review GUI.")
    debug_item("Drag empty area to create a new box.")
    debug_item("Drag inside a box to move it.")
    debug_item("Drag yellow corners to resize it.")
    debug_item("Ctrl+S saves, Delete removes selected box, Left/Right changes image.")

    root = ttk.Window(themename="darkly")
    BBoxReviewApp(root, image_files=image_files, label_dir=label_dir)
    root.mainloop()
