import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import os
import threading
import time
import datetime
import logging
import socket
import shutil
import concurrent.futures # <--- Parallel Processing
from typing import List, Tuple, Optional
from PIL import Image, ImageTk
import qrcode # <--- New import for QR generation

# Import our image processing classes
from classic_image_processor import ClassicImageProcessor
from advanced_image_processor import AdvancedImageProcessor
from tiered_contour_processor import TieredContourProcessor
from sharp_detail_processor import SharpDetailProcessor
from fast_eye_tier_processor import FastEyeTierProcessor
from smart_auto_eye_processor import SmartAutoEyeProcessor
from real_image_drawing_processor import RealImageDrawingProcessor
from smooth_auto_eye_processor import SmoothAutoEyeProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Constants (Consolidated) ---
SCRIPT_DIR = os.getenv("SCRIPT_DIR", ".")
DATA_DIR = os.getenv("DATA_DIR", ".")

TMP_CAPTURE_PATH = os.path.join(DATA_DIR, "temp_capture.png")
TMP_EDGE_OUTPUT_PATH = os.path.join(DATA_DIR, "temp_edges_{}.png")

REAL_ROBOT_HOST = '192.168.125.1'
REAL_ROBOT_PORT = 1025
SIMULATION_HOST = '127.0.0.1'
SIMULATION_PORT = 55000

# Drawing Specific Constants
FINAL_ROBOT_POSITION = (0, -120, 0) # Use X, Z, Y format (X, Depth, Y) - NOTE: Z is depth here
ROBOT_HOME_POSITION = (300, -350.922061873, 300) # Use X, Z, Y format

A4_WIDTH_MM = 170  # Drawing area width
A4_HEIGHT_MM = 207 # Drawing area height
DEFAULT_PEN_DOWN_Z = -10   # Default pen down position (depth)

MIN_CONTOUR_LENGTH_PX = 30 # Minimum contour length in pixels to consider

# Threshold options for Canny edge detection
THRESHOLD_OPTIONS = [
    ("Option {}".format(i), i*10, i*20) for i in range(1, 8)
]

# Time estimation factor (seconds per command)
TIME_ESTIMATE_FACTOR = 0.02

class RUNME_GUI:
    """Main GUI application for the Robotics System."""

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Robotics Drawing GUI")
        self.main_frame = tk.Frame(self.window)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # --- History & Queue Setup ---
        self.history_dir = os.path.join(DATA_DIR, "temp_history")
        if os.path.exists(self.history_dir):
            shutil.rmtree(self.history_dir, ignore_errors=True)
        os.makedirs(self.history_dir, exist_ok=True)
        self.image_queue = []
        
        # --- Background QR Server ---
        self.tunnel_url = None
        threading.Thread(target=self.start_qr_server_bg, daemon=True).start()

        # Initialize Image Processors
        self.classic_processor = ClassicImageProcessor(A4_WIDTH_MM, A4_HEIGHT_MM, MIN_CONTOUR_LENGTH_PX)
        self.advanced_processor = AdvancedImageProcessor(A4_WIDTH_MM, A4_HEIGHT_MM, MIN_CONTOUR_LENGTH_PX)
        self.tiered_processor = TieredContourProcessor(A4_WIDTH_MM, A4_HEIGHT_MM, MIN_CONTOUR_LENGTH_PX)
        self.sharp_processor = SharpDetailProcessor(A4_WIDTH_MM, A4_HEIGHT_MM, MIN_CONTOUR_LENGTH_PX)
        self.fast_eye_processor = FastEyeTierProcessor(A4_WIDTH_MM, A4_HEIGHT_MM, MIN_CONTOUR_LENGTH_PX)
        self.smart_auto_processor = SmartAutoEyeProcessor(A4_WIDTH_MM, A4_HEIGHT_MM, MIN_CONTOUR_LENGTH_PX)
        self.real_image_processor = RealImageDrawingProcessor(A4_WIDTH_MM, A4_HEIGHT_MM, MIN_CONTOUR_LENGTH_PX)
        self.smooth_eye_processor = SmoothAutoEyeProcessor(A4_WIDTH_MM, A4_HEIGHT_MM, MIN_CONTOUR_LENGTH_PX) 

        # Processing Mode Variables
        self.processing_mode_var = tk.StringVar(value="classic")
        self.tier_var = tk.IntVar(value=20)

        # Connection related variables
        self.connection_var = tk.StringVar(value="simulation")
        self.socket = None
        self.connected = False
        self.connection_established = False
        self.testing_mode = False  

        # Drawing process related variables
        self.image_path_var = tk.StringVar() 
        self.current_image_path = None
        self.threshold_options_data = {}
        self.edge_preview_paths = {}
        self.selected_commands = None
        self.drawing_in_progress = False
        self.cancel_requested = False
        self.progress_bar = None
        self.status_label = None
        self.cancel_button = None
        self.reconnect_button = None
        self.test_mode_button = None 
        
        # Interactive state
        self.user_eye_points = []
        
        # Image TK references to prevent garbage collection
        self.orig_imgtk = None
        self.proc_imgtk = None
        self.history_thumbnails = [] # To hold filmstrip thumbnails
        self.is_on_input_page = False # Tracking for auto-refresh

        # Packing Checklist variables
        self.pack_check_1 = tk.BooleanVar()
        self.pack_check_2 = tk.BooleanVar()
        self.pack_check_3 = tk.BooleanVar()

        # Pen position and control variables
        self.pen_down_z_var = tk.StringVar(value=str(DEFAULT_PEN_DOWN_Z))
        self.safe_center_z_var = tk.StringVar(value=str(-120.0))
        self.pause_event = threading.Event()
        self.pause_resume_button = None

        # ETA Countdown variables
        self.eta_update_id = None
        self.drawing_start_time = 0
        self.total_paused_time = 0
        self.pause_start_time = 0
        self.progress_text_var = tk.StringVar()

        # Status tracking for previous drawing attempts
        self.last_drawing_status = {
            "total_commands": 0, "completed_commands": 0, "status": "None", "error_message": ""
        }
        
        # Resume-related variables
        self.resume_needed = False
        self.resume_commands = None
        self.resume_start_index_global = 0

        # Start the application
        self.main_page()

    # --- QR Server & Background Handling ---
    def start_qr_server_bg(self):
        """Starts the Flask server and Cloudflare tunnel silently on app launch."""
        try:
            from qr_upload_server import start_server_and_tunnel
            url, proc = start_server_and_tunnel(self.on_qr_image_received)
            if url:
                self.tunnel_url = url
                logging.info(f"QR Background Server live at: {url}")
            else:
                logging.error("Failed to generate Cloudflare Tunnel URL. Check if cloudflared is installed.")
        except Exception as e:
            logging.error(f"Error starting QR server: {e}")

    def show_qr_popup(self):
        """Universal popup window to show the QR code without leaving the current page."""
        popup = tk.Toplevel(self.window)
        popup.title("Scan to Upload")
        popup.geometry("350x400")
        popup.attributes("-topmost", True) 
        
        if not self.tunnel_url:
            tk.Label(popup, text="Server is still starting...\nPlease check your internet or cloudflared installation.", fg="orange").pack(pady=50)
            tk.Button(popup, text="Close", command=popup.destroy).pack(pady=10)
            return
            
        tk.Label(popup, text="Scan with phone to upload to Queue:", font=("Arial", 11, "bold")).pack(pady=10)
        
        qr = qrcode.QRCode(box_size=8, border=2)
        qr.add_data(self.tunnel_url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        
        popup.qr_imgtk = ImageTk.PhotoImage(image=img)
        tk.Label(popup, image=popup.qr_imgtk).pack(pady=10)

        tk.Button(popup, text="Close", command=popup.destroy, width=15).pack(pady=10)

    def on_qr_image_received(self, filepath):
        """Silently queues uploaded images from the server."""
        def handle_upload():
            hist_path = self._copy_to_history(filepath)
            self.image_queue.append(hist_path)
            logging.info(f"Image received and queued: {hist_path}")
            
            # Update UI elements dynamically if they are currently rendered on screen
            if hasattr(self, 'queue_notification_label') and self.queue_notification_label.winfo_exists():
                self.queue_notification_label.config(text=f"New images in queue: {len(self.image_queue)}")
            
            # Auto-refresh the visual gallery if the user is on the input page
            if self.is_on_input_page:
                self.input_image_page()
                
        self.window.after(0, handle_upload)

    def _copy_to_history(self, filepath):
        """Copies an image into the history directory with a timestamp."""
        if not filepath or not os.path.exists(filepath): 
            return filepath
            
        ext = os.path.splitext(filepath)[1]
        if not ext: ext = ".png"
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}{ext}"
        new_path = os.path.join(self.history_dir, filename)
        
        shutil.copy2(filepath, new_path)
        return new_path

    # --- Page Navigation ---
    def main_page(self):
        self.clear_frame()
        tk.Label(self.main_frame, text="Robotics Drawing System", font=("Arial", 16)).pack(pady=10)
        tk.Button(self.main_frame, text="Setup Connection & Draw", command=self.connection_setup_page, width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Exit", command=self.on_window_close, width=30).pack(pady=5)

    def connection_setup_page(self):
        self.clear_frame()
        tk.Label(self.main_frame, text="Robot Connection Setup", font=("Arial", 16)).pack(pady=10)

        connection_frame = tk.Frame(self.main_frame)
        connection_frame.pack(pady=10)
        tk.Radiobutton(connection_frame, text=f"Simulation: {SIMULATION_HOST}:{SIMULATION_PORT}", variable=self.connection_var, value="simulation").pack(anchor='w')
        tk.Radiobutton(connection_frame, text=f"Real Robot: {REAL_ROBOT_HOST}:{REAL_ROBOT_PORT}", variable=self.connection_var, value="real").pack(anchor='w')

        self.connect_button = tk.Button(self.main_frame, text="Connect", command=self.establish_connection, width=20)
        self.reconnect_button = tk.Button(self.main_frame, text="Reconnect & Resume", command=self.establish_connection, width=20)
        
        self.test_mode_button = tk.Button(self.main_frame, text="Testing Mode (Offline)", command=self.start_testing_mode, width=20, bg="#e6e6e6")

        if self.resume_needed:
            self.reconnect_button.pack(pady=5)
            tk.Label(self.main_frame, text="Connection lost during last drawing. Reconnect to resume.", fg="orange").pack()
        else:
            self.connect_button.pack(pady=5)
            self.test_mode_button.pack(pady=15)

        tk.Button(self.main_frame, text="Back", command=self.main_page, width=20).pack(pady=5)

    def start_testing_mode(self):
        logging.info("Starting Offline Testing Mode.")
        self.testing_mode = True
        self.connection_established = True
        self.connected = False
        self.drawing_options_page()

    def drawing_options_page(self):
        if not self.connection_established:
            messagebox.showerror("Connection Required", "Please establish connection first.")
            self.connection_setup_page()
            return

        self.clear_frame()
        tk.Label(self.main_frame, text="Robot Drawing Options", font=("Arial", 16)).pack(pady=10)
        
        if self.testing_mode:
            conn_type = "Offline Testing Mode"
            fg_color = "orange"
        else:
            conn_type = "Simulation" if self.connection_var.get() == "simulation" else "Real Robot"
            fg_color = "green"
            
        tk.Label(self.main_frame, text=f"Connected to: {conn_type}", fg=fg_color).pack(pady=5)
        
        last_status = self.last_drawing_status["status"]
        if last_status not in ["None", "Completed"]:
            status_frame = tk.Frame(self.main_frame, relief=tk.RIDGE, borderwidth=2)
            status_frame.pack(pady=10, padx=10, fill='x')
            tk.Label(status_frame, text="Previous Drawing Status:", font=("Arial", 10, "bold")).pack(anchor='w')
            status_text = f"Status: {last_status}"
            if self.last_drawing_status["total_commands"] > 0:
                status_text += f" (Stopped at command {self.last_drawing_status['completed_commands'] + 1} of {self.last_drawing_status['total_commands']})"
            tk.Label(status_frame, text=status_text).pack(anchor='w', padx=5)

        controls_frame = tk.Frame(self.main_frame, pady=5, relief=tk.GROOVE, borderwidth=2)
        controls_frame.pack(pady=10, padx=10, fill='x')
        
        tk.Label(controls_frame, text="Testing & Calibration Controls", font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=3, pady=5)

        tk.Label(controls_frame, text="Pen Down Z (for drawing):").grid(row=1, column=0, sticky='w', padx=5)
        tk.Entry(controls_frame, textvariable=self.pen_down_z_var, width=10).grid(row=1, column=1, padx=5)
        self.send_z_button = tk.Button(controls_frame, text="Test at (0, 0, Z)", command=self.send_to_test_z_action)
        self.send_z_button.grid(row=1, column=2, padx=10)

        tk.Label(controls_frame, text="Safe Center Z:").grid(row=2, column=0, sticky='w', padx=5)
        tk.Entry(controls_frame, textvariable=self.safe_center_z_var, width=10).grid(row=2, column=1, padx=5)
        self.safe_center_button = tk.Button(controls_frame, text="Go to Safe Center", command=self.send_to_safe_center_action)
        self.safe_center_button.grid(row=2, column=2, padx=10)

        self.test_workspace_button = tk.Button(controls_frame, text="Test Workspace Area", command=self.test_workspace_action)
        self.test_workspace_button.grid(row=3, column=0, columnspan=3, pady=5)

        self.go_home_button = tk.Button(controls_frame, text="Go Home", command=self.go_home_action)
        self.go_home_button.grid(row=4, column=0, columnspan=3, pady=5)

        self.pack_button = tk.Button(controls_frame, text="Packing Position", command=self.packing_checklist_page, bg="#FFCCCB")
        self.pack_button.grid(row=5, column=0, columnspan=3, pady=5)

        # --- Main Navigation ---
        tk.Button(self.main_frame, text="Show QR Upload Code", command=self.show_qr_popup, width=30, bg="#d4edda").pack(pady=5)
        tk.Button(self.main_frame, text="Input Image to Draw", command=self.input_image_page, width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Disconnect", command=self.close_and_return_main, width=30).pack(pady=5)

    # --- Packing Position Workflow ---
    def packing_checklist_page(self):
        self.clear_frame()
        tk.Label(self.main_frame, text="Packing Position Checklist", font=("Arial", 16, "bold"), fg="red").pack(pady=10)
        tk.Label(self.main_frame, text="Warning: This moves the robot to shipping angles.", font=("Arial", 10)).pack(pady=5)

        checklist_frame = tk.Frame(self.main_frame, relief=tk.SUNKEN, borderwidth=1, padx=20, pady=20)
        checklist_frame.pack(pady=10)

        self.pack_check_1.set(False)
        self.pack_check_2.set(False)
        self.pack_check_3.set(False)

        c1 = tk.Checkbutton(checklist_frame, text="1. All tools have been removed.", variable=self.pack_check_1, command=self.check_packing_conditions)
        c1.pack(anchor='w', pady=5)
        c2 = tk.Checkbutton(checklist_frame, text="2. I know what I am doing (MoveAbsJ).", variable=self.pack_check_2, command=self.check_packing_conditions)
        c2.pack(anchor='w', pady=5)
        c3 = tk.Checkbutton(checklist_frame, text="3. I am ready to pack.", variable=self.pack_check_3, command=self.check_packing_conditions)
        c3.pack(anchor='w', pady=5)

        self.confirm_pack_button = tk.Button(self.main_frame, text="Confirm & Pack", command=self.execute_packing_sequence, state=tk.DISABLED, bg="#ff9999", width=20)
        self.confirm_pack_button.pack(pady=20)

        tk.Button(self.main_frame, text="Back", command=self.drawing_options_page, width=20).pack(pady=5)

    def check_packing_conditions(self):
        if self.pack_check_1.get() and self.pack_check_2.get() and self.pack_check_3.get():
            self.confirm_pack_button.config(state=tk.NORMAL)
        else:
            self.confirm_pack_button.config(state=tk.DISABLED)

    def execute_packing_sequence(self):
        if hasattr(self, 'confirm_pack_button') and self.confirm_pack_button.winfo_exists():
            self.confirm_pack_button.config(state=tk.DISABLED, text="Sending...")
        threading.Thread(target=self._send_packing_thread, daemon=True).start()

    def _send_packing_thread(self):
        if self.send_message_internal("PACK"):
            response = self.receive_message_internal(timeout=10.0)
            if response == "R":
                self.window.after(0, lambda: messagebox.showinfo("Packing", "Command sent. Robot should be moving to packing position."))
            else:
                self.window.after(0, lambda: messagebox.showerror("Error", f"Robot did not confirm packing command. Got: {response}"))
        else:
            self.window.after(0, lambda: messagebox.showerror("Connection Error", "Failed to send packing command."))
        
        self.window.after(0, self.drawing_options_page)

    # --- Other Action Methods ---
    def send_to_test_z_action(self):
        try:
            test_z = float(self.pen_down_z_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "The Pen Down Z-coordinate must be a valid number.")
            return

        if hasattr(self, 'send_z_button') and self.send_z_button.winfo_exists():
            self.send_z_button.config(state=tk.DISABLED)
        threading.Thread(target=self._send_command_sequence_thread, args=([(0.0, test_z, 0.0)], self.send_z_button), daemon=True).start()

    def send_to_safe_center_action(self):
        try:
            safe_z = float(self.safe_center_z_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "The Safe Center Z-coordinate must be a valid number.")
            return
        
        if hasattr(self, 'safe_center_button') and self.safe_center_button.winfo_exists():
            self.safe_center_button.config(state=tk.DISABLED)
        
        threading.Thread(target=self._send_command_sequence_thread, args=([(0, safe_z, 0)], self.safe_center_button), daemon=True).start()

    def go_home_action(self):
        if hasattr(self, 'go_home_button') and self.go_home_button.winfo_exists():
            self.go_home_button.config(state=tk.DISABLED)
        threading.Thread(target=self._send_command_sequence_thread, args=([ROBOT_HOME_POSITION], self.go_home_button), daemon=True).start()

    def test_workspace_action(self):
        try:
            test_z = float(self.pen_down_z_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "The Z-coordinate for testing must be a valid number.")
            return
        
        pen_up_z =  test_z / 10 if test_z > 0 else test_z * 2.0

        if hasattr(self, 'test_workspace_button') and self.test_workspace_button.winfo_exists():
            self.test_workspace_button.config(state=tk.DISABLED)
        
        w = A4_WIDTH_MM / 3
        h = A4_HEIGHT_MM / 3
        workspace_path = [
            (w, pen_up_z, h),   (w, test_z, h),     (w, test_z, -h),    
            (-w, test_z, -h),   (-w, test_z, h),    (w, test_z, h),     
            (0, pen_up_z, 0)
        ]
        
        threading.Thread(target=self._send_command_sequence_thread, args=(workspace_path, self.test_workspace_button), daemon=True).start()

    def _send_command_sequence_thread(self, commands: List[Tuple], button_to_re_enable: tk.Button):
        original_text = button_to_re_enable.cget("text")
        self.window.after(0, lambda: button_to_re_enable.config(text="Moving..."))

        for i, (x, z, y) in enumerate(commands):
            if self.cancel_requested:
                break
            
            command_str = f"{x:.2f},{z:.2f},{y:.2f}"
            if self.send_message_internal(command_str):
                response_r = self.receive_message_internal(timeout=10.0)
                if response_r != "R":
                    error_msg = f"Robot did not confirm receipt (R) for command {i+1}. Got: '{response_r}'"
                    self.window.after(0, lambda: messagebox.showerror("Test Failed", error_msg))
                    break
            else:
                self.window.after(0, lambda: messagebox.showerror("Connection Error", "Failed to send test command. Connection may be lost."))
                break
        
        if button_to_re_enable and button_to_re_enable.winfo_exists():
            self.window.after(0, lambda: button_to_re_enable.config(state=tk.NORMAL, text=original_text))

    # --- Input Image Workflow ---
    def input_image_page(self):
        self.clear_frame()
        self.is_on_input_page = True # Flag for auto-refresh
        
        tk.Label(self.main_frame, text="Input Image to Draw", font=("Arial", 16)).pack(pady=10)

        entry_frame = tk.Frame(self.main_frame)
        entry_frame.pack(pady=5, fill='x', padx=10)
        tk.Label(entry_frame, text="Active Image:").pack(side=tk.LEFT)
        
        path_entry = tk.Entry(entry_frame, textvariable=self.image_path_var, width=50)
        path_entry.pack(side=tk.LEFT, fill='x', expand=True, padx=5)
        tk.Button(entry_frame, text="Browse...", command=self.browse_image_file).pack(side=tk.LEFT)

        # --- Rendered Visual History / Queue Gallery ---
        history_frame = tk.Frame(self.main_frame, pady=10, relief=tk.SUNKEN, borderwidth=1)
        history_frame.pack(fill='x', padx=10)
        
        header_frame = tk.Frame(history_frame)
        header_frame.pack(fill='x', padx=5, pady=5)
        tk.Label(header_frame, text="Image Queue & History (Click image to select):", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        tk.Button(header_frame, text="Show QR Upload Code", command=self.show_qr_popup, bg="#d4edda", padx=10).pack(side=tk.RIGHT)
        
        # Scrollable Canvas for Thumbnails
        canvas_frame = tk.Frame(history_frame, height=150)
        canvas_frame.pack(fill='x', expand=True, padx=5, pady=5)
        canvas_frame.pack_propagate(False) # Keep height constrained
        
        canvas = tk.Canvas(canvas_frame, bg="#f9f9f9", highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="horizontal", command=canvas.xview)
        scrollable_frame = tk.Frame(canvas, bg="#f9f9f9")
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(xscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        def load_selected_from_history(full_path):
            if full_path in self.image_queue:
                self.image_queue.remove(full_path)
            self.image_path_var.set(full_path)
            self.current_image_path = full_path
            self.user_eye_points = []
            self.input_image_page() # Refresh UI to remove queued tags

        if os.path.exists(self.history_dir):
            hist_files = sorted(os.listdir(self.history_dir), reverse=True) # Newest first
            for f in hist_files:
                full_path = os.path.join(self.history_dir, f)
                try:
                    img = Image.open(full_path)
                    img.thumbnail((100, 100))
                    photo = ImageTk.PhotoImage(img)
                    self.history_thumbnails.append(photo)
                    
                    item_frame = tk.Frame(scrollable_frame, bg="#f9f9f9", padx=5)
                    item_frame.pack(side=tk.LEFT, fill=tk.Y)
                    
                    # Apply styling for Queued vs History
                    is_queued = full_path in self.image_queue
                    border_color = "red" if is_queued else "gray"
                    border_width = 3 if is_queued else 1
                    
                    btn = tk.Button(item_frame, image=photo, command=lambda p=full_path: load_selected_from_history(p), 
                                    relief=tk.SOLID, bd=border_width, activebackground=border_color)
                    btn.pack()
                    
                    if is_queued:
                        tk.Label(item_frame, text="QUEUED", fg="red", font=("Arial", 8, "bold"), bg="#f9f9f9").pack()
                    else:
                        tk.Label(item_frame, text="History", fg="gray", font=("Arial", 8), bg="#f9f9f9").pack()
                        
                except Exception as e:
                    logging.error(f"Error loading thumbnail for {f}: {e}")

        # --- Modes ---
        mode_frame = tk.Frame(self.main_frame, pady=10)
        mode_frame.pack()
        tk.Label(mode_frame, text="Select Algorithm Mode:", font=("Arial", 10, "bold")).pack()
        tk.Radiobutton(mode_frame, text="Mode 1: Classic (Outlines Only)", variable=self.processing_mode_var, value="classic").pack(anchor='w')
        tk.Radiobutton(mode_frame, text="Mode 2: Advanced (Outlines + Shaded Hatching)", variable=self.processing_mode_var, value="advanced").pack(anchor='w')
        tk.Radiobutton(mode_frame, text="Mode 3: Topographical (Volume + Edge Details)", variable=self.processing_mode_var, value="tiered").pack(anchor='w')
        tk.Radiobutton(mode_frame, text="Mode 4: Raw Detail (No Blur)", variable=self.processing_mode_var, value="sharp").pack(anchor='w')
        tk.Radiobutton(mode_frame, text="Mode 5: Fast Tiered (Interactive Eye Fill)", variable=self.processing_mode_var, value="fast_eye").pack(anchor='w')        
        tk.Radiobutton(mode_frame, text="Mode 6: Smart Auto-Pair Eye Fill + Interactive", variable=self.processing_mode_var, value="smart_auto").pack(anchor='w')
        tk.Radiobutton(mode_frame, text="Mode 7: Real Image Shortcut (60-Tier Medium Pass)", variable=self.processing_mode_var, value="real_image").pack(anchor='w')
        tk.Radiobutton(mode_frame, text="Mode 8: Enhanced Smooth Auto-Eye (Adjustable + Dense Fill)", variable=self.processing_mode_var, value="smooth_eye").pack(anchor='w')

        # Slider specifically for Modes 3, 5, 6, 8
        tier_frame = tk.Frame(mode_frame)
        tier_frame.pack(pady=5)
        tk.Label(tier_frame, text="Number of Tiers (Modes 3, 5, 6 & 8):").pack(side=tk.LEFT)
        tk.Scale(tier_frame, variable=self.tier_var, from_=2, to=60, orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT, padx=10)

        tk.Button(self.main_frame, text="Process Image", command=self.process_input_image, width=20, bg="#cce5ff").pack(pady=10)
        tk.Button(self.main_frame, text="Back", command=self.drawing_options_page, width=20).pack(pady=10)

    def browse_image_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Image to Draw", 
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All Files", "*.*")]
        )
        if filepath:
            hist_path = self._copy_to_history(filepath)
            self.image_path_var.set(hist_path)
            self.current_image_path = hist_path

    def process_input_image(self):
        filepath = self.image_path_var.get()
        if not filepath or not os.path.isfile(filepath):
            messagebox.showerror("Error", f"Invalid or non-existent file path:\n{filepath}")
            return
        
        self.current_image_path = filepath
        self.user_eye_points = [] 
        self.show_threshold_options(self.current_image_path)

    # --- Threshold Selection Workflow ---
    def show_threshold_options(self, image_path):
        self.clear_frame()
        
        mode = self.processing_mode_var.get()
        
        mode_names = {
            "classic": "Classic Mode", 
            "advanced": "Advanced Mode", 
            "tiered": "Topographical Mode", 
            "sharp": "Raw Detail Mode",
            "fast_eye": "Fast Tiered Mode", 
            "smart_auto": "Smart Auto-Eye Mode",
            "real_image": "Real Image Shortcut Mode",
            "smooth_eye": "Enhanced Smooth Auto-Eye Mode"
        }
        mode_str = mode_names.get(mode, mode)

        tk.Label(self.main_frame, text=f"Select Drawing Style ({mode_str})", font=("Arial", 16)).pack(pady=10)

        try:
            pen_down_z = float(self.pen_down_z_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "The Pen Down Z-coordinate must be a valid number.")
            return

        self.threshold_options_data = {}
        self.selected_threshold_option = tk.StringVar(value=None)
        
        content_frame = tk.Frame(self.main_frame)
        content_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, padx=20, fill=tk.BOTH, expand=True)
        tk.Label(left_frame, text="Preview", font=("Arial", 12, "bold")).pack(pady=5)
        
        self.preview_label = tk.Label(left_frame, cursor="crosshair" if mode in ["fast_eye", "smart_auto", "real_image", "smooth_eye"] else "arrow")
        self.preview_label.pack(pady=5)

        right_frame = tk.Frame(content_frame)
        right_frame.pack(side=tk.LEFT, padx=20, fill=tk.BOTH, expand=True)
        tk.Label(right_frame, text="Options", font=("Arial", 12, "bold")).pack(pady=5)

        options_frame = tk.Frame(right_frame)
        options_frame.pack(pady=5, fill=tk.BOTH, expand=True)

        if mode in ["fast_eye", "smart_auto", "real_image", "smooth_eye"]:
            instruction_frame = tk.Frame(options_frame)
            instruction_frame.pack(pady=5)
            tk.Label(instruction_frame, text="💡 Click on the PREVIEW IMAGE to fill eyes.\n(Candidate areas are outlined in pink)", fg="blue", justify=tk.CENTER).pack()
            if mode in ["smart_auto", "real_image", "smooth_eye"]:
                tk.Label(instruction_frame, text="(Green dots = Auto-detected, Blue dots = Manual clicks)", fg="green").pack()
            tk.Button(instruction_frame, text="Clear Selections", command=self.clear_eye_selections).pack(pady=5)

        loading_label = tk.Label(options_frame, text="Processing options in parallel...\nThis may take a moment.")
        loading_label.pack()
        self.window.update()

        threading.Thread(
            target=self._process_threshold_options_thread, 
            args=(image_path, options_frame, loading_label, pen_down_z), 
            daemon=True
        ).start()

    def clear_eye_selections(self):
        """Clears user clicks and regenerates the previews"""
        self.user_eye_points = []
        self.show_threshold_options(self.current_image_path)

    def on_preview_click(self, event):
        """Handles user clicking the preview image to add eye fill targets"""
        if self.processing_mode_var.get() not in ["fast_eye", "smart_auto", "real_image", "smooth_eye"]: 
            return
        if not hasattr(self, 'preview_thumb_size'): 
            return
            
        x, y = event.x, event.y
        thumb_w, thumb_h = self.preview_thumb_size
        orig_w, orig_h = self.preview_orig_size
        
        if x > thumb_w or y > thumb_h: 
            return
            
        orig_x = int((x / thumb_w) * orig_w)
        orig_y = int((y / thumb_h) * orig_h)
        
        self.user_eye_points.append((orig_x, orig_y))
        logging.info(f"Added eye fill target at: {orig_x}, {orig_y}")
        
        self.show_threshold_options(self.current_image_path)

    def _process_threshold_options_thread(self, image_path, options_frame, loading_label, pen_down_z):
        results = {}
        preview_paths = {}
        mode = self.processing_mode_var.get()

        options_to_run = []
        if mode in ["classic", "advanced", "sharp"]:
            options_to_run = THRESHOLD_OPTIONS 
        elif mode == "real_image":
            options_to_run.append(("Optimal Real Image (60 Tiers, Medium Pass)", 0, 0))
        elif mode == "smooth_eye":
            base_tiers = self.tier_var.get()
            options_to_run.append((f"{base_tiers} Tiers (Smooth Detail - Lightest 1)", base_tiers, 1))
            options_to_run.append((f"{base_tiers} Tiers (Smooth Detail - Lightest 2)", base_tiers, 2))
            options_to_run.append((f"{base_tiers} Tiers (Smooth Detail - Lightest 3)", base_tiers, 3))
            options_to_run.append((f"{base_tiers} Tiers (Smooth Detail - Lightest 4)", base_tiers, 4))
            options_to_run.append((f"{base_tiers} Tiers (Smooth Detail - Lightest 5)", base_tiers, 5))
            options_to_run.append((f"{base_tiers} Tiers (Smooth Detail - Light)", base_tiers, 6))
            options_to_run.append((f"{base_tiers} Tiers (Smooth Detail - Medium)", base_tiers, 7))
            options_to_run.append((f"{base_tiers} Tiers (Smooth Detail - Heavy)", base_tiers, 8))
            options_to_run.append((f"{base_tiers} Tiers (Smooth Detail - Heavier 1)", base_tiers, 9))
            options_to_run.append((f"{base_tiers} Tiers (Smooth Detail - Heavier 2)", base_tiers, 10))
            options_to_run.append((f"{base_tiers} Tiers (Smooth Detail - Heavier 3)", base_tiers, 11))
            options_to_run.append((f"{base_tiers} Tiers (Smooth Detail - Heavier 4)", base_tiers, 12))
            options_to_run.append((f"{base_tiers} Tiers (Smooth Detail - Maximum)", base_tiers, 13))
        else:
            base_tiers = self.tier_var.get()
            options_to_run.append((f"{base_tiers} Tiers (High Detail Edge Pass)", base_tiers, 1))
            options_to_run.append((f"{base_tiers} Tiers (Medium Detail Edge Pass)", base_tiers, 2))
            options_to_run.append((f"{base_tiers} Tiers (Low Detail Edge Pass)", base_tiers, 3))

        def _run_processor_for_option(i, label, t1, t2):
            logging.info(f"Processing option: {label}")
            preview_path = TMP_EDGE_OUTPUT_PATH.format(i)
            commands = []
            
            if mode == "advanced":
                contours_xy, w, h = self.advanced_processor.image_to_contours_and_hatching(image_path, t1, t2, save_edge_path=preview_path)
                if contours_xy: commands = self.advanced_processor.create_drawing_paths(contours_xy, w, h, pen_down_z, optimize_paths=True)
            elif mode == "tiered":
                contours_xy, w, h = self.tiered_processor.image_to_tiered_contours(image_path, t1, t2, save_edge_path=preview_path)
                if contours_xy: commands = self.tiered_processor.create_drawing_paths(contours_xy, w, h, pen_down_z, optimize_paths=True)
            elif mode == "sharp":
                contours_xy, w, h = self.sharp_processor.image_to_contours(image_path, t1, t2, save_edge_path=preview_path)
                if contours_xy: commands = self.sharp_processor.create_drawing_paths(contours_xy, w, h, pen_down_z, optimize_paths=True)
            elif mode == "fast_eye":
                contours_xy, w, h = self.fast_eye_processor.image_to_fast_eye_tiers(
                    image_path, t1, t2, save_edge_path=preview_path, user_eye_points=self.user_eye_points)
                if contours_xy: commands = self.fast_eye_processor.create_drawing_paths(contours_xy, w, h, pen_down_z, optimize_paths=True)                
            elif mode == "smart_auto":
                contours_xy, w, h = self.smart_auto_processor.image_to_smart_auto_tiers(
                    image_path, t1, t2, save_edge_path=preview_path, user_eye_points=self.user_eye_points)
                if contours_xy: commands = self.smart_auto_processor.create_drawing_paths(contours_xy, w, h, pen_down_z, optimize_paths=True)
            elif mode == "real_image":
                contours_xy, w, h = self.real_image_processor.image_to_real_image_drawing(
                    image_path, save_edge_path=preview_path, user_eye_points=self.user_eye_points)
                if contours_xy: commands = self.real_image_processor.create_drawing_paths(contours_xy, w, h, pen_down_z, optimize_paths=True)
            elif mode == "smooth_eye": 
                contours_xy, w, h = self.smooth_eye_processor.image_to_smooth_auto_tiers(
                    image_path, t1, t2, save_edge_path=preview_path, user_eye_points=self.user_eye_points)
                if contours_xy: commands = self.smooth_eye_processor.create_drawing_paths(contours_xy, w, h, pen_down_z, optimize_paths=True)
            else:
                contours_xy, w, h = self.classic_processor.image_to_contours(image_path, t1, t2, save_edge_path=preview_path)
                if contours_xy: commands = self.classic_processor.create_drawing_paths(contours_xy, w, h, pen_down_z, optimize_paths=True)

            result_data = None
            if commands:
                num_commands = len(commands)
                est_time_sec = num_commands * TIME_ESTIMATE_FACTOR
                est_time_min = est_time_sec / 60
                result_data = {
                    "commands": commands, 
                    "count": num_commands,
                    "time_str": f"{est_time_min:.1f} min"
                }
            else:
                logging.warning(f"No commands generated for option {label}")
            
            valid_preview = preview_path if os.path.exists(preview_path) else None
            return label, result_data, valid_preview

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(_run_processor_for_option, i, label, t1, t2) for i, (label, t1, t2) in enumerate(options_to_run)]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    label, result_data, valid_preview = future.result()
                    results[label] = result_data
                    preview_paths[label] = valid_preview
                except Exception as e:
                    logging.error(f"Error executing future: {e}")

        self.window.after(0, lambda: self._display_threshold_options(options_frame, loading_label, results, preview_paths, options_to_run))

    def _display_threshold_options(self, options_frame, loading_label, results, preview_paths, options_to_run):
         loading_label.destroy()
         self.threshold_options_data = results
         self.edge_preview_paths = preview_paths
         default_selected = False
         mode = self.processing_mode_var.get()

         for label, t1, t2 in options_to_run:
             option_data = results.get(label)
             if option_data:
                 count = option_data["count"]
                 time_str = option_data["time_str"]
                 
                 if mode in ["tiered", "fast_eye", "smart_auto", "real_image", "smooth_eye"]:
                     radio_text = f"{label} - Cmds: {count}, Est: {time_str}"
                 else:
                     radio_text = f"{label} (t1={t1}, t2={t2}) - Cmds: {count}, Est: {time_str}"

                 rb = tk.Radiobutton(
                    options_frame, text=radio_text, variable=self.selected_threshold_option,
                    value=label, command=lambda l=label: self.show_edge_preview(l)
                 )
                 rb.pack(anchor='w', pady=2)
                 
                 if not default_selected:
                      self.selected_threshold_option.set(label)
                      self.show_edge_preview(label)
                      default_selected = True
             else:
                 tk.Label(options_frame, text=f"{label} - No drawing generated", fg="gray").pack(anchor='w')

         button_frame = tk.Frame(self.main_frame)
         button_frame.pack(pady=20)
         tk.Button(button_frame, text="Confirm and Draw", command=self.confirm_and_start_drawing, width=20).pack(side=tk.LEFT, padx=5)
         tk.Button(button_frame, text="Save Points to File", command=self.save_points_to_file, width=20).pack(side=tk.LEFT, padx=5)
         tk.Button(button_frame, text="Save Processed Image", command=self.save_processed_image_to_file, width=22).pack(side=tk.LEFT, padx=5) 
         tk.Button(button_frame, text="Back", command=self.input_image_page, width=20).pack(side=tk.LEFT, padx=5)

    def save_processed_image_to_file(self):
        selected_label = self.selected_threshold_option.get()
        if not selected_label:
            messagebox.showwarning("Selection Needed", "Please select a drawing style option first.")
            return

        preview_path = self.edge_preview_paths.get(selected_label)
        if not preview_path or not os.path.exists(preview_path):
            messagebox.showerror("Error", "Processed image not found for the selected option.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save Processed Image", defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")], initialfile="processed_image.png"
        )

        if not filepath: return

        try:
            shutil.copy(preview_path, filepath)
            messagebox.showinfo("Success", f"Processed image successfully saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save the image.\nError: {e}")

    def save_points_to_file(self):
        selected_label = self.selected_threshold_option.get()
        if not selected_label:
            messagebox.showwarning("Selection Needed", "Please select a drawing style option first.")
            return

        option_data = self.threshold_options_data.get(selected_label)
        if not option_data or not option_data.get("commands"):
            messagebox.showerror("Error", "Selected option has no drawing commands to save.")
            return

        commands = option_data["commands"]
        
        filepath = filedialog.asksaveasfilename(
            title="Save Drawing Points", defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")], initialfile="drawing_points.txt"
        )

        if not filepath: return

        try:
            with open(filepath, 'w') as f:
                f.write("X, Z, Y\n") 
                for x, z, y in commands: f.write(f"{x:.3f},{z:.3f},{y:.3f}\n")
            messagebox.showinfo("Success", f"Drawing points successfully saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save the file.\nError: {e}")

    def show_edge_preview(self, option_label):
         preview_path = self.edge_preview_paths.get(option_label)
         if preview_path and os.path.exists(preview_path):
              try:
                   img = Image.open(preview_path)
                   
                   self.preview_orig_size = img.size
                   img.thumbnail((600, 600))
                   self.preview_thumb_size = img.size
                   
                   imgtk = ImageTk.PhotoImage(image=img)
                   self.preview_label.imgtk = imgtk
                   self.preview_label.configure(image=imgtk)
                   
                   self.preview_label.bind("<Button-1>", self.on_preview_click)
              except Exception as e:
                   logging.error(f"Error loading preview image {preview_path}: {e}")
                   self.preview_label.configure(image=None, text="Preview error")
         else:
              self.preview_label.configure(image=None, text="No Preview")

    def confirm_and_start_drawing(self):
        selected_label = self.selected_threshold_option.get()
        if not selected_label:
            messagebox.showwarning("Selection Needed", "Please select a drawing style option.")
            return

        option_data = self.threshold_options_data.get(selected_label)
        if not option_data or not option_data.get("commands"):
             messagebox.showerror("Error", "Selected option has no drawing commands.")
             return

        self.selected_commands = option_data["commands"]

        if not self.drawing_in_progress:
             self.drawing_in_progress = True
             self.cancel_requested = False
             self.resume_needed = False
             self.pause_event.set()
             
             self.drawing_start_time = time.time()
             self.total_paused_time = 0
             self.pause_start_time = 0

             full_command_list = self.selected_commands
             logging.info(f"Starting drawing with {len(self.selected_commands)} image commands.")

             threading.Thread(target=self.run_drawing_loop, args=(full_command_list,), daemon=True).start()
             self.show_drawing_progress_page(len(full_command_list))
        else:
            messagebox.showwarning("Busy", "Drawing already in progress.")

    # --- Drawing Execution Workflow ---
    def show_drawing_progress_page(self, total_commands, current_progress=0, status_message="Starting..."):
         self.clear_frame()
         
         header_frame = tk.Frame(self.main_frame)
         header_frame.pack(fill=tk.X, pady=10)
         tk.Label(header_frame, text="Drawing in Progress...", font=("Arial", 16)).pack(side=tk.LEFT)
         tk.Button(header_frame, text="Show QR Upload Code", command=self.show_qr_popup, bg="#d4edda", padx=10).pack(side=tk.RIGHT)

         image_frame = tk.Frame(self.main_frame)
         image_frame.pack(pady=10)

         orig_frame = tk.Frame(image_frame)
         orig_frame.pack(side=tk.LEFT, padx=10)
         tk.Label(orig_frame, text="Original Image", font=("Arial", 10, "bold")).pack()
         
         if self.current_image_path and os.path.exists(self.current_image_path):
             try:
                 orig_img = Image.open(self.current_image_path)
                 orig_img.thumbnail((500, 500))
                 self.orig_imgtk = ImageTk.PhotoImage(image=orig_img)
                 tk.Label(orig_frame, image=self.orig_imgtk).pack()
             except Exception as e:
                 logging.error(f"Error loading original image for progress page: {e}")
                 tk.Label(orig_frame, text="Preview Unavailable").pack()

         proc_frame = tk.Frame(image_frame)
         proc_frame.pack(side=tk.LEFT, padx=10)
         tk.Label(proc_frame, text="Processed Paths", font=("Arial", 10, "bold")).pack()

         selected_label = self.selected_threshold_option.get()
         preview_path = self.edge_preview_paths.get(selected_label) if selected_label else None
         
         if preview_path and os.path.exists(preview_path):
             try:
                 proc_img = Image.open(preview_path)
                 proc_img.thumbnail((500, 500))
                 self.proc_imgtk = ImageTk.PhotoImage(image=proc_img)
                 tk.Label(proc_frame, image=self.proc_imgtk).pack()
             except Exception as e:
                 logging.error(f"Error loading processed image for progress page: {e}")
                 tk.Label(proc_frame, text="Preview Unavailable").pack()

         self.queue_notification_label = tk.Label(self.main_frame, text="", fg="green", font=("Arial", 10, "bold"))
         self.queue_notification_label.pack(pady=5)
         if hasattr(self, 'image_queue') and self.image_queue:
             self.queue_notification_label.config(text=f"New images in queue: {len(self.image_queue)}")

         self.status_label = tk.Label(self.main_frame, textvariable=self.progress_text_var)
         self.status_label.pack(pady=5)

         self.progress_bar = ttk.Progressbar(self.main_frame, orient="horizontal", length=400, mode="determinate", maximum=total_commands, value=current_progress)
         self.progress_bar.pack(pady=10)

         controls_frame = tk.Frame(self.main_frame)
         controls_frame.pack(pady=5)

         self.pause_resume_button = tk.Button(controls_frame, text="Pause", command=self.toggle_pause_resume, width=15)
         self.pause_resume_button.pack(side=tk.LEFT, padx=5)
         
         self.cancel_button = tk.Button(controls_frame, text="Cancel Drawing", command=self.request_cancel_drawing, width=15)
         self.cancel_button.pack(side=tk.LEFT, padx=5)
         
         self.update_drawing_status(current_progress, total_commands)
         self._update_eta_countdown()

    def _update_eta_countdown(self):
        if not self.drawing_in_progress: return

        completed_cmds = self.progress_bar['value']
        total_cmds = self.progress_bar['maximum']
        remaining_time = 0
        
        if not self.pause_event.is_set():
            self.progress_text_var.set(f"Sent {completed_cmds} / {total_cmds} commands | PAUSED")
        elif completed_cmds > 5:
            active_drawing_time = (time.time() - self.drawing_start_time) - self.total_paused_time
            if active_drawing_time > 0:
                avg_time_per_cmd = active_drawing_time / completed_cmds
                remaining_time = (total_cmds - completed_cmds) * avg_time_per_cmd
        else:
            elapsed_time = (time.time() - self.drawing_start_time) - self.total_paused_time
            remaining_time = max(0, (total_cmds * TIME_ESTIMATE_FACTOR) - elapsed_time)

        mins, secs = divmod(int(remaining_time), 60)
        if self.pause_event.is_set():
            self.progress_text_var.set(f"Sent {completed_cmds} / {total_cmds} commands | ETA: {mins:02d}:{secs:02d}")

        self.eta_update_id = self.window.after(1000, self._update_eta_countdown)

    def toggle_pause_resume(self):
        if self.pause_event.is_set():
            self.pause_event.clear()
            logging.info("Drawing paused by user.")
            if self.pause_resume_button and self.pause_resume_button.winfo_exists(): self.pause_resume_button.config(text="Resume")
            self.pause_start_time = time.time()
        else:
            if self.pause_start_time > 0:
                self.total_paused_time += time.time() - self.pause_start_time
                self.pause_start_time = 0
            self.pause_event.set()
            logging.info("Drawing resumed by user.")
            if self.pause_resume_button and self.pause_resume_button.winfo_exists(): self.pause_resume_button.config(text="Pause")

    def update_drawing_status(self, current_command_index, total_commands, message=""):
        if self.progress_bar and self.progress_bar.winfo_exists(): self.progress_bar['value'] = current_command_index
        if message: self.progress_text_var.set(f"Sent {current_command_index} / {total_commands} commands | {message}")

    def request_cancel_drawing(self):
        if self.drawing_in_progress:
            logging.info("Cancel requested by user.")
            self.cancel_requested = True
            self.pause_event.set() 
            if self.cancel_button and self.cancel_button.winfo_exists(): self.cancel_button.config(text="Cancelling...", state=tk.DISABLED)
            if self.pause_resume_button and self.pause_resume_button.winfo_exists(): self.pause_resume_button.config(state=tk.DISABLED)
            self.progress_text_var.set("Cancellation requested...")

    def _send_final_position_and_cleanup(self, success_message, failure_message):
        self.drawing_in_progress = False
        logging.info("Attempting to move robot to final position.")
        final_x, final_z, final_y = FINAL_ROBOT_POSITION
        command_str_final = f"{final_x:.3f},{final_z:.3f},{final_y:.3f}"

        move_ok = False
        if self.connected and self.socket:
            if self.send_message_internal(command_str_final):
                response_r_final = self.receive_message_internal(timeout=20.0)
                if response_r_final == "R":
                    logging.info("Robot received final move command.")
                    move_ok = True
                else: logging.error(f"Robot didn't confirm final move receipt, got '{response_r_final}'") 
            else: logging.error("Failed to send final position command.") 

        final_status = f"{success_message} Final move command sent." if move_ok else f"{failure_message} Failed to send final move command."
        self.last_drawing_status["status"] = success_message
        self.last_drawing_status["error_message"] = "" if move_ok else "Failed to send final move command."

        self.window.after(0, lambda fs=final_status: self.update_final_status(fs))
        self.selected_commands = None
        self.cancel_requested = False
        if not self.resume_needed:
            self.resume_commands = None
            self.resume_start_index_global = 0

        self.window.after(2000, self.input_image_page)

    def update_final_status(self, message):
        if self.eta_update_id:
            self.window.after_cancel(self.eta_update_id)
            self.eta_update_id = None
        if self.status_label and self.status_label.winfo_exists(): self.progress_text_var.set(message)
        if self.cancel_button and self.cancel_button.winfo_exists(): self.cancel_button.pack_forget()
        if self.pause_resume_button and self.pause_resume_button.winfo_exists(): self.pause_resume_button.pack_forget()

    def run_drawing_loop(self, commands_to_send: List[Tuple], start_index=0):
        total_commands = len(commands_to_send)
        if start_index > 0: self.window.after(0, lambda: self.show_drawing_progress_page(total_commands, start_index, "Resuming drawing..."))
        
        try:
            for i, (x, z, y) in enumerate(commands_to_send[start_index:], start=start_index):
                self.pause_event.wait() 
                if self.cancel_requested:
                    logging.info(f"Cancellation detected at command {i+1}.")
                    self._send_final_position_and_cleanup("Drawing Cancelled.", "Drawing Cancelled.")
                    return

                command_str = f"{x:.2f},{z:.2f},{y:.2f}"
                
                # --- Testing Mode Bypass ---
                if not self.send_message_internal(command_str):
                    logging.error(f"Connection lost while sending command {i+1}. Preparing to resume.")
                    self.resume_needed = True
                    self.resume_commands = commands_to_send
                    self.resume_start_index_global = i
                    self.last_drawing_status = {"total_commands": total_commands, "completed_commands": i, "status": "Connection Lost", "error_message": f"Lost connection before sending command {i+1}"}
                    self.window.after(0, lambda idx=i: self.update_drawing_status(idx, total_commands, "Connection Lost!"))
                    self.window.after(1000, self.connection_setup_page)
                    self.drawing_in_progress = False
                    return

                response_r = self.receive_message_internal(timeout=20.0)
                if response_r is None or response_r != "R":
                    error_msg = f"Robot did not confirm receipt (R) for command {i+1}, got '{response_r}'."
                    logging.error(error_msg + " Preparing to resume.")
                    self.resume_needed = True
                    self.resume_commands = commands_to_send
                    self.resume_start_index_global = i
                    self.last_drawing_status = {"total_commands": total_commands, "completed_commands": i, "status": f"Protocol Error", "error_message": error_msg}
                    self.window.after(0, lambda idx=i, r=response_r: self.update_drawing_status(idx, total_commands, f"Error: No 'R' (Got {r}). Reconnect to resume."))
                    self.window.after(1000, self.connection_setup_page)
                    self.drawing_in_progress = False
                    return

                self.window.after(0, lambda idx=i + 1: self.update_drawing_status(idx, total_commands))

            logging.info("All drawing commands sent successfully.")
            self._send_final_position_and_cleanup("Drawing Complete.", "Drawing Complete.")

        except Exception as e:
            logging.error(f"Unexpected error during drawing process: {e}", exc_info=True)
            self.drawing_in_progress = False
            self.cancel_requested = False

    # --- Internal Socket Methods ---
    def send_message_internal(self, message: str) -> bool:
        if self.testing_mode:
            return True
        if not self.connected or not self.socket: return False
        try:
            self.socket.sendall(message.encode('utf-8'))
            return True
        except (socket.error, ConnectionResetError, BrokenPipeError, socket.timeout) as e:
            self.handle_connection_loss()
            return False

    def receive_message_internal(self, timeout=20.0) -> Optional[str]:
         if self.testing_mode:
             time.sleep(0.001) # Tiny sleep so the progress bar can breathe and animate
             return "R"
         if not self.connected or not self.socket: return None
         try:
             self.socket.settimeout(timeout)
             data = self.socket.recv(1024)
             self.socket.settimeout(None)
             if not data:
                 self.handle_connection_loss()
                 return None
             return data.decode('utf-8').strip()
         except socket.timeout:
             self.handle_connection_loss()
             return None
         except (socket.error, ConnectionResetError, BrokenPipeError) as e:
             self.handle_connection_loss()
             return None

    def handle_connection_loss(self):
        was_connected = self.connected
        self.close_socket()
        if was_connected and not self.drawing_in_progress and not self.resume_needed:
            self.window.after(0, lambda: messagebox.showinfo("Connection Lost", "Robot connection lost."))

    # --- Connection Handling ---
    def establish_connection(self):
        if hasattr(self, 'connect_button') and self.connect_button.winfo_exists(): self.connect_button.config(state=tk.DISABLED)
        if hasattr(self, 'reconnect_button') and self.reconnect_button.winfo_exists(): self.reconnect_button.config(state=tk.DISABLED)

        host, port = (SIMULATION_HOST, SIMULATION_PORT) if self.connection_var.get() == "simulation" else (REAL_ROBOT_HOST, REAL_ROBOT_PORT)
        threading.Thread(target=self._connection_attempt_thread, args=(host, port), daemon=True).start()

    def _connection_attempt_thread(self, host, port):
        try:
            self.close_socket()
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            self.socket.connect((host, port))
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.socket.settimeout(None)
            self.connected = True
            self.window.after(0, lambda: self.handle_connection_result(True))
        except (socket.error, socket.timeout, ConnectionRefusedError) as e:
            self.connected = False
            self.close_socket()
            self.window.after(0, lambda: self.handle_connection_result(False))

    def handle_connection_result(self, connected):
        if hasattr(self, 'connect_button') and self.connect_button.winfo_exists(): self.connect_button.config(state=tk.NORMAL)
        if hasattr(self, 'reconnect_button') and self.reconnect_button.winfo_exists(): self.reconnect_button.config(state=tk.NORMAL)

        if connected:
            self.connection_established = True
            if self.resume_needed and self.resume_commands is not None:
                self.move_to_final_before_resume()
            else:
                self.drawing_options_page()
        else:
            if self.resume_needed:
                messagebox.showerror("Reconnection Failed", "Failed to reconnect. Cannot resume the previous drawing.")
                self.resume_needed = False
                self.resume_commands = None
                self.resume_start_index_global = 0
                self.drawing_options_page()
            else:
                messagebox.showerror("Connection Failed", "Failed to establish connection.")

    def move_to_final_before_resume(self):
        def move_and_resume_thread():
            self.show_drawing_progress_page(len(self.resume_commands), self.resume_start_index_global, "Moving to resume position...")
            final_x, final_z, final_y = FINAL_ROBOT_POSITION
            command_str_final = f"{final_x:.3f},{final_z:.3f},{final_y:.3f}"
            move_ok = False
            if self.connected and self.socket:
                if self.send_message_internal(command_str_final):
                    response_r = self.receive_message_internal(timeout=20.0)
                    if response_r == "R": move_ok = True

            if move_ok:
                 self.drawing_in_progress = True
                 self.cancel_requested = False
                 self.pause_event.set()
                 self.run_drawing_loop(self.resume_commands, self.resume_start_index_global)
            else:
                error_msg = "Failed to move robot to safe resume position."
                self.window.after(0, lambda: messagebox.showwarning("Resume Warning", error_msg + "\nYou can try 'Reconnect & Resume' again."))
                self.drawing_in_progress = False
                self.window.after(1000, self.connection_setup_page)

        threading.Thread(target=move_and_resume_thread, daemon=True).start()

    def close_socket(self):
        if self.socket:
            try: self.socket.shutdown(socket.SHUT_RDWR)
            except: pass
            finally:
                try: self.socket.close()
                except: pass
                self.socket = None
        self.connected = False
        self.connection_established = False
        self.testing_mode = False

    def close_and_return_main(self):
         self.close_socket()
         self.resume_needed = False
         self.resume_commands = None
         self.resume_start_index_global = 0
         self.main_page()

    # --- Utility Methods ---
    def clear_frame(self):
        self.is_on_input_page = False # Reset flag whenever we leave the page
        
        if hasattr(self, 'eta_update_id') and self.eta_update_id:
            self.window.after_cancel(self.eta_update_id)
            self.eta_update_id = None
        for widget in self.main_frame.winfo_children(): widget.destroy()
        
        # Free up Image resources
        self.orig_imgtk = None
        self.proc_imgtk = None
        self.history_thumbnails = [] # Clears out refs so Tkinter doesn't leak memory
        
        self.progress_bar = None; self.status_label = None; self.cancel_button = None
        self.connect_button = None; self.reconnect_button = None; self.preview_label = None; self.pause_resume_button = None
        self.test_mode_button = None

    def on_window_close(self):
        self.cancel_requested = True
        self.close_socket()
        time.sleep(0.2)
        self.window.destroy()

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    app = RUNME_GUI()
    app.window.protocol("WM_DELETE_WINDOW", app.on_window_close)
    app.window.mainloop()