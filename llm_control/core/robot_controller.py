import socket
import logging
import threading
import time
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class RobotController:
    def __init__(self):
        self.socket = None
        self.connected = False
        
        self.REAL_ROBOT_HOST, self.REAL_ROBOT_PORT = '192.168.125.1', 1025
        self.SIMULATION_HOST, self.SIMULATION_PORT = '127.0.0.1', 55000
        
        self.ROBOT_HOME_POSITION = (300, -350.922, 300)
        self.FINAL_ROBOT_POSITION = (0, -120, 0)
        self.A4_WIDTH_MM = 170
        self.A4_HEIGHT_MM = 207

        # Background Threading state
        self.drawing_in_progress = False
        self.pause_event = threading.Event()
        self.cancel_requested = False
        self.current_cmd_idx = 0
        self.total_cmds = 0

    def connect(self, mode: str) -> bool:
        host, port = (self.SIMULATION_HOST, self.SIMULATION_PORT) if mode == "simulation" else (self.REAL_ROBOT_HOST, self.REAL_ROBOT_PORT)
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            self.socket.connect((host, port))
            self.socket.settimeout(None)
            self.connected = True
            logging.info(f"Connected to {host}:{port}")
            return True
        except Exception as e:
            logging.error(f"Connect failed: {e}")
            self.connected = False
            return False

    def _send(self, command: str) -> bool:
        if not self.connected or not self.socket: return False
        try:
            self.socket.sendall(command.encode('utf-8'))
            response = self.socket.recv(1024).decode('utf-8').strip()
            return response == "R"
        except Exception as e:
            logging.error(f"Connection lost: {e}")
            self.connected = False
            return False

    def move_to(self, x: float, z: float, y: float) -> str:
        if not self.connected: return "Error: Not connected."
        success = self._send(f"{x:.2f},{z:.2f},{y:.2f}")
        return "Moved successfully." if success else "Error: Robot rejected command or disconnected."

    def pack_robot(self) -> str:
        if not self.connected: return "Error: Not connected."
        logging.info("Sending PACK command...")
        return "Packing successful." if self._send("PACK") else "Failed to pack."

    def test_workspace(self, test_z: float) -> str:
        if not self.connected: return "Error: Not connected."
        pen_up_z = test_z / 10 if test_z > 0 else test_z * 1.5
        w, h = self.A4_WIDTH_MM / 3, self.A4_HEIGHT_MM / 3
        path = [
            (w, pen_up_z, h), (w, test_z, h), (w, test_z, -h), 
            (-w, test_z, -h), (-w, test_z, h), (w, test_z, h), (0, pen_up_z, 0)
        ]
        for x, z, y in path:
            if not self._send(f"{x:.2f},{z:.2f},{y:.2f}"):
                return "Failed during workspace test."
        return "Workspace test completed."

    def start_drawing(self, commands: List[Tuple[float, float, float]]) -> str:
        if self.drawing_in_progress: return "Drawing already in progress."
        if not self.connected: return "Error: Not connected."
        
        self.drawing_in_progress = True
        self.cancel_requested = False
        self.pause_event.set()
        self.current_cmd_idx = 0
        self.total_cmds = len(commands)
        
        threading.Thread(target=self._drawing_loop, args=(commands,), daemon=True).start()
        return f"Started drawing {self.total_cmds} commands in the background."

    def _drawing_loop(self, commands):
        try:
            for i, (x, z, y) in enumerate(commands):
                self.pause_event.wait()
                if self.cancel_requested: break
                
                if not self._send(f"{x:.2f},{z:.2f},{y:.2f}"):
                    logging.error("Connection lost during drawing.")
                    break
                self.current_cmd_idx = i + 1
            
            # Send to final position when done/cancelled
            fx, fz, fy = self.FINAL_ROBOT_POSITION
            self._send(f"{fx:.2f},{fz:.2f},{fy:.2f}")
        finally:
            self.drawing_in_progress = False
            self.cancel_requested = False

    def control_drawing(self, action: str) -> str:
        if not self.drawing_in_progress and action != "status":
            return "No drawing currently in progress."
            
        if action == "pause":
            self.pause_event.clear()
            return "Drawing paused."
        elif action == "resume":
            self.pause_event.set()
            return "Drawing resumed."
        elif action == "cancel":
            self.cancel_requested = True
            self.pause_event.set()
            return "Cancellation requested. Robot will return to safe position."
        elif action == "status":
            if not self.drawing_in_progress: return "Idle."
            state = "Paused" if not self.pause_event.is_set() else "Running"
            return f"Status: {state}. Progress: {self.current_cmd_idx}/{self.total_cmds}."
        return "Unknown control command."