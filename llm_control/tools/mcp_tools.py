from core.robot_controller import RobotController
from core.drawing_processor import DrawingProcessor

robot = RobotController()
processor = DrawingProcessor()

# Global states for Z heights
CURRENT_PEN_DOWN_Z = -10.0
CURRENT_SAFE_CENTER_Z = -120.0

def connect_robot(mode: str) -> str:
    """Connects to the robotic arm. Mode must be 'simulation' or 'real'."""
    return "Connected successfully." if robot.connect(mode) else "Connection failed."

def robot_action(action_type: str, z_val: float = None) -> str:
    """
    Executes a single test or movement action.
    action_type can be: "home", "safe_center", "test_z", "test_workspace".
    z_val is optional. If omitted, it uses the respective global Z variable.
    """
    if action_type == "home":
        x, z, y = robot.ROBOT_HOME_POSITION
        return robot.move_to(x, z, y)
        
    elif action_type == "safe_center":
        target_z = z_val if z_val is not None else CURRENT_SAFE_CENTER_Z
        return robot.move_to(0, target_z, 0)
        
    elif action_type == "test_z":
        target_z = z_val if z_val is not None else CURRENT_PEN_DOWN_Z
        return robot.move_to(0, target_z, 0)
        
    elif action_type == "test_workspace":
        target_z = z_val if z_val is not None else CURRENT_PEN_DOWN_Z
        return robot.test_workspace(target_z)
        
    return "Unknown action type."

def pack_robot(confirm_checklist: bool) -> str:
    """Folds the robot for shipping. User MUST be asked before confirming."""
    if not confirm_checklist: return "Packing aborted."
    return robot.pack_robot()

def process_image(filepath: str) -> str:
    """Reads an image path and returns threshold options for the user to pick."""
    return processor.process_and_cache_image(filepath)

def set_pen_down_z(z_val: float) -> str:
    """Updates the global pen down Z height."""
    global CURRENT_PEN_DOWN_Z
    try:
        CURRENT_PEN_DOWN_Z = float(z_val)
        return f"Pen down Z successfully updated to {CURRENT_PEN_DOWN_Z}"
    except ValueError:
        return "Error: z_val must be a valid float number."

def set_safe_center_z(z_val: float) -> str:
    """Updates the global safe center Z height."""
    global CURRENT_SAFE_CENTER_Z
    try:
        CURRENT_SAFE_CENTER_Z = float(z_val)
        return f"Safe center Z successfully updated to {CURRENT_SAFE_CENTER_Z}"
    except ValueError:
        return "Error: z_val must be a valid float number."

def start_drawing(option_label: str, pen_down_z: float = None) -> str:
    """Starts the background drawing loop. option_label must be like 'Option 1'."""
    if pen_down_z is None:
        pen_down_z = CURRENT_PEN_DOWN_Z
        
    commands = processor.get_commands(option_label, pen_down_z)
    if not commands:
        return f"Error: {option_label} not found. Did you run process_image first?"
    return robot.start_drawing(commands)

def control_drawing(command: str) -> str:
    """Controls background drawing. command must be 'pause', 'resume', 'cancel', or 'status'."""
    return robot.control_drawing(command)