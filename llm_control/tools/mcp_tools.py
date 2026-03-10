from core.robot_controller import RobotController
from core.drawing_processor import DrawingProcessor

robot = RobotController()
processor = DrawingProcessor()

def connect_robot(mode: str) -> str:
    """Connects to the robotic arm. Mode must be 'simulation' or 'real'."""
    return "Connected successfully." if robot.connect(mode) else "Connection failed."

def robot_action(action_type: str, z_val: float = -10.0) -> str:
    """
    Executes a single test or movement action.
    action_type can be: "home", "safe_center", "test_z", "test_workspace".
    z_val is required for safe_center, test_z, and test_workspace.
    """
    if action_type == "home":
        x, z, y = robot.ROBOT_HOME_POSITION
        return robot.move_to(x, z, y)
    elif action_type == "safe_center":
        return robot.move_to(0, z_val, 0)
    elif action_type == "test_z":
        return robot.move_to(0, z_val, 0)
    elif action_type == "test_workspace":
        return robot.test_workspace(z_val)
    return "Unknown action type."

def pack_robot(confirm_checklist: bool) -> str:
    """Folds the robot for shipping. User MUST be asked before confirming."""
    if not confirm_checklist: return "Packing aborted."
    return robot.pack_robot()

def process_image(filepath: str) -> str:
    """Reads an image path and returns threshold options for the user to pick."""
    return processor.process_and_cache_image(filepath)

def start_drawing(option_label: str, pen_down_z: float = -10.0) -> str:
    """Starts the background drawing loop. option_label must be like 'Option 1'."""
    commands = processor.get_commands(option_label, pen_down_z)
    if not commands:
        return f"Error: {option_label} not found. Did you run process_image first?"
    return robot.start_drawing(commands)

def control_drawing(command: str) -> str:
    """Controls background drawing. command must be 'pause', 'resume', 'cancel', or 'status'."""
    return robot.control_drawing(command)