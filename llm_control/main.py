# robot_drawing_main/llm_control/main.py features: similar to robot_drawing_main/main.py but using custom mcp and self host a llm to control instead of buttons. user interact with text input. It must match all the steps in robot_drawing_main/main.py. It relys on its own python environment, so readme.md and requirements.txt must be set up correctly. 
import sys
import os
from llm_engine.agent import LLMAgent

def main():
    print("=====================================================")
    print("        ABB GoFa LLM Control Interface               ")
    print("        Type 'exit' or 'quit' to stop.               ")
    print("=====================================================")
    
    agent = LLMAgent()
    
    # Scan image directory so the AI secretly knows what's available for later
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, "image")
    
    available_images = []
    if os.path.exists(image_dir):
        valid_exts = {'.png', '.jpg', '.jpeg'}
        for f in os.listdir(image_dir):
            if os.path.splitext(f)[1].lower() in valid_exts:
                abs_path = os.path.join(image_dir, f)
                abs_path = abs_path.replace("\\", "/") # Sanitize paths for JSON tool calls
                available_images.append((f, abs_path))
    
    images_info = "\n".join([f"- {name} (Path: {path})" for name, path in available_images]) if available_images else "No images found."
    
    # --- NEW FEATURE: Initial Prompt enforces the GUI Workflow ---
    initial_prompt = (
        "System Notification: The application has just started. "
        "You MUST guide the user through the following strict workflow, mimicking a GUI application:\n"
        "1. CONNECTION: First, you must ask the user to connect to the robot (either 'simulation' or 'real').\n"
        "2. CALIBRATION/TESTING: Once connected, suggest testing the workspace, going to the safe center, or testing the Z-height.\n"
        "3. DRAWING: Only after connection and optional testing should you process an image to draw.\n\n"
        f"Available local images for step 3 are:\n{images_info}\n\n"
        "ACTION REQUIRED NOW: Greet the user. State that the first step is to establish a connection. "
        "Ask them if they would like to connect to the 'simulation' or the 'real' robot. "
        "Do NOT mention the available images or drawing yet."
    )
    
    print("\nAssistant > ", end="", flush=True)
    # Process the hidden system prompt to generate the opening message
    for chunk in agent.process_user_input(initial_prompt):
        print(chunk, end="", flush=True)
    print()
    
    while True:
        try:
            user_input = input("\nUser > ")
            if user_input.strip().lower() in ['exit', 'quit']:
                print("Shutting down...")
                break
                
            if not user_input.strip():
                continue
                
            print("\nAssistant > ", end="", flush=True)
            
            # Streams chunks, executes tools in the background, and streams the follow-up seamlessly
            for chunk in agent.process_user_input(user_input):
                print(chunk, end="", flush=True)
            print() # Print a final newline when the conversational turn is fully resolved
            
        except KeyboardInterrupt:
            print("\nShutting down...")
            break

if __name__ == "__main__":
    main()