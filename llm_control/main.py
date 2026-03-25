import sys
import os
from llm_engine.agent import LLMAgent
from tools import mcp_tools

def main():
    print("=====================================================")
    print("        ABB GoFa LLM Control Interface               ")
    print("        Type 'exit' or 'quit' to stop.               ")
    print("=====================================================")
    
    agent = LLMAgent()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, "image")
    
    available_images = []
    if os.path.exists(image_dir):
        valid_exts = {'.png', '.jpg', '.jpeg'}
        for f in os.listdir(image_dir):
            if os.path.splitext(f)[1].lower() in valid_exts:
                abs_path = os.path.join(image_dir, f)
                abs_path = abs_path.replace("\\", "/") 
                available_images.append((f, abs_path))
    
    images_info = "\n".join([f"- {name} (Path: {path})" for name, path in available_images]) if available_images else "No images found."
    
    # Updated prompt to explicitly block premature tool usage
    initial_prompt = (
        "System Notification: The application has just started. "
        "You MUST guide the user through the following strict workflow, mimicking a GUI application:\n"
        "1. CONNECTION: First, you must ask the user to connect to the robot (either 'simulation' or 'real').\n"
        "2. CALIBRATION/TESTING: Once connected, suggest testing the workspace, going to the safe center, or testing the Z-height.\n"
        "3. DRAWING: Only after connection and optional testing should you process an image to draw.\n\n"
        f"Available local images for step 3 are:\n{images_info}\n\n"
        "ACTION REQUIRED NOW: Greet the user. State that the first step is to establish a connection. "
        "Ask them if they would like to connect to the 'simulation' or the 'real' robot. "
        "CRITICAL: DO NOT OUTPUT ANY TOOL CALLS RIGHT NOW. Just speak to the user naturally."
    )
    
    print("\nAssistant > ", end="", flush=True)
    for chunk in agent.process_user_input(initial_prompt):
        print(chunk, end="", flush=True)
    print()
    
    while True:
        try:
            # Display both current Z values directly in the user prompt!
            user_input = input(f"\nUser (Pen Z: {mcp_tools.CURRENT_PEN_DOWN_Z} | Safe Z: {mcp_tools.CURRENT_SAFE_CENTER_Z}) > ")
            if user_input.strip().lower() in ['exit', 'quit']:
                print("Shutting down...")
                break
                
            if not user_input.strip():
                continue
                
            print("\nAssistant > ", end="", flush=True)
            
            for chunk in agent.process_user_input(user_input):
                print(chunk, end="", flush=True)
            print()
            
        except KeyboardInterrupt:
            print("\nShutting down...")
            break

if __name__ == "__main__":
    main()