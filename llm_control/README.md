
# Robot Drawing - LLM Control Module

This module replaces the traditional graphical user interface (GUI) with a conversational AI agent. Using a self-hosted Large Language Model (LLM) and a custom Model Context Protocol (MCP), you can interact with the ABB GoFa robotic arm using natural language text commands.

The entire AI brain runs 100% locally on your machine, requiring no cloud APIs, subscriptions, or internet connection once set up.

---

## Robot Details
**ABB CRB 15000 Industrial Manipulator (GoFa)**
* **Manufacturer:** ABB Engineering (Shanghai) Ltd.
* **Type/Product:** Manipulator, CRB 15000
* **Payload:** 5 kg
* **Reach:** 0.95 m
* **Date of Manufacturing:** 20230913 (September 13, 2023)

---

## Prerequisites

Because this module uses `llama-cpp-python` to run the LLM locally, your system needs a C++ compiler to build the underlying `llama.cpp` engine. 

*(Note: If you want GPU acceleration for the LLM, you will need an NVIDIA GPU and the CUDA Toolkit installed prior to building the package) [official documentation](https://github.com/abetlen/llama-cpp-python).*

### For Windows:
1. **Python 3.x**: Ensure Python is installed. **CRITICAL:** Check the "Add Python to PATH" box during installation.
2. **Visual Studio C++ Build Tools**: Required to compile the Python bindings. 
   * Download the [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
   * During installation, ensure **"Desktop development with C++"** is checked.
3. **NVIDIA CUDA Toolkit (Optional but Highly Recommended)**: If you want GPU acceleration, download and install the [CUDA Toolkit for Windows](https://developer.nvidia.com/cuda-downloads).
4. **ABB RobotStudio**: Required if you are running the simulation mode.

### For Linux (Ubuntu/Debian-based):
1. **Python 3.x**: Usually pre-installed, but ensure you have `python3-venv` and `python3-dev`.
2. **Build Tools & CUDA**: Install the required compilers and GPU toolkit via your terminal:
```bash
sudo apt update
sudo apt install build-essential cmake python3-dev
sudo apt install nvidia-cuda-toolkit -y

```

Verify the compiler is ready:

```bash
nvcc --version

```

---

## Installation Setup

1. Open your terminal or command prompt and navigate to the `llm_control` directory:

**Windows (PowerShell):**

```powershell
cd path\to\robot_drawing_main\llm_control

```

**Linux:**

```bash
cd path/to/robot_drawing_main/llm_control

```

2. Create and activate a virtual environment to keep your packages isolated:

**Windows (PowerShell):**
*Note: If you get an unauthorized script error, run `Set-ExecutionPolicy Unrestricted -Scope CurrentUser` first.*

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

```

**Linux:**

```bash
python3 -m venv venv
source venv/bin/activate

```

3. Install the required Python libraries:

```bash
pip install -r requirements.txt

```

4. **Install GPU-Accelerated LLM Bindings (Optional):**
If you want the AI to respond quickly using your NVIDIA GPU, you need to compile `llama-cpp-python` with CUDA enabled. *(If `llama-cpp-python` was already installed without CUDA, uninstall it first: `pip uninstall llama-cpp-python -y`)*

**Windows (PowerShell):**

```powershell
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

```

**Linux:**

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

```

*(Tip: To monitor GPU usage on Windows, open Task Manager -> Performance -> GPU. On Linux, run `watch -n 1 nvidia-smi`).*

---

## 🧠 Downloading the Local LLM

The project expects a compiled `.gguf` model file to act as the brain. **We recommend Qwen 2.5 (7B) Instruct** because it is exceptionally good at tool-calling.

1. Go to HuggingFace and download a quantized GGUF model file (e.g., `qwen2.5-7b-instruct-q4_k_m.gguf`).
2. Move that exact file into the designated model directory:

```text
robot_drawing_main/llm_control/model/

```

3. *Important:* If you downloaded a differently named model, open `llm_engine/agent.py` and update the `model_filename` parameter in the `__init__` function to match your file exactly.

make sure the model is not currupted. 

---

## How to Run

1. **Start the Robot/Simulation:**
* **Real Robot:** Ensure the controller is on, the robot is in Auto mode, and the RAPID code is running (`SocketAccept` state, IP: `192.168.125.1`).
* **Simulation:** Open RobotStudio, start the simulation, and ensure the RAPID code is running.


2. **Start the Agent:**
Ensure your virtual environment is activated, then run:

**Windows:**

```powershell
python main.py

```

**Linux:**

```bash
python3 main.py

```

### Example Conversation Flow

Once the terminal boots up, the Assistant will guide you:

> **Assistant:** Hello! Please connect to the robot first. Would you like to connect to the 'simulation' or 'real' robot?
> **User:** Connect to the simulation.
> **Assistant:** *[Agent -> connect_robot({'mode': 'simulation'})]* Connected successfully. Would you like to test the workspace before drawing?
> **User:** Let's process the dog image.
> **Assistant:** *[Agent -> process_image({'filepath': 'C:/.../image/dog.png'})]* Image processed successfully. Option 1 has 150 commands. Option 2 has 400 commands. Which would you like?
> **User:** Let's draw Option 1.
> **Assistant:** *[Agent -> start_drawing({'option_label': 'Option 1', 'pen_down_z': -10.0})]* Started drawing 150 commands in the background.

While the robot is drawing, the interface remains active. You can type "status", "pause", "resume", or "cancel" at any time.

---

## Troubleshooting

* **Model Not Found Error:** Double-check that your `.gguf` file is actually inside `model/` and that the filename perfectly matches the one defined in `agent.py`.
* **Failed to build `llama-cpp-python` (Windows):** This strictly means your system lacks a C++ compiler. Install Visual Studio C++ Build Tools and ensure "Desktop development with C++" is checked.
* **Virtual Environment Won't Activate (Windows):** Run PowerShell as Administrator and execute `Set-ExecutionPolicy RemoteSigned`, or run `Set-ExecutionPolicy Unrestricted -Scope CurrentUser` in your normal PowerShell window.
* **Connection Refused / Failed:** Ensure your firewall is not blocking port `1025` (Real) or `55000` (Sim), and that the RAPID code is actively running on the controller/simulation before you ask the LLM to connect.

