# robot_drawing_main

## Robot Details
**ABB CRB 15000 Industrial Manipulator (GoFa)**
* **Manufacturer:** ABB Engineering (Shanghai) Ltd.
* **Type/Product:** Manipulator, CRB 15000
* **Payload:** 5 kg
* **Reach:** 0.95 m
* **Date of Manufacturing:** 20230913 (September 13, 2023)

## Prerequisites (Windows)

1.  **Python 3.x**: Ensure Python is installed on your Windows machine.
    * *Note:* When installing Python on Windows, make sure to check the box **"tcl/tk and IDLE"** during the installation process. This is required for the GUI (`tkinter`) to work.
2.  **ABB RobotStudio**: Required if you are running the simulation mode.
3.  **Physical Robot**: Required if running in "Real Robot" mode (IP: `192.168.125.1`).

## Installation

1.  Open **Command Prompt** (cmd) or PowerShell.
2.  Navigate to the project directory:
    ```powershell
    cd path\to\robot_drawing_main
    ```
3.  (Optional) Create a virtual environment to keep packages isolated:
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```
4.  Install the required external libraries:
    ```powershell
    pip install -r requirements.txt
    ```

## How to Run

1.  **Start the Robot/Simulation:**
    * **Real Robot:** Ensure the controller is on, the robot is in Auto mode, and the RAPID code is running (waiting for connection).
    * **Simulation:** Open RobotStudio, start the simulation, and ensure the RAPID code is running.

2.  **Run the Python Script:**
    In your terminal/command prompt:
    ```powershell
    python main.py
    ```

3.  **Using the GUI:**
    * Select **Simulation** or **Real Robot** and click **Connect**.
    * Upload an image to generate drawing paths.
    * Use the **Packing Position** button (located under "Go Home") to safely fold the robot for transport.

## Troubleshooting

* **"No module named tkinter":** This usually means Python was installed without Tcl/Tk support. Re-run the Python installer, choose "Modify", and ensure "tcl/tk and IDLE" is checked.
* **Connection Refused:** Ensure the firewall is not blocking port `1025` (Real) or `55000` (Sim), and that the RAPID code is actively running in the `SocketAccept` state.