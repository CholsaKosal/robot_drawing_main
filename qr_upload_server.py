import os
import threading
import subprocess
import time
import re
import logging
from flask import Flask, request, render_template_string

app = Flask(__name__)
UPLOAD_FOLDER = os.getenv("DATA_DIR", ".")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

_server_running = False
_tunnel_url = None
on_image_received_callback = None

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Send Image to Robot</title>
    <style>
      body { font-family: Arial, sans-serif; text-align: center; padding: 20px; background-color: #f4f4f9; margin: 0; }
      .container { max-width: 500px; margin: 0 auto; background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
      h2 { color: #333; margin-top: 0; }
      input[type=file], input[type=submit] { 
        width: 100%; 
        padding: 14px; 
        margin: 12px 0; 
        border-radius: 8px; 
        border: 1px solid #ccc; 
        font-size: 16px; 
        box-sizing: border-box; 
      }
      input[type=submit] { 
        background-color: #4CAF50; 
        color: white; 
        border: none; 
        cursor: pointer; 
        font-weight: bold; 
        font-size: 18px;
        transition: background-color 0.3s;
      }
      input[type=submit]:disabled { background-color: #9e9e9e; cursor: not-allowed; }
    </style>
  </head>
  <body>
    <div class="container">
        <h2>Upload Image for Robot</h2>
        <p style="color: #666; margin-bottom: 20px;">Select an image to send directly to the robot operator.</p>
        <form method=post enctype=multipart/form-data action="/upload" onsubmit="document.getElementById('uploadBtn').disabled=true; document.getElementById('uploadBtn').value='Sending...';">
          
          <input type=file name=file accept="image/*" required>
          
          <input type=submit id="uploadBtn" value="Send to Robot">
        </form>
    </div>
  </body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], "qr_uploaded_image.png")
        file.save(filepath)
        if on_image_received_callback:
            # We only send the filepath back now
            on_image_received_callback(filepath)
        return "<h2 style='font-family: Arial; text-align: center; margin-top: 50px; color: green;'>Image successfully sent! You can look at the computer screen now.</h2>", 200

def run_flask():
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='127.0.0.1', port=5000, use_reloader=False, debug=False)

def start_cloudflare_tunnel():
    cmd = ['cloudflared', 'tunnel', '--url', 'http://127.0.0.1:5000']
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
    except FileNotFoundError:
        logging.error("cloudflared executable not found in PATH.")
        return None, None

    url = None
    start_time = time.time()
    
    for line in iter(process.stdout.readline, ''):
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if match:
            url = match.group(0)
            break
        if time.time() - start_time > 15:
            break
            
    def consume_stdout(proc):
        for _ in iter(proc.stdout.readline, ''):
            pass
            
    if process:
        threading.Thread(target=consume_stdout, args=(process,), daemon=True).start()

    return url, process

def start_server_and_tunnel(callback):
    global _server_running, _tunnel_url, on_image_received_callback
    on_image_received_callback = callback
    
    if _server_running:
        return _tunnel_url, None

    threading.Thread(target=run_flask, daemon=True).start()
    _tunnel_url, proc = start_cloudflare_tunnel()
    
    if _tunnel_url:
        _server_running = True
        
    return _tunnel_url, proc