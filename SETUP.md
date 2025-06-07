# Setup Instructions

These steps will prepare a Windows lab machine to run your assigned experiment.

---

## Step 1: Install Python (Version 3.11.8)

1. Download Python 3.11.8  
   Visit: https://www.python.org/downloads/release/python-3118/  
   Scroll down to "Windows Installer (64-bit)" and download it.

2. Install Python  
   - Open the downloaded file.  
   - Important:  
     - Check the box that says "Add Python to PATH"  
     - Uncheck the box that says "Use admin privileges"  
   - Click "Install Now" and wait for installation to finish.

3. Verify installation  
   - Open Command Prompt (press Win + R, type `cmd`, and hit Enter)  
   - Type:
     ```
     python --version
     ```
   - You should see:
     ```
     Python 3.11.8
     ```

4. Troubleshooting  
   If you see an error:  
   - Close and reopen Command Prompt and try again.  
   - Restart your computer if needed.

---

## Step 2: Install Required Python Packages

1. Open Command Prompt.

2. Run:
   ```
   pip install pyserial==3.5 pygame==2.5.2 numpy==1.26.4 pandas==2.2.1 matplotlib==3.8.3 seaborn==0.13.2
   ```

3. Check installation  
   Launch Python:
   ```
   python
   ```
   Then run:
   ```
   import serial, pygame, numpy as np, pandas as pd, matplotlib, seaborn
   print(serial.__version__, pygame.__version__, np.__version__, pd.__version__, matplotlib.__version__, seaborn.__version__)
   ```
   You should see:
   ```
   3.5 2.5.2 1.26.4 2.2.1 3.8.3 0.13.2
   ```

4. Exit Python:
   ```
   exit()
   ```

---

## Step 3: Install Git

1. Download Git for Windows:  
   https://git-scm.com/download/win

2. Run the installer.  
   Keep all default settings during installation.

3. Verify installation:
   ```
   git --version
   ```
   You should see something like:
   ```
   git version 2.xx.x.windows.x
   ```

---

## Step 4: Create a GitHub Account (if needed)

1. Go to https://github.com and click "Sign Up".

2. Use your real name or a recognizable username.

3. Verify your email and log in.

---

Once setup is complete, move on to [GIT_WORKFLOW.md](GIT_WORKFLOW.md) to fork the repo and clone it to your computer.
