# Running Your Experiment

Follow these steps to locate your assigned experiment, set up configuration options, and run the code on a lab machine.

---

## Step 1: Locate Your Experiment Folder

1. Open Command Prompt.

2. Navigate to the main repository folder:
   ```
   cd %USERPROFILE%\Desktop\mq_honours_2025
   ```

3. List the contents:
   ```
   dir
   ```

4. Find the folder for your assigned project:
   ```
   cd projects\<your_project_name>\code
   ```

Replace `<your_project_name>` with your actual project folder name.

---

## Step 2: Set the Subject Number

1. Open the `run_exp.py` file in a plain text editor like Notepad or Notepad++.

2. Find the line of code that sets the subject number:
   ```
   subject = 1
   ```

3. Choose the next available subject number that has not been used yet.

4. If you are only piloting and do not intend to use the data for publication, use a number like:
   ```
   subject = 999
   ```

---

## Step 3: Configure Motion Tracking (if applicable)

Some experiments use a motion tracker. If yours does:

1. Switch on the Liberty tracker.

2. Wait for the blinking red light to become a steady green light.

3. In `run_exp.py`, find and set:
   ```
   use_liberty = True
   ```

If you are piloting or using a trackpad instead of the tracker, set:
   ```
   use_liberty = False
   ```

---

## Step 4: Run the Experiment

**Do not double click** or otherwise use the mouse to run
any Python scripts. It can create intermediate file that are
annoying to clean up and track with git. 

In Command Prompt, run the experiment:
```
python run_exp.py
```

Follow the on-screen instructions.

If the experiment uses motion tracking, you will be guided through a calibration procedure before the actual experiment begins. This may involve placing the sensor at specific locations and adjusting the physical setup as needed.

---

## Troubleshooting

### If Python is not recognized:
- Close and reopen Command Prompt and try again.
- Restart your computer.
- Reinstall Python and make sure to check "Add Python to PATH" during installation.

### If you see a ModuleNotFoundError:
- Install the required packages again:
  ```
  pip install -r requirements.txt
  ```

If you continue to experience issues, contact your supervisor.
