# mq_honours_2025

This repo holds code for a variety 2025 honours projects at Macquarie University.

# **Setting Up a Lab Windows Machine to Run These Experiment**

## **Step 1: Install Python (Version 3.11.8)**

1. **Download Python 3.11.8:**  
   - Visit: [https://www.python.org/downloads/release/python-3118/](https://www.python.org/downloads/release/python-3118/)  
   - Scroll down to **Windows Installer (64-bit)** and click to download.

2. **Install Python:**  
   - Open the downloaded file.  
   - **IMPORTANT:** **Uncheck** the box that says **“Use admin privileges"”** before clicking **Install Now**.  
   - **IMPORTANT:** **Check** the box that says **“Add Python to PATH”** before clicking **Install Now**.  
   - Wait for installation to complete, then close the installer.

3. **Verify the installation:**  
   - Open **Command Prompt** (Press **Win + R**, type `cmd`, and press **Enter**).  
   - Type:  
     ```
     python --version
     ```
   - You should see:  
     ```
     Python 3.11.8
     ```

4. **Common issues**
    - If you see an error, try closing and restarting Command Prompt.

---

## **Step 2: Install Required Python Libraries**
We will now install the required libraries.

1. **Open Command Prompt**.

2. **Run this command**:
   ```
   pip install pyserial==3.5 pygame==2.5.2 numpy==1.26.4 pandas==2.2.1 matplotlib==3.8.3
   ```

3. **Check installation**:
   ```
   python
   ```
   Then type:
   ```
   import serial, pygame, numpy as np, pandas as pd, matplotlib
   print(serial.__version__, pygame.__version__, np.__version__, pd.__version__, matplotlib.__version__)
   ```
   Expected output:
   ```
   3.5 2.5.2 1.26.4 2.2.1 3.8.3
   ```

4. **Exit Python**:
   ```
   exit()
   ```

---

## **Step 3: Create a GitHub Account (If You Don't Have One)**
Since we will be working with GitHub, you need an account.

1. **Go to** - [https://github.com](https://github.com).  

2. Click **Sign Up** and follow the instructions.  

3. **Use your real name or a recognizable username** (so I can identify you).  

4. Verify your email and log in.

---

## **Step 4: Install Git**
Git is required to interact with GitHub.

1. **Download Git for Windows:**  
   - [https://git-scm.com/download/win](https://git-scm.com/download/win)  

2. **Install Git:**  
   - Open the installer and **keep all default settings**.  

3. **Verify installation:**  
   - Open **Command Prompt** and type:  
     ```
     git --version
     ```
   - You should see something like:
     ```
     git version 2.xx.x.windows.x
     ```

---

## **Step 5: Fork and Clone the Experiment Repository**

### **5.1 Fork the Repository (Create Your Own Copy on GitHub)**

1. **Go to the main experiment repository**:  
   - [https://github.com/crossley/mq_honours_2025](https://github.com/crossley/mq_honours_2025)  

2. Click the **Fork** button (top-right corner).  
   - This creates a copy of the repository under **your GitHub account**.

### **5.2 Clone the Repository (Download it to the lab computer)**

1. **Go to your forked version** at:  
   ```
   https://github.com/YOUR-GITHUB-USERNAME/mq_honours_2025
   ```

2. Click the green **“Code”** button and copy the URL.  

3. **Open Command Prompt** and navigate to a folder (e.g., your Desktop):  
   ```
   cd %USERPROFILE%\Desktop
   ```

4. **Run the following command (replace `YOUR-GITHUB-USERNAME` with your actual username):**  
   ```
   git clone https://github.com/YOUR-GITHUB-USERNAME/mq_honours_2025.git
   ```

5. Navigate into the repository:
   ```
   cd mq_honours_2025
   ```

6. **Confirm it’s linked to your GitHub account**:  
   ```
   git remote -v
   ```
   You should see a URL that includes your GitHub username.

---

## **Step 6: Follow My GitHub Account and Let Me Follow You**

Since we will be sharing data via GitHub, make sure we follow each other.

1. **Follow my GitHub account**:  
   - [https://github.com/crossley](https://github.com/crossley)  

2. **Send me a message or email with your GitHub username**, so I can follow you back.  

---

## **Step 7: Navigate to Your Experiment Folder and Run the Code**
1. **Find your experiment folder** inside `mq_honours_2025`:  
   ```
   dir
   ```

2. **Navigate to your assigned experiment folder**:  
   ```
   cd <path_to_your_experiment_name>
   ```

3. **Move into the `code` folder**:  
   ```
   cd code
   ```

4. **Run the experiment**:  
   ```
   python run_exp.py
   ```

---

## **Step 8: Submit Your Data to GitHub**

Once you have collected data, you need to **upload it to GitHub** and submit a pull request so I can review and merge your changes.

### **8.1 Add Your Data Files**

1. **Make sure you're inside the repository**:
   ```
   cd mq_honours_2025
   ```

2. **Check which files have changed**:
   ```
   git status
   ```
   You should see newly created **data files**.

3. **Stage the new data files**:
   ```
   git add data/*
   ```

4. **Commit the changes with a message**:
   ```
   git commit -m "Added experiment data for <your_name>"
   ```

### **8.2 Push Changes to Your Forked Repository**
1. **Push the changes to your GitHub repository**:
   ```
   git push origin main
   ```

### **8.3 Submit a Pull Request**

1. **Go to your GitHub repository** (the forked one under your username).  

2. Click the **Pull Requests** tab.  

3. Click **New Pull Request**.  

4. Set the **base repository** to `crossley/mq_honours_2025` and the **head repository** to your fork.  

5. Add a brief description (e.g., *"Added my experiment data"*) and click **Create Pull Request**.  

I will then review and merge your data into the main repository.

---

## **Troubleshooting**

### **If `git push` asks for credentials repeatedly:**

1. Set up GitHub authentication:
   ```
   git config --global credential.helper wincred
   ```
   Next time you enter your username and password, Windows should remember it.

### **If `ModuleNotFoundError` appears when running the experiment:**

- **Reinstall missing libraries:**  
  ```
  pip install -r requirements.txt
  ```

### **If Python is not recognized:**

- **Restart your computer** and try again.
- **Reinstall Python and check "Add Python to PATH".**

---

## **Congratulations!**
Your Windows machine is now set up to **run the psychology experiment**, **push your data to GitHub**, and **submit pull requests** to keep the main repository updated.
