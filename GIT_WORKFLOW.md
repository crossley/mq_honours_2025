# Git Workflow

This document describes how to fork the main repository, clone it to a lab computer, work with your data files, and submit your changes to your supervisor.

---

## Step 1: Fork the Repository

1. Go to the main repository:  
   https://github.com/crossley/mq_honours_2025

2. Click the "Fork" button in the top-right corner of the page.  
   This creates a copy of the repository under your GitHub account.

---

## Step 2: Clone Your Fork to the Lab Computer

1. Go to your forked version of the repository:  
   ```
   https://github.com/YOUR-GITHUB-USERNAME/mq_honours_2025
   ```

2. Click the green "Code" button and copy the HTTPS URL.

3. Open Command Prompt and navigate to a folder (for example, your Desktop):
   ```
   cd %USERPROFILE%\Desktop
   ```

4. Run the following command, replacing `YOUR-GITHUB-USERNAME`:
   ```
   git clone https://github.com/YOUR-GITHUB-USERNAME/mq_honours_2025.git
   ```

5. Navigate into the repository:
   ```
   cd mq_honours_2025
   ```

6. Confirm the repository is linked to your GitHub account:
   ```
   git remote -v
   ```

---

## Step 3: Pull Before You Push

Before staging and committing your data, always pull the latest version from GitHub to make sure your local repository is up to date.

1. Inside your local repository:
   ```
   cd mq_honours_2025
   ```

2. Pull the latest changes:
   ```
   git pull
   ```

This helps avoid conflicts and prevents your push from being rejected.

---

## Step 4: Add, Commit, and Push Your Data

Once you have collected data, it exists only on the local machine. Follow these steps to push it to your GitHub fork:

1. Make sure you're inside the repository folder:
   ```
   cd mq_honours_2025
   ```

2. Check which files have changed:
   ```
   git status
   ```

3. Stage your new data files:
   ```
   git add projects/<your_project_name>/data/*
   ```

4. Commit your changes with a message:
   ```
   git commit -m "Added experiment data for <your_name>"
   ```

5. Push the changes to your GitHub repository:
   ```
   git push origin main
   ```

---

## Step 5: Submit a Pull Request

To share your data with your supervisor, submit a pull request.

1. Go to your GitHub repository (your fork).

2. Click the "Pull Requests" tab.

3. Click "New Pull Request".

4. Set the base repository to `crossley/mq_honours_2025` and the head repository to your fork.

5. Add a short description (for example, "Added experiment data") and click "Create Pull Request".

Your supervisor will review and merge your data into the main repository.

Great — here’s a lightweight, plain-Markdown troubleshooting guide you can include at the bottom of `GIT_WORKFLOW.md` or link to separately if you prefer.

---

## Troubleshooting Common Git Issues

If something goes wrong while using Git, here are a few common problems and how to fix them.

---

### Problem: Git rejects your push

You might see:
```
error: failed to push some refs to 'https://github.com/YOUR-USERNAME/repo.git'
hint: Updates were rejected because the remote contains work that you do
not have locally.
```

**What to do:**

1. Pull the latest changes before pushing:
   ```
   git pull
   ```

2. If there are no conflicts, try pushing again:
   ```
   git push origin main
   ```

---

### Problem: You have uncommitted changes and can't pull

You might see:
```
error: Your local changes would be overwritten by merge
```

**What to do:**

1. Save your work by committing your changes:
   ```
   git add .
   git commit -m "Saving local changes before pulling"
   ```

2. Then pull and push:
   ```
   git pull
   git push
   ```

---

### Problem: Git asks for your username and password every time

This can happen on some Windows machines. To fix:

1. Enable Git credential helper:
   ```
   git config --global credential.helper wincred
   ```

2. Next time you enter your username and password, Git will remember it.

---

### Still stuck?

If you're unsure what's going on, it's always safe to:

- **Copy your data files to a safe folder**
- **Delete your local repository folder**
- **Re-clone your fork** and re-add your data
- Then:
  ```
  git add .
  git commit -m "Re-added data"
  git push
  ```

When in doubt, ask your supervisor for help.
