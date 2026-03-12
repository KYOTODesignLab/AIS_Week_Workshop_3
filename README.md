# AIS_Week_Workshop_3

# Getting Started

## Software requirements

1. Install the following on your system:

* VS Code: https://code.visualstudio.com/
* GIT : https://git-scm.com/


> NOTE: On Mac, GIT is usually pre-installed, you do not need to install it again. To check if it is installed, open a terminal window and type `git`, if the result does not indicate any error, it means you already have it in your system.
>

* Miniconda: https://www.anaconda.com/download/success
>
Make sure to add installationto my PATH variable!
>
<img width="380" height="303" alt="image" src="https://github.com/user-attachments/assets/4cfaa5f0-4e73-40ca-ad57-cb58a58a7b3f" />



2. Open VS Code, and in the file menu, open your documents folder. Navigate to the terminal window (ctr + @).

If you don't have a GitHub subfolder yet, create it by pasting the following lines:
```
md GitHub
```

Once it is done, navigate to the folder and clone the repository.
```
cd GitHub
git clone https://github.com/grgle/AIS_Week_Workshop_3.git
```

3. In VS Code, set the default terminal profile to command prompt, by clicking on the down arrow next to the + symbol at the top right corner of the terminal window, click "Select Default Profile", and then click "Command Prompt" from the menu in the top middle.

Open your repository folder (`GitHub/AIS_Week_Workshop_3'), and create the necessary virtual environment with conda:

```
conda env create -f environment.yml
```

To update the environment with conda, navigate to your repository root folder, and input the following line to the terminal:

```
conda env update -f environment.yml
```

Then activate the environmet:

```
conda activate AIS26
```