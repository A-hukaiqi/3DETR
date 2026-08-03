# 3DETR - Unity Scene Reconstruction

This project aims to reconstruct indoor scenes from RGB-D data using modern 3D deep learning methods (PointNet++, VoteNet, 3DETR), with the eventual goal of generating Unity-compatible scene representations.

---

# Repository Structure

```
3DETR/
│
├── rawdata/           # Raw datasets (ignored by git)
│
├── src/               # Source code
│
├── notebooks/         # Jupyter notebooks
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

---

# Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/<your_username>/3DETR.git

cd 3DETR
```

---

## 2. Create a Python virtual environment

### macOS / Linux

```bash
python3 -m venv .venv
```

Activate it

```bash
source .venv/bin/activate
```

---

### Windows (PowerShell)

```powershell
python -m venv .venv
```

Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install project dependencies

Upgrade pip

```bash
python -m pip install --upgrade pip
```

Install all required packages

```bash
pip install -r requirements.txt
```

---

## 4. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name=3detr --display-name="Python (3DETR)"
```

---

## 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Select the kernel

```
Python (3DETR)
```

---

## 6. Download the dataset

Download the SUN RGB-D dataset.

Place it inside

```
rawdata/

└── SUNRGBD/
```

Since `rawdata/` is ignored by git, the dataset will not be committed.

---

## 7. Verify the installation

Run

```python
import numpy
import scipy
import cv2
import matplotlib
import open3d
```

If no errors occur, the environment is ready.

---

# Git Workflow

The project follows a feature-branch workflow.

```
main
│
├── alex
└── partner
```

Never commit directly to `main`.

Create a feature branch

```bash
git checkout -b feature-name
```

Push the branch

```bash
git push -u origin feature-name
```

Open a Pull Request on GitHub once the feature is complete.

---

# Notes

- Raw datasets are **not** tracked by git.
- Large generated files (processed datasets, checkpoints, etc.) should also remain outside version control.
- Reusable code should live in `src/`.
- Exploratory work should be done in `notebooks/`.