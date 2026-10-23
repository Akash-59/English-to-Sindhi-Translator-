# Installation and Setup Guide

## System Requirements
- **Operating System**: Windows 10 / Windows 11 (64-bit) or Linux (Ubuntu 20.04+, Debian 11+, Fedora 36+).
- **RAM**: 8 GB physical memory (System uses < 3.5 GB peak RSS).
- **Processor**: Dual-core or Quad-core CPU (Intel Core i3/i5/i7/i9 or AMD Ryzen).
- **GPU**: None required (100% CPU-only inference).
- **Disk Space**: ~4 GB free disk space.
- **Python**: Version 3.10 or higher.

---

## 1. Quick Installation (Online Setup Phase)

### Step 1: Clone or Open Workspace
```bash
git clone https://github.com/your-username/sindhi-ai-translator.git
cd sindhi-ai-translator
```

### Step 2: Install PyTorch CPU-Only Wheel
Installing the dedicated CPU build avoids downloading unnecessary 2+ GB CUDA binaries:
```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install Core Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 4 (Windows Specific): Unblock Installed DLLs
On Windows 11 with Smart App Control or AppLocker enabled, user-profile downloaded DLLs can receive a security block (WinError 4551). Unblock them once via PowerShell:
```powershell
Get-ChildItem -Path $env:LOCALAPPDATA\Programs\Python -Recurse -Include *.dll,*.pyd | Unblock-File
```

---

## 2. Model Preparation & Offline Packaging

To use the system entirely offline, download the translation checkpoint into the local models directory:
```bash
python download_weights.py
```
This downloads `pytorch_model.bin` directly into:
`models/translation/nllb-200-distilled-600M/`

Train the local diacritic restoration model:
```bash
python -m src.diacritics.train_diacritics
```
This produces `models/diacritics/diacritic_restorer.joblib`.

---

## 3. Running the Application

Launch the local Streamlit user interface:
```bash
streamlit run app.py
```
The interface will automatically open in your default browser at `http://localhost:8501`.

---

## 4. Running the Test Suite

Execute the comprehensive test suite verifying Unicode, Diacritics, Preprocessing, and Offline guarantees:
```bash
python -m pytest tests/ -v
```

---

## 5. Offline Verification Procedure

To strictly prove that the application executes without internet connectivity:
1. Disconnect your computer from Wi-Fi and Ethernet (or enable Airplane Mode).
2. Set the environment variable:
   - On Windows PowerShell: `$env:FORCE_OFFLINE="1"`
   - On Linux/Bash: `export FORCE_OFFLINE=1`
3. Launch the application:
   ```bash
   streamlit run app.py
   ```
4. Enter an English sentence (e.g. `I am studying at the university.`) and click **Translate to Sindhi**.
5. Observe instantaneous generation in Perso-Arabic Sindhi script with zero outbound network calls!
