# EEG Vocabulary Memory System

This project integrates a vocabulary memorization application with EEG data


## 📦 Requirements

- Python 3.8+
- Install packages:

```
pip install pyautogui pyperclip numpy
```

> macOS users must grant "Accessibility" permission to the terminal for automation.

## 🛠 Setup

1. Create a folder for results:

```
mkdir results
```

2. Prepare a vocabulary file:

```
data/vocab.txt
```

Each line should follow this format:

```
apple[TAB]蘋果
```

## ▶️ How to Run

### On the Biopac PC

Start the EEG recording server:

```
cd biopac_pc
python server.py
```

### On the Host PC

Run the vocabulary memory app:

```
cd host
python main.py
```

This will:

- Display a word and trigger EEG recording
- Stop after 10 seconds and collect EEG from clipboard
- Infer cognitive states and select test words accordingly
- Repeat for 5 rounds (100 words total)