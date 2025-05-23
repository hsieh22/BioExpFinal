import time
import base64
import numpy as np
import pyautogui
import pyperclip

recording = False
mock_data = None

def start_recording():
    global recording, mock_data
    print("🎥 Start to record EEG data...")
    recording = True
    pyautogui.hotkey('ctrl', 'space')

def stop_recording():
    global recording, mock_data
    print("⏸️ Stop recording EEG data, sending data back...")

    recording = False


def stop_recording():

	pyautogui.hotkey('ctrl', 'space')
	time.sleep(0.5)
	pyautogui.hotkey('ctrl', 'a')
	time.sleep(0.2)
	pyautogui.hotkey('ctrl', 'l')
	time.sleep(0.5)
	# os.system("start notepad")
	# time.sleep(1.5)  # 等待記事本開啟
	# pyautogui.hotkey('ctrl', 'a')
	# time.sleep(0.2)
	# pyautogui.hotkey('ctrl', 'v')
	# time.sleep(0.5)
	# pyautogui.hotkey('ctrl', 's')
	# time.sleep(0.5)
	# pyautogui.hotkey('alt', 'f4')
	# time.sleep(0.5)
    
	text = pyperclip.paste()  # 從剪貼簿取得純文字資料
	lines = text.strip().splitlines()
	values = np.array([float(x) for x in lines], dtype='float32')

