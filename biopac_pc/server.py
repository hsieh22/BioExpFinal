import socket
import json
import time
import base64
import numpy as np

IP = "127.0.0.1"
CMD_PORT = 65432
RECV_PORT = 65433

def mock_eeg_data():
    raw = np.random.randn(32, 10).astype('float32')  # 模擬 10s, 1000Hz
    return base64.b64encode(raw.tobytes()).decode()

print("Server is starting...")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    print("📥 Server is waiting for connection from client...")
    s.bind(('0.0.0.0', CMD_PORT))
    s.listen(1)
    while True:
        conn, addr = s.accept()
        with conn:
            data = conn.recv(1024).decode()
            msg = json.loads(data)
            print(f"Received command: {msg['cmd']}, word_id: {msg['word_id']}")
            if msg['cmd'] == 'PING':  # test connection
              print("🏓receive PING test connection")
              conn.sendall("PONG".encode())
              continue
            elif msg['cmd'] == 'START':
                #TODO: 開始錄製 EEG 資料
                print("🎥 Start to record EEG data...")
            elif msg['cmd'] == 'STOP':
                #TODO: 停止錄製 EEG 資料
                print("⏸️ Stop recording EEG data, sending data back...")
                time.sleep(1)  # 模擬延遲
                eeg_json = {
                    "word_id": msg['word_id'],
                    "data_b64": mock_eeg_data()
                }
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                    s2.connect((IP, RECV_PORT))
                    s2.sendall(json.dumps(eeg_json).encode())
