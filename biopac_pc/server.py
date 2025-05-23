import socket
import json

from biopac_control import start_recording, stop_recording

IP = "127.0.0.1"
CMD_PORT = 65432
RECV_PORT = 65433

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
                start_recording()
            elif msg['cmd'] == 'STOP':
                eeg_json = stop_recording()
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                    s2.connect((IP, RECV_PORT))
                    s2.sendall(json.dumps(eeg_json).encode())
