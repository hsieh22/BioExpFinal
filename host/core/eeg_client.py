import socket
import json

class EEGClient:
    def __init__(self, biopac_ip, cmd_port, recv_port):
        self.biopac_ip = biopac_ip
        self.cmd_port = cmd_port
        self.recv_port = recv_port

    def test_connection(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.biopac_ip, self.cmd_port))
                msg = json.dumps({"cmd": "PING", "word_id": 0}) 
                s.sendall(msg.encode())
                response = s.recv(1024).decode()
                if response.strip() == "PONG":
                    print("✅ successfully connected to Biopac Server")
                    return True
                else:
                    print("⚠️ Response is wrong : ", response)
                    return False
        except Exception as e:
            print("❌ Cannot connect to Biopac Server : ", e)
            return False

    def send_command(self, cmd: str, word_id: int):
        msg = json.dumps({"cmd": cmd, "word_id": word_id})
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.biopac_ip, self.cmd_port))
            s.sendall(msg.encode())

    def send_and_receive(self, cmd: str, word_id: int):
        msg = json.dumps({"cmd": cmd, "word_id": word_id})
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.biopac_ip, self.cmd_port))
            s.sendall(msg.encode())

            data = b''
            while True:
                packet = s.recv(4096)
                if not packet:
                    break
                data += packet
            return json.loads(data.decode())