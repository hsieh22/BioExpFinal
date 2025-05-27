from core import Controller
import socket
import time

#TODO : 視覺化狀態機率

# Get the local IP address
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
print("Local IP:", local_ip)

VOCAB_FILE = "data/vocab.txt"
BIOPAC_IP = "172.20.10.3"
ROUND = 2
VOCAB_PER_ROUND = 5
TEST_COUNT = 3

controller = Controller(
	vocab_file=VOCAB_FILE,
	model_path='models/eeg_state_classifier.pth',
	biopac_ip=BIOPAC_IP, 
	cmd_port=65432, 
	recv_port=65433,
	eeg_based=True,
	round=ROUND,
	vocab_per_round=VOCAB_PER_ROUND,
)

# Memory phase and test phase
for round_num in range(ROUND):
	print(f"\n==== 📚 Round {round_num+1} memory phase ====")
	controller.run_memory_phase(round_num)
	print("\n" * 50)
	print(f"\n==== 📝 Round {round_num+1} test phase ====")
	controller.run_test_phase(round=round_num, test_count = TEST_COUNT, )
	print("\n" * 50)

controller.save_results("results/final_results.json")

# Final review
print("\n" * 50)
print("\n==== 📖 Final review ====")
controller.show_all_vocab()
time.sleep(10)
print("\n" * 50)

# Final examination
print("\n" * 50)
print("\n==== 📝 Final examination ====")
controller.examine()
controller.save_exam_results("results/final_exam_results.json")
controller.save_statistics("results/statistics.json")