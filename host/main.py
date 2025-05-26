from core import Controller
import socket
import time

# Get the local IP address
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
print("Local IP:", local_ip)

VOCAB_FILE = "data/vocab1.txt"
BIOPAC_IP = "172.20.10.3"
ROUND = 2
VOCAB_PER_ROUND = 5
TEST_COUNT = 3

controller = Controller(
	vocab_file=VOCAB_FILE,
	model_path='models/eeg_state_classifier.pth',
	biopac_ip=BIOPAC_IP, 
	cmd_port=65432, 
	recv_port=65433
)

# Memory phase and test phase
for round_num in range(ROUND):
	print(f"\n==== 📚 Round {round_num+1} memory phase ====")
	controller.run_memory_phase(
			start_idx = round_num * VOCAB_PER_ROUND, 
			count = VOCAB_PER_ROUND
		)
	print("\n" * 50)
	print(f"\n==== 📝 Round {round_num+1} test phase ====")
	controller.run_test_phase(
			start_idx = round_num * VOCAB_PER_ROUND, 
			count = VOCAB_PER_ROUND, 
			test_count = TEST_COUNT, 
			eeg_based = True
		)
	print("\n" * 50)
    
    # controller.save_results(round_num * VOCAB_PER_ROUND, f"results/memory_round{round_num+1}.json")

controller.save_results(ROUND * VOCAB_PER_ROUND, "results/final_results.json")

# Final review
print("\n" * 50)
print("\n==== 📖 Final review ====")
controller.show_all_vocab(VOCAB_PER_ROUND * ROUND)
time.sleep(10)
print("\n" * 50)

# Final examination
print("\n" * 50)
print("\n==== 📝 Final examination ====")
controller.examine(VOCAB_PER_ROUND * ROUND)
controller.save_exam_results(VOCAB_PER_ROUND * ROUND, "results/final_exam_results.json")