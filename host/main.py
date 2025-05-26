from core import Controller
import socket

# Get the local IP address
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
print("Local IP:", local_ip)

VOCAB_FILE = "data/vocab1.txt"
BIOPAC_IP = "172.20.10.3"
VOCAB_PER_ROUND = 20
TEST_COUNT = 5

controller = Controller(
	vocab_file=VOCAB_FILE,
	model_path='models/eeg_state_classifier.pt',
	biopac_ip=BIOPAC_IP, 
	cmd_port=65432, 
	recv_port=65433
)


for round_num in range(5):
    print(f"\n==== 📚 Round {round_num+1} remember phase ====")
    controller.run_memory_round(
			start_idx = round_num * VOCAB_PER_ROUND, 
			count = VOCAB_PER_ROUND
        )
    print(f"\n==== 📝 Round {round_num+1} test phase ====")
    controller.run_test_round(
			start_idx = round_num * VOCAB_PER_ROUND, 
			count = VOCAB_PER_ROUND, 
			test_count = TEST_COUNT, 
			eeg_based = True
        )
    
    controller.save_results(f"results/memory_round{round_num+1}.json")

controller.save_results("results/final_results.json")

