import random
import time
import json
import numpy as np
from core.eeg_client import EEGClient
from core.inference import InferenceEngine
from core.vocabulary import VocabularyManager

class Controller:
	def __init__(self, vocab_file, model_path, biopac_ip, cmd_port, recv_port):
		print("🧠 Controller is starting...")
		# Init EEG client
		self.client = EEGClient(biopac_ip, cmd_port, recv_port)
		self.client.test_connection()

		# Init vocabulary manager
		self.vocab = VocabularyManager(vocab_file)
		self.vocab.load()
		self.vocab.shuffle_words()

		# Init inference engine
		self.infer = InferenceEngine(model_path)
	
	def run_memory_phase(self, start_idx, count):
		for i in range(start_idx, start_idx + count):
			if i >= self.vocab.size():
				print("Exceeding vocabulary size, stopping the round.")
				break

			# Display the word in the console
			word = self.vocab.get_word(i)
			print(f"\n🧠 {i+1} : {word.english} ({word.chinese})")

			# Send command to start EEG recording
			self.client.send_command("START", word.id)
			time.sleep(11)	# Set the memory time per word here
			print("\n" * 50)

			# Send command to stop EEG recording and receive data
			eeg_raw_data = self.client.send_and_receive("STOP", word.id)
			if not eeg_raw_data:
				print("❌ No EEG data received, skipping this word.")
				continue

			# parse the received EEG data
			lines = eeg_raw_data.strip().splitlines()[1:]
			float_data = [float(line) for line in lines]
			data = np.array(float_data)[-5000:]
			# print(len(data), "data points received")
			if len(data) < 5000:
				print(len(data), "data points received")
				print("❌ Not enough data points, skipping this word.")
				continue
			
			# Run inference on the EEG data
			probs = self.infer.predict_from_json(data)

			# Store the state probabilities in the word object
			word.state_probs = probs
			print(f"✅ Status probs: {probs}")
        
	def run_test_phase(self, start_idx, count, test_count, eeg_based):
		print("\n📝 Starting testing phase...")
		# 依照 memory 機率由低到高挑選
		candidates = self.vocab.get_all()[start_idx:start_idx + count]
		if eeg_based:
			# candidates = sorted(candidates, key=lambda w: w.state_probs['memory'] + w.state_probs['focus'], reverse=False)
			candidates = sorted(candidates, key=lambda w: w.state_probs['relax'], reverse=True)
		else:
			random.shuffle(candidates)
		test_words = candidates[:test_count]

		for word in test_words:
			print(f"👉 : {word.chinese}")
			ans = input("Please answer the English word: ")
			correct = ans.strip().lower() == word.english.lower()
			word.test_history.append(correct)
			if correct:
				print("✅ Correct!")
				time.sleep(1)  # Give some time before the next word
			else:
				print(f"❌ Wrong: {word.english} ({word.chinese})")
				time.sleep(5)  # Give some time before the next word
			print("\n" * 50)

	def show_all_vocab(self, vocab_num):
		print("\n📝 Showing all vocabulary...")
		test_vocab = self.vocab.get_all()[:vocab_num]
		for idx, word in enumerate(test_vocab):
			print(f"{idx+1} : {word.english} ({word.chinese})")

	def save_results(self, vocab_num, path):
		result = []
		test_vocab = self.vocab.get_all()[:vocab_num]
		for word in test_vocab:
			result.append({
				"id": word.id,
				"english": word.english,
				"chinese": word.chinese,
				"state_probs": word.state_probs,
				"test_history": word.test_history
			})
		with open(path, 'w', encoding='utf-8') as f:
			json.dump(result, f, ensure_ascii=False, indent=2)
		print("💾 Save result in :", path)

	def examine(self, vocab_num):
		correct_count = 0
		print("\n📝 Starting final examination...")
		test_vocab = self.vocab.get_all()[:vocab_num]
		random.shuffle(test_vocab)
		for idx, word in enumerate(test_vocab):
			print(f"\n👉 {idx+1} : {word.chinese}")
			ans = input("Please answer the English word: ")
			correct = ans.strip().lower() == word.english.lower()
			word.exam_history.append(correct)
			if correct:
				print("✅ Correct!")
				correct_count += 1
			else:
				print(f"❌ Wrong, answer :{word.english}")

		score = correct_count / vocab_num * 100
		print(f"\n🎉 Final exam completed! Your score: {score:.2f}%")

	def save_exam_results(self, vocab_num, path):
		exam_result = []
		test_vocab = self.vocab.get_all()[:vocab_num]
		for word in test_vocab:
			exam_result.append({
				"id": word.id,
				"english": word.english,
				"chinese": word.chinese,
				"exam_history": word.exam_history
			})
		with open(path, 'w', encoding='utf-8') as f:
			json.dump(exam_result, f, ensure_ascii=False, indent=2)
		print("💾 Save exam result in :", path)
