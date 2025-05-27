import random
import time
import json
import numpy as np
from core.eeg_client import EEGClient
from core.inference import InferenceEngine
from core.vocabulary import VocabularyManager
from collections import Counter

class Controller:
	def __init__(self, vocab_file, model_path, biopac_ip, cmd_port, recv_port, eeg_based, round, vocab_per_round):
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

		# Settings
		self.eeg_based = eeg_based
		if self.eeg_based:
			print("🧠 EEG-based mode is enabled.")
		else:
			print("🧠 EEG-based mode is disabled.")

		self.round = round
		self.vocab_per_round = vocab_per_round
		self.vocab_num = round * vocab_per_round

	def run_memory_phase(self, round):
		start_idx = round * self.vocab_per_round
		end_idx = start_idx + self.vocab_per_round
		for i in range(start_idx, end_idx):
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
        
	def run_test_phase(self, round, test_count):
		print("\n📝 Starting testing phase...")
		start_idx = round * self.vocab_per_round
		end_idx = start_idx + self.vocab_per_round

		# pick test words
		candidates = self.vocab.get_all()[start_idx:end_idx]
		if self.eeg_based:
			# A.sort 
			# candidates = sorted(candidates, key=lambda w: w.state_probs['relax'], reverse=True)

			# B.randomly select 
			alpha = 5
			relax_scores = np.array([w.state_probs['relax'] for w in candidates])
			exp_scores = np.exp(alpha * relax_scores)
			probs = exp_scores / np.sum(exp_scores)
			# random pick 3 words based on the probabilities
			selected = np.random.choice(candidates, size=3, replace=False, p=probs)
			candidates = list(selected)
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

	def show_all_vocab(self):

		print("\n📝 Showing all vocabulary...")
		test_vocab = self.vocab.get_all()[:self.vocab_num]
		for idx, word in enumerate(test_vocab):
			print(f"{idx+1} : {word.english} ({word.chinese})")

	def examine(self):
		correct_count = 0
		print("\n📝 Starting final examination...")
		test_vocab = self.vocab.get_all()[:self.vocab_num]
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

		score = correct_count / self.vocab_num * 100
		print(f"\n🎉 Final exam completed! Your score: {score:.2f}%")

	def save_results(self, path):
		result = []
		test_vocab = self.vocab.get_all()[:self.vocab_num]
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

	def save_exam_results(self, path):
		exam_result = []
		test_vocab = self.vocab.get_all()[:self.vocab_num]
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

	def save_statistics(self, path):
		test_vocab = self.vocab.get_all()[:self.vocab_num]
		# calculate average probabilities
		avg_probs = Counter()
		for word in test_vocab:
			avg_probs.update(word.state_probs)
		for state in avg_probs:
			avg_probs[state] /= self.vocab_num

		# calculate average test and exam history
		avg_test_history = {'correct': 0, 'wrong': 0}
		avg_exam_history = {'correct': 0, 'wrong': 0}
		for word in test_vocab:
			for history, record in [('test_history', avg_test_history), ('exam_history', avg_exam_history)]:
				hist = getattr(word, history)
				record['correct'] += sum(hist)
				record['wrong'] += len(hist) - sum(hist)

		# save statistics
		stats = {
			'avg_probs': dict(avg_probs),
			'avg_test_history': avg_test_history,
			'avg_exam_history': avg_exam_history
		}

		with open(path, 'w', encoding='utf-8') as f:
			json.dump(stats, f, ensure_ascii=False, indent=2)

