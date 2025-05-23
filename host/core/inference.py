import numpy as np

class InferenceEngine:
    def __init__(self, model_path):
        print("🤖 Model loaded")

    def predict_from_json(self, eeg_json):
        # 模擬輸出機率
        return {
            'relax': float(np.random.rand()),
            'focus': float(np.random.rand()),
            'memory': float(np.random.rand()),
            'stress': float(np.random.rand())
        }