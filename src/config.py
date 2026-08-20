from pathlib import Path

# 获取当前项目根目录 E:\code\ai_review
ROOT_PATH = Path(__file__).parents[1]
DATA_PATH = ROOT_PATH / 'data'
LOG_PATH = ROOT_PATH / 'log'
MODEL_PATH = ROOT_PATH / 'model'

BATCH_SIZE = 2
LR = 1e-5
EPOCH=1
STEP_SIZE = 200

if __name__ == "__main__":
    print(ROOT_PATH)
