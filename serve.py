# serve.py
import json
import sys

########################################################
# 서빙 모델
########################################################

r = json.load(open("/eval_out/results.json"))

if not r["pass"]:
    print("❌❌❌❌❌❌❌❌❌❌ evaluation failed. not serving.")
    sys.exit(1)

print("🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀 serving model... 🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀 ")
print("Hello from model!")
