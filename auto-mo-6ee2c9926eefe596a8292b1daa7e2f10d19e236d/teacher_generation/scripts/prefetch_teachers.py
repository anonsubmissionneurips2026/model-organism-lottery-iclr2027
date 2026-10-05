import os, sys, json, time
from huggingface_hub import snapshot_download
os.environ.setdefault("HF_HUB_DISABLE_XET","1")
teachers=json.load(open("scripts/gemma_milsub_samearch_subliminal_teachers.json"))["teachers"]
base=("model-organisms-for-real/gemma-3-1b-vanilla-dpo-123-seed","gemma_3_1b_dpo__123__1777552336")
targets=[(t["repo"],t["revision"]) for t in teachers]+[base]
for repo,rev in targets:
    for attempt in range(1,5):
        try:
            p=snapshot_download(repo_id=repo, revision=rev, allow_patterns=["*.safetensors","*.json","*.model","tokenizer*"])
            print(f"[prefetch] OK {repo}@{rev}", flush=True); break
        except Exception as e:
            print(f"[prefetch] retry {attempt} {repo}: {type(e).__name__}", flush=True); time.sleep(attempt*15)
print("[prefetch] done", flush=True)
