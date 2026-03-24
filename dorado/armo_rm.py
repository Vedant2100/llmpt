import json
import argparse
import os
from typing import Dict, List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

class ArmoRMPipeline:
    def __init__(self, model_id, device_map="cuda", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[List[Dict[str, str]]]) -> List[Dict[str, float]]:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            scores = outputs.logits.squeeze().float().tolist()  # Handle batch outputs
        return [{"score": score} for score in scores]

def process_file(args):
    rm = ArmoRMPipeline(args.model_id, trust_remote_code=args.trust_remote_code, device_map=args.device_map, max_length=args.max_length)

    with open(args.input_file, 'r') as f:
        data = json.load(f)

    batched_data = [data[i:i + args.batch_size] for i in range(0, len(data), args.batch_size)]
    
    for batch in tqdm(batched_data, desc="Processing items in batches"):
        prompts = []
        responses1 = []
        responses2 = []

        for item in batch:
            prompt = item["synthesized_prompt"]
            response1 = item["synthesized_response_1"]
            response2 = item["synthesized_response_2"]
            prompts.append([{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}])
            responses1.append([{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}])
            responses2.append([{"role": "user", "content": prompt}, {"role": "assistant", "content": response2}])

        scores1 = rm(responses1)
        scores2 = rm(responses2)

        for idx, item in enumerate(batch):
            score1 = scores1[idx]
            score2 = scores2[idx]

            item["synthesized_response_1_score"] = score1["score"]
            item["synthesized_response_2_score"] = score2["score"]

            if score1["score"] >= score2["score"]:
                item["rm_order"] = "1 > 2"
                item["chosen"] = item["synthesized_response_1"]
                item["rejected"] = item["synthesized_response_2"]
            else:
                item["rm_order"] = "2 > 1"
                item["chosen"] = item["synthesized_response_2"]
                item["rejected"] = item["synthesized_response_1"]

    input_dir = os.path.dirname(args.input_file)
    output_file = os.path.join(input_dir, "rm_" + os.path.basename(args.input_file))

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Processing complete. Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process JSON file with ArmoRMPipeline")
    parser.add_argument("--input_file", type=str, default="/scratch/avt2gy/persona-hub/score_test_response.json", help="Input JSON file path")
    parser.add_argument("--model_id", type=str, default="RLHFlow/ArmoRM-Llama3-8B-v0.1", help="Model ID for ArmoRMPipeline")
    parser.add_argument("--trust_remote_code", type=bool, default=True, help="Trust remote code for model loading")
    parser.add_argument("--device_map", type=str, default="cuda", help="Device map for model")
    parser.add_argument("--max_length", type=int, default=4096, help="Max length for tokenizer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")

    args = parser.parse_args()
    process_file(args)

if __name__ == "__main__":
    main()
# python code/ArmoRM.py  --input_file /scratch/avt2gy/persona-hub/70B_rank/meta-llama_Meta-Llama-3.1-70B-Instruct-Meta-Llama-3-8B-Instruct-100.json
# python code/ArmoRM.py  --input_file /scratch/avt2gy/persona-hub/70B_rank/meta-llama_Meta-Llama-3.1-70B-Instruct-llama-gen-instruct-1e-6-all-55000.json
# python code/ArmoRM.py  --input_file /scratch/avt2gy/persona-hub/70B_rank/meta-llama_Meta-Llama-3.1-70B-Instruct-llama-gen-instruct-1e-6-it2-all-55000.json