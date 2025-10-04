# data.py
import random, json
from config import CLASSES, POKEMON_SYNONYMS, ACTION_WORDS, NEGATION_WORDS, TACTICAL_FILLER

class PromptProcessor:
    """Generate synthetic Pokémon tactical prompts for training NeoBERT"""

    def __init__(self):
        self.filler_phrases = TACTICAL_FILLER

    def load_train_prompts(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        processed = []
        for item in data:
            prompt = item['prompt']
            target = prompt.split(": ")[1].strip().lower() if ": " in prompt else prompt.split(" ")[1].lower()
            if target in CLASSES:
                processed.append({"prompt": prompt, "target": target, "image_id": item.get("image_id", "")})
        return processed

    def generate_synthetic_prompts(self, num_prompts=10000):
        print(f"🧠 Generating {num_prompts} prompts…")
        prompts = []
        per_class = num_prompts // len(CLASSES)
        for target in CLASSES:
            for _ in range(per_class):
                prompts.append(self._gen_tactical_report(target))
        random.shuffle(prompts)
        return prompts

    def _get_synonyms(self, p): return [k for k,v in POKEMON_SYNONYMS.items() if v==p] + [p]

    def _gen_tactical_report(self, target):
        alias = random.choice(self._get_synonyms(target))
        action = random.choice(ACTION_WORDS)
        filler = random.choice(self.filler_phrases)
        return {
            "prompt": f"{filler}. Mission objective: {action} the {alias}.",
            "target": target
        }
