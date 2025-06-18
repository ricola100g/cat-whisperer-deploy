import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class KnowledgeRAG:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.texts = []
        self.entries = []

        for section in raw.get("behaviors", []):
            name = section.get("behavior_name_zh_tw", "")
            for interp in section.get("interpretations", []):
                meaning = interp.get("meaning_zh_tw", "")
                context = interp.get("context_zh_tw", "")
                emotion = ",".join(interp.get("emotion_tags", []))
                self.texts.append(f"{name}：{meaning}（{context}）〔情緒：{emotion}〕")
                self.entries.append(interp)

        self.vecs = self.model.encode(self.texts, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.vecs.shape[1])
        self.index.add(self.vecs)

    def query(self, text, k=3):
        qv = self.model.encode([text], convert_to_numpy=True)
        D, I = self.index.search(qv, k)
        return [self.texts[i] for i in I[0]]
