"""SAE features of a text.

The text is read by Llama-3.1-8B-Instruct, and the residual stream of layer 19 is passed through the encoder of the
Goodfire sparse autoencoder (65,536 features). A feature is present in a text if it is among the 24 most active
features on at least one of the first 512 tokens. For every present feature we keep its largest activation and the
token on which it occurs. Both models are frozen.
"""
import numpy as np

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SAE_REPO, SAE_FILE = "Goodfire/Llama-3.1-8B-Instruct-SAE-l19", "Llama-3.1-8B-Instruct-SAE-l19.pth"
LAYER, TOP_K, MAX_TOKENS = 19, 24, 512


class Encoder:
    def __init__(self, model=MODEL, sae_repo=SAE_REPO, sae_file=SAE_FILE, device=None):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(hf_hub_download(sae_repo, sae_file), map_location=self.device, weights_only=True)
        self.W = state["encoder_linear.weight"].to(self.device).float().T.contiguous()
        self.b = state["encoder_linear.bias"].to(self.device).float()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            llm = AutoModel.from_pretrained(model, dtype=torch.bfloat16)
        except TypeError:                                  # older versions of transformers
            llm = AutoModel.from_pretrained(model, torch_dtype=torch.bfloat16)
        self.model = llm.to(self.device).eval()

    def encode(self, texts, batch_size=32, progress=False):
        """For every text: (feature ids, largest activation of each, token position of that activation)."""
        torch, out = self.torch, []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = [t if t.strip() else "x" for t in texts[start:start + batch_size]]
                enc = self.tokenizer(chunk, return_tensors="pt", truncation=True, max_length=MAX_TOKENS,
                                     padding=True).to(self.device)
                hidden = self.model(**enc, output_hidden_states=True).hidden_states[LAYER].float()
                mask = enc["attention_mask"].bool()
                for j in range(len(chunk)):
                    h = hidden[j][mask[j]]
                    h = h[1:] if h.shape[0] > 1 else h                      # drop the begin-of-text token
                    z = torch.relu(h @ self.W + self.b)
                    top = torch.topk(z, min(TOP_K, z.shape[1]), dim=1)
                    ids, values = top.indices.reshape(-1), top.values.reshape(-1)
                    tokens = torch.arange(z.shape[0], device=z.device).repeat_interleave(top.indices.shape[1])
                    present, inverse = torch.unique(ids, return_inverse=True)
                    peak = torch.full((len(present),), -1.0, device=z.device).scatter_reduce(
                        0, inverse, values, reduce="amax", include_self=False)
                    position = torch.zeros(len(present), dtype=torch.long, device=z.device)
                    position.scatter_reduce_(0, inverse, torch.where(values >= peak[inverse] - 1e-6, tokens,
                                                                     torch.zeros_like(tokens)),
                                             reduce="amax", include_self=False)
                    out.append((present.cpu().numpy().astype(np.int32), peak.cpu().numpy().astype(np.float16),
                                position.cpu().numpy().astype(np.int16)))
                if progress and (start // batch_size) % 20 == 0:
                    print(f"  encoded {min(start + batch_size, len(texts)):,}/{len(texts):,}", flush=True)
        return out

    def features(self, text):
        """Ids of the SAE features present in one text."""
        return self.encode([text])[0][0]
