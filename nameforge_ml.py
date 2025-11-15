# nameforge_ml.py
# Character-level LSTM name generator + scoring + dual logs + final suggestions
# Training data: ./seeds/*.txt  (one name per line)

# to a devolper to devopler you can see female_names.txt not inside seeds folde
# why you ask ? cause this way its create better jappanese style name if you dont like 
# change it ? or add your own data

from pyexpat import model
import os, re, sys, time, math, random, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED_DIR      = "seeds"
LOG_SHORT     = "generated_short.txt"
LOG_LONG      = "generated_long.txt"
FINAL_TOP     = "finalsuggestion.txt"
TOP_CAP       = 50
REFRESH_N     = 25
REF_N         = 4

SHORT_RANGE   = (2, 7)
LONG_RANGE    = (8, 12)

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
RAND          = random.Random(42)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

VOWELS        = set("aeiouyäöüıi")
SOFT_CONS     = set("lmnrshfvy")

RARE_LETTERS  = set("qxw")

# Bad clusters
BAD_CLUSTERS = {
    "hsf", "rtz", "sdw", "psk", "tsk", "tsh", "dgh", "ghs", "zbt",
    "qwr", "wrt", "plk", "bnm", "xtr", "vsk", "ndr", "rts"
}

# ================== UTIL ==================
def precompute_ref_ngrams(refs, n=REF_N):
    out = []
    for r in refs:
        s = r.lower()
        g = {s[i:i+n] for i in range(len(s)-n+1)} or {s}
        out.append(g)
    return out

def normalize_name(s: str) -> str:
    s = s.strip()
    s = s.replace("’","'").replace("ʼ","'")
    s = re.sub(r"[^0-9A-Za-zÀ-ÖØ-öø-ÿĀ-žẞßÇĞİıŞÖÜçğıöşü -]", "", s)
    return s.strip(" -")

def is_clean(s: str) -> bool:
    return True  

def capitalize_locale(s: str) -> str:
    if not s: return s
    overrides = {'i': 'İ', 'ı': 'I', 'ß': 'ẞ'}
    return overrides.get(s[0], s[0].upper()) + s[1:].lower()

# data
def load_corpus() -> list[str]:
    p = Path(SEED_DIR)
    names = []
    if p.exists():
        for f in p.glob("*.txt"):
            with f.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    nm = normalize_name(line)
                    if nm:
                        names.append(nm)
    if not names:
        # varsayılan isimler
        names = [
            "Lilith","Selene","Aurelia","Mystra","Freya","Isis","Bastet","Elara",
            "Amaterasu","Sakura","Miyuki","Hinata","Rin","Ayame","Nozomi","Yuna",
            "Thalia","Elowen","Isolde","Nyx","Lyra","Aenwyn","Elysia","Astraea"
        ]
    # dedup, clean
    uniq, seen = [], set()
    for n in names:
        if n not in seen and len(n) >= 1 and is_clean(n):
            uniq.append(n); seen.add(n)
    return uniq

def build_vocab(names: list[str]):
    chars = set()
    for n in names:
        for ch in n.lower():
            chars.add(ch)
    itos = ['<pad>','<bos>','<eos>'] + sorted(chars)
    stoi = {ch:i for i,ch in enumerate(itos)}
    return stoi, itos

def encode_name(stoi, name: str):
    return [stoi['<bos>']] + [stoi.get(ch, None) or 0 for ch in name.lower()] + [stoi['<eos>']]

# model
class CharLSTM(nn.Module):
    
    def __init__(self, vocab_size, emb=80, hid=320, layers=2, dropout=0.20):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hid, num_layers=layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hid, vocab_size)

    def forward(self, x, h=None):
        e = self.emb(x)
        o, h = self.lstm(e, h)
        logits = self.fc(o)
        return logits, h

# eğitim
def batchify(encoded, batch_size=64, shuffle=True):
    if shuffle: RAND.shuffle(encoded)
    for i in range(0, len(encoded), batch_size):
        batch = encoded[i:i+batch_size]
        maxlen = max(len(seq) for seq in batch)
        x, y = [], []
        for seq in batch:
            pad = [0] * (maxlen - len(seq))
            x.append(seq[:-1] + pad)  
            y.append(seq[1:] + pad)   
        yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def eval_val_loss(model, enc_batches):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    tot = cnt = 0
    with torch.no_grad():
        for xb, yb in batchify(enc_batches, batch_size=256, shuffle=False):
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            logits, _ = model(xb)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            tot += loss.item(); cnt += 1
    return tot / max(1, cnt)

def train_model(model, enc_seqs, epochs=8, lr=3e-3, batch_size=64, val_enc=None, patience=3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    best_loss = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, epochs+1):
        model.train()  
        total = cnt = 0
        for xb, yb in batchify(enc_seqs, batch_size=batch_size, shuffle=True):
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); cnt += 1

        train_loss = total / max(1, cnt)
        msg = f"[train] epoch {ep:02d} | loss {train_loss:.4f}"

        if val_enc:
            vloss = eval_val_loss(model, val_enc)  
            model.train()  
            msg += f" | val {vloss:.4f}"
            if vloss < best_loss - 1e-4:
                best_loss = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            print(msg)
            if bad >= patience:
                print("[early stop] restoring best weights")
                if best_state:
                    model.load_state_dict(best_state)
                return
        else:
            print(msg)

    if best_state:
        model.load_state_dict(best_state)

# örnekleme / sampling
@torch.no_grad()
def sample_name(model, stoi, itos, max_len=16, temperature=0.9):
    model.eval()
    bos = torch.tensor([[stoi['<bos>']]], dtype=torch.long, device=DEVICE)
    h = None
    seq = [stoi['<bos>']]
    cur = bos
    for _ in range(max_len*2):  
        logits, h = model(cur, h)
        next_logits = logits[0, -1] / max(1e-5, temperature)
        probs = F.softmax(next_logits, dim=-1)

        # top-k sampling
        k = 30
        topv, topi = torch.topk(probs, k)
        idx = topi[torch.multinomial(topv, 1)].item()

        if itos[idx] == '<eos>':
            break
        if itos[idx] not in ('<bos>','<pad>'):
            seq.append(idx)
        cur = torch.tensor([[idx]], dtype=torch.long, device=DEVICE)
        if len([i for i in seq if i not in (0, stoi['<bos>'])]) >= max_len:
            break
    chars = [itos[i] for i in seq if i not in (0, stoi['<bos>'])]
    return ''.join(chars)

# puanlama sistemi / scoring
def score_uniqueness_proxy(name: str, ref_grams) -> float:
    s = name.lower()
    grams = {s[i:i+REF_N] for i in range(len(s)-REF_N+1)} or {s}
    total = 0.0
    for g2 in ref_grams:
        total += len(grams & g2) / max(1, len(grams))
    avg = total / max(1, len(ref_grams))
    return max(0.0, 1.0 - avg)

def score_length_compactness(name: str) -> float:
    s = name.strip()
    extra = max(0, len(s) - 8)
    penalty = 0.10 * extra + 0.02 * (extra * (extra - 1) / 2.0) 
    return max(0.0, 1.0 - penalty)

def score_phonetic_elegance(name: str) -> float:
    s = name.lower()
    if len(s) < 2: 
        return 0.5
    val = 0.0
    for i in range(1, len(s)):
        a, b = s[i-1] in VOWELS, s[i] in VOWELS
        val += 1.0 if (a ^ b) else -0.5
    val -= 0.15 * abs(len(s) - 6)  
    return max(0.0, (val + 6.0) / 10.0)

def score_pronounceability(name: str) -> float:
    s = name.lower()

    longest = 0
    run = 0
    for ch in s:
        if ch.isalpha() and ch not in VOWELS:
            run += 1
            longest = max(longest, run)
        else:
            run = 0

    penalty = 0.0
    if longest >= 3:
        penalty += 0.35 + 0.20 * (longest - 3)  

    
    for bad in BAD_CLUSTERS:
        if bad in s:
            penalty += 0.25 

    # rare letters penalty
    penalty += 0.05 * sum(ch in RARE_LETTERS for ch in s)

    return max(0.0, 1.0 - min(1.0, penalty))

def score_balance_vowels_consonants(name: str) -> float:
    s = name.lower()
    vowels = sum(ch in VOWELS for ch in s)
    consonants = sum(ch.isalpha() and ch not in VOWELS for ch in s)
    diff = abs(vowels - consonants)
    penalty = max(0, diff - 1) * 0.15
    return max(0.0, 1.0 - penalty)

def composite_score(name: str, ref_grams) -> float:
    u = score_uniqueness_proxy(name, ref_grams)           # n-gram benzersizlik
    p = score_phonetic_elegance(name)                     # V/C alternasyonu + uzunluk tatlı noktası
    r = score_pronounceability(name)                      # 3+ sessiz koşuları, kötü kümeler, nadir harfler
    b = score_balance_vowels_consonants(name)             # sesli/sessiz oranı
    L = score_length_compactness(name)                    # 8+ harfe ceza

    # Toplam 1.00 olacak şekilde basit ve işlevsel ağırlıklar:
    return round(
        0.28*u + 0.22*p + 0.22*r + 0.18*b + 0.10*L,
        4
    )

 #loop
def coinflip_length():
    short = RAND.choice([True, False])
    return SHORT_RANGE if short else LONG_RANGE

def generate_and_filter(model, stoi, itos, refs, n_samples=1, temperature=0.9,
                        train_sets=None):
    out = []
    minlen, maxlen = coinflip_length()
    for _ in range(n_samples):
        raw = sample_name(model, stoi, itos, max_len=maxlen, temperature=temperature)
        cand = capitalize_locale(re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿĀ-žẞßÇĞİıŞÖÜçğıöşü]", "", raw))
        if len(cand) < minlen or len(cand) > maxlen:
            continue
        if not is_clean(cand):
            continue

        # eğitim verisinde varsa reddet (büyük/küçük harf ve boşluk duyarsız)
        if train_sets is not None:
            s_simple = cand.lower()
            s_nospace = re.sub(r"\s+", "", cand).lower()
            ts_simple, ts_nospace = train_sets
            if s_simple in ts_simple or s_nospace in ts_nospace:
                continue
        

        out.append(cand)
    return out


def log_name(name: str, score: float):
    line = f"{name:<18} | score={score:.4f}\n"
    path = LOG_SHORT if len(name) <= SHORT_RANGE[1] else LOG_LONG
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)

def refresh_finalsuggestion(toplist):
    toplist.sort(reverse=True)  # by score
    with open(FINAL_TOP, "w", encoding="utf-8") as f:
        f.write("# Top suggestions (composite score)\n")
        for i, (sc, nm) in enumerate(toplist[:TOP_CAP], 1):
            tag = "S" if len(nm) <= SHORT_RANGE[1] else "L"
            f.write(f"{i:2d}. {nm:<18} | {tag} | score={sc:.4f}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_epochs", type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--generate", type=int, default=900)
    ap.add_argument("--temperature", type=float, default=0.9) #9 aradağıma en yakın sonuç
    args = ap.parse_args()

    names = load_corpus()
    print(f"[data] names: {len(names)}")

    # Hard uniqueness vs training data
    TRAIN_SET_SIMPLE  = {normalize_name(n).lower() for n in names}
    TRAIN_SET_NOSPACE = {re.sub(r"\s+","", normalize_name(n)).lower() for n in names}

    refs_for_uniqueness = names[:]
    ref_grams = precompute_ref_ngrams(refs_for_uniqueness, n=REF_N)

    stoi, itos = build_vocab(names)
    enc = [encode_name(stoi, n) for n in names]

    # 90/10 bölünmüş eğitim/doğrulama seti ve early stopplama
    if len(enc) > 1:
        split = max(1, int(0.9 * len(enc)))
        train_enc, val_enc = enc[:split], enc[split:]
    else:
        train_enc, val_enc = enc, None

    model = CharLSTM(vocab_size=len(itos)).to(DEVICE)

    # Resume if exists
    if Path("model.pt").exists():
        model.load_state_dict(torch.load("model.pt", map_location=DEVICE))

    train_model(model, train_enc, epochs=args.train_epochs, lr=args.lr,
                batch_size=args.batch_size, val_enc=val_enc, patience=3)

    # En iyi kontrol noktasını kaydet
    torch.save(model.state_dict(), "model.pt")

    best = []
    generated = 0
    seen = set()

    for _ in range(args.generate):
        cands = generate_and_filter(
            model, stoi, itos, refs_for_uniqueness,
            n_samples=4, temperature=args.temperature,
            train_sets=(TRAIN_SET_SIMPLE, TRAIN_SET_NOSPACE)
        )
        for nm in cands:
            if nm in seen:
                continue
            sc = composite_score(nm, ref_grams)
            log_name(nm, sc)
            best.append((sc, nm))
            seen.add(nm)
            generated += 1
            if generated % REFRESH_N == 0:
                refresh_finalsuggestion(best)

    refresh_finalsuggestion(best)
    print(f"[done] wrote {LOG_SHORT}, {LOG_LONG}, and {FINAL_TOP}")
    print(f"[model] total parameters: {sum(p.numel() for p in model.parameters()):,}")    

if __name__ == "__main__":
    main()
