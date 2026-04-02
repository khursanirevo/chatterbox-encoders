# SETUP GUIDE - Chatterbox Encoders

## Quick Setup (5 minutes)

### Step 1: Install Package

\`\`\`bash
cd chatterbox-encoders
pip install -e .
\`\`\`

### Step 2: Download Weights

**Option A: Automatic (Recommended)**

\`\`\`bash
python download_weights.py
\`\`\`

This downloads ~1.01 GB from HuggingFace:
- \`s3gen.safetensors\` (1007 MB)
- \`ve.safetensors\` (5.4 MB)
- \`tokenizer.json\` (25 KB)

**Option B: Manual Download**

Visit: https://huggingface.co/khursanirevo/chatterbox-encoders-weights/tree/main

Download all files to \`chatterbox_encoders/weights/\`:
- Click each file
- Click "Download"
- Save to \`chatterbox_encoders/weights/\`

### Step 3: Verify Installation

\`\`\`bash
python setup.py
\`\`\`

This will:
- Check all weights are present
- Test reconstruction pipeline
- Verify quality metrics

### Step 4: Test Reconstruction

\`\`\`bash
python -m chatterbox_encoders.pipelines.standalone test_audio.wav
\`\`\`

Expected output:
\`\`\`
MAE:  0.0758
MSE:  0.0214
Threshold: < 0.1

MAE < 0.1:  ✅ PASS
MSE < 0.1:  ✅ PASS
\`\`\`

## Full Package Contents

After setup, your package structure will be:

\`\`\`
chatterbox_encoders/
├── weights/                    # 1.01 GB total
│   ├── s3gen.safetensors      # 1007 MB - Speech token → audio
│   ├── ve.safetensors         # 5.4 MB - Speaker embedding
│   └── tokenizer.json         # 25 KB - Text tokenizer
├── audio/                      # Audio components
│   ├── s3_tokenizer.py        # Encode audio → tokens
│   ├── voice_encoder.py       # Extract speaker embeddings
│   ├── decoder.py             # Decode tokens → audio
│   └── ...
├── pipelines/
│   └── standalone.py          # Reconstruction pipeline
└── ...
\`\`\`

## Requirements

- Python >= 3.10
- 2 GB free disk space (for weights)
- CUDA/MPS recommended (CPU works but slower)

## Troubleshooting

### Weights not downloading?

Check your internet connection and try:
\`\`\`bash
pip install --upgrade huggingface_hub
python download_weights.py
\`\`\`

### CUDA out of memory?

Use CPU mode:
\`\`\`python
from chatterbox_encoders.pipelines.standalone import StandaloneReconstructionPipeline

pipeline = StandaloneReconstructionPipeline(device="cpu")
\`\`\`

### Import errors?

Make sure you installed the package:
\`\`\`bash
pip install -e .
\`\`\`

## Production Deployment

For production, you can:

1. **Bundle weights with package**: Include \`chatterbox_encoders/weights/\` in your git repository
2. **Use custom HF repo**: Modify \`REPO_ID\` in \`download_weights.py\`
3. **Air-gapped deployment**: Manually download weights and copy to target machine

## Support

- HuggingFace: https://huggingface.co/khursanirevo/chatterbox-encoders-weights
- Issues: Open an issue in this repository

---

**Setup complete! Ready to reconstruct audio.** ✅
