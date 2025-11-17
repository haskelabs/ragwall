#!/usr/bin/env python3
"""
Pre-download and cache models for faster loading
"""

print("Downloading and caching models...")
print("This will take a few minutes on first run, but will speed up subsequent runs")
print("=" * 60)

# Download sentence-transformers model
try:
    print("1. Downloading sentence-transformers model (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("   ✅ Model downloaded and cached")
    del model  # Free memory
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\nModels are now cached and will load much faster!")
print("You can now run test_sota_improvements.py")
