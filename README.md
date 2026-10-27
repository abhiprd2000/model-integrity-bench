# model-integrity-bench

A checker for AI model files on small devices. Each model file is split into 4 KiB blocks under a signed, versioned Merkle tree (BLAKE3 leaves, Ed25519-signed root):

- a read checks only the blocks it touches,
- an honest update re-hashes one path to the root and re-signs it,
- an older signed version is rejected (rollback check),
- a crash mid-update leaves either the old or the new version.

It grew out of my kernel, Axiom (github.com/abhiprd2000/axiom-os-kernel). Measured on an Intel i5-5200U laptop and a Snapdragon 7 Gen 4 phone.

## Try it yourself (about 2 minutes)

Needs Rust (stable), `xxd`, and Linux or macOS.

```bash
# 1. Build and get a model (MobileNetV2, 14 MB)
cargo build --release
mkdir -p models
wget -O models/mid.tflite "https://huggingface.co/litert-community/MobileNet-v2/resolve/main/mobilenet_v2.tflite"

# 2. Setup
alias mib=./target/release/mib
D=/tmp/mib_demo; M=models/mid.tflite; ID="--model-id mobilenet_v2"
mkdir -p $D && mib keygen $D/seed $D/pub
flip(){ b=$(xxd -s $2 -l 1 -p $1); printf "$(printf '\\x%02x' $((0x$b ^ 0x40)))" | dd of=$1 bs=1 seek=$2 conv=notrunc status=none; }

# 3. Seal the model: build the hash tree and sign the root (version 1)
mib seal $M --key $D/seed --version 1 $ID --out $D/v1.mib

# 4. Checked read of the whole file: prints the byte count, 13975904
mib verify-read $M --pubkey $D/pub --sidecar $D/v1.mib $ID | wc -c

# 5. Flip one byte in a copy: the read is refused and names the block
cp $M $D/copy.tflite && flip $D/copy.tflite 5000000
mib verify-read $D/copy.tflite --pubkey $D/pub --sidecar $D/v1.mib $ID > /dev/null
#   -> block 1220 (offset 4997120) failed verification

# 6. Release the same change properly as version 2: accepted
cp $D/copy.tflite $D/v2.tflite
mib seal $D/v2.tflite --key $D/seed --version 2 $ID --out $D/v2.mib
mib verify-read $D/v2.tflite --pubkey $D/pub --sidecar $D/v2.mib $ID --min-version 2 > /dev/null && echo OK

# 7. Try to roll back to version 1: refused
mib verify-read $M --pubkey $D/pub --sidecar $D/v1.mib $ID --min-version 2 > /dev/null
#   -> open refused: Rollback

rm -rf $D
```
