# Installation

## Python Environment

```bash
conda create -n tlsf python=3.12
conda activate tlsf
pip install -r requirements.txt
```

## Required Tools

### Bolt (Modified Fork)

Our modified version of Bolt with safety/liveness mining support.

```bash
git clone git@github.com:SamKouteili/Bolt.git
cd Bolt
cargo build --release
```

Add to PATH:
```bash
export PATH="$PATH:/path/to/Bolt/target/release"
```

### Issy (TSL Synthesis)

```bash
git clone https://github.com/phheim/issy.git
cd issy
stack build
```

Add to PATH:
```bash
export PATH="$PATH:/path/to/issy/.stack-work/install/.../bin"
# Or copy the binary:
cp $(stack path --local-install-root)/bin/issy /usr/local/bin/
```

### CVC5 (SyGuS Solver)

```bash
# macOS
brew install cvc5

# Or download from https://github.com/cvc5/cvc5/releases
```

### Other Dependencies (Optional)

These are only needed for specific features:

- [tsl](https://github.com/Barnard-PL-Labs/tsltools) - TSL tooling
- [spot](https://spot.lre.epita.fr/index.html) - LTL manipulation
- [syfco](https://github.com/reactive-systems/syfco) - TLSF format conversion

## Verify Installation

```bash
conda activate tlsf
bolt --help
issy --help
cvc5 --version
```
