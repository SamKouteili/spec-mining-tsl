import os
import subprocess

CONFIG_PATH = "./bin/TraceGen_Random.toml"

# NOTE: Need to handle HOA counterstrategy case
def run_tsl(cmd: str, tsl: str) -> str:
    """$> tsl cmd < tsl"""
    try:
        result = subprocess.run(["tsl", cmd], input=tsl, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Error running tsl:", e.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"TSL failed on \n{tsl}\n with error: {result.stderr}")
    
    return result.stdout

def run_hoax(hoa: str, hoa_n="tmp.hoa") -> str:
    """$> hoax hoa --config bin/TraceGen_Random.toml"""
    if not os.path.exists(hoa_n):
        with open(hoa_n, "w") as f:
            f.write(hoa)
    try:
        result = subprocess.run(["hoax", hoa_n, "--config", CONFIG_PATH], 
                                capture_output=True, 
                                text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Error running hoax:", e.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Hoax failed on {hoa} with erroxr: {result.stderr}")
    return result.stdout


def run_complement(hoa: str, flags=[]) -> str:
    """$> autfilt --complement < hoa"""
    cmd = ["autfilt", "--complement"] + flags
    try:
        result = subprocess.run(cmd, input=hoa, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Error running autfilt --complement:", e.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"autfilt --complement failed on \n{hoa}\n with error: {result.stderr}")
    
    return result.stdout

def run_to_finite(hoa: str) -> str:
    """$> autfilt --to-finite < hoa"""
    cmd = ["autfilt", "--to-finite"]
    try:
        result = subprocess.run(cmd, input=hoa, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Error running autfilt --to-finite:", e.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"autfilt --to-finite failed on \n{hoa}\n with error: {result.stderr}")
    
    return result.stdout
    
def run_ltlf(ltl: str) -> str:
    """$> ltlfilt --from-ltlf < ltl"""
    cmd = ["ltlfilt", "--from-ltlf"]
    try:
        result = subprocess.run(cmd, input=ltl, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Error running ltlfilt --from-ltlf:", e.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"ltlfilt --from-ltlf failed on {ltl} with error: {result.stderr}")
    
    return result.stdout

def run_neg_ltl(ltl: str) -> str:
    """$> ltlfilt --negate < ltl"""
    cmd = ["ltlfilt", "--negate"]
    try:
        result = subprocess.run(cmd, input=ltl, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Error running ltlfilt --from-ltlf:", e.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"ltlfilt --from-ltlf failed on {ltl} with error: {result.stderr}")
    
    return result.stdout

def run_accept_word(hoa: str, word: str) -> bool:
    """$> autfilt --accept-word=WORD < hoa"""
    try:
        result = subprocess.run(["autfilt", f"--accept-word={word}"], input=hoa, capture_output=True, text=True)
    except :
        print("Error running autfilt hoa --acept")
        return False
    if result.returncode != 0:
        return False
    return result.stdout != ""

def run_dot_gen(hoa: str) -> str:
    """$> autfilt --hoa --dot < hoa"""
    try:
        result = subprocess.run(["autfilt", "--hoa", "--dot"], input=hoa, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Error running autfilt --hoa --dot:", e.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"autfilt --hoa --dot failed on \n{hoa}\n with error: {result.stderr}")
    
    start_idx = result.stdout.find("digraph")
    if start_idx != -1:
        return result.stdout[start_idx:]
    else :
        raise RuntimeError(f"autfilt --hoa --dot output parsing failed on \n{hoa}\n with error: {result.stderr}")

def run_ltl2tgba(ltl: str, flags: list[str] = ["-D", "-M", "-H"]) -> str:
    """$> ltl2tgba [flags] < ltl."""
    cmd = ["ltl2tgba"] + flags + [ltl]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Error running ltl2tgba:", e.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"ltl2tgba failed on {ltl} with error: {result.stderr}")
    return result.stdout

def run_syfco(inp: str, flags: list[str] = ["-f", "ltl"]) -> str:
    """$> syfco [flags] < input"""
    cmd = ["syfco"] + flags + ["-in"]
    try:
        result = subprocess.run(cmd, input=inp, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Error running syfco:", e.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"syfco failed on {inp} with error: {result.stderr}")
    return result.stdout

def run_ltlf2dfa(ltl: str) -> str:
    """$> ltlf2dfa < ltl. Converts LTL to DFA with finite trace semantics."""
    cmd = ["ltlf2dfa"]
    try:
        result = subprocess.run(cmd, input=ltl, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Error running ltlf2dfa:", e.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"ltlf2dfa failed on {ltl} with error: {result.stderr}")
    return result.stdout

