#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Run the improved ty-mcc binary on an MCC input set and grade every answer
# against the CONSENSUS ORACLE (the "estimated result" column of the official
# raw-result-analysis.csv). The headline number is WRONG: a definite TY verdict
# that disagrees with the multi-tool consensus = a real soundness error on
# held-out models. Grades single-verdict, StateSpace, and the 16-formula
# bool-vector examinations (cardinality/fireability/CTL/LTL).
#
#   scripts/mcc_oracle_eval.py --inputs DIR --oracle answer.csv [--bin ...]
#       [--timeout 60] [--jobs 3] [--max-states 20000000] [--memfrac 0.06]
#       [--sample N | --models "a b"] [--exams "..."] [--out runs.csv]
import csv, os, sys, subprocess, tempfile, shutil, random, argparse, concurrent.futures, re, resource, signal, xml.etree.ElementTree as ET

SINGLE = {"ReachabilityDeadlock","OneSafe","QuasiLiveness","StableMarking","Liveness"}
MULTI  = {"ReachabilityCardinality","ReachabilityFireability",
          "CTLCardinality","CTLFireability","LTLCardinality","LTLFireability"}

def _mem_limiter(gb):
    def _f():
        b = int(gb * 1024 * 1024 * 1024)
        try: resource.setrlimit(resource.RLIMIT_AS, (b, b))
        except Exception: pass
    return _f

def load_oracle(path):
    o = {}
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 16 or row[0] == "### tool": continue
            o.setdefault((row[1], row[2]), row[15].strip())
    return o

def run_ty(binpath, model_dir, exam, timeout, memfrac, memlimit_gb, max_states):
    d = tempfile.mkdtemp(prefix="oracle-")
    env = dict(os.environ, TY_MCC_REQUIRE_BACKEND_EVIDENCE="0", TY_MCC_FPSET_BACKEND="cas")
    # Launch in its OWN session (start_new_session) so the whole process TREE
    # (ty-mcc + its `ay` solver grandchildren) can be killed as one group. The
    # grandchildren inherit ty-mcc's stdout pipe; if the watchdog SIGKILLs only
    # ty-mcc, an orphaned solver keeps the pipe open and communicate() hangs
    # forever (the 2.8h coverage-run hang). killpg on timeout closes the pipe.
    p = None
    try:
        # NOTE: `ty-mcc` takes the model dir as a BARE POSITIONAL — there is no
        # `mcc` subcommand (that's the unified `ty` binary). A stray "mcc" arg
        # made the binary error and the script silently grade every cell CC.
        p = subprocess.Popen([binpath, model_dir, "--examination", exam, "--threads", "4",
              "--memory-fraction", str(memfrac), "--max-states", str(max_states),
              "--storage", "auto", "--storage-dir", d, "--timeout", str(timeout)],
              env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
              preexec_fn=_mem_limiter(memlimit_gb), start_new_session=True)
        try:
            out, _ = p.communicate(timeout=timeout + 90)
            return out or ""
        except subprocess.TimeoutExpired:
            _killpg(p)
            try: p.communicate(timeout=15)
            except Exception: pass
            return ""
    except (MemoryError, OSError):
        return ""
    finally:
        if p is not None and p.poll() is None:
            _killpg(p)
        shutil.rmtree(d, ignore_errors=True)

def _killpg(p):
    try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        try: p.kill()
        except Exception: pass

def ty_single_verdict(out):
    m = re.search(r'FORMULA\s+\S+\s+(TRUE|FALSE)\b', out)
    if m: return m.group(1)[0]
    if "CANNOT_COMPUTE" in out: return "CC"
    return None

def ty_statespace(out):
    def g(k):
        m = re.search(r'STATE_SPACE\s+'+k+r'\s+(\d+)', out)
        return int(m.group(1)) if m else None
    s,t,mp,ms = g("STATES"),g("TRANSITIONS"),g("MAX_TOKEN_IN_PLACE"),g("MAX_TOKEN_PER_MARKING")
    if None in (s,t,mp,ms):
        return "CC" if "CANNOT_COMPUTE" in out else None
    return (s,t,mp,ms)

def parse_num(x):
    try: return int(round(float(x.strip())))
    except: return None

# --- multi-formula (16-bool) grading ---------------------------------------
def xml_formula_order(model_dir, exam):
    """Ordered list of property ids from <exam>.xml (document order)."""
    p = os.path.join(model_dir, exam + ".xml")
    if not os.path.exists(p): return None
    try:
        root = ET.parse(p).getroot()
    except Exception:
        return None
    ids = []
    for el in root.iter():
        if el.tag.endswith("}id") or el.tag == "id":
            # only top-level property ids (skip nested transition/place ids):
            # property ids look like <model>-<Exam>-NN
            t = (el.text or "").strip()
            if exam in t:
                ids.append(t)
    return ids or None

def ty_upperbounds_verdicts(out):
    """Map formula id -> int bound or None (CANNOT_COMPUTE) from FORMULA lines.

    UpperBounds emits `FORMULA <id> <number> TECHNIQUES ...` for a resolved bound
    and `FORMULA <id> CANNOT_COMPUTE` otherwise."""
    v = {}
    for m in re.finditer(r'FORMULA\s+(\S+)\s+(\d+|CANNOT_COMPUTE)\b', out):
        fid, tok = m.group(1), m.group(2)
        v[fid] = None if tok == "CANNOT_COMPUTE" else int(tok)
    return v

def grade_upper_bounds(model_dir, out, oracle):
    """Grade UpperBounds: per-formula numeric comparison against the oracle's
    space-separated bound list (ordered by SORTED formula id, like grade_multi).

    A TY number that disagrees with a definite oracle number => WRONG (a too-high
    or too-low bound is a wrong answer, never merely imprecise). A TY
    CANNOT_COMPUTE => CC. Oracle '?' (unknown) => CC for that cell."""
    order = xml_formula_order(model_dir, "UpperBounds")
    if order: order = sorted(order)
    ty = ty_upperbounds_verdicts(out)
    toks = oracle.strip().split()
    if not order or len(toks) != len(order):
        if not ty: return ("CC", "no-formula-lines")
        return ("UNJUDGED", f"align-fail order={bool(order)} olen={len(toks)} ordlen={len(order) if order else 0}")
    corr=wrong=cc=0; details=[]
    for pos, fid in enumerate(order):
        orc = parse_num(toks[pos]) if toks[pos] not in ("?","-","") else None
        tv = ty.get(fid, None)
        if tv is None or orc is None:
            cc += 1
        elif tv == orc:
            corr += 1
        else:
            wrong += 1; details.append(f"{fid}:ty={tv}!=or={orc}")
    if wrong: return ("WRONG", f"{wrong} formulas wrong: " + "; ".join(details[:6]))
    if corr: return ("CORRECT", f"{corr}/{len(order)} ok, {cc} cc")
    return ("CC", f"all {cc} cc")

def ty_formula_verdicts(out):
    """Map formula id -> 'T'/'F'/'?' from FORMULA lines."""
    v = {}
    for m in re.finditer(r'FORMULA\s+(\S+)\s+(TRUE|FALSE|CANNOT_COMPUTE)\b', out):
        fid, verd = m.group(1), m.group(2)
        v[fid] = {'TRUE':'T','FALSE':'F','CANNOT_COMPUTE':'?'}[verd]
    return v

def grade_multi(model_dir, exam, out, oracle):
    # CRITICAL: the MCC consensus bit-string (and every tool's col-7 string) is
    # ordered by SORTED formula id, not XML document order. Reused formulas keep
    # their original year tag (e.g. -2023-12), which sorts BEFORE the current
    # -2025-NN block. Using document order silently permutes ~the last 4 vs first
    # 12 formulas and manufactures phantom "wrong answers". Sort the ids.
    order = xml_formula_order(model_dir, exam)
    if order: order = sorted(order)
    ty = ty_formula_verdicts(out)
    # The consensus writes unknown formulas as "? " (question mark + SPACE),
    # inflating the bit-string past 16 chars (e.g. "FTFFFFFTF? FFFTTF" is 16
    # formulas). Strip spaces before aligning, else these grade UNJUDGED.
    oracle = oracle.strip().upper().replace(" ", "")
    if not order or not re.fullmatch(r'[TF?]+', oracle) or len(oracle) != len(order):
        # can't align; fall back to a global check (any definite TY verdict that
        # conflicts with a same-position oracle char if counts match)
        if not ty: return ("CC", "no-formula-lines")
        return ("UNJUDGED", f"align-fail order={bool(order)} olen={len(oracle)}")
    corr=wrong=cc=0; details=[]
    for pos, fid in enumerate(order):
        oc = oracle[pos]
        tv = ty.get(fid, '?')
        if tv == '?' or oc == '?':
            cc += 1
        elif tv == oc:
            corr += 1
        else:
            wrong += 1; details.append(f"{fid}:ty={tv}!=or={oc}")
    if wrong: return ("WRONG", f"{wrong} formulas wrong: " + "; ".join(details[:6]))
    if corr: return ("CORRECT", f"{corr}/{len(order)} ok, {cc} cc")
    return ("CC", f"all {cc} cc")

def grade(model_dir, exam, out, oracle):
    oracle = (oracle or "").strip()
    if exam in SINGLE:
        ty = ty_single_verdict(out)
        if ty == "CC" or ty is None: return ("CC", "")
        if oracle in ("?","","DNC","CC"): return ("UNJUDGED", f"ty={ty} oracle={oracle}")
        return ("CORRECT" if ty == oracle[0].upper() else "WRONG", f"ty={ty} oracle={oracle}")
    if exam == "StateSpace":
        ty = ty_statespace(out)
        if ty == "CC" or ty is None: return ("CC", "")
        toks = oracle.split()
        if len(toks) != 4 or "?" in oracle: return ("UNJUDGED", f"ty={ty} oracle={oracle}")
        orc = tuple(parse_num(t) for t in toks)
        if None in orc: return ("UNJUDGED", f"ty={ty} oracle={orc}")
        close = lambda a,b,rel=2e-3: a==b or (b!=0 and abs(a-b) <= max(1,abs(b)*rel))
        ok = close(ty[0],orc[0]) and close(ty[1],orc[1]) and ty[2]==orc[2] and ty[3]==orc[3]
        return ("CORRECT" if ok else "WRONG", f"ty={ty} oracle={orc}")
    if exam in MULTI:
        return grade_multi(model_dir, exam, out, oracle)
    if exam == "UpperBounds":
        return grade_upper_bounds(model_dir, out, oracle)
    return ("UNJUDGED", "exam-not-graded")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--bin", default="./target/release/ty-mcc")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--memfrac", type=float, default=0.06)
    ap.add_argument("--memlimit-gb", type=float, default=12.0)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--max-states", type=int, default=20_000_000)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-pnml", type=int, default=0, help="only models whose model.pnml <= this many bytes (0=any)")
    ap.add_argument("--models", default="")
    ap.add_argument("--exams", default="ReachabilityDeadlock OneSafe QuasiLiveness StableMarking StateSpace")
    ap.add_argument("--out", default="/tmp/ty_oracle_runs.csv")
    a = ap.parse_args()

    oracle = load_oracle(a.oracle)
    exams = a.exams.split()
    all_models = sorted(d for d in os.listdir(a.inputs) if os.path.isdir(os.path.join(a.inputs, d)))
    if a.max_pnml > 0:
        def small(m):
            try: return os.path.getsize(os.path.join(a.inputs, m, "model.pnml")) <= a.max_pnml
            except OSError: return False
        all_models = [m for m in all_models if small(m)]
    if a.models.strip():
        models = a.models.split()
    elif a.sample > 0:
        random.seed(a.seed); models = random.sample(all_models, min(a.sample, len(all_models)))
    else:
        models = all_models
    cells = [(m,e) for m in models for e in exams if (m,e) in oracle]
    print(f"binary={a.bin}  models={len(models)}  cells={len(cells)}  timeout={a.timeout}s  jobs={a.jobs}  max_states={a.max_states}")

    tally = {}; wrongs = []
    outf = open(a.out, "w", newline=""); w = csv.writer(outf)
    w.writerow(["model","examination","grade","detail","wall_s"])
    def work(cell):
        m,e = cell
        import time; t0=time.time()
        # CRASH-PROOF: any exception in a worker (incl. a watchdog SIGKILL of the
        # child, a parse error, an OSError from the rlimit preexec) must NOT
        # propagate through ex.map and kill the whole run — grade it CC and go on.
        try:
            out = run_ty(a.bin, os.path.join(a.inputs, m), e, a.timeout, a.memfrac, a.memlimit_gb, a.max_states)
            g, detail = grade(os.path.join(a.inputs, m), e, out, oracle[cell]); dt=time.time()-t0
            return (m,e,g,detail,dt)
        except Exception as ex:
            return (m,e,"CC",f"work-exc:{type(ex).__name__}:{str(ex)[:50]}", time.time()-t0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for (m,e,g,detail,dt) in ex.map(work, cells):
            tally[(e,g)] = tally.get((e,g),0)+1
            w.writerow([m,e,g,detail,f"{dt:.0f}"]); outf.flush()
            mark = "  <<< WRONG" if g=="WRONG" else ""
            print(f"{m:36} {e:24} {g:9} {detail[:70]}{mark}")
            if g=="WRONG": wrongs.append((m,e,detail))
    outf.close()

    print("\n===== ORACLE GRADE SUMMARY =====")
    totals = {}
    for (e,g),n in tally.items(): totals[g] = totals.get(g,0)+n
    for g in ("CORRECT","WRONG","CC","UNJUDGED"):
        print(f"  {g:9} {totals.get(g,0)}")
    print("\nper-examination CORRECT / WRONG / CC / UNJUDGED:")
    for e in exams:
        c=tally.get((e,"CORRECT"),0); ww=tally.get((e,"WRONG"),0)
        cc=tally.get((e,"CC"),0); u=tally.get((e,"UNJUDGED"),0)
        if c+ww+cc+u: print(f"  {e:26} {c:5} {ww:5} {cc:5} {u:5}")
    if wrongs:
        print(f"\n!!!!! {len(wrongs)} WRONG ANSWERS vs the 2025 consensus oracle (real errors) !!!!!")
        for x in wrongs: print("   ", x)
    else:
        print("\n*** ZERO WRONG ANSWERS vs the 2025 consensus oracle on the graded cells ***")
    sys.exit(1 if wrongs else 0)

if __name__ == "__main__":
    main()
