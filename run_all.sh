#!/bin/bash
# All experiments of the paper (two shards in parallel on a 2-core CPU).
# Each worker's exit status is checked; the script stops at the first failed or stale group.
set -u
fail=0
run_pair () {   # run_pair <script> <args...>: two shards, wait for both PIDs, propagate failure
  TORCH_THREADS=1 python "$@" 0/2 & p0=$!
  TORCH_THREADS=1 python "$@" 1/2 & p1=$!
  wait $p0 || fail=1
  wait $p1 || fail=1
  if [ $fail -ne 0 ]; then echo "FAILED: $*" >&2; exit 1; fi
}
python data_prep.py || exit 1            # regenerates the split manifest and checks MANIFEST_SHA256.txt
python features.py || exit 1
python test_secagg.py && python test_dp.py && python test_protocol.py || exit 1
for g in baseline robust joint dpctl dp nullspace sketchdim fsweep elsa; do
  run_pair exp.py $g
done
run_pair tune.py
for s in near_dup_audit.py moderation.py sketch_fidelity.py leak.py xai.py tamper.py; do
  TORCH_THREADS=2 python $s || { echo "FAILED: $s" >&2; exit 1; }
done
TORCH_THREADS=2 python bench.py || exit 1
echo "ALL DONE"
