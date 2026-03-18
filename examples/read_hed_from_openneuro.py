"""
===========================================
Read HED annotations from an OpenNeuro dataset
===========================================

Download a BIDS dataset with HED (Hierarchical Event Descriptors) annotations
from OpenNeuro and read them using MNE-BIDS.

Dataset: ds004117 — Sternberg Working Memory (Onton et al., 2005)
HED annotations are stored in the JSON sidecar (task-WorkingMemory_events.json)
as per-column mappings (event_type, task_role, trial, letter, memory_cond).
"""

# %%
# Imports
# -------

import json
from pathlib import Path

import openneuro

from mne_bids import BIDSPath, read_raw_bids

# %%
# Download one subject from OpenNeuro
# ------------------------------------

openneuro.download(
    dataset="ds004117",
    target_dir="/tmp/ds004117",
    include=[
        "dataset_description.json",
        "participants.*",
        "task-*_events.json",
        "sub-002/**",
    ],
    verify_hash=False,
)

# %%
# Read with MNE-BIDS
# -------------------

bids_path = BIDSPath(
    subject="002",
    session="01",
    task="WorkingMemory",
    run="1",
    datatype="eeg",
    root="/tmp/ds004117",
)
raw = read_raw_bids(bids_path, verbose="warning")

# %%
# Inspect annotations
# --------------------
# When HED strings validate, the annotations will be HEDAnnotations with a
# ``hed_string`` attribute.  Otherwise they fall back to regular Annotations
# (with a warning).

print(f"Annotation type: {type(raw.annotations).__name__}")
print(f"Number of events: {len(raw.annotations)}")
print()

# Show first 10 annotations
for desc, onset, dur in zip(
    raw.annotations.description[:10],
    raw.annotations.onset[:10],
    raw.annotations.duration[:10],
):
    print(f"  {onset:8.2f}s  {dur:8.2f}s  {desc}")

# %%
# Check if HED strings are available
# ------------------------------------

if hasattr(raw.annotations, "hed_string"):
    print(f"\nHED version: {raw.annotations._hed_version}")
    print("First 10 HED strings:")
    for desc, hed in zip(
        raw.annotations.description[:10],
        raw.annotations.hed_string[:10],
    ):
        print(f"  {desc}: {hed}")
else:
    print("\nHEDAnnotations not created (see warnings above).")
    print("HED data is still accessible via the JSON sidecar.")

# %%
# Read HED from the sidecar directly
# ------------------------------------
# Even without HEDAnnotations, you can always read the sidecar manually.

sidecar = json.loads(
    (Path("/tmp/ds004117") / "task-WorkingMemory_events.json").read_text()
)

print("\nHED mappings in sidecar:")
for col, meta in sidecar.items():
    if isinstance(meta, dict) and "HED" in meta:
        hed = meta["HED"]
        if isinstance(hed, dict):
            print(f"\n  {col} (categorical):")
            for val, tag in list(hed.items())[:3]:
                print(f"    {val}: {tag}")
            if len(hed) > 3:
                print(f"    ... ({len(hed)} values total)")
        else:
            print(f"\n  {col} (template): {hed}")
