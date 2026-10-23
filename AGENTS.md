# Template stewardship

Work on `main` for scoped verified changes. Preserve unrelated edits and use
isolation when ownership overlaps. Commit and push only reviewed public content.

`export-manifest.json` owns the canonical Python source list and exact hashes.
`reuse-manifest.json` owns the byte-preserving public tcg-agent example export.
Keep its original LICENSE, relative paths, source pin and per-file hashes.
Refresh exported files from an accepted immutable commit with the exporter.
Do not edit a copied file and retain its old provenance. Keep reusable fixes in
the canonical implementation, then import the accepted change here.

Keep restricted engine source, binaries, vendor copies, card-program dumps, raw observations,
weights, local bindings, private paths, and credentials out of this repository.
Only reviewed project-owned code, source inventories and aggregate receipts may
be public. The reusable public kernel and Twinleaf conformance tools are in
`examples/tcg-agent/`; Twinleaf source is retrieved separately into ignored cache.
See PROVENANCE.md for attribution, license boundaries and omitted artifacts.

Verify the real artifact with `python scripts/verify_template.py`. The
`--require-full-standard` command must exit 2 until an independently reviewed
full-Standard gate validates complete evidence. A summary flag is insufficient.
Use retained experiments, native parity, and fixed evaluation for scientific
claims. Do not add unit or mock suites that imply unobserved gameplay coverage.

This repository has one export writer. Coordinate accepted source commits and
sanitized coverage receipts with their owners before updating public claims.
Public checks do not authorize engine execution, training, GPU use, cloud
spending, or live play.
