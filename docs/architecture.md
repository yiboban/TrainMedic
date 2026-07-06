# TrainMedic Architecture

TrainMedic uses a `src` layout so tests import the installed package instead of importing
modules directly from the repository root. This catches packaging mistakes early and keeps
local development close to CI behavior.

The first MVP will use deterministic rules rather than language-model judgment. A diagnosis
must be supported by observed model state, optimizer membership, tensor summaries, hooks,
or other explicit runtime evidence.

Diagnostics use a unified data model so console output, JSON output, tests, and future
integrations all describe findings in the same shape. Phase 0 defines `Severity`,
`Evidence`, and `Diagnostic` with JSON-compatible serialization. Standard TrainMedic
evidence values should use stable primitive types. Arbitrary user objects are converted
with `str(value)` only as a compatibility fallback, and that string form may include
runtime-specific details.

Future phases are expected to separate responsibilities as follows:

- `inspectors`: static model and optimizer inspection.
- `monitors`: runtime activation, gradient, parameter, and mode observation.
- `rules`: deterministic checks that turn observations into diagnostics.
- `reports`: console and JSON rendering.

Phase 1 implements the first `inspectors`, `rules`, and `reports` modules for static
model/optimizer parameter checks. Parameter identity comparisons use `id(parameter)` so
shared parameters and tied weights are handled without Tensor equality comparisons.

Monitoring code must not keep full tensors by default. It should store bounded summaries
such as shape, dtype, device, min, max, mean, standard deviation, NaN/Inf counts, and norm.
This keeps memory overhead predictable and avoids retaining computation graphs.

Current phase status: Phase 1 implements static model/optimizer inspection only. It does
not implement hooks, gradient monitoring, activation monitoring, parameter update
monitoring, training-loop monitoring, CLI commands, or automatic fixes.
