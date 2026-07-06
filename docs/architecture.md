# TrainMedic Architecture

TrainMedic uses a `src` layout so tests import the installed package instead of importing
modules directly from the repository root. This catches packaging mistakes early and keeps
local development close to CI behavior.

The first MVP will use deterministic rules rather than language-model judgment. A diagnosis
must be supported by observed model state, optimizer membership, tensor summaries, hooks,
or other explicit runtime evidence.

Diagnostics use a unified data model so console output, JSON output, tests, and future
integrations all describe findings in the same shape. Phase 0 defines `Severity`,
`Evidence`, and `Diagnostic` with stable JSON-compatible serialization.

Future phases are expected to separate responsibilities as follows:

- `inspectors`: static model and optimizer inspection.
- `monitors`: runtime activation, gradient, parameter, and mode observation.
- `rules`: deterministic checks that turn observations into diagnostics.
- `reports`: console and JSON rendering.

Monitoring code must not keep full tensors by default. It should store bounded summaries
such as shape, dtype, device, min, max, mean, standard deviation, NaN/Inf counts, and norm.
This keeps memory overhead predictable and avoids retaining computation graphs.

Current phase status: Phase 0 only provides project infrastructure and diagnostic data
structures. It does not implement hooks, optimizer inspection, training-loop monitoring,
CLI commands, or diagnostic rules.
