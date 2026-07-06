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

Phase 2 adds forward output monitoring. `ForwardMonitor` registers local forward hooks on
unique modules collected with `named_modules(remove_duplicate=False)`, so shared modules
receive one hook while diagnostics retain all aliases. The default `module_scope="all"`
monitors every unique module to catch functional operations inside non-leaf modules.
`module_scope="leaf"` monitors leaf modules and always includes the root model for lower
overhead.

When a shared module has multiple registered aliases, TrainMedic reports all aliases and
uses one stable primary name for `object_name`. A normal PyTorch forward hook does not
tell TrainMedic which attribute path was used for a specific call, so the primary name is
not claimed to be the actual call path.

Forward hooks traverse only tensors, lists, tuples, and mappings in module outputs. They
store tensor summaries only: shape, dtype, device, numel, NaN count, and Inf count. They
do not store original tensors, views, masks, modules, containers, inputs, or outputs. Hook
functions return `None`, so model outputs and computation graphs are not replaced.

Monitoring code must not keep full tensors by default. It should store bounded summaries
such as shape, dtype, device, min, max, mean, standard deviation, NaN/Inf counts, and norm.
This keeps memory overhead predictable and avoids retaining computation graphs.

Current phase status: Phase 2 implements static model/optimizer inspection and forward
output NaN/Inf monitoring only. It does not implement backward hooks, gradient
monitoring, parameter update monitoring, train/eval mode diagnostics, training-loop
monitoring, CLI commands, or automatic fixes.
