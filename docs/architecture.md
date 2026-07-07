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

Phase 3 adds parameter gradient monitoring using `Parameter.register_post_accumulate_grad_hook`.
The monitor observes the accumulated `.grad` value after autograd updates a leaf
Parameter. It stores only summaries and first abnormal observations; it does not retain
gradient tensors, masks, grad_fn objects, optimizer objects, or hook handles after close.
When an optimizer is provided, only trainable model parameters managed by that optimizer
are selected. Without an optimizer, all trainable model parameters are selected.

Sparse COO gradients are checked through their coalesced values without densifying.
Global gradient norm diagnostics are user-threshold based only; TrainMedic does not set
default exploding or vanishing thresholds and never clips gradients.

Phase 4 adds parameter update monitoring using optimizer step pre-hooks and post-hooks.
TrainMedic observes real `optimizer.step()` calls without replacing the method, changing
step arguments, modifying closures, calling backward, calling step, or touching optimizer
state. The monitor selects the intersection of trainable model parameters and optimizer
parameters by object identity. It refreshes the model/optimizer relationship and learning
rates at session start and each step pre-hook because schedulers, param groups, and
`requires_grad` can change during a session. It distinguishes attempted steps from
completed steps.

Parameter update monitoring uses bounded before/after samples instead of copying whole
models. Small parameters can be compared exactly when the global snapshot budget allows
it. Large parameters use deterministic sampled flat indices, so the diagnostic language
must say that no sampled element changed rather than claiming a full-parameter proof.
Snapshots are cleared after each post-hook and on close.

Phase 5 adds explicit train/eval mode monitoring using local forward pre-hooks. The user
must provide `expected_mode`; TrainMedic does not infer phase and never calls `train()` or
`eval()`. The monitor records only actual module calls, so uncalled branches do not
produce mode diagnostics. It stores bounded observation summaries and never stores input
or output tensors.

Monitoring code must not keep full tensors by default. It should store bounded summaries
such as shape, dtype, device, min, max, mean, standard deviation, NaN/Inf counts, and norm.
This keeps memory overhead predictable and avoids retaining computation graphs.

Current phase status: Phase 5 implements static model/optimizer inspection, forward
output NaN/Inf monitoring, accumulated parameter gradient monitoring, bounded parameter
update monitoring around optimizer steps, and explicit train/eval mode diagnostics. It
does not implement Module grad_input/grad_output hooks, a unified watch entry point,
training-loop monitoring, CLI commands, or automatic fixes.
