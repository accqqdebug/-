# EventSNN model source only

Includes the continuous-time frontend and EAS-SNN/YOLOX model modules.
No experiment runners, datasets, logs, evaluation suites, checkpoints, or results.
Original license and copyright headers are retained.

Install dependencies from requirements.txt. Use matching PyTorch/torchvision builds.
SpikingJelly version: 0.0.0.0.14. Original experiments used torch 2.0.1 / torchvision 0.15.2.

Construction:
    from model import build_model
    model = build_model(num_classes=2)

Model input: dict(events=[tensor(N,4), ...], readouts=[tensor(T), ...]).
Each event row is x, y, integer timestamp in microseconds, polarity (0/1).
Events and readouts must be chronological, within sensor bounds, with no event
beyond the final readout. Output readout count must match steps.

For inference call model.eval(); for training provide detection targets.
Reset SNN state between independent samples/batches using
spikingjelly.activation_based.functional.reset_net(model).
For explicit continuous frontend state use model.embedding.encode(..., states=...).

Model class implementations are preserved. model.py is a simplified construction
entry point; its random initialization sequence differs from the former experiment
runner. No dataset or experiment modules are needed to import this package.
