# debug/ -- scratch space

One-off diagnostics for whatever is being chased right now.

Nothing here is expected to survive. Once a question is answered, delete the
script that answered it -- git keeps the history, and a stale diagnostic is
worse than no diagnostic: it still runs, and its output still looks
authoritative, long after the assumptions it encodes have stopped holding.

If a script turns out to be worth re-running months later, it does not belong
here. Operator accuracy and performance go to `../bench/`; checkpoint and device
inspection goes to `../checks/`.
