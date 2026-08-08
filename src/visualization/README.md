# ATP League Lab

Open `index.html` through a local web server, for example from the project root:

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000/src/visualization/`.

## Connect the trained estimator

Edit [`model-adapter.js`](./model-adapter.js) and replace `predictMatch()` with your model call. It receives two player objects and an optional context object, and must return the probability that `playerA` wins (`0` to `1`). The rest of the league simulation is model-agnostic.
