# 3D models

## `space-station.glb`

The scene on the training screen, rendered by
`components/dashboard/TrainingRig.tsx`: a figure standing in still water with
planets orbiting overhead. Credits and licence are in [CREDITS.md](CREDITS.md);
note that it is **non-commercial**.

### How it moves

The file has no animation clips. The component animates the model's parts by
node name:

| Node | What happens |
|---|---|
| `body` | the figure; bobs gently |
| `waves`, `waves1`, `waves2` | ripple rings; pulse outward, harder while aggregating |
| `particles` | the star field; drifts around the vertical axis |
| `Sphere*` | planets; each orbits the figure at its own radius, inner ones faster |
| `Cube` | a black ground slab; hidden |

Speed follows how much of the mesh is training. The model's black material
takes the run-state colour: indigo while training, cyan while aggregating,
green when finished, red on failure.

### Replacing it

Drop another GLB at the same path. Anything works, but to get the motion keep
those node names, or extend the name patterns at the top of `TrainingRig.tsx`.
Keep it under about 5 MB (every dashboard visitor loads it) with embedded
textures. If the file is missing or fails to load, a procedural scene with the
same idea is drawn instead, so the screen never breaks.
