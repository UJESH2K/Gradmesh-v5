/**
 * The space scene, built once and shared by the landing page backdrop and the
 * training screen.
 *
 * The model is `public/models/space-station.glb` ("space boi" by silvercrow101,
 * CC BY-NC 4.0, credited in public/models/CREDITS.md): a figure standing in
 * still water with planets overhead. It has no animation clips, so the motion is
 * built here from its parts, found by node name:
 *
 *   body          the figure; bobs gently
 *   waves*        ripple rings; pulse outward
 *   particles     the star field; drifts around the vertical axis
 *   Sphere*       planets; each orbits the figure at its own radius, inner
 *                 ones faster, as orbits do
 *   Cube          a black ground slab that hides the stars; removed
 *
 * GLTFLoader strips dots from node names, so "Sphere.001" arrives as
 * "Sphere001". The model's materials are unlit black and off-white; the black
 * parts take a tint colour, which is how run state (or, on the landing page,
 * the accent) shows on an unlit model.
 */

import type * as ThreeNS from "three";

type Three = typeof ThreeNS;

export const MODEL_URL = "/models/space-station.glb";
export const CREDIT_URL = "https://sketchfab.com/3d-models/space-boi-f6a8c6a6727b4f2cb020c8b50bb2ee60";

const SKIP = /^cube/i;
const BODY = /^body$/i;
const WAVES = /^waves\d*$/i;
const STARS = /^particles$/i;
const PLANET = /^sphere\d*$/i;

export type Station = {
  /** Add this to the scene. */
  group: ThreeNS.Group;
  /** Centre and size of the figure plus planets, after normalisation. */
  centre: ThreeNS.Vector3;
  radius: number;
  /** Advance the animation. `speed` scales orbits, ripples and drift. */
  update: (delta: number, time: number, speed: number, pulse?: number) => void;
  /** Colour of the model's dark parts, glow and orbit guides. */
  setTint: (colour: ThreeNS.Color, ringOpacity?: number) => void;
  /** Recolour each orbit guide separately (for the vendor colours on the landing page). */
  setRingColours: (colours: number[]) => void;
  source: "model" | "fallback";
};

function glowTexture(THREE: Three): ThreeNS.Texture {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 128;
  const context = canvas.getContext("2d")!;
  const gradient = context.createRadialGradient(64, 64, 0, 64, 64, 64);
  gradient.addColorStop(0, "rgba(255,255,255,0.55)");
  gradient.addColorStop(0.35, "rgba(255,255,255,0.18)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  context.fillStyle = gradient;
  context.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(canvas);
}

/**
 * Load the model and wire its motion. Falls back to a procedural scene with the
 * same idea, a glowing core with moons on tilted orbits, if the file is missing.
 */
export async function buildStation(
  THREE: Three,
  { tint = 0x6d7cff, size = 2.6, signal }: { tint?: number; size?: number; signal?: { cancelled: boolean } } = {}
): Promise<Station> {
  const group = new THREE.Group();
  const tinted: { color: ThreeNS.Color }[] = [];
  const rings: ThreeNS.LineBasicMaterial[] = [];
  const orbiters: { pivot: ThreeNS.Object3D; speed: number }[] = [];
  const ripples: { object: ThreeNS.Object3D; base: ThreeNS.Vector3; offset: number; vertical: number }[] = [];
  let stars: ThreeNS.Object3D | null = null;
  let figure: ThreeNS.Object3D | null = null;
  let figureBaseY = 0;

  const makeGlow = (scale: number) => {
    const material = new THREE.SpriteMaterial({
      map: glowTexture(THREE),
      color: tint,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    tinted.push(material);
    const sprite = new THREE.Sprite(material);
    sprite.scale.setScalar(scale);
    return sprite;
  };

  const orbitRing = (radius: number, y: number) => {
    const points: ThreeNS.Vector3[] = [];
    for (let index = 0; index <= 128; index += 1) {
      const angle = (index / 128) * Math.PI * 2;
      points.push(new THREE.Vector3(Math.cos(angle) * radius, y, Math.sin(angle) * radius));
    }
    const material = new THREE.LineBasicMaterial({ color: tint, transparent: true, opacity: 0.16, depthWrite: false });
    rings.push(material);
    return new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(points), material);
  };

  const finish = (source: "model" | "fallback", centre: ThreeNS.Vector3, radius: number): Station => ({
    group,
    centre,
    radius,
    source,
    update(delta, time, speed, pulse = 0.035) {
      for (const orbiter of orbiters) orbiter.pivot.rotation.y += delta * speed * orbiter.speed;
      for (const ripple of ripples) {
        const wave = 1 + Math.sin(time * (0.8 + speed * 1.6) + ripple.offset * Math.PI) * pulse;
        ripple.object.scale.set(
          ripple.base.x * (ripple.vertical === 0 ? 1 : wave),
          ripple.base.y * (ripple.vertical === 1 ? 1 : wave),
          ripple.base.z * (ripple.vertical === 2 ? 1 : wave)
        );
      }
      if (stars) stars.rotation.y += delta * speed * 0.06;
      if (figure) figure.position.y = figureBaseY + Math.sin(time * 1.2) * 0.012;
    },
    setTint(colour, ringOpacity) {
      for (const material of tinted) material.color.copy(colour);
      for (const material of rings) {
        material.color.copy(colour);
        if (ringOpacity !== undefined) material.opacity = ringOpacity;
      }
    },
    setRingColours(colours) {
      rings.forEach((material, index) => material.color.setHex(colours[index % colours.length]));
    },
  });

  // --- the model --------------------------------------------------------------
  let available = false;
  try {
    const probe = await fetch(MODEL_URL, { method: "HEAD" });
    available = probe.ok && !(probe.headers.get("content-type") ?? "").includes("html");
  } catch {
    available = false;
  }

  if (available) {
    try {
      const { GLTFLoader } = await import("three/examples/jsm/loaders/GLTFLoader.js");
      const gltf = await new GLTFLoader().loadAsync(MODEL_URL);
      if (signal?.cancelled) return finish("model", new THREE.Vector3(), 1);
      const model = gltf.scene;
      model.updateMatrixWorld(true);

      const parts: ThreeNS.Object3D[] = [];
      model.traverse((object) => {
        if (object === model || !object.name) return;
        const name = object.name;
        if (SKIP.test(name) || BODY.test(name) || WAVES.test(name) || STARS.test(name) || PLANET.test(name)) {
          parts.push(object);
        }
      });
      for (const part of parts) if (SKIP.test(part.name)) part.visible = false;

      // Black becomes the tint; off-white stays.
      model.traverse((object) => {
        const mesh = object as ThreeNS.Mesh;
        if (!mesh.isMesh) return;
        const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        const replaced = materials.map((material) => {
          const basic = material as ThreeNS.MeshBasicMaterial;
          const clone = basic.clone();
          if (basic.color && basic.color.getHex() === 0x000000) {
            clone.color = new THREE.Color(tint);
            tinted.push(clone);
          }
          return clone;
        });
        mesh.material = Array.isArray(mesh.material) ? replaced : replaced[0];
      });

      // Frame on the figure and planets; the star field is far wider.
      const focus = new THREE.Box3();
      for (const part of parts) if (BODY.test(part.name) || PLANET.test(part.name)) focus.expandByObject(part);
      if (focus.isEmpty()) focus.setFromObject(model);
      const extent = focus.getSize(new THREE.Vector3());
      const middle = focus.getCenter(new THREE.Vector3());
      const scale = size / Math.max(extent.x, extent.y, extent.z || 1);
      model.scale.setScalar(scale);
      model.position.sub(middle.clone().multiplyScalar(scale));
      group.add(model);
      model.updateMatrixWorld(true);

      const bodyPart = parts.find((part) => BODY.test(part.name));
      const axis = new THREE.Vector3();
      if (bodyPart) new THREE.Box3().setFromObject(bodyPart).getCenter(axis);
      figure = bodyPart ?? null;
      if (figure) figureBaseY = figure.position.y;

      for (const part of parts) {
        if (PLANET.test(part.name)) {
          const position = new THREE.Box3().setFromObject(part).getCenter(new THREE.Vector3());
          const radius = Math.hypot(position.x - axis.x, position.z - axis.z) || 0.3;
          const pivot = new THREE.Group();
          pivot.position.set(axis.x, 0, axis.z);
          group.add(pivot);
          pivot.attach(part);
          orbiters.push({ pivot, speed: 0.9 / Math.sqrt(Math.max(0.2, radius)) });
        } else if (WAVES.test(part.name)) {
          // Ring nodes lie flat, but the exporter's axis swaps mean "flat" is
          // not the same local axes on every node; find the one pointing up.
          const quaternion = part.getWorldQuaternion(new THREE.Quaternion());
          const up = new THREE.Vector3(0, 1, 0);
          const alignment = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)].map(
            (axisVector) => Math.abs(axisVector.applyQuaternion(quaternion).dot(up))
          );
          ripples.push({
            object: part,
            base: part.scale.clone(),
            offset: ripples.length * 0.33,
            vertical: alignment.indexOf(Math.max(...alignment)),
          });
        } else if (STARS.test(part.name)) {
          const pivot = new THREE.Group();
          pivot.position.set(axis.x, 0, axis.z);
          group.add(pivot);
          pivot.attach(part);
          stars = pivot;
        }
      }

      const seen = new Set<number>();
      for (const { pivot } of orbiters) {
        const child = pivot.children[0];
        if (!child) continue;
        const position = new THREE.Box3().setFromObject(child).getCenter(new THREE.Vector3());
        const radius = Math.round(Math.hypot(position.x - axis.x, position.z - axis.z) * 20) / 20;
        if (seen.has(radius) || radius < 0.1) continue;
        seen.add(radius);
        const ring = orbitRing(radius, position.y);
        ring.position.set(axis.x, 0, axis.z);
        group.add(ring);
      }

      const glow = makeGlow(size * 1.3);
      glow.position.set(axis.x, axis.y + 0.2, axis.z - 0.4);
      group.add(glow);

      return finish("model", axis.clone(), size / 2);
    } catch {
      // fall through to the procedural scene
    }
  }

  // --- the stand-in -----------------------------------------------------------
  const core = new THREE.MeshStandardMaterial({ color: tint, emissive: tint, emissiveIntensity: 0.55, roughness: 0.4 });
  tinted.push(core);
  tinted.push({ color: core.emissive });
  group.add(new THREE.Mesh(new THREE.SphereGeometry(size * 0.16, 48, 32), core));
  group.add(makeGlow(size * 0.9));
  const moon = new THREE.MeshStandardMaterial({ color: 0xdfe4ec, roughness: 0.6 });
  [0.36, 0.52, 0.68].forEach((fraction, index) => {
    const radius = size * fraction;
    const pivot = new THREE.Group();
    pivot.rotation.x = 0.35 - index * 0.28;
    const body = new THREE.Mesh(new THREE.SphereGeometry(size * (0.027 + index * 0.01), 24, 16), moon);
    body.position.set(radius, 0, 0);
    pivot.add(body);
    pivot.add(orbitRing(radius, 0));
    group.add(pivot);
    orbiters.push({ pivot, speed: 1 / Math.sqrt(radius) });
  });
  return finish("fallback", new THREE.Vector3(), size / 2);
}

/** Release every geometry, material and texture under `root`. */
export function disposeTree(root: ThreeNS.Object3D) {
  root.traverse((object) => {
    const mesh = object as ThreeNS.Mesh;
    mesh.geometry?.dispose?.();
    const material = mesh.material as ThreeNS.Material | ThreeNS.Material[] | undefined;
    const release = (item: ThreeNS.Material) => {
      const map = (item as ThreeNS.SpriteMaterial).map;
      map?.dispose?.();
      item.dispose();
    };
    if (Array.isArray(material)) material.forEach(release);
    else if (material) release(material);
  });
}
