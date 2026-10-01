"use client";

import { useEffect, useRef, useState } from "react";

/**
 * The 3D scene on the training screen: a figure standing in still water, with
 * planets orbiting overhead. While the mesh trains the planets orbit, the
 * ripples spread and the stars drift; how fast follows how much of the mesh is
 * busy, so the scene visibly works harder under load and settles when idle.
 *
 * The model is `public/models/space-station.glb`, "space boi" by silvercrow101,
 * CC BY-NC 4.0 (credited on screen and in public/models/CREDITS.md). It ships
 * with no animation clips, so the motion is built here from its parts:
 *
 *   body          the figure; bobs gently
 *   waves*        the ripple rings; pulse outward
 *   particles     the star field; drifts
 *   Sphere*       the planets; each orbits the figure at its own radius, the
 *                 inner ones faster, as orbits do
 *   Cube          a black ground slab that hides the stars; removed
 *
 * The model's materials are unlit black and off-white. The black parts take the
 * run-state colour (indigo training, cyan aggregating, green done, red failed),
 * which on an unlit model is the only way state can show.
 *
 * A Sketchfab embed was considered and rejected: it needs internet access the
 * mesh does not otherwise need, it cannot react to training state, and it
 * brings Sketchfab's own interface into the dashboard.
 *
 * If the file is missing or fails to load, a procedural scene with the same
 * idea (a core with orbiting moons) is drawn instead.
 */

const MODEL_URL = "/models/space-station.glb";
const CREDIT_URL = "https://sketchfab.com/3d-models/space-boi-f6a8c6a6727b4f2cb020c8b50bb2ee60";

export type RigState = "idle" | "training" | "aggregating" | "done" | "failed";

const STATE_COLOUR: Record<RigState, number> = {
  idle: 0x5b6478,
  training: 0x6d7cff,
  aggregating: 0x22d3ee,
  done: 0x34d399,
  failed: 0xf76b6b,
};

/** Names as GLTFLoader leaves them: dots are stripped, so "Sphere.001" is "Sphere001". */
const SKIP = /^cube/i;
const BODY = /^body$/i;
const WAVES = /^waves\d*$/i;
const STARS = /^particles$/i;
const PLANET = /^sphere\d*$/i;

export default function TrainingRig({
  state = "idle",
  activity = 0,
  height = 260,
}: {
  /** Drives colour and whether the scene is moving at full speed. */
  state?: RigState;
  /** 0 to 1: the share of the mesh that is training right now. */
  activity?: number;
  height?: number;
}) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const [source, setSource] = useState<"model" | "fallback" | "loading">("loading");

  // Live values the render loop reads, so changing props never rebuilds the scene.
  const stateRef = useRef(state);
  const activityRef = useRef(activity);
  stateRef.current = state;
  activityRef.current = activity;

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let disposed = false;
    let cleanup = () => {};

    (async () => {
      const THREE = await import("three");
      if (disposed) return;

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 200);

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "low-power" });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setSize(mount.clientWidth, mount.clientHeight, false);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      mount.appendChild(renderer.domElement);

      // Only the fallback is lit; the model's materials are unlit.
      scene.add(new THREE.AmbientLight(0xffffff, 0.55));
      const key = new THREE.DirectionalLight(0xffffff, 1.4);
      key.position.set(3, 4, 2);
      scene.add(key);

      const world = new THREE.Group();
      scene.add(world);

      type Orbiter = { pivot: import("three").Object3D; speed: number; phase: number };
      const orbiters: Orbiter[] = [];
      const ripples: {
        object: import("three").Object3D;
        base: import("three").Vector3;
        offset: number;
        vertical: number;
      }[] = [];
      let stars: import("three").Object3D | null = null;
      let figure: import("three").Object3D | null = null;
      let figureBaseY = 0;
      const tinted: { color: import("three").Color }[] = [];
      const orbitLines: import("three").LineBasicMaterial[] = [];

      // A soft glow behind the centre, in the state colour, so the scene reads
      // on a dark panel and changes visibly with the run.
      function makeGlow(radius: number) {
        const canvas = document.createElement("canvas");
        canvas.width = canvas.height = 128;
        const context = canvas.getContext("2d")!;
        const gradient = context.createRadialGradient(64, 64, 0, 64, 64, 64);
        gradient.addColorStop(0, "rgba(255,255,255,0.55)");
        gradient.addColorStop(0.35, "rgba(255,255,255,0.18)");
        gradient.addColorStop(1, "rgba(255,255,255,0)");
        context.fillStyle = gradient;
        context.fillRect(0, 0, 128, 128);
        const material = new THREE.SpriteMaterial({
          map: new THREE.CanvasTexture(canvas),
          color: STATE_COLOUR.training,
          transparent: true,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
        });
        const sprite = new THREE.Sprite(material);
        sprite.scale.setScalar(radius);
        tinted.push(material);
        return sprite;
      }

      function orbitRing(radius: number, y: number) {
        const points: import("three").Vector3[] = [];
        for (let index = 0; index <= 96; index += 1) {
          const angle = (index / 96) * Math.PI * 2;
          points.push(new THREE.Vector3(Math.cos(angle) * radius, y, Math.sin(angle) * radius));
        }
        const material = new THREE.LineBasicMaterial({
          color: STATE_COLOUR.training,
          transparent: true,
          opacity: 0.16,
          depthWrite: false,
        });
        orbitLines.push(material);
        return new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(points), material);
      }

      /** The stand-in: a glowing core with moons on tilted orbits. */
      function buildFallback() {
        const coreMaterial = new THREE.MeshStandardMaterial({
          color: STATE_COLOUR.training,
          emissive: STATE_COLOUR.training,
          emissiveIntensity: 0.55,
          roughness: 0.4,
        });
        tinted.push(coreMaterial);
        tinted.push({ color: coreMaterial.emissive });
        world.add(new THREE.Mesh(new THREE.SphereGeometry(0.42, 48, 32), coreMaterial));
        world.add(makeGlow(2.4));
        const moonMaterial = new THREE.MeshStandardMaterial({ color: 0xdfe4ec, roughness: 0.6 });
        [0.95, 1.35, 1.75].forEach((radius, index) => {
          const pivot = new THREE.Group();
          pivot.rotation.x = 0.35 - index * 0.28;
          const moon = new THREE.Mesh(new THREE.SphereGeometry(0.07 + index * 0.025, 24, 16), moonMaterial);
          moon.position.set(radius, 0, 0);
          pivot.add(moon);
          pivot.add(orbitRing(radius, 0));
          world.add(pivot);
          orbiters.push({ pivot, speed: 1 / Math.sqrt(radius), phase: index * 2.1 });
        });
        camera.position.set(0, 0.9, 4.6);
        camera.lookAt(0, 0, 0);
        setSource("fallback");
      }

      async function buildModel(): Promise<boolean> {
        try {
          const probe = await fetch(MODEL_URL, { method: "HEAD" });
          const type = probe.headers.get("content-type") ?? "";
          if (!probe.ok || type.includes("html")) return false;
        } catch {
          return false;
        }
        const { GLTFLoader } = await import("three/examples/jsm/loaders/GLTFLoader.js");
        const gltf = await new GLTFLoader().loadAsync(MODEL_URL);
        if (disposed) return true;

        const model = gltf.scene;
        model.updateMatrixWorld(true);

        // Classify the model's top-level parts by name.
        const parts: import("three").Object3D[] = [];
        model.traverse((object) => {
          if (!object.name || object === model) return;
          if (SKIP.test(object.name) || BODY.test(object.name) || WAVES.test(object.name) || STARS.test(object.name) || PLANET.test(object.name)) {
            parts.push(object);
          }
        });
        for (const part of parts) {
          if (SKIP.test(part.name)) part.visible = false;
        }

        // Recolour: black becomes the state colour, off-white stays.
        model.traverse((object) => {
          const mesh = object as import("three").Mesh;
          if (!mesh.isMesh) return;
          const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
          const replaced = materials.map((material) => {
            const basic = material as import("three").MeshBasicMaterial;
            const clone = basic.clone();
            if (basic.color && basic.color.getHex() === 0x000000) {
              clone.color = new THREE.Color(STATE_COLOUR.training);
              tinted.push(clone);
            }
            return clone;
          });
          mesh.material = Array.isArray(mesh.material) ? replaced : replaced[0];
        });

        // Frame on the figure and planets; the star field is far wider and
        // would shrink everything else to a dot.
        const focus = new THREE.Box3();
        for (const part of parts) {
          if (BODY.test(part.name) || PLANET.test(part.name)) focus.expandByObject(part);
        }
        if (focus.isEmpty()) focus.setFromObject(model);
        const size = focus.getSize(new THREE.Vector3());
        const centre = focus.getCenter(new THREE.Vector3());
        const scale = 2.6 / Math.max(size.x, size.y, size.z || 1);
        model.scale.setScalar(scale);
        model.position.sub(centre.clone().multiplyScalar(scale));
        world.add(model);
        model.updateMatrixWorld(true);

        // The vertical axis through the figure is what the planets orbit.
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
            world.add(pivot);
            pivot.attach(part); // keeps the planet exactly where the artist put it
            orbiters.push({ pivot, speed: 0.9 / Math.sqrt(Math.max(0.2, radius)), phase: 0 });
          } else if (WAVES.test(part.name)) {
            // The rings lie flat, but the exporter's axis swaps mean "flat" is
            // not the same local axes on every node. Find the local axis that
            // points up and scale the other two.
            const up = new THREE.Vector3(0, 1, 0);
            const quaternion = part.getWorldQuaternion(new THREE.Quaternion());
            const alignment = [
              new THREE.Vector3(1, 0, 0),
              new THREE.Vector3(0, 1, 0),
              new THREE.Vector3(0, 0, 1),
            ].map((axisVector) => Math.abs(axisVector.applyQuaternion(quaternion).dot(up)));
            const vertical = alignment.indexOf(Math.max(...alignment));
            ripples.push({ object: part, base: part.scale.clone(), offset: ripples.length * 0.33, vertical });
          } else if (STARS.test(part.name)) {
            const pivot = new THREE.Group();
            pivot.position.set(axis.x, 0, axis.z);
            world.add(pivot);
            pivot.attach(part);
            stars = pivot;
          }
        }

        // Faint orbit guides at each planet's radius and height.
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
          world.add(ring);
        }

        const glow = makeGlow(3.4);
        glow.position.set(axis.x, axis.y + 0.2, axis.z - 0.4);
        world.add(glow);

        camera.position.set(0, 0.55, 5.2);
        camera.lookAt(0, 0.05, 0);
        setSource("model");
        return true;
      }

      let loaded = false;
      try {
        loaded = await buildModel();
      } catch {
        loaded = false;
      }
      if (disposed) return;
      if (!loaded) buildFallback();

      const clock = new THREE.Clock();
      const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      const colour = new THREE.Color(STATE_COLOUR.training);
      const target = new THREE.Color();
      let raf = 0;
      let visible = true;

      function frame() {
        if (disposed) return;
        raf = requestAnimationFrame(frame);
        if (!visible) return;
        const delta = Math.min(clock.getDelta(), 0.1);
        const time = clock.elapsedTime;
        const current = stateRef.current;
        const running = current === "training" || current === "aggregating";

        // Idle still moves, slowly, so the scene never looks frozen or broken.
        const speed = reduceMotion ? 0 : running ? 0.35 + activityRef.current * 1.4 : 0.12;

        for (const orbiter of orbiters) {
          orbiter.pivot.rotation.y += delta * speed * orbiter.speed;
        }
        ripples.forEach((ripple) => {
          // Rings breathe outward; aggregation sends a stronger pulse.
          const pulse = current === "aggregating" ? 0.09 : 0.035;
          const wave = 1 + Math.sin(time * (0.8 + speed * 1.6) + ripple.offset * Math.PI) * pulse * (reduceMotion ? 0 : 1);
          ripple.object.scale.set(
            ripple.base.x * (ripple.vertical === 0 ? 1 : wave),
            ripple.base.y * (ripple.vertical === 1 ? 1 : wave),
            ripple.base.z * (ripple.vertical === 2 ? 1 : wave)
          );
        });
        if (stars) stars.rotation.y += delta * speed * 0.06;
        if (figure && !reduceMotion) figure.position.y = figureBaseY + Math.sin(time * 1.2) * 0.012;

        world.rotation.y = reduceMotion ? 0 : Math.sin(time * 0.12) * 0.35;

        // Ease the colour rather than snapping it when a run changes state.
        target.setHex(STATE_COLOUR[current] ?? STATE_COLOUR.idle);
        colour.lerp(target, Math.min(1, delta * 3));
        for (const material of tinted) material.color.copy(colour);
        for (const material of orbitLines) {
          material.color.copy(colour);
          material.opacity = running ? 0.24 : 0.12;
        }

        renderer.render(scene, camera);
      }
      frame();

      // Stop rendering when scrolled off screen or the tab is hidden; this
      // component sits on a page people leave open for hours during a run.
      const intersection = new IntersectionObserver(([entry]) => {
        visible = Boolean(entry?.isIntersecting) && !document.hidden;
      });
      intersection.observe(mount);
      const onVisibility = () => {
        visible = !document.hidden;
        clock.getDelta();
      };
      document.addEventListener("visibilitychange", onVisibility);

      function resize() {
        const width = mount!.clientWidth;
        const height = mount!.clientHeight;
        if (!width) return;
        renderer.setSize(width, height, false);
        camera.aspect = width / Math.max(1, height);
        // Keep the whole scene in view on narrow screens.
        camera.fov = camera.aspect < 1 ? 46 : 34;
        camera.updateProjectionMatrix();
      }
      const observer = new ResizeObserver(resize);
      observer.observe(mount);
      resize();

      cleanup = () => {
        cancelAnimationFrame(raf);
        observer.disconnect();
        intersection.disconnect();
        document.removeEventListener("visibilitychange", onVisibility);
        scene.traverse((object) => {
          const mesh = object as import("three").Mesh;
          mesh.geometry?.dispose?.();
          const material = mesh.material as import("three").Material | import("three").Material[] | undefined;
          if (Array.isArray(material)) material.forEach((item) => item.dispose());
          else material?.dispose?.();
        });
        renderer.dispose();
        renderer.domElement.remove();
      };
    })();

    return () => {
      disposed = true;
      cleanup();
    };
  }, []);

  return (
    <div className="rig" style={{ height }}>
      <div ref={mountRef} className="rig-canvas" aria-hidden="true" />
      {source === "model" ? (
        <a className="rig-note" href={CREDIT_URL} target="_blank" rel="noreferrer" title="Model licence: CC BY-NC 4.0">
          “space boi” by silvercrow101 · CC BY-NC
        </a>
      ) : null}
    </div>
  );
}
