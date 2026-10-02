"use client";

import { useEffect, useRef, useState } from "react";

import { CREDIT_URL, buildStation, disposeTree } from "@/components/three/station";

/**
 * The 3D scene on the training screen: a figure in still water with planets
 * orbiting overhead (components/three/station.ts). How fast the planets orbit
 * and the ripples spread follows how much of the mesh is training, and the
 * scene takes the run-state colour: indigo training, cyan aggregating, green
 * done, red failed. A Sketchfab embed was considered and rejected: it needs
 * internet the mesh does not otherwise need, cannot react to training state,
 * and brings Sketchfab's interface into the dashboard.
 */

export type RigState = "idle" | "training" | "aggregating" | "done" | "failed";

const STATE_COLOUR: Record<RigState, number> = {
  idle: 0x5b6478,
  training: 0x6d7cff,
  aggregating: 0x22d3ee,
  done: 0x34d399,
  failed: 0xf76b6b,
};

export default function TrainingRig({
  state = "idle",
  activity = 0,
  height = 260,
}: {
  state?: RigState;
  /** 0 to 1: the share of the mesh that is training right now. */
  activity?: number;
  height?: number;
}) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const [source, setSource] = useState<"model" | "fallback" | "loading">("loading");
  const stateRef = useRef(state);
  const activityRef = useRef(activity);
  stateRef.current = state;
  activityRef.current = activity;

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const signal = { cancelled: false };
    let cleanup = () => {};

    (async () => {
      const THREE = await import("three");
      if (signal.cancelled) return;

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 200);
      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "low-power" });
      // Capped at 1.5x: a Retina MacBook Air would otherwise render 4x the
      // pixels for no visible gain on a panel this size.
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      mount.appendChild(renderer.domElement);
      scene.add(new THREE.AmbientLight(0xffffff, 0.55));
      const key = new THREE.DirectionalLight(0xffffff, 1.4);
      key.position.set(3, 4, 2);
      scene.add(key);

      const station = await buildStation(THREE, { tint: STATE_COLOUR.training, signal });
      if (signal.cancelled) return;
      const world = new THREE.Group();
      world.add(station.group);
      scene.add(world);
      setSource(station.source);
      if (station.source === "model") {
        camera.position.set(0, 0.55, 5.2);
        camera.lookAt(0, 0.05, 0);
      } else {
        camera.position.set(0, 0.9, 4.6);
        camera.lookAt(0, 0, 0);
      }

      const clock = new THREE.Clock();
      const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      const colour = new THREE.Color(STATE_COLOUR.training);
      const target = new THREE.Color();
      let raf = 0;
      let visible = true;

      const frame = () => {
        raf = requestAnimationFrame(frame);
        if (!visible) return;
        const delta = Math.min(clock.getDelta(), 0.1);
        const current = stateRef.current;
        const running = current === "training" || current === "aggregating";
        const speed = reduceMotion ? 0 : running ? 0.35 + activityRef.current * 1.4 : 0.12;
        station.update(delta, clock.elapsedTime, speed, reduceMotion ? 0 : current === "aggregating" ? 0.09 : 0.035);
        world.rotation.y = reduceMotion ? 0 : Math.sin(clock.elapsedTime * 0.12) * 0.35;
        target.setHex(STATE_COLOUR[current] ?? STATE_COLOUR.idle);
        colour.lerp(target, Math.min(1, delta * 3));
        station.setTint(colour, running ? 0.24 : 0.12);
        renderer.render(scene, camera);
      };
      frame();

      // Stop rendering off screen or in a background tab; this page is often
      // left open for the whole of a long run.
      const intersection = new IntersectionObserver(([entry]) => {
        visible = Boolean(entry?.isIntersecting) && !document.hidden;
      });
      intersection.observe(mount);
      const onVisibility = () => {
        visible = !document.hidden;
        clock.getDelta();
      };
      document.addEventListener("visibilitychange", onVisibility);

      const resize = () => {
        const width = mount.clientWidth;
        const heightNow = mount.clientHeight;
        if (!width) return;
        renderer.setSize(width, heightNow, false);
        camera.aspect = width / Math.max(1, heightNow);
        camera.fov = camera.aspect < 1 ? 46 : 34;
        camera.updateProjectionMatrix();
      };
      const observer = new ResizeObserver(resize);
      observer.observe(mount);
      resize();

      cleanup = () => {
        cancelAnimationFrame(raf);
        observer.disconnect();
        intersection.disconnect();
        document.removeEventListener("visibilitychange", onVisibility);
        disposeTree(scene);
        renderer.dispose();
        renderer.domElement.remove();
      };
    })();

    return () => {
      signal.cancelled = true;
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
