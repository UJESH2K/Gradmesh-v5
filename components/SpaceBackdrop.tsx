"use client";

import { useEffect, useRef } from "react";

import { buildStation, disposeTree } from "@/components/three/station";

/**
 * The landing page's background: the space scene, fixed behind the whole page,
 * always slowly turning, its planets orbiting.
 *
 * As the page scrolls, the model glides to wherever the current section has
 * left room for it. Each section says where it wants the model with data
 * attributes, and the camera eases between them:
 *
 *   data-scene-x     -1 (left) to 1 (right), as a share of half the viewport
 *   data-scene-y     -1 (down) to 1 (up)
 *   data-scene-zoom  1 is the default framing; above 1 is closer
 *   data-scene-dim   0 to 1, how far the model fades back behind the text
 *
 * On a narrow screen the model always centres and fades, so text stays legible.
 * It renders at most 1.5x device pixels, pauses in a background tab and honours
 * reduced motion, so it stays light on a fanless MacBook Air.
 */

const VENDOR_RINGS = [0xa3d65c, 0x4ea8ff, 0xd3d5de];

type Keyframe = { x: number; y: number; zoom: number; dim: number; top: number; height: number };

function readKeyframes(): Keyframe[] {
  return Array.from(document.querySelectorAll<HTMLElement>("[data-scene-x]")).map((element) => {
    const rect = element.getBoundingClientRect();
    return {
      x: Number(element.dataset.sceneX ?? 0),
      y: Number(element.dataset.sceneY ?? 0),
      zoom: Number(element.dataset.sceneZoom ?? 1),
      dim: Number(element.dataset.sceneDim ?? 0),
      top: rect.top + window.scrollY,
      height: rect.height,
    };
  });
}

/** Where the model should be for the current scroll position. */
function sample(frames: Keyframe[]): Omit<Keyframe, "top" | "height"> {
  if (frames.length === 0) return { x: 0, y: 0, zoom: 1, dim: 0 };
  const probe = window.scrollY + window.innerHeight * 0.5;
  const centres = frames.map((frame) => frame.top + frame.height / 2);
  // Always a fresh object: the caller adjusts it for narrow screens.
  const pose = ({ x, y, zoom, dim }: Keyframe) => ({ x, y, zoom, dim });
  if (probe <= centres[0]) return pose(frames[0]);
  if (probe >= centres[centres.length - 1]) return pose(frames[frames.length - 1]);
  for (let index = 0; index < frames.length - 1; index += 1) {
    if (probe >= centres[index] && probe <= centres[index + 1]) {
      const raw = (probe - centres[index]) / Math.max(1, centres[index + 1] - centres[index]);
      // Hold each pose through the middle of its section, move between them.
      const t = raw < 0.3 ? 0 : raw > 0.7 ? 1 : (raw - 0.3) / 0.4;
      const eased = t * t * (3 - 2 * t);
      const a = frames[index];
      const b = frames[index + 1];
      const mix = (from: number, to: number) => from + (to - from) * eased;
      return { x: mix(a.x, b.x), y: mix(a.y, b.y), zoom: mix(a.zoom, b.zoom), dim: mix(a.dim, b.dim) };
    }
  }
  return pose(frames[frames.length - 1]);
}

export default function SpaceBackdrop() {
  const mountRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const signal = { cancelled: false };
    let cleanup = () => {};

    (async () => {
      const THREE = await import("three");
      if (signal.cancelled) return;

      const narrow = () => window.innerWidth < 820;
      const scene = new THREE.Scene();
      scene.fog = new THREE.FogExp2(0x05060a, 0.035);
      const camera = new THREE.PerspectiveCamera(32, window.innerWidth / window.innerHeight, 0.1, 400);
      camera.position.set(0, 0.4, 7);

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, narrow() ? 1.25 : 1.5));
      renderer.setSize(window.innerWidth, window.innerHeight, false);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      mount.appendChild(renderer.domElement);

      // Deep field: three shells of stars at different depths and sizes,
      // drifting at different rates, which is what makes the scene read as
      // deep rather than painted on.
      // A soft round sprite for every star: without one, WebGL draws points as
      // squares, which the near layer makes obvious.
      const starCanvas = document.createElement("canvas");
      starCanvas.width = starCanvas.height = 64;
      const starContext = starCanvas.getContext("2d");
      if (starContext) {
        const glow = starContext.createRadialGradient(32, 32, 0, 32, 32, 32);
        glow.addColorStop(0, "rgba(255,255,255,1)");
        glow.addColorStop(0.25, "rgba(255,255,255,0.85)");
        glow.addColorStop(0.6, "rgba(255,255,255,0.18)");
        glow.addColorStop(1, "rgba(255,255,255,0)");
        starContext.fillStyle = glow;
        starContext.fillRect(0, 0, 64, 64);
      }
      const starSprite = new THREE.CanvasTexture(starCanvas);
      starSprite.colorSpace = THREE.SRGBColorSpace;

      const starLayers: import("three").Points[] = [];
      const layers = narrow()
        ? [{ count: 700, radius: 40, size: 0.05 }, { count: 260, radius: 22, size: 0.08 }]
        : [
            { count: 1600, radius: 60, size: 0.05 },
            { count: 700, radius: 30, size: 0.07 },
            { count: 160, radius: 16, size: 0.11 },
          ];
      for (const layer of layers) {
        const positions = new Float32Array(layer.count * 3);
        for (let index = 0; index < layer.count; index += 1) {
          const u = Math.random() * 2 - 1;
          const phi = Math.random() * Math.PI * 2;
          const r = layer.radius * (0.6 + Math.random() * 0.4);
          const s = Math.sqrt(1 - u * u);
          positions.set([r * s * Math.cos(phi), r * u * 0.6, r * s * Math.sin(phi) - layer.radius * 0.3], index * 3);
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        const points = new THREE.Points(
          geometry,
          new THREE.PointsMaterial({
            color: 0xcfd6ff,
            map: starSprite,
            size: layer.size * 1.8,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.75,
            depthWrite: false,
          })
        );
        starLayers.push(points);
        scene.add(points);
      }

      const station = await buildStation(THREE, { tint: 0x6d7cff, size: 2.8, signal });
      if (signal.cancelled) return;
      station.setRingColours(VENDOR_RINGS);

      // pivot: where the section wants the model. spin: the model turning on
      // its own, all the time.
      const pivot = new THREE.Group();
      const spin = new THREE.Group();
      spin.rotation.x = 0.08;
      spin.add(station.group);
      pivot.add(spin);
      scene.add(pivot);

      const fadeTargets: { material: import("three").Material & { opacity: number }; base: number }[] = [];
      station.group.traverse((object) => {
        const mesh = object as import("three").Mesh;
        const material = mesh.material as (import("three").Material & { opacity: number }) | undefined;
        if (!material || Array.isArray(material)) return;
        material.transparent = true;
        fadeTargets.push({ material, base: material.opacity ?? 1 });
      });

      let frames = readKeyframes();
      const refreshFrames = () => {
        frames = readKeyframes();
      };
      const resizeObserver = new ResizeObserver(refreshFrames);
      resizeObserver.observe(document.body);

      const pointer = { x: 0, y: 0 };
      const onPointer = (event: PointerEvent) => {
        pointer.x = event.clientX / window.innerWidth - 0.5;
        pointer.y = event.clientY / window.innerHeight - 0.5;
      };
      window.addEventListener("pointermove", onPointer, { passive: true });

      const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      const clock = new THREE.Clock();
      const current = { x: 0, y: 0, zoom: 1, dim: 0 };
      let first = true;
      let visible = !document.hidden;
      let raf = 0;

      const frame = () => {
        raf = requestAnimationFrame(frame);
        if (!visible) return;
        const delta = Math.min(clock.getDelta(), 0.1);
        const time = clock.elapsedTime;

        const wanted = sample(frames);
        if (narrow()) {
          // Text runs the full width on a phone, so the scene sits a little
          // right of centre, smaller and further back.
          wanted.x = 0.3;
          wanted.zoom *= 0.85;
          wanted.dim = Math.max(wanted.dim, 0.6);
        }
        const ease = first ? 1 : Math.min(1, delta * 2.2);
        first = false;
        current.x += (wanted.x - current.x) * ease;
        current.y += (wanted.y - current.y) * ease;
        current.zoom += (wanted.zoom - current.zoom) * ease;
        current.dim += (wanted.dim - current.dim) * ease;

        // Convert the section's "share of half the viewport" into world units
        // at the model's distance, so placement holds on any aspect ratio.
        // In portrait, back the camera off until the scene fits the width.
        const distance = Math.max(7 / current.zoom, camera.aspect < 1 ? 4.9 / camera.aspect / current.zoom : 0);
        const halfHeight = Math.tan((camera.fov * Math.PI) / 360) * distance;
        const halfWidth = halfHeight * camera.aspect;
        pivot.position.set(current.x * halfWidth * 0.62, current.y * halfHeight * 0.5, 0);

        camera.position.x += (pointer.x * 0.35 - camera.position.x) * Math.min(1, delta * 2);
        camera.position.y += (0.4 - pointer.y * 0.25 - camera.position.y) * Math.min(1, delta * 2);
        camera.position.z = distance;
        camera.lookAt(pivot.position.x * 0.15, pivot.position.y * 0.15, 0);

        if (!reduceMotion) {
          spin.rotation.y += delta * 0.16;
          station.update(delta, time, 0.55, 0.04);
          starLayers.forEach((layer, index) => {
            layer.rotation.y += delta * (0.004 + index * 0.003);
          });
        }
        const opacity = 1 - current.dim * 0.75;
        for (const target of fadeTargets) target.material.opacity = target.base * opacity;

        renderer.render(scene, camera);
      };
      frame();
      document.documentElement.classList.add("scene-ready");

      const onResize = () => {
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, narrow() ? 1.25 : 1.5));
        renderer.setSize(window.innerWidth, window.innerHeight, false);
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        refreshFrames();
      };
      window.addEventListener("resize", onResize);
      const onVisibility = () => {
        visible = !document.hidden;
        clock.getDelta();
      };
      document.addEventListener("visibilitychange", onVisibility);

      cleanup = () => {
        cancelAnimationFrame(raf);
        resizeObserver.disconnect();
        window.removeEventListener("pointermove", onPointer);
        window.removeEventListener("resize", onResize);
        document.removeEventListener("visibilitychange", onVisibility);
        document.documentElement.classList.remove("scene-ready");
        disposeTree(scene);
        starSprite.dispose();
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
    <div className="ld-backdrop" aria-hidden="true">
      <div className="ld-nebula" />
      <div ref={mountRef} className="ld-canvas" />
      <div className="ld-vignette" />
      <div className="ld-grain" />
    </div>
  );
}
