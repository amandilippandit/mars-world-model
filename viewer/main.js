// Mars World Model — Three.js web viewer
//
// Third-person Mars exploration: a stylized astronaut figure walks across
// procedural Mars terrain under correct Mars gravity. Camera orbits the
// character with the mouse; WASD moves; Shift sprints; Space jumps (with
// the right Mars airtime); scroll zooms.
//
// Loads:
//   /data/processed/terrain.bin     — float32 raw heightmap
//   /data/processed/terrain.json    — heightmap metadata
//   /src/mars/assets/regolith.png   — baked sand albedo
//   /src/mars/assets/regolith_normal.png — baked normal map

import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

// ──────────────────────────────────────────────────────────────────────
//  Mars constants
// ──────────────────────────────────────────────────────────────────────
const MARS = {
  gravity: 3.721,
  zenith:  new THREE.Color(0.83, 0.50, 0.30),
  horizon: new THREE.Color(0.95, 0.65, 0.40),
  sun_color: new THREE.Color(0.95, 0.78, 0.55),
  regolith: new THREE.Color(0.55, 0.34, 0.22),
  ground_ambient: new THREE.Color(0.30, 0.18, 0.12),
};

// ──────────────────────────────────────────────────────────────────────
//  Renderer & scene
// ──────────────────────────────────────────────────────────────────────
const host = document.getElementById("canvas-host");
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0.92, 0.62, 0.40).convertSRGBToLinear();

// Sky sphere — radius < camera.far so it never clips
{
  const skyGeo = new THREE.SphereGeometry(4500, 64, 32);
  const skyMat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: {
      topColor:    { value: MARS.zenith },
      bottomColor: { value: MARS.horizon },
      exponent:    { value: 0.65 },
    },
    vertexShader: `
      varying vec3 vWorldPos;
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldPos = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }`,
    fragmentShader: `
      uniform vec3 topColor;
      uniform vec3 bottomColor;
      uniform float exponent;
      varying vec3 vWorldPos;
      void main() {
        float h = normalize(vWorldPos).y;
        float t = pow(max(h, 0.0), exponent);
        gl_FragColor = vec4(mix(bottomColor, topColor, t), 1.0);
      }`,
  });
  scene.add(new THREE.Mesh(skyGeo, skyMat));
}

scene.fog = new THREE.FogExp2(
  new THREE.Color(0.92, 0.62, 0.40).convertSRGBToLinear(),
  0.0009
);

// ──────────────────────────────────────────────────────────────────────
//  Lighting
// ──────────────────────────────────────────────────────────────────────
const sun = new THREE.DirectionalLight(MARS.sun_color, 2.4);
sun.position.set(220, 260, 160);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.near = 1;
sun.shadow.camera.far = 1500;
sun.shadow.camera.left = -350;
sun.shadow.camera.right = 350;
sun.shadow.camera.top = 350;
sun.shadow.camera.bottom = -350;
sun.shadow.bias = -0.0005;
sun.shadow.normalBias = 0.5;
scene.add(sun);
scene.add(sun.target);

const hemi = new THREE.HemisphereLight(MARS.horizon, MARS.ground_ambient, 0.55);
scene.add(hemi);
scene.add(new THREE.AmbientLight(MARS.regolith, 0.10));

// ──────────────────────────────────────────────────────────────────────
//  Camera (third-person — its position is computed from character pose)
// ──────────────────────────────────────────────────────────────────────
const camera = new THREE.PerspectiveCamera(
  55, window.innerWidth / window.innerHeight, 0.05, 8000
);

// ──────────────────────────────────────────────────────────────────────
//  Vertex AO baking (free at runtime)
// ──────────────────────────────────────────────────────────────────────
function bakeVertexAO(geom, heights, meta) {
  const pos = geom.attributes.position;
  const n = pos.count;
  const colors = new Float32Array(n * 3);
  const samplesPerRing = 8;
  const ringRadii = [4, 10, 20];
  for (let i = 0; i < n; i++) {
    const px = pos.getX(i);
    const py = pos.getY(i);
    const pz = pos.getZ(i);
    const colF = (px + meta.width_m / 2) / meta.width_m * (meta.cols - 1);
    const rowF = (pz + meta.height_m / 2) / meta.height_m * (meta.rows - 1);
    let occluded = 0, total = 0;
    for (const r of ringRadii) {
      for (let s = 0; s < samplesPerRing; s++) {
        const a = (s + 0.5) * 2 * Math.PI / samplesPerRing;
        const nc = Math.max(0, Math.min(meta.cols - 1, Math.round(colF + Math.cos(a) * r)));
        const nr = Math.max(0, Math.min(meta.rows - 1, Math.round(rowF + Math.sin(a) * r)));
        const nh = heights[nr * meta.cols + nc];
        const dist_m = r * meta.m_per_cell;
        if ((nh - py) / dist_m > 0.55) occluded++;
        total++;
      }
    }
    const ao = 1.0 - 0.55 * (occluded / total);
    colors[i * 3] = ao;
    colors[i * 3 + 1] = ao;
    colors[i * 3 + 2] = ao;
  }
  geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
}

// ──────────────────────────────────────────────────────────────────────
//  Terrain loader
// ──────────────────────────────────────────────────────────────────────
async function loadTerrain() {
  const loader = new THREE.TextureLoader();
  const [meta, binBuf, regolithTex, normalTex] = await Promise.all([
    fetch("/data/processed/terrain.json").then(r => r.json()),
    fetch("/data/processed/terrain.bin").then(r => r.arrayBuffer()),
    loader.loadAsync("/src/mars/assets/regolith.png"),
    loader.loadAsync("/src/mars/assets/regolith_normal.png"),
  ]);

  document.getElementById("meta-line").textContent =
    `${meta.cols}×${meta.rows} cells · ${(meta.width_m / 1000).toFixed(2)} km × ${(meta.height_m / 1000).toFixed(2)} km · ` +
    `elev ${meta.z_min.toFixed(1)}…${meta.z_max.toFixed(1)} m`;

  const heights = new Float32Array(binBuf);

  const target = 512;
  const stride = Math.max(1, Math.ceil(meta.cols / target));
  const cols = Math.floor(meta.cols / stride);
  const rows = Math.floor(meta.rows / stride);
  const widthM = (cols - 1) * meta.m_per_cell * stride;
  const heightM = (rows - 1) * meta.m_per_cell * stride;

  const geom = new THREE.PlaneGeometry(widthM, heightM, cols - 1, rows - 1);
  geom.rotateX(-Math.PI / 2);
  const pos = geom.attributes.position;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const srcIdx = (r * stride) * meta.cols + (c * stride);
      pos.setY(r * cols + c, heights[srcIdx]);
    }
  }
  pos.needsUpdate = true;
  geom.computeVertexNormals();
  geom.computeTangents();

  bakeVertexAO(geom, heights, meta);

  const aniso = Math.min(4, renderer.capabilities.getMaxAnisotropy());
  regolithTex.wrapS = regolithTex.wrapT = THREE.RepeatWrapping;
  regolithTex.repeat.set(40, 40);
  regolithTex.anisotropy = aniso;
  regolithTex.colorSpace = THREE.SRGBColorSpace;

  normalTex.wrapS = normalTex.wrapT = THREE.RepeatWrapping;
  normalTex.repeat.set(40, 40);
  normalTex.anisotropy = aniso;
  normalTex.colorSpace = THREE.NoColorSpace;

  const material = new THREE.MeshStandardMaterial({
    map: regolithTex,
    normalMap: normalTex,
    normalScale: new THREE.Vector2(1.2, 1.2),
    roughness: 0.96,
    metalness: 0.0,
    flatShading: false,
    vertexColors: true,
  });

  const mesh = new THREE.Mesh(geom, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  scene.add(mesh);

  // Distant ground plane
  const groundColor = regolithTex.clone();   groundColor.needsUpdate = true;
  const groundNormal = normalTex.clone();    groundNormal.needsUpdate = true;
  groundColor.repeat.set(800, 800);
  groundNormal.repeat.set(800, 800);
  const groundGeo = new THREE.PlaneGeometry(8000, 8000, 1, 1);
  groundGeo.rotateX(-Math.PI / 2);
  const groundMat = new THREE.MeshStandardMaterial({
    map: groundColor,
    normalMap: groundNormal,
    normalScale: new THREE.Vector2(0.5, 0.5),
    roughness: 1.0,
    metalness: 0,
  });
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.position.y = meta.z_min - 0.5;
  ground.receiveShadow = true;
  scene.add(ground);

  // PMREM env map for PBR reflections
  const pmrem = new THREE.PMREMGenerator(renderer);
  pmrem.compileEquirectangularShader();
  const envMap = pmrem.fromScene(scene, 0.1).texture;
  scene.environment = envMap;
  material.envMap = envMap;
  material.envMapIntensity = 0.45;
  groundMat.envMap = envMap;
  groundMat.envMapIntensity = 0.35;
  pmrem.dispose();

  sun.target.position.set(0, 0, 0);
  return { heights, meta, cols, rows, stride, widthM, heightM };
}


// ──────────────────────────────────────────────────────────────────────
//  Astronaut humanoid — hierarchical skeleton with biomechanical joints
// ──────────────────────────────────────────────────────────────────────
//
// Joint hierarchy (each level is a THREE.Group with its own rotation pivot):
//
//   root (character.position = ground contact)
//     hipsBob               — vertical bob + lateral sway
//       pelvis              — pelvis tilt, hip yaw counter-rotation
//         spine             — sprint lean + counter-twist + breathing
//           torso/chest mesh, backpack, accent stripe
//           neck            — head nod
//             head + helmet + antenna
//           leftArm.shoulder
//             upperArm mesh
//             leftArm.elbow
//               forearm mesh + glove
//           rightArm.{shoulder, elbow}
//         leftLeg.hip
//           upperLeg mesh
//           leftLeg.knee
//             lowerLeg mesh
//             leftLeg.ankle
//               foot mesh
//         rightLeg.{hip, knee, ankle}
//
// Each joint takes its rotation independently so the animation can drive
// real biomechanics: knees flexing only during the swing phase, pelvis
// tilting (Trendelenburg) on the airborne side, spine counter-twisting,
// arms swinging in opposition to legs with elbow bend, etc.
// ──────────────────────────────────────────────────────────────────────
//  Scatter rocks via GPU instancing
// ──────────────────────────────────────────────────────────────────────
// Real Mars rover panoramas show the ground LITERALLY DOTTED with basalt
// rocks at every scale. We instance ~10–15k rocks across the terrain
// using THREE.InstancedMesh — all of them render in a handful of draw
// calls, regardless of count. Without this, the surface looks like a
// smooth dust desert; with it, the surface reads as Mars.
function scatterRocks(terrain, opts) {
  const {
    count, sizeMin, sizeMax,
    color, roughness = 0.95,
    castShadow = false, slopeLimit = 0.7,
  } = opts;

  // 4 base shapes for variety. Per-vertex distortion gives unique
  // silhouettes. Low-poly (~20 tris each) so 8k+ instances stay cheap.
  const baseGeos = [];
  for (let v = 0; v < 4; v++) {
    const g = new THREE.IcosahedronGeometry(1, 0);
    const pos = g.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
      const d = 0.55 + 0.5 * Math.random();
      pos.setXYZ(i, x * d, y * d * (0.45 + Math.random() * 0.3), z * d);
    }
    g.computeVertexNormals();
    baseGeos.push(g);
  }

  const material = new THREE.MeshStandardMaterial({
    color, roughness, metalness: 0.0, flatShading: true,
  });
  if (scene.environment) {
    material.envMap = scene.environment;
    material.envMapIntensity = 0.4;
  }

  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const e = new THREE.Euler();
  const p = new THREE.Vector3();
  const s = new THREE.Vector3();

  const perGeo = Math.ceil(count / baseGeos.length);
  let totalPlaced = 0;

  for (const g of baseGeos) {
    const inst = new THREE.InstancedMesh(g, material, perGeo);
    inst.castShadow = castShadow;
    inst.receiveShadow = true;

    let placed = 0, attempts = 0;
    while (placed < perGeo && attempts < perGeo * 6) {
      attempts++;
      const x = (Math.random() - 0.5) * terrain.widthM * 0.96;
      const z = (Math.random() - 0.5) * terrain.heightM * 0.96;
      const y = sampleTerrainHeight(terrain, x, z);
      if (y < terrain.meta.z_min - 0.5) continue;
      // Reject cliff faces — rocks would clip through and float.
      const eps = 1.0;
      const dyx = sampleTerrainHeight(terrain, x + eps, z) - sampleTerrainHeight(terrain, x - eps, z);
      const dyz = sampleTerrainHeight(terrain, x, z + eps) - sampleTerrainHeight(terrain, x, z - eps);
      const slope = Math.hypot(dyx, dyz) / (2 * eps);
      if (slope > slopeLimit) continue;

      const sz = sizeMin + Math.pow(Math.random(), 2.2) * (sizeMax - sizeMin);
      e.set(
        (Math.random() - 0.5) * 0.6,
        Math.random() * Math.PI * 2,
        (Math.random() - 0.5) * 0.6,
      );
      q.setFromEuler(e);
      s.set(sz, sz * (0.45 + Math.random() * 0.55), sz);
      // Sink slightly into the ground so they look embedded, not stuck on top.
      p.set(x, y - sz * 0.18, z);
      m.compose(p, q, s);
      inst.setMatrixAt(placed, m);
      placed++;
    }
    inst.instanceMatrix.needsUpdate = true;
    inst.count = placed;
    scene.add(inst);
    totalPlaced += placed;
  }
  return totalPlaced;
}


function buildHumanoid() {
  const root = new THREE.Group();

  // Materials — clothed human (no spacesuit). Dark fabric body, skin-tone
  // head and hands, black boots. Designed for humanoid-robot pretraining
  // context where the avatar represents a real human walker, not a Mars
  // astronaut.
  const fabric = new THREE.MeshStandardMaterial({
    color: 0x2d3640, roughness: 0.88, metalness: 0.02,
    envMap: scene.environment, envMapIntensity: 0.3,
  });
  const skin = new THREE.MeshStandardMaterial({
    color: 0xc69377, roughness: 0.72, metalness: 0.0,
    envMap: scene.environment, envMapIntensity: 0.25,
  });
  const hair = new THREE.MeshStandardMaterial({
    color: 0x1f1612, roughness: 0.55, metalness: 0.05,
    envMap: scene.environment, envMapIntensity: 0.3,
  });
  const boot = new THREE.MeshStandardMaterial({
    color: 0x14161a, roughness: 0.55, metalness: 0.15,
    envMap: scene.environment, envMapIntensity: 0.4,
  });

  // hipsBob: top-of-pelvis height ~ 0.94 m above feet. Total body height
  // works out to ~1.72 m given the leg lengths below — a realistic adult.
  const hipsBob = new THREE.Group();
  hipsBob.position.y = 0.94;
  root.add(hipsBob);

  // Pelvis: tilts laterally, yaws around vertical.
  const pelvis = new THREE.Group();
  hipsBob.add(pelvis);

  // Spine: leans forward when running, counter-twists vs pelvis.
  const spine = new THREE.Group();
  pelvis.add(spine);

  // ── Visible pelvis (sits at the top of the legs) ─────────────────
  const pelvisMesh = new THREE.Mesh(
    new THREE.BoxGeometry(0.30, 0.18, 0.22), fabric
  );
  pelvisMesh.position.y = -0.06;
  pelvisMesh.castShadow = true;
  // Slightly round the corners by scaling — boxes look too rigid otherwise
  pelvis.add(pelvisMesh);

  // ── Torso: chest (broader, flatter F-B) → abdomen (waist taper) ──
  // Real human torso silhouette: oval cross-section (wider side-to-side
  // than front-to-back), broadest at the chest, narrowest at the waist,
  // flaring back out at the hips. We approximate with two stacked
  // capsules with non-uniform scaling.
  const chest = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.20, 0.10, 10, 18), fabric
  );
  chest.position.y = 0.32;
  chest.scale.set(1.18, 1.0, 0.74);
  chest.castShadow = true;
  spine.add(chest);

  const abdomen = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.155, 0.10, 10, 18), fabric
  );
  abdomen.position.y = 0.12;
  abdomen.scale.set(1.10, 1.0, 0.78);
  abdomen.castShadow = true;
  spine.add(abdomen);

  // ── Neck + head ──────────────────────────────────────────────────
  const neck = new THREE.Group();
  neck.position.y = 0.58;     // above the chest
  spine.add(neck);

  const neckMesh = new THREE.Mesh(
    new THREE.CylinderGeometry(0.055, 0.063, 0.08, 14), skin
  );
  neckMesh.position.y = -0.04;
  neckMesh.castShadow = true;
  neck.add(neckMesh);

  // Head: ellipsoid with the long axis vertical (head is taller than wide)
  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.105, 26, 20), skin
  );
  head.position.y = 0.08;
  head.scale.set(0.95, 1.10, 0.92);
  head.castShadow = true;
  neck.add(head);

  // Subtle jawline shadow (ellipsoid below head, slightly darker skin)
  const jaw = new THREE.Mesh(
    new THREE.SphereGeometry(0.080, 16, 12),
    new THREE.MeshStandardMaterial({
      color: 0xb38063, roughness: 0.78, metalness: 0,
      envMap: scene.environment, envMapIntensity: 0.2,
    })
  );
  jaw.position.y = 0.005;
  jaw.scale.set(0.95, 0.6, 0.85);
  neck.add(jaw);

  // Hair cap
  const hairCap = new THREE.Mesh(
    new THREE.SphereGeometry(0.108, 24, 16, 0, Math.PI * 2, 0, Math.PI * 0.55),
    hair
  );
  hairCap.position.y = 0.08;
  hairCap.scale.set(0.95, 1.10, 0.92);
  hairCap.castShadow = true;
  neck.add(hairCap);

  // ── Arms: shoulder → deltoid → upperArm → elbow → forearm → hand ──
  function makeArm(side) {
    const shoulder = new THREE.Group();
    // Shoulder offset 0.21 → 42 cm shoulder span (typical adult male)
    shoulder.position.set(side * 0.21, 0.42, 0);
    spine.add(shoulder);

    // Deltoid cap — gives the shoulder visible muscle volume
    const deltoid = new THREE.Mesh(
      new THREE.SphereGeometry(0.078, 14, 12), fabric
    );
    deltoid.scale.set(1.0, 0.85, 1.0);
    deltoid.castShadow = true;
    shoulder.add(deltoid);

    // Upper arm (sleeve)
    const upper = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.054, 0.20, 6, 10), fabric
    );
    upper.position.y = -0.14;
    upper.castShadow = true;
    shoulder.add(upper);

    const elbow = new THREE.Group();
    elbow.position.y = -0.28;     // upper arm length 28 cm
    shoulder.add(elbow);

    // Forearm (slightly thinner at wrist than elbow)
    const forearm = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.046, 0.18, 6, 10), fabric
    );
    forearm.position.y = -0.11;
    forearm.castShadow = true;
    elbow.add(forearm);

    // Wrist (small skin band)
    const wrist = new THREE.Mesh(
      new THREE.CylinderGeometry(0.040, 0.044, 0.025, 10), skin
    );
    wrist.position.y = -0.218;
    elbow.add(wrist);

    // Hand: flat palm (oval, much wider than thick)
    const hand = new THREE.Mesh(
      new THREE.SphereGeometry(0.05, 16, 12), skin
    );
    hand.position.y = -0.27;
    hand.scale.set(0.70, 1.45, 0.42);    // flat palm
    hand.castShadow = true;
    elbow.add(hand);

    return { shoulder, elbow };
  }

  // ── Legs: hip → upperLeg (pants) → knee → lowerLeg (pants) → ankle + shoe ──
  function makeLeg(side) {
    const hip = new THREE.Group();
    // Hip offset 0.09 → 18 cm hip span (anatomically realistic)
    hip.position.set(side * 0.09, 0, 0);
    pelvis.add(hip);

    // Thigh: slightly thicker at top than at knee
    const upper = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.078, 0.30, 6, 10), fabric
    );
    upper.position.y = -0.225;
    upper.castShadow = true;
    hip.add(upper);

    const knee = new THREE.Group();
    knee.position.y = -0.45;       // thigh length 45 cm
    hip.add(knee);

    // Knee cap (subtle bump)
    const kneeCap = new THREE.Mesh(
      new THREE.SphereGeometry(0.072, 12, 10), fabric
    );
    kneeCap.scale.set(1.0, 0.65, 1.0);
    knee.add(kneeCap);

    // Calf
    const lower = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.062, 0.30, 6, 10), fabric
    );
    lower.position.y = -0.21;
    lower.castShadow = true;
    knee.add(lower);

    const ankle = new THREE.Group();
    ankle.position.y = -0.42;      // calf length 42 cm
    knee.add(ankle);

    // ── Shoe: sole + upper + slight toe taper ──────────────────────
    const sole = new THREE.Mesh(
      new THREE.BoxGeometry(0.10, 0.030, 0.27), boot
    );
    sole.position.set(0, -0.055, 0.05);
    sole.castShadow = true;
    ankle.add(sole);

    const shoeUpper = new THREE.Mesh(
      new THREE.BoxGeometry(0.095, 0.05, 0.22), boot
    );
    shoeUpper.position.set(0, -0.015, 0.05);
    shoeUpper.castShadow = true;
    ankle.add(shoeUpper);

    // Toe wedge — narrower than the rest of the shoe
    const toe = new THREE.Mesh(
      new THREE.BoxGeometry(0.07, 0.04, 0.05), boot
    );
    toe.position.set(0, -0.025, 0.20);
    ankle.add(toe);

    return { hip, knee, ankle };
  }

  const leftArm  = makeArm(+1);
  const rightArm = makeArm(-1);
  const leftLeg  = makeLeg(+1);
  const rightLeg = makeLeg(-1);

  root.userData.parts = {
    hipsBob, pelvis, spine, neck,
    leftArm, rightArm, leftLeg, rightLeg,
  };
  root.userData.walkPhase = 0;
  root.userData.airBlend = 0;       // 0 = grounded, 1 = airborne
  root.userData.sprintBlend = 0;
  root.userData.moveBlend = 0;
  return root;
}


// ──────────────────────────────────────────────────────────────────────
//  Walk-cycle animator — drives every joint from a single phase variable.
// ──────────────────────────────────────────────────────────────────────
//
// Phase convention (cycles 0 → 2π = one full stride, two steps):
//   phase=0:   left toe-off (left leg behind body, lifting)
//   phase=π/2: left mid-swing (knee max bend, leg passing under body)
//   phase=π:   left heel-strike (leg ahead, knee straight, right toe-off)
//   phase=3π/2: right mid-swing
//
// Hip angle = -A·cos(phase): leg goes from −A (behind) to +A (forward).
// Knee bends most during swing (max at mid-swing), nearly straight in stance.
// Pelvis tilts down on the airborne side (Trendelenburg gait).
// Spine counter-rotates against the pelvis. Arms swing in opposition
// with elbow flexion that increases during forward swing.
function animateHumanoid(humanoid, dt) {
  const ud = humanoid.userData;
  const p  = ud.parts;
  const moving = !!ud.isMoving;
  const sprinting = !!ud.isSprinting && moving;
  const grounded = !!ud.isOnGround;

  // Smoothly blend state factors so animation crossfades cleanly.
  const lerp = (a, b, k) => a + (b - a) * Math.min(1, Math.max(0, k));
  ud.airBlend     = lerp(ud.airBlend,     grounded ? 0 : 1, dt * 8);
  ud.sprintBlend  = lerp(ud.sprintBlend,  sprinting ? 1 : 0, dt * 4);
  ud.moveBlend    = lerp(ud.moveBlend,    moving ? 1 : 0, dt * 6);

  // Step rate: walks ~ 0.95 cycles/s (≈ 1.9 steps/s), sprints ~ 1.45 cycles/s
  const cyclesPerSec = 0.95 + ud.sprintBlend * 0.5;
  if (moving && grounded) {
    ud.walkPhase += dt * Math.PI * 2 * cyclesPerSec;
  } else if (!grounded) {
    // Hold in air — no advancing
  } else {
    // Decay phase toward the nearest neutral position so the legs settle
    // back to a standing pose smoothly.
    const target = Math.round(ud.walkPhase / Math.PI) * Math.PI;
    ud.walkPhase = lerp(ud.walkPhase, target, dt * 4);
  }

  const phase = ud.walkPhase;
  const sinP = Math.sin(phase);
  const cosP = Math.cos(phase);
  const m  = ud.moveBlend;
  const s  = ud.sprintBlend;
  const intens = (1 + s * 0.45);   // sprint amplifies amplitudes ~45%

  // ── Hip angle (forward/back) ───────────────────────────────────────
  const hipSwing = 0.65 * intens * m;
  p.leftLeg.hip.rotation.x  = -hipSwing * cosP;
  p.rightLeg.hip.rotation.x = +hipSwing * cosP;

  // ── Knee flexion ───────────────────────────────────────────────────
  // Bends only during swing (when sin > 0 for left, sin < 0 for right).
  // Squared so it ramps in/out smoothly. Negative rotation = knee bends backward.
  const kneeRest = 0.06;
  const kneePeak = 1.55 * intens * m;
  const lk = Math.max(0, sinP);
  const rk = Math.max(0, -sinP);
  p.leftLeg.knee.rotation.x  = -(kneeRest + kneePeak * lk * lk);
  p.rightLeg.knee.rotation.x = -(kneeRest + kneePeak * rk * rk);

  // ── Ankle (heel strike + toe-off) ──────────────────────────────────
  const ankleAmp = 0.32 * intens * m;
  p.leftLeg.ankle.rotation.x  = -ankleAmp * sinP;
  p.rightLeg.ankle.rotation.x = +ankleAmp * sinP;

  // ── Pelvis lateral tilt (Trendelenburg) ───────────────────────────
  // Hip drops on the swing-leg side as the support leg holds the body up.
  p.pelvis.rotation.z = -sinP * 0.10 * intens * m;

  // ── Pelvis horizontal rotation (counter to spine) ─────────────────
  p.pelvis.rotation.y = +sinP * 0.16 * intens * m;

  // ── Spine counter-twist ───────────────────────────────────────────
  p.spine.rotation.y = -sinP * 0.22 * intens * m;

  // ── Spine forward lean (sprint) ───────────────────────────────────
  const sprintLean = -0.24 * s;
  p.spine.rotation.x = lerp(p.spine.rotation.x, sprintLean, dt * 5);

  // ── Vertical body bob ─────────────────────────────────────────────
  // Body rises twice per cycle (once per step) — uses 2× phase.
  const bobAmp = 0.045 * intens * m;
  p.hipsBob.position.y = 0.92 + (1 - Math.cos(2 * phase)) * 0.5 * bobAmp;

  // ── Lateral sway ──────────────────────────────────────────────────
  p.hipsBob.position.x = -sinP * 0.025 * intens * m;

  // ── Arm pump (shoulder rotation) ──────────────────────────────────
  // Arms move opposite the legs of the same side: when left leg goes
  // forward (cosP < 0 → left hip negative), left arm goes back (positive
  // shoulder rotation).
  const armSwing = 0.55 * intens * m;
  p.leftArm.shoulder.rotation.x  = +armSwing * cosP;
  p.rightArm.shoulder.rotation.x = -armSwing * cosP;
  // Constant slight outward arm flare so they don't clip the torso
  p.leftArm.shoulder.rotation.z  = +0.12 + 0.06 * s;
  p.rightArm.shoulder.rotation.z = -0.12 - 0.06 * s;

  // ── Elbow flexion ────────────────────────────────────────────────
  // Constant baseline + extra bend during forward swing of that arm.
  const elbowBase = 0.45 + 0.10 * s;
  const elbowExtra = 0.40 * intens * m;
  const lArmFwd = Math.max(0, +cosP);
  const rArmFwd = Math.max(0, -cosP);
  p.leftArm.elbow.rotation.x  = -(elbowBase + elbowExtra * lArmFwd);
  p.rightArm.elbow.rotation.x = -(elbowBase + elbowExtra * rArmFwd);

  // ── Head nod ──────────────────────────────────────────────────────
  p.neck.rotation.x = 0.04 * intens * m * Math.sin(2 * phase);

  // ── Airborne pose blend (jump tuck) ───────────────────────────────
  const air = ud.airBlend;
  if (air > 0.001) {
    // Both knees tuck up
    p.leftLeg.knee.rotation.x  -= 0.7 * air;
    p.rightLeg.knee.rotation.x -= 0.7 * air;
    // Hips bend forward slightly
    p.leftLeg.hip.rotation.x  += 0.35 * air;
    p.rightLeg.hip.rotation.x += 0.35 * air;
    // Arms reach forward to balance
    p.leftArm.shoulder.rotation.x  -= 0.55 * air;
    p.rightArm.shoulder.rotation.x -= 0.55 * air;
    // Elbows extend a bit
    p.leftArm.elbow.rotation.x  += 0.25 * air;
    p.rightArm.elbow.rotation.x += 0.25 * air;
  }

  // ── Idle breathing ────────────────────────────────────────────────
  const idleAmt = 1 - ud.moveBlend;
  if (idleAmt > 0.01) {
    const breath = 0.008 * Math.sin(performance.now() * 0.0018);
    p.spine.position.y = breath * idleAmt;
    // Tiny shoulder rise/fall with breath
    p.leftArm.shoulder.rotation.z  += breath * 0.5 * idleAmt;
    p.rightArm.shoulder.rotation.z -= breath * 0.5 * idleAmt;
  } else {
    p.spine.position.y = 0;
  }
}


// ──────────────────────────────────────────────────────────────────────
//  Third-person camera + character controls
// ──────────────────────────────────────────────────────────────────────
const overlay = document.getElementById("click-to-play");
let pointerLocked = false;

renderer.domElement.addEventListener("click", () => {
  if (!pointerLocked) renderer.domElement.requestPointerLock();
});
overlay.addEventListener("click", () => {
  renderer.domElement.requestPointerLock();
});
document.addEventListener("pointerlockchange", () => {
  pointerLocked = document.pointerLockElement === renderer.domElement;
  overlay.style.display = pointerLocked ? "none" : "flex";
});

// Camera orbit state
let camYaw = 0;             // radians around character (Y axis)
let camPitch = -0.15;       // radians (pitch). 0 = level. Negative = looking down.
let camDistance = 4.5;

document.addEventListener("mousemove", (e) => {
  if (!pointerLocked) return;
  camYaw   -= e.movementX * 0.0025;
  camPitch -= e.movementY * 0.0025;
  camPitch = Math.max(-1.30, Math.min(0.55, camPitch));
});

renderer.domElement.addEventListener("wheel", (e) => {
  e.preventDefault();
  camDistance = Math.max(2.0, Math.min(20.0, camDistance + e.deltaY * 0.01));
}, { passive: false });

const keys = new Set();
window.addEventListener("keydown", e => keys.add(e.code));
window.addEventListener("keyup",   e => keys.delete(e.code));

// Character state
const character = buildHumanoid();
scene.add(character);
const charVel = new THREE.Vector3();
let charOnGround = false;
// 3 m/s vertical impulse — a normal human jump push gives:
//   apex on Mars = v²/(2g) = 9 / 7.44 = 1.21 m
//   apex on Earth = 9 / 19.6 = 0.46 m
// So Mars hop is ~2.6× higher than Earth, but still proportionate to a
// real human's leg push instead of cartoon-gamey heights.
const jumpV = 3.0;

function sampleTerrainHeight(t, x, z) {
  if (!t) return 0;
  const col = (x + t.widthM / 2) / (t.widthM) * (t.cols - 1);
  const row = (z + t.heightM / 2) / (t.heightM) * (t.rows - 1);
  if (col < 0 || col >= t.cols - 1 || row < 0 || row >= t.rows - 1)
    return t.meta.z_min - 1;
  const c0 = Math.floor(col), c1 = c0 + 1;
  const r0 = Math.floor(row), r1 = r0 + 1;
  const fc = col - c0, fr = row - r0;
  const idx = (r, c) => (r * t.stride) * t.meta.cols + (c * t.stride);
  const h00 = t.heights[idx(r0, c0)];
  const h10 = t.heights[idx(r0, c1)];
  const h01 = t.heights[idx(r1, c0)];
  const h11 = t.heights[idx(r1, c1)];
  const a = h00 * (1 - fc) + h10 * fc;
  const b = h01 * (1 - fc) + h11 * fc;
  return a * (1 - fr) + b * fr;
}

function updateCharacter(dt, terrain) {
  const sprintHeld = keys.has("ShiftLeft") || keys.has("ShiftRight");

  // Forward/right vectors in XZ from camera yaw. With our convention,
  // a yaw of 0 means the camera is looking along -Z, so character forward
  // is also -Z when yaw=0.
  const fx = -Math.sin(camYaw), fz = -Math.cos(camYaw);
  const rx =  Math.cos(camYaw), rz = -Math.sin(camYaw);

  let mx = 0, mz = 0;
  if (keys.has("KeyW")) { mx += fx; mz += fz; }
  if (keys.has("KeyS")) { mx -= fx; mz -= fz; }
  if (keys.has("KeyD")) { mx += rx; mz += rz; }
  if (keys.has("KeyA")) { mx -= rx; mz -= rz; }

  const moveLen = Math.hypot(mx, mz);
  const moving = moveLen > 0;
  if (moving) { mx /= moveLen; mz /= moveLen; }

  // Realistic human pace. Walking ~1.8 m/s (6.5 km/h), running ~5 m/s
  // (~18 km/h, a steady jog). Mars gravity makes the gait look slightly
  // floatier than the same speed would on Earth.
  const speed = sprintHeld ? 5.0 : 1.8;
  charVel.x = mx * speed;
  charVel.z = mz * speed;
  charVel.y -= MARS.gravity * dt;
  if (keys.has("Space") && charOnGround) {
    charVel.y = jumpV;
    charOnGround = false;
  }

  character.position.x += charVel.x * dt;
  character.position.y += charVel.y * dt;
  character.position.z += charVel.z * dt;

  // Ground collision
  const groundY = sampleTerrainHeight(terrain, character.position.x, character.position.z);
  if (character.position.y < groundY) {
    character.position.y = groundY;
    charVel.y = 0;
    charOnGround = true;
  } else {
    charOnGround = false;
  }

  // Rotate body to face movement direction (smooth lerp)
  if (moving) {
    const targetYaw = Math.atan2(mx, mz);
    let dy = targetYaw - character.rotation.y;
    while (dy >  Math.PI) dy -= Math.PI * 2;
    while (dy < -Math.PI) dy += Math.PI * 2;
    character.rotation.y += dy * Math.min(1.0, dt * 12.0);
  }

  // Animation state — animateHumanoid() reads these every frame.
  character.userData.isMoving = moving;
  character.userData.isSprinting = sprintHeld;
  character.userData.isOnGround = charOnGround;
  character.userData.vy = charVel.y;

  // HUD speed/mode line
  const horizSpeed = Math.hypot(charVel.x, charVel.z);
  const label = !charOnGround ? "JUMP"
              : sprintHeld && moving ? "RUN"
              : (moving ? "WALK" : "IDLE");
  document.getElementById("mode-line").textContent =
    `${label} · ${horizSpeed.toFixed(1)} m/s · g = ${MARS.gravity} m/s²`;
}

function updateCamera(terrain) {
  // Look-at point: chest height of the character
  const target = new THREE.Vector3(
    character.position.x,
    character.position.y + 1.30,
    character.position.z,
  );

  // Camera offset from target. With yaw=0 and pitch=0, camera sits on +Z
  // (behind the character looking toward -Z). As yaw rotates, the camera
  // sweeps around horizontally; pitch tilts vertically.
  const cx = Math.sin(camYaw) * Math.cos(camPitch) * camDistance;
  const cy = Math.sin(camPitch) * camDistance;
  const cz = Math.cos(camYaw) * Math.cos(camPitch) * camDistance;

  let camX = target.x + cx;
  let camY = target.y + cy;
  let camZ = target.z + cz;

  // Don't let the camera dip below the terrain
  if (terrain) {
    const groundY = sampleTerrainHeight(terrain, camX, camZ);
    if (camY < groundY + 0.4) camY = groundY + 0.4;
  }

  camera.position.set(camX, camY, camZ);
  camera.lookAt(target);
}

// ──────────────────────────────────────────────────────────────────────
//  Post-processing
// ──────────────────────────────────────────────────────────────────────
const composer = new EffectComposer(renderer);
composer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
composer.setSize(window.innerWidth, window.innerHeight);
composer.addPass(new RenderPass(scene, camera));
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight),
  0.30, 0.4, 1.20
);
composer.addPass(bloomPass);
composer.addPass(new OutputPass());

// ──────────────────────────────────────────────────────────────────────
//  Main loop
// ──────────────────────────────────────────────────────────────────────
let terrain = null;
const clock = new THREE.Clock();
let frame = 0; let fpsAcc = 0;
const fpsEl = document.getElementById("fps");

function tick() {
  const dt = Math.min(0.05, clock.getDelta());
  if (terrain) {
    if (pointerLocked) {
      updateCharacter(dt, terrain);
    } else {
      // While paused, force the character into idle so animation settles.
      character.userData.isMoving = false;
      character.userData.isSprinting = false;
      character.userData.isOnGround = true;
    }
    // Animation always runs — character idles + breathes when paused.
    animateHumanoid(character, dt);
    updateCamera(terrain);
  }
  composer.render();
  fpsAcc += dt; frame++;
  if (fpsAcc >= 0.5) {
    fpsEl.textContent = `${Math.round(frame / fpsAcc)} fps`;
    fpsAcc = 0; frame = 0;
  }
  requestAnimationFrame(tick);
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
  bloomPass.setSize(window.innerWidth, window.innerHeight);
});

loadTerrain().then(t => {
  terrain = t;
  character.position.set(0, sampleTerrainHeight(t, 0, 0), 0);
  character.traverse(obj => {
    if (obj.isMesh && obj.material && "envMap" in obj.material) {
      obj.material.envMap = scene.environment;
      obj.material.needsUpdate = true;
    }
  });

  // ── Scatter rocks across the terrain (the defining Mars surface look) ──
  // Three size tiers matching reference Mars rover photos: a few big
  // boulders that cast shadows, lots of mid-sized debris that forms the
  // "rock carpet", and even more small pebbles for ground texture.
  console.log("scattering rocks…");
  const rockT0 = performance.now();
  const boulders = scatterRocks(t, {
    count: 600, sizeMin: 0.8, sizeMax: 2.4,
    color: 0x3a2516, roughness: 0.96,
    castShadow: true, slopeLimit: 0.55,
  });
  const rocks = scatterRocks(t, {
    count: 4000, sizeMin: 0.20, sizeMax: 0.7,
    color: 0x2e1d10, roughness: 0.95,
    castShadow: false, slopeLimit: 0.75,
  });
  const pebbles = scatterRocks(t, {
    count: 8000, sizeMin: 0.05, sizeMax: 0.18,
    color: 0x261810, roughness: 0.92,
    castShadow: false, slopeLimit: 0.85,
  });
  console.log(
    `scattered ${boulders} boulders + ${rocks} rocks + ${pebbles} pebbles in ` +
    `${(performance.now() - rockT0).toFixed(0)} ms`
  );

  tick();
}).catch(err => {
  document.getElementById("meta-line").textContent =
    `error loading terrain: ${err.message}. Run: mars terrain --size 1024`;
  console.error(err);
  tick();
});
