# C++ Physics Engine (raylib + raygui)

A minimal but complete starting point: bouncing circles under gravity, with
wall collisions and circle-circle collision resolution (impulse-based,
same approach used by Box2D). A raygui panel at the top lets you tweak
gravity and bounciness live, pause the sim, and clear all bodies. Click
anywhere below the panel to spawn a new circle.

## Project layout

```
physics-engine/
├── CMakeLists.txt      # auto-downloads raylib via CMake FetchContent
├── external/
│   └── raygui.h        # single-header raygui library (already included)
└── src/
    └── main.cpp         # the whole demo
```

## Build & run

You need CMake (3.11+) and a C++17 compiler. You do **not** need to
install raylib yourself — CMake will fetch and build it automatically
the first time (this takes a few minutes; subsequent builds are fast).

### Linux

```bash
sudo apt install build-essential cmake git \
    libasound2-dev libx11-dev libxrandr-dev libxi-dev \
    libgl1-mesa-dev libglu1-mesa-dev libxcursor-dev libxinerama-dev

cmake -B build -S .
cmake --build build -j
./build/physics_engine
```

### macOS

```bash
brew install cmake git
cmake -B build -S .
cmake --build build -j
./build/physics_engine
```

### Windows (with Visual Studio)

```powershell
cmake -B build -S . -G "Visual Studio 17 2022"
cmake --build build --config Release
build\Release\physics_engine.exe
```

(MinGW works too: `cmake -B build -S . -G "MinGW Makefiles"`.)

## How the code is organized

- **`Vec2`** — tiny 2D vector struct with the operations physics needs
  (add, subtract, scale, dot, normalize).
- **`Body`** — a circle with position, velocity, radius, mass, and
  `invMass` (inverse mass — `0` means "infinitely heavy / immovable",
  which is the standard trick for static bodies).
- **`integrate()`** — applies gravity and moves bodies forward
  (semi-implicit Euler: update velocity, then position).
- **`resolveWallCollisions()`** — bounces bodies off the screen edges.
- **`resolveCircleCollisions()`** — the real physics: for every pair of
  overlapping circles, it (1) pushes them apart so they don't sink into
  each other (positional correction), then (2) applies an impulse along
  the collision normal so they bounce off each other realistically.
- **Fixed timestep loop** in `main()` — physics always steps in fixed
  `1/60s` chunks regardless of frame rate, which keeps the simulation
  stable and reproducible.

## Where to go next

This covers circle-circle collision only, which is the easiest full
pipeline to learn on. Natural next steps, roughly in order of difficulty:

1. **Axis-aligned boxes (AABB)** — add a `Box` body type and AABB-AABB /
   AABB-circle collision checks.
2. **Rotated boxes / convex polygons** — requires the Separating Axis
   Theorem (SAT) instead of the simple distance check used here.
3. **Rotation & torque** — give bodies an orientation and angular
   velocity, and apply torque so collisions can spin things.
4. **Spatial partitioning** — the current collision check is O(n²) (every
   body checked against every other). Fine for a few hundred circles;
   swap in a grid or quadtree once you want thousands.
5. **Joints/constraints** — distance joints, hinges, springs — usually
   solved with the same "sequential impulses" idea already used here.

Erin Catto's Box2D source and GDC talks are the best reference once you
want to go past this point — this demo follows the same impulse +
positional-correction pattern Box2D uses internally.

## Window and look

The app opens as a maximized, resizable window in **dark mode**. Press **F11** for borderless fullscreen;
**ESC** leaves fullscreen, or quits when you're in the normal window.

## Two modes

**Spawner** (default): click to drop shapes, with live sliders for gravity, bounciness, size, sides and spawn color.
Click **Open Sandbox Editor** (top right) to switch.

**Sandbox Editor**: a separate world for building scenes by hand.

- **Palette (left):** drag Circle / Triangle / Square / Rectangle / Polygon into the world. The Sides slider sets the polygon's side count.
- **Move shapes:** click and drag any shape. Let go while moving and it gets thrown.
- **Gravity is OFF for every new shape.** Select a shape and tick **Enable gravity** in the inspector, then set the amount in m/s^2 (9.8 = Earth). The World section has **Gravity: all on / all off** buttons.
- **Inspector (right)** for the selected shape: Size, Enable gravity + amount, Hex color (type `#RRGGBB`, plus a color picker), Delete (or press the Delete key).
- **Ground:** drag the ground line up/down, or use the Ground slider. Bounce sets the world's bounciness.
- **Pause** freezes physics so you can arrange things; **Back to Spawner** returns to the first mode.

### Assests (images from your computer)

The shelf along the bottom of the editor is called **Assests** (the label is the `ASSETS_TITLE` constant in `main.cpp`).

1. **Drag image files** (png, jpg, bmp, tga, gif, qoi) from your file explorer onto the window, **or** paste a file path into the box and press **Load** / Enter.
2. Thumbnails appear in the shelf (mouse wheel scrolls it).
3. **Drag a thumbnail up into the world** to place the image. It's a physics body with a rectangular collider matching the picture.
4. Select it to change its size and gravity like any shape. The hex color acts as a **tint** (white = original colors).
