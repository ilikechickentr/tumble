#include "raylib.h"

#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <cstring>
#include <cctype>
#include <cstdio>
#include <string>
#include <cstdint>
#include <cwchar>

// ---------------------------------------------------------
// Minimal 2D vector math
// ---------------------------------------------------------
struct Vec2 {
    float x = 0.0f, y = 0.0f;

    Vec2 operator+(const Vec2& o) const { return { x + o.x, y + o.y }; }
    Vec2 operator-(const Vec2& o) const { return { x - o.x, y - o.y }; }
    Vec2 operator*(float s) const { return { x * s, y * s }; }
    Vec2& operator+=(const Vec2& o) { x += o.x; y += o.y; return *this; }
    Vec2& operator-=(const Vec2& o) { x -= o.x; y -= o.y; return *this; }

    float dot(const Vec2& o) const { return x * o.x + y * o.y; }
    float length() const { return std::sqrt(x * x + y * y); }

    Vec2 normalized() const {
        float len = length();
        if (len < 1e-6f) return { 0, 0 };
        return { x / len, y / len };
    }
};

// ---------------------------------------------------------
// Shapes
// ---------------------------------------------------------
enum class ShapeType { CIRCLE, POLYGON };

// Regular N-sided polygon (triangle, square, pentagon, hexagon, ...),
// vertices in local space centered at the origin.
std::vector<Vec2> makeRegularPolygon(int sides, float circumradius, float phase = 0.0f) {
    std::vector<Vec2> verts;
    verts.reserve(sides);
    for (int i = 0; i < sides; ++i) {
        float angle = -PI / 2.0f + phase + (2.0f * PI * i) / sides; // start pointing up (+ optional phase)
        verts.push_back({ circumradius * cosf(angle), circumradius * sinf(angle) });
    }
    return verts;
}

// Rectangle with independent width/height, centered at the origin.
std::vector<Vec2> makeRectangle(float width, float height) {
    float hw = width * 0.5f, hh = height * 0.5f;
    return { {-hw, -hh}, {hw, -hh}, {hw, hh}, {-hw, hh} };
}

// ---------------------------------------------------------
// A rigid body: either a circle or a convex polygon
// ---------------------------------------------------------
struct Body {
    ShapeType shape;
    Vec2 position;
    Vec2 velocity;
    float rotation = 0.0f;            // fixed at spawn for visual variety; not simulated (no torque)
    std::vector<Vec2> localVertices;  // empty for circles, convex polygon verts otherwise
    float radius;                     // circle radius, OR bounding radius for polygons (spawn/UI use)
    float mass;
    float invMass;                    // 0 = infinite mass (static/immovable)
    Color color;
    float gravityScale = 1.0f;        // multiplier on global gravity (set per shape in the sandbox editor)
    bool gravityEnabled = true;       // sandbox editor shapes start with this OFF
    int assetIndex = -1;              // >= 0: draw this loaded image instead of a flat color

    float effectiveGravityScale() const { return gravityEnabled ? gravityScale : 0.0f; }

    // Per-object properties (edited in the sandbox editor's Properties panel)
    std::string name;
    bool isStatic = false;            // immovable: physics never moves it, but other shapes bounce off it
    float restitution = 0.4f;         // bounciness, 0..1
    float friction = 0.4f;            // sliding friction, 0..1

    void refreshInvMass() { invMass = (isStatic || mass <= 0.0f) ? 0.0f : 1.0f / mass; }

    // Circle constructor
    Body(Vec2 pos, float r, float m, Color c)
        : shape(ShapeType::CIRCLE), position(pos), velocity{ 0, 0 },
          radius(r), mass(m), color(c) {
        invMass = (m <= 0.0f) ? 0.0f : 1.0f / m;
    }

    // Polygon constructor
    Body(Vec2 pos, std::vector<Vec2> verts, float boundingRadius, float rot, float m, Color c)
        : shape(ShapeType::POLYGON), position(pos), velocity{ 0, 0 }, rotation(rot),
          localVertices(std::move(verts)), radius(boundingRadius), mass(m), color(c) {
        invMass = (m <= 0.0f) ? 0.0f : 1.0f / m;
    }

    // World-space vertex positions (rotated by `rotation`, then translated to `position`)
    std::vector<Vec2> worldVertices() const {
        std::vector<Vec2> out;
        out.reserve(localVertices.size());
        float c = cosf(rotation), s = sinf(rotation);
        for (const auto& v : localVertices) {
            Vec2 rotated{ v.x * c - v.y * s, v.x * s + v.y * c };
            out.push_back(rotated + position);
        }
        return out;
    }

    // Uniformly resize the body by `f` (mass scales with area)
    void scaleBy(float f) {
        radius *= f;
        for (auto& v : localVertices) v = v * f;
        mass *= f * f;
        refreshInvMass();
    }
};

// ---------------------------------------------------------
// Simulation parameters (tweakable live via raygui)
// ---------------------------------------------------------
struct SimParams {
    float gravity = 500.0f;      // pixels/sec^2
    float restitution = 0.8f;    // 0 = no bounce, 1 = perfectly elastic
    bool paused = false;

    // Used by the sandbox editor (the spawner leaves these at their defaults):
    bool useBodyMaterials = false;   // true: each body's own bounciness + friction are used
    float groundRestitution = 0.5f;  // bounciness of the ground and walls
    float groundFriction = 0.6f;     // friction of the ground
    float airDrag = 0.0f;            // slows everything down (per second)
};

// ---------------------------------------------------------
// Collision manifold: result of a narrow-phase test.
// normal always points from body A toward body B.
// ---------------------------------------------------------
struct Manifold {
    bool colliding = false;
    Vec2 normal;
    float penetration = 0.0f;
};

// Outward-facing unit normal for each edge of a world-space convex polygon.
// Orientation-agnostic: we verify "outward" against the centroid rather
// than assuming vertex winding order.
std::vector<Vec2> polygonWorldNormals(const std::vector<Vec2>& worldVerts) {
    size_t n = worldVerts.size();
    Vec2 centroid{ 0, 0 };
    for (const auto& v : worldVerts) centroid += v;
    centroid = centroid * (1.0f / n);

    std::vector<Vec2> normals;
    normals.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        Vec2 edge = worldVerts[(i + 1) % n] - worldVerts[i];
        Vec2 normal = Vec2{ edge.y, -edge.x }.normalized();
        Vec2 midpoint = (worldVerts[i] + worldVerts[(i + 1) % n]) * 0.5f;
        if (normal.dot(midpoint - centroid) < 0.0f) normal = normal * -1.0f;
        normals.push_back(normal);
    }
    return normals;
}

// ---------------------------------------------------------
// Narrow-phase: circle vs circle
// ---------------------------------------------------------
Manifold collideCircleCircle(const Body& a, const Body& b) {
    Manifold m;
    Vec2 delta = b.position - a.position;
    float dist = delta.length();
    float minDist = a.radius + b.radius;
    if (dist >= minDist || dist < 1e-6f) return m;

    m.colliding = true;
    m.normal = delta.normalized();
    m.penetration = minDist - dist;
    return m;
}

// ---------------------------------------------------------
// Narrow-phase: polygon vs polygon (SAT)
// ---------------------------------------------------------
Manifold collidePolygonPolygon(const Body& a, const Body& b) {
    Manifold m;
    std::vector<Vec2> vertsA = a.worldVertices();
    std::vector<Vec2> vertsB = b.worldVertices();
    std::vector<Vec2> normalsA = polygonWorldNormals(vertsA);
    std::vector<Vec2> normalsB = polygonWorldNormals(vertsB);

    float minOverlap = std::numeric_limits<float>::max();
    Vec2 smallestAxis{ 0, 0 };

    auto testAxes = [&](const std::vector<Vec2>& normals) -> bool {
        for (const auto& axis : normals) {
            float minA = std::numeric_limits<float>::max(), maxA = -std::numeric_limits<float>::max();
            for (const auto& v : vertsA) { float p = v.dot(axis); minA = std::min(minA, p); maxA = std::max(maxA, p); }
            float minB = std::numeric_limits<float>::max(), maxB = -std::numeric_limits<float>::max();
            for (const auto& v : vertsB) { float p = v.dot(axis); minB = std::min(minB, p); maxB = std::max(maxB, p); }

            float overlap = std::min(maxA, maxB) - std::max(minA, minB);
            if (overlap <= 0.0f) return false; // separating axis found -> no collision
            if (overlap < minOverlap) { minOverlap = overlap; smallestAxis = axis; }
        }
        return true;
    };

    if (!testAxes(normalsA)) return m;
    if (!testAxes(normalsB)) return m;

    // Make sure the chosen axis points from A toward B
    Vec2 centerDelta = b.position - a.position;
    if (smallestAxis.dot(centerDelta) < 0.0f) smallestAxis = smallestAxis * -1.0f;

    m.colliding = true;
    m.normal = smallestAxis;
    m.penetration = minOverlap;
    return m;
}

// ---------------------------------------------------------
// Narrow-phase: circle vs polygon.
// `poly` is treated as body A, `circle` as body B: the returned
// normal points from the polygon toward the circle.
// ---------------------------------------------------------
Manifold collideCirclePolygon(const Body& poly, const Body& circle) {
    Manifold m;
    std::vector<Vec2> verts = poly.worldVertices();
    std::vector<Vec2> normals = polygonWorldNormals(verts);
    size_t n = verts.size();

    float bestSeparation = -std::numeric_limits<float>::max();
    size_t bestEdge = 0;
    for (size_t i = 0; i < n; ++i) {
        float separation = normals[i].dot(circle.position - verts[i]);
        if (separation > circle.radius) return m; // separating axis found
        if (separation > bestSeparation) { bestSeparation = separation; bestEdge = i; }
    }

    Vec2 v1 = verts[bestEdge];
    Vec2 v2 = verts[(bestEdge + 1) % n];

    if (bestSeparation < 1e-6f) {
        // Circle's center is inside the polygon
        m.colliding = true;
        m.normal = normals[bestEdge];
        m.penetration = circle.radius - bestSeparation;
        return m;
    }

    // Circle center is outside: figure out if it's closest to v1, v2, or the edge face
    float d1 = (circle.position - v1).dot(v2 - v1);
    float d2 = (circle.position - v2).dot(v1 - v2);

    if (d1 <= 0.0f) {
        float dist = (circle.position - v1).length();
        if (dist > circle.radius) return m;
        m.colliding = true;
        m.normal = (circle.position - v1).normalized();
        m.penetration = circle.radius - dist;
    } else if (d2 <= 0.0f) {
        float dist = (circle.position - v2).length();
        if (dist > circle.radius) return m;
        m.colliding = true;
        m.normal = (circle.position - v2).normalized();
        m.penetration = circle.radius - dist;
    } else {
        float dist = bestSeparation;
        if (dist > circle.radius) return m;
        m.colliding = true;
        m.normal = normals[bestEdge];
        m.penetration = circle.radius - dist;
    }
    return m;
}

// Dispatches to the right narrow-phase test based on each body's shape.
// Always returns a manifold whose normal points from `a` toward `b`.
Manifold collide(const Body& a, const Body& b) {
    // Cheap broad-phase: if the bounding circles don't touch, the shapes can't either.
    Vec2 d = b.position - a.position;
    float reach = a.radius + b.radius;
    if (d.dot(d) > reach * reach) return Manifold{};

    if (a.shape == ShapeType::CIRCLE && b.shape == ShapeType::CIRCLE) {
        return collideCircleCircle(a, b);
    }
    if (a.shape == ShapeType::POLYGON && b.shape == ShapeType::POLYGON) {
        return collidePolygonPolygon(a, b);
    }
    if (a.shape == ShapeType::POLYGON && b.shape == ShapeType::CIRCLE) {
        return collideCirclePolygon(a, b); // already poly(a) -> circle(b)
    }
    // a = CIRCLE, b = POLYGON
    Manifold m = collideCirclePolygon(b, a); // poly(b) -> circle(a)
    m.normal = m.normal * -1.0f;             // flip to a -> b
    return m;
}

// ---------------------------------------------------------
// Step 1: Integrate forces (gravity) -> velocity -> position
// Each body can scale gravity with gravityScale (used by the sandbox editor).
// ---------------------------------------------------------
void integrate(std::vector<Body>& bodies, const SimParams& params, float dt) {
    for (auto& b : bodies) {
        if (b.invMass == 0.0f) continue; // static body, skip
        b.velocity.y += params.gravity * b.effectiveGravityScale() * dt;
        if (params.airDrag > 0.0f) b.velocity = b.velocity * std::max(0.0f, 1.0f - params.airDrag * dt);
        b.position += b.velocity * dt;
    }
}

// ---------------------------------------------------------
// Step 2: Resolve collisions against a rectangular play area.
// `bottom` is the floor (the ground), `top` the ceiling, etc.
// ---------------------------------------------------------
struct Bounds {
    float left, top, right, bottom;
};

void resolveWallCollisions(std::vector<Body>& bodies, const Bounds& bd, const SimParams& params) {
    for (auto& b : bodies) {
        // Below this speed a floor hit just stops the body, which prevents
        // endless micro-bouncing while something is resting on the ground.
        float restSpeed = params.gravity * b.effectiveGravityScale() * (2.0f / 60.0f);

        // Bounciness / friction against the ground and walls
        float wallE = params.useBodyMaterials ? std::min(b.restitution, params.groundRestitution) : params.restitution;
        float groundMu = params.useBodyMaterials ? std::sqrt(b.friction * params.groundFriction) : 0.0f;

        // Axis-aligned extents of the body (exact for circles, bounding box for polygons)
        float minX, maxX, minY, maxY;
        if (b.shape == ShapeType::CIRCLE) {
            minX = b.position.x - b.radius; maxX = b.position.x + b.radius;
            minY = b.position.y - b.radius; maxY = b.position.y + b.radius;
        } else {
            std::vector<Vec2> verts = b.worldVertices();
            minX = minY = std::numeric_limits<float>::max();
            maxX = maxY = -std::numeric_limits<float>::max();
            for (const auto& v : verts) {
                minX = std::min(minX, v.x); maxX = std::max(maxX, v.x);
                minY = std::min(minY, v.y); maxY = std::max(maxY, v.y);
            }
        }

        // Left / right
        if (minX < bd.left) {
            b.position.x += bd.left - minX;
            if (b.velocity.x < 0.0f) b.velocity.x = -b.velocity.x * wallE;
        } else if (maxX > bd.right) {
            b.position.x -= maxX - bd.right;
            if (b.velocity.x > 0.0f) b.velocity.x = -b.velocity.x * wallE;
        }

        // Ceiling
        if (minY < bd.top) {
            b.position.y += bd.top - minY;
            if (b.velocity.y < 0.0f) b.velocity.y = -b.velocity.y * wallE;
        }
        // Floor (the ground)
        else if (maxY > bd.bottom) {
            b.position.y -= maxY - bd.bottom;
            if (b.velocity.y > 0.0f) {
                float vn = b.velocity.y;
                bool resting = vn < restSpeed;
                float e = resting ? 0.0f : wallE;

                // Ground friction: slows sideways sliding, limited by how hard the body presses down
                if (groundMu > 0.0f) {
                    float maxDv = groundMu * (1.0f + e) * vn;
                    if (std::fabs(b.velocity.x) <= maxDv) b.velocity.x = 0.0f;
                    else b.velocity.x -= (b.velocity.x > 0.0f ? maxDv : -maxDv);
                }
                b.velocity.y = resting ? 0.0f : -vn * e;
            }
        }
    }
}

// ---------------------------------------------------------
// Step 3: Narrow-phase + resolution for every body pair.
// (Naive O(n^2) check -- fine for a few hundred bodies;
//  swap in a spatial grid later if you need more.)
// ---------------------------------------------------------
void resolveCollisions(std::vector<Body>& bodies, const SimParams& params) {
    for (size_t i = 0; i < bodies.size(); ++i) {
        for (size_t j = i + 1; j < bodies.size(); ++j) {
            Body& a = bodies[i];
            Body& b = bodies[j];

            Manifold m = collide(a, b);
            if (!m.colliding) continue;

            float totalInvMass = a.invMass + b.invMass;
            if (totalInvMass <= 0.0f) continue; // both immovable

            // --- Positional correction (stop sinking into each other) ---
            const float percent = 0.8f;
            Vec2 correction = m.normal * (m.penetration / totalInvMass * percent);
            a.position -= correction * a.invMass;
            b.position += correction * b.invMass;

            // --- Impulse resolution ---
            Vec2 relativeVelocity = b.velocity - a.velocity;
            float velAlongNormal = relativeVelocity.dot(m.normal);
            if (velAlongNormal > 0) continue; // already separating

            // Very slow impacts don't bounce (resting contact) - otherwise stacked
            // shapes jitter forever, because gravity re-creates a tiny impact every step.
            // With per-body materials the less bouncy of the two wins.
            float baseE = params.useBodyMaterials ? std::min(a.restitution, b.restitution) : params.restitution;
            float restVel = std::max(30.0f, params.gravity * (2.0f / 60.0f));
            float e = (-velAlongNormal < restVel) ? 0.0f : baseE;

            float impulseMag = -(1.0f + e) * velAlongNormal / totalInvMass;
            Vec2 impulse = m.normal * impulseMag;

            a.velocity -= impulse * a.invMass;
            b.velocity += impulse * b.invMass;

            // --- Friction: resists sliding along the contact, limited by the normal impulse ---
            if (params.useBodyMaterials) {
                float mu = std::sqrt(a.friction * b.friction);
                if (mu > 0.0f) {
                    Vec2 tangent{ -m.normal.y, m.normal.x };
                    float vt = (b.velocity - a.velocity).dot(tangent);
                    float jt = std::clamp(-vt / totalInvMass, -mu * impulseMag, mu * impulseMag);
                    a.velocity -= tangent * (jt * a.invMass);
                    b.velocity += tangent * (jt * b.invMass);
                }
            }
        }
    }
}

// ---------------------------------------------------------
// One full physics step: move everything, then run several solver passes.
// A single pass can't hold a stack of shapes apart (they sink into each other
// and jitter), so collisions + walls are resolved a few times per step.
// ---------------------------------------------------------
constexpr int SOLVER_ITERATIONS = 6;

void stepWorld(std::vector<Body>& bodies, const Bounds& bd, const SimParams& params, float dt) {
    integrate(bodies, params, dt);
    for (int i = 0; i < SOLVER_ITERATIONS; ++i) {
        resolveCollisions(bodies, params);
        resolveWallCollisions(bodies, bd, params);
    }
}

// =========================================================
// Theme (dark mode)
// =========================================================
namespace Theme {
constexpr Color BG         = { 17, 18, 23, 255 };    // window background
constexpr Color WORLD      = { 24, 26, 34, 255 };    // editor play area
constexpr Color PANEL      = { 32, 34, 44, 255 };    // bars and side panels
constexpr Color CELL       = { 44, 47, 60, 255 };    // palette + asset cells
constexpr Color BORDER     = { 72, 76, 94, 255 };
constexpr Color TEXT       = { 228, 231, 240, 255 };
constexpr Color TEXT_DIM   = { 140, 146, 162, 255 };
constexpr Color ACCENT     = { 90, 160, 255, 255 };
constexpr Color SELECT     = { 255, 205, 70, 255 };  // selection highlight
constexpr Color GROUND     = { 68, 54, 42, 255 };
constexpr Color GROUND_TOP = { 120, 96, 72, 255 };
}

// Restyles every raygui control (sliders, buttons, text boxes, ...) for dark mode.
void applyDarkTheme() {
    auto rgb = [](Color c) { return ColorToInt(c); };
    GuiSetStyle(DEFAULT, BACKGROUND_COLOR, rgb(Theme::PANEL));
    GuiSetStyle(DEFAULT, LINE_COLOR, rgb(Theme::BORDER));

    GuiSetStyle(DEFAULT, BORDER_COLOR_NORMAL, rgb(Theme::BORDER));
    GuiSetStyle(DEFAULT, BASE_COLOR_NORMAL, rgb(Theme::CELL));
    GuiSetStyle(DEFAULT, TEXT_COLOR_NORMAL, rgb(Theme::TEXT));

    GuiSetStyle(DEFAULT, BORDER_COLOR_FOCUSED, rgb(Theme::ACCENT));
    GuiSetStyle(DEFAULT, BASE_COLOR_FOCUSED, rgb(Color{ 58, 64, 84, 255 }));
    GuiSetStyle(DEFAULT, TEXT_COLOR_FOCUSED, rgb(WHITE));

    GuiSetStyle(DEFAULT, BORDER_COLOR_PRESSED, rgb(Theme::ACCENT));
    GuiSetStyle(DEFAULT, BASE_COLOR_PRESSED, rgb(Color{ 54, 104, 184, 255 }));
    GuiSetStyle(DEFAULT, TEXT_COLOR_PRESSED, rgb(WHITE));

    GuiSetStyle(DEFAULT, BORDER_COLOR_DISABLED, rgb(Color{ 52, 55, 68, 255 }));
    GuiSetStyle(DEFAULT, BASE_COLOR_DISABLED, rgb(Color{ 36, 38, 48, 255 }));
    GuiSetStyle(DEFAULT, TEXT_COLOR_DISABLED, rgb(Color{ 90, 95, 110, 255 }));
}

// =========================================================
// Assets: images loaded from the user's computer
// =========================================================
const char* ASSETS_TITLE = "Assests";   // label of the image area (change the spelling here if you like)

struct Asset {
    std::string name;
    std::string path;
    Texture2D tex;
};

std::vector<Asset> g_assets;
std::string g_assetMsg = "No images yet";
bool g_textEditing = false;   // true while a text box is being typed into (so ESC/Delete don't fire)

// Loads an image file into the asset library. Returns true on success.
bool loadAsset(const std::string& rawPath) {
    // Trim spaces and the quotes Windows adds with "Copy as path"
    std::string path = rawPath;
    while (!path.empty() && (path.front() == ' ' || path.front() == '"' || path.front() == '\'')) path.erase(path.begin());
    while (!path.empty() && (path.back() == ' ' || path.back() == '"' || path.back() == '\'' ||
                             path.back() == '\n' || path.back() == '\r')) path.pop_back();

    if (path.empty()) { g_assetMsg = "Type or paste a file path first"; return false; }
    if (!FileExists(path.c_str())) { g_assetMsg = "File not found"; return false; }
    if (!IsFileExtension(path.c_str(), ".png;.jpg;.jpeg;.bmp;.tga;.gif;.qoi")) {
        g_assetMsg = "Not a supported image (png, jpg, bmp, tga, gif, qoi)";
        return false;
    }
    for (const auto& a : g_assets) {
        if (a.path == path) { g_assetMsg = "Already loaded: " + a.name; return false; }
    }

    Image img = LoadImage(path.c_str());
    if (img.data == nullptr) { g_assetMsg = "Couldn't read that image"; return false; }

    // Cap huge images so they don't eat GPU memory
    const int MAX_DIM = 2048;
    if (img.width > MAX_DIM || img.height > MAX_DIM) {
        float s = (float)MAX_DIM / (float)std::max(img.width, img.height);
        ImageResize(&img, std::max(1, (int)(img.width * s)), std::max(1, (int)(img.height * s)));
    }

    Texture2D tex = LoadTextureFromImage(img);
    UnloadImage(img);
    if (tex.id == 0) { g_assetMsg = "Couldn't create a texture from that image"; return false; }

    GenTextureMipmaps(&tex);
    SetTextureFilter(tex, TEXTURE_FILTER_TRILINEAR); // smooth when scaled down

    g_assets.push_back({ GetFileName(path.c_str()), path, tex });
    g_assetMsg = TextFormat("Loaded %s (%dx%d)", g_assets.back().name.c_str(), tex.width, tex.height);
    return true;
}

// Loads several image files at once (from the file dialog or a drag-and-drop)
int loadAssets(const std::vector<std::string>& paths) {
    int ok = 0;
    for (const auto& p : paths) {
        if (loadAsset(p)) ++ok;
    }
    if (paths.size() > 1) g_assetMsg = TextFormat("Loaded %d of %d files", ok, (int)paths.size());
    return ok;
}

// ---------------------------------------------------------
// Native "open file" dialog: browse your computer for images.
// raylib has no file picker, so this asks the operating system for one.
//   Windows: the standard Open dialog (comdlg32, loaded at runtime - no extra linking)
//   macOS:   osascript's "choose file"
//   Linux:   zenity, or kdialog as a fallback
// Returns the chosen files (empty if cancelled). `note` explains an empty result.
// ---------------------------------------------------------
#ifdef _WIN32
// Win32 pieces are declared by hand because <windows.h> clashes with raylib's names
// (Rectangle, CloseWindow, DrawText, ...). Layout matches the real OPENFILENAMEW.
namespace winapi {
struct OpenFileNameW {
    unsigned long lStructSize;
    void* hwndOwner;
    void* hInstance;
    const wchar_t* lpstrFilter;
    wchar_t* lpstrCustomFilter;
    unsigned long nMaxCustFilter;
    unsigned long nFilterIndex;
    wchar_t* lpstrFile;
    unsigned long nMaxFile;
    wchar_t* lpstrFileTitle;
    unsigned long nMaxFileTitle;
    const wchar_t* lpstrInitialDir;
    const wchar_t* lpstrTitle;
    unsigned long Flags;
    unsigned short nFileOffset;
    unsigned short nFileExtension;
    const wchar_t* lpstrDefExt;
    intptr_t lCustData;
    void* lpfnHook;
    const wchar_t* lpTemplateName;
    void* pvReserved;
    unsigned long dwReserved;
    unsigned long FlagsEx;
};
extern "C" {
__declspec(dllimport) void* __stdcall LoadLibraryA(const char* name);
__declspec(dllimport) void* __stdcall GetProcAddress(void* module, const char* name);
__declspec(dllimport) int   __stdcall FreeLibrary(void* module);
__declspec(dllimport) void* __stdcall GetActiveWindow(void);
__declspec(dllimport) int   __stdcall WideCharToMultiByte(unsigned int codePage, unsigned long flags,
                                                          const wchar_t* wide, int wideLen,
                                                          char* out, int outLen,
                                                          const char* defaultChar, int* usedDefault);
}
} // namespace winapi

static std::string wideToUtf8(const wchar_t* w) {
    int n = winapi::WideCharToMultiByte(65001 /* CP_UTF8 */, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) return std::string();
    std::string s((size_t)n - 1, '\0');
    winapi::WideCharToMultiByte(65001, 0, w, -1, &s[0], n, nullptr, nullptr);
    return s;
}

std::vector<std::string> openImageFilesDialog(std::string& note) {
    std::vector<std::string> result;

    void* lib = winapi::LoadLibraryA("comdlg32.dll");
    if (!lib) { note = "Couldn't open the Windows file dialog"; return result; }
    using GetOpenFileNameWFn = int(__stdcall*)(winapi::OpenFileNameW*);
    GetOpenFileNameWFn getOpenFileNameW =
        reinterpret_cast<GetOpenFileNameWFn>(winapi::GetProcAddress(lib, "GetOpenFileNameW"));
    if (!getOpenFileNameW) { winapi::FreeLibrary(lib); note = "Couldn't open the Windows file dialog"; return result; }

    std::vector<wchar_t> buf(65536, L'\0'); // room for lots of selected files

    winapi::OpenFileNameW ofn = {};
    ofn.lStructSize = (unsigned long)sizeof(ofn);
    ofn.hwndOwner = winapi::GetActiveWindow();
    ofn.lpstrFilter = L"Images (png, jpg, bmp, tga, gif, qoi)\0*.png;*.jpg;*.jpeg;*.bmp;*.tga;*.gif;*.qoi\0"
                      L"All files (*.*)\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrFile = buf.data();
    ofn.nMaxFile = (unsigned long)buf.size();
    ofn.lpstrTitle = L"Add images to Assests";
    ofn.Flags = 0x00080000   // OFN_EXPLORER
              | 0x00000200   // OFN_ALLOWMULTISELECT
              | 0x00001000   // OFN_FILEMUSTEXIST
              | 0x00000800   // OFN_PATHMUSTEXIST
              | 0x00000008   // OFN_NOCHANGEDIR (don't change the program's working folder)
              | 0x00000004;  // OFN_HIDEREADONLY

    if (getOpenFileNameW(&ofn)) {
        // One file:  "C:\dir\a.png\0\0"      Several: "C:\dir\0a.png\0b.png\0\0"
        const wchar_t* p = buf.data();
        std::string first = wideToUtf8(p);
        p += std::wcslen(p) + 1;
        if (*p == L'\0') {
            result.push_back(first);
        } else {
            std::string dir = first;
            if (!dir.empty() && dir.back() != '\\' && dir.back() != '/') dir += '\\';
            while (*p) {
                result.push_back(dir + wideToUtf8(p));
                p += std::wcslen(p) + 1;
            }
        }
    }
    winapi::FreeLibrary(lib);
    if (result.empty()) note = "No file chosen";
    return result;
}

#else // macOS / Linux

static bool commandExists(const char* name) {
    std::string cmd = std::string("command -v ") + name + " >/dev/null 2>&1";
    return std::system(cmd.c_str()) == 0;
}

static std::string runCommand(const std::string& cmd) {
    std::string out;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return out;
    char chunk[4096];
    while (std::fgets(chunk, sizeof(chunk), pipe)) out += chunk;
    pclose(pipe);
    return out;
}

std::vector<std::string> openImageFilesDialog(std::string& note) {
    std::string out;
#if defined(__APPLE__)
    out = runCommand(
        "osascript -e 'set f to choose file with prompt \"Add images to Assests\" of type {\"public.image\"} "
        "with multiple selections allowed' -e 'set out to \"\"' -e 'repeat with i in f' "
        "-e 'set out to out & POSIX path of i & linefeed' -e 'end repeat' -e 'out' 2>/dev/null");
#else
    if (commandExists("zenity")) {
        out = runCommand("zenity --file-selection --multiple --separator='\n' --title='Add images to Assests' "
                         "--file-filter='Images | *.png *.jpg *.jpeg *.bmp *.tga *.gif *.qoi' "
                         "--file-filter='All files | *' 2>/dev/null");
    } else if (commandExists("kdialog")) {
        out = runCommand("kdialog --getopenfilename . 'Images (*.png *.jpg *.jpeg *.bmp *.tga *.gif *.qoi)' "
                         "--multiple --separate-output 2>/dev/null");
    } else {
        note = "No file dialog found - install zenity (or kdialog), or drag files onto the window";
        return {};
    }
#endif
    std::vector<std::string> result;
    std::string line;
    for (char c : out + "\n") {
        if (c == '\n' || c == '\r') {
            if (!line.empty()) result.push_back(line);
            line.clear();
        } else {
            line += c;
        }
    }
    if (result.empty()) note = "No file chosen";
    return result;
}
#endif

// Files dragged from the file explorer onto the window
void handleDroppedFiles() {
    if (!IsFileDropped()) return;
    FilePathList files = LoadDroppedFiles();
    std::vector<std::string> paths;
    for (unsigned int i = 0; i < files.count; ++i) paths.push_back(files.paths[i]);
    UnloadDroppedFiles(files);
    loadAssets(paths);
}

// =========================================================
// Drawing, picking and shape-building helpers
// =========================================================

// Outline color that stays visible on both light and dark fills
Color outlineFor(Color fill, bool ghost) {
    float lum = 0.299f * fill.r + 0.587f * fill.g + 0.114f * fill.b;
    Color o = (lum < 90.0f) ? Color{ 255, 255, 255, 170 } : Color{ 10, 10, 14, 230 };
    return ghost ? Fade(o, 0.55f) : o;
}

// Draws a body. `selected` adds a highlight outline, `ghost` makes it translucent
// (used for the drag-and-drop preview).
void drawBody(const Body& b, bool selected = false, bool ghost = false) {
    Color fill = ghost ? Fade(b.color, 0.55f) : b.color;
    Color line = outlineFor(b.color, ghost);

    if (b.shape == ShapeType::CIRCLE) {
        Vector2 c = { b.position.x, b.position.y };
        if (selected) DrawRing(c, b.radius, b.radius + 4.0f, 0.0f, 360.0f, 48, Theme::SELECT);
        DrawCircleV(c, b.radius, fill);
        DrawCircleLines((int)b.position.x, (int)b.position.y, b.radius, line);
        return;
    }

    std::vector<Vec2> verts = b.worldVertices();
    size_t n = verts.size();

    // Image: the polygon is a rectangle, so draw the texture stretched over it.
    if (b.assetIndex >= 0 && b.assetIndex < (int)g_assets.size()) {
        float hw = 0.0f, hh = 0.0f;
        for (const auto& v : b.localVertices) {
            hw = std::max(hw, std::fabs(v.x));
            hh = std::max(hh, std::fabs(v.y));
        }
        const Texture2D& tex = g_assets[b.assetIndex].tex;
        Rectangle src = { 0.0f, 0.0f, (float)tex.width, (float)tex.height };
        Rectangle dst = { b.position.x, b.position.y, hw * 2.0f, hh * 2.0f };
        DrawTexturePro(tex, src, dst, Vector2{ hw, hh }, b.rotation * RAD2DEG, fill); // fill = tint
        if (selected) {
            for (size_t k = 0; k < n; ++k) {
                Vec2 p1 = verts[k], p2 = verts[(k + 1) % n];
                DrawLineEx({ p1.x, p1.y }, { p2.x, p2.y }, 3.0f, Theme::SELECT);
            }
        }
        return;
    }

    // Fill: one triangle per edge, fanned out from the center. raylib culls
    // triangles by winding order, so each triangle is drawn in both orders -
    // that way the fill shows up no matter which way the vertices were listed.
    Vector2 center = { b.position.x, b.position.y };
    for (size_t k = 0; k < n; ++k) {
        Vector2 p1 = { verts[k].x, verts[k].y };
        Vector2 p2 = { verts[(k + 1) % n].x, verts[(k + 1) % n].y };
        DrawTriangle(center, p1, p2, fill);
        DrawTriangle(center, p2, p1, fill);
    }

    if (selected) {
        for (size_t k = 0; k < n; ++k) {
            Vec2 p1 = verts[k], p2 = verts[(k + 1) % n];
            DrawLineEx({ p1.x, p1.y }, { p2.x, p2.y }, 6.0f, Theme::SELECT);
        }
    }
    for (size_t k = 0; k < n; ++k) {
        Vec2 p1 = verts[k], p2 = verts[(k + 1) % n];
        DrawLineEx({ p1.x, p1.y }, { p2.x, p2.y }, 2.0f, line);
    }
}

// Is point p inside the body? (used for mouse picking)
bool containsPoint(const Body& b, Vec2 p) {
    if (b.shape == ShapeType::CIRCLE) return (p - b.position).length() <= b.radius;

    std::vector<Vec2> verts = b.worldVertices();
    std::vector<Vec2> normals = polygonWorldNormals(verts);
    for (size_t i = 0; i < verts.size(); ++i) {
        if (normals[i].dot(p - verts[i]) > 0.0f) return false; // outside this edge
    }
    return true;
}

// shapeSel: 0=Circle, 1=Triangle, 2=Square, 3=Rectangle, 4=Polygon(sides)
// Polygons are rotated so they sit flat on their bottom edge.
Body makeShape(int shapeSel, Vec2 pos, float size, int sides, float rot, Color c) {
    switch (shapeSel) {
    case 0:
        return Body(pos, size, size * size * 0.05f, c);
    case 1:
        return Body(pos, makeRegularPolygon(3, size), size, rot, size * size * 0.05f, c);
    case 2:
        return Body(pos, makeRegularPolygon(4, size, PI / 4.0f), size, rot, size * size * 0.08f, c);
    case 3: {
        float w = size * 1.8f, h = size * 0.9f;
        float bounding = std::sqrt((w * 0.5f) * (w * 0.5f) + (h * 0.5f) * (h * 0.5f));
        return Body(pos, makeRectangle(w, h), bounding, rot, w * h * 0.002f, c);
    }
    default: {
        float phase = (sides % 2 == 0) ? PI / sides : 0.0f; // even N: flat edge on the bottom
        return Body(pos, makeRegularPolygon(sides, size, phase), size, rot, size * size * 0.05f, c);
    }
    }
}

// An image as a physics body: a rectangle (same aspect ratio as the picture) that shows the texture.
Body makeImageBody(int assetIndex, Vec2 pos) {
    const Texture2D& t = g_assets[assetIndex].tex;
    float scale = 140.0f / (float)std::max(t.width, t.height); // longest side starts at 140px
    float w = t.width * scale, h = t.height * scale;
    float bounding = std::sqrt((w * 0.5f) * (w * 0.5f) + (h * 0.5f) * (h * 0.5f));
    Body b(pos, makeRectangle(w, h), bounding, 0.0f, w * h * 0.002f, WHITE);
    b.assetIndex = assetIndex;
    return b;
}

Color randomBrightColor() {
    return Color{ (unsigned char)GetRandomValue(80, 255),
                  (unsigned char)GetRandomValue(80, 255),
                  (unsigned char)GetRandomValue(80, 255), 255 };
}

// Accepts "#RRGGBB" or "RRGGBB". Returns false if the text isn't a complete hex color yet.
bool parseHexColor(const char* s, Color& out) {
    if (*s == '#') ++s;
    if (std::strlen(s) != 6) return false;
    for (int i = 0; i < 6; ++i) {
        if (!std::isxdigit((unsigned char)s[i])) return false;
    }
    unsigned long v = std::strtoul(s, nullptr, 16);
    out = Color{ (unsigned char)((v >> 16) & 0xFF), (unsigned char)((v >> 8) & 0xFF),
                 (unsigned char)(v & 0xFF), 255 };
    return true;
}

const char* shapeName(const Body& b) {
    if (b.assetIndex >= 0 && b.assetIndex < (int)g_assets.size())
        return TextFormat("Image: %s", g_assets[b.assetIndex].name.c_str());
    if (b.shape == ShapeType::CIRCLE) return "Circle";
    return TextFormat("Polygon (%d sides)", (int)b.localVertices.size());
}

// =========================================================
// App state
// =========================================================
constexpr float PIXELS_PER_METER = 50.0f;
constexpr float EARTH_G = 9.8f;          // m/s^2
constexpr float FIXED_DT = 1.0f / 60.0f; // physics timestep
constexpr int   TOP_H = 50;              // editor: top bar height
constexpr int   LEFT_W = 250;            // editor: Add + Scene list (left)
constexpr int   INS_W = 300;             // editor: Properties panel (right)
constexpr int   ASSET_H = 150;           // editor: Assests shelf height (bottom)
constexpr int   PALETTE_COUNT = 5;

enum class AppMode { SPAWNER, EDITOR };

struct SpawnerState {
    std::vector<Body> bodies;
    SimParams params;
    int shapeSelection = 0;       // 0=Circle, 1=Triangle, 2=Square, 3=Rectangle, 4=Polygon
    float polygonSidesF = 5.0f;
    float sizeSlider = 20.0f;
    float rectWidthSlider = 60.0f;
    float rectHeightSlider = 30.0f;
    Color spawnColor = RED;
    bool randomColor = true;
};

struct HexField {
    char buf[16] = "#FFFFFF";
    bool edit = false;
    int owner = -99;          // what the box currently shows; editing restarts when this changes
};

struct Editor {
    std::vector<Body> bodies;
    SimParams params;         // gravity is driven by the Environment settings below
    bool paused = false;

    // Selection: index of the selected object, or -1 = the Environment is selected
    int selected = -1;
    int lastSelected = -99;
    int dragging = -1;        // index of the body being dragged, -1 = none
    Vec2 dragOffset{ 0, 0 };
    Vec2 dragVelocity{ 0, 0 };    // smoothed mouse velocity, applied when you let go (throw)

    int paletteDrag = -1;         // "Add" item being dragged into the world, -1 = none
    Color paletteColor = SKYBLUE;
    float paletteSides = 5.0f;

    int assetDrag = -1;           // asset being dragged into the world, -1 = none
    float assetScroll = 0.0f;
    char pathBuf[512] = "";
    bool pathEdit = false;
    bool browseRequested = false; // set by the Browse button, handled after the frame is drawn

    int nextObjectId = 1;         // used for default names like "Circle 3"

    // ---- Environment properties ----
    float gravityMs2 = EARTH_G;   // world gravity in m/s^2 (objects opt in to it)
    float timeScale = 1.0f;       // slow motion / fast forward
    float groundOffset = 110.0f;  // how far the ground surface sits above the bottom of the world
    Color groundColor = Theme::GROUND;
    Color backgroundColor = Theme::WORLD;
    bool showGrid = true;
    int envColorTarget = 0;       // which color the Environment color editor shows: 0 ground, 1 background
    bool draggingGround = false;

    // ---- Panel state ----
    float listScroll = 0.0f;
    float inspectorScroll = 0.0f;
    float inspectorContentH = 0.0f;
    bool objSection[3] = { true, true, true };
    bool envSection[4] = { true, true, true, true };
    HexField hex;
    char nameBuf[64] = "";
    bool nameEdit = false;
    int nameOwner = -99;
};

// "Add" grid on the left: 3 cells per row
Rectangle paletteCell(int i) {
    return { 12.0f + (i % 3) * 78.0f, (float)(TOP_H + 36) + (i / 3) * 78.0f, 70.0f, 70.0f };
}

Rectangle assetCell(int i, const Rectangle& shelf, float scroll) {
    return { shelf.x + 10.0f + i * 110.0f - scroll, shelf.y + 36.0f, 100.0f, (float)ASSET_H - 46.0f };
}

// =========================================================
// Spawner mode (click-to-spawn playground)
// =========================================================
void spawnerFrame(SpawnerState& s, float& accumulator, AppMode& mode) {
    const int W = GetScreenWidth();
    const int H = GetScreenHeight();
    const float PANEL_HEIGHT = 175.0f;
    g_textEditing = false;

    // Click to spawn the selected shape
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && GetMouseY() > PANEL_HEIGHT) {
        Vec2 spawnPos{ (float)GetMouseX(), (float)GetMouseY() };
        Color c = s.randomColor ? randomBrightColor() : s.spawnColor;
        float randRot = (float)GetRandomValue(0, 628) / 100.0f; // cosmetic spawn rotation, 0-2pi

        if (s.shapeSelection == 0) {
            float r = s.sizeSlider;
            s.bodies.emplace_back(spawnPos, r, r * r * 0.05f, c);
        } else if (s.shapeSelection == 1) {
            float r = s.sizeSlider;
            s.bodies.emplace_back(spawnPos, makeRegularPolygon(3, r), r, randRot, r * r * 0.05f, c);
        } else if (s.shapeSelection == 2) {
            float r = s.sizeSlider;
            s.bodies.emplace_back(spawnPos, makeRegularPolygon(4, r), r, randRot, r * r * 0.08f, c);
        } else if (s.shapeSelection == 3) {
            float w = s.rectWidthSlider, h = s.rectHeightSlider;
            float bounding = std::sqrt((w * 0.5f) * (w * 0.5f) + (h * 0.5f) * (h * 0.5f));
            s.bodies.emplace_back(spawnPos, makeRectangle(w, h), bounding, randRot, w * h * 0.002f, c);
        } else {
            float r = s.sizeSlider;
            s.bodies.emplace_back(spawnPos, makeRegularPolygon((int)s.polygonSidesF, r), r, randRot, r * r * 0.05f, c);
        }
    }

    // Fixed-timestep physics
    if (!s.params.paused) {
        accumulator = std::min(accumulator + GetFrameTime(), 0.25f);
        Bounds bounds{ 0.0f, 0.0f, (float)W, (float)H };
        while (accumulator >= FIXED_DT) {
            stepWorld(s.bodies, bounds, s.params, FIXED_DT);
            accumulator -= FIXED_DT;
        }
    }

    // Render
    BeginDrawing();
    ClearBackground(Theme::BG);

    for (const auto& b : s.bodies) drawBody(b);

    DrawRectangle(0, 0, W, (int)PANEL_HEIGHT, Fade(Theme::PANEL, 0.96f));
    DrawLine(0, (int)PANEL_HEIGHT, W, (int)PANEL_HEIGHT, Theme::BORDER);

    GuiSlider({ 120, 15, 200, 20 }, "Gravity", TextFormat("%.2f m/s^2", s.params.gravity / PIXELS_PER_METER),
              &s.params.gravity, 0.0f, 2000.0f);
    GuiCheckBox({ 400, 15, 20, 20 }, "Paused", &s.params.paused);

    GuiSlider({ 120, 45, 200, 20 }, "Bounciness", TextFormat("%.2f", s.params.restitution),
              &s.params.restitution, 0.0f, 1.0f);
    if (GuiButton({ 350, 45, 100, 25 }, "Clear all")) s.bodies.clear();

    DrawText("Shape:", 20, 85, 16, Theme::TEXT);
    GuiToggleGroup({ 80, 80, 70, 25 }, "Circle;Triangle;Square;Rectangle;Polygon", &s.shapeSelection);
    if (s.shapeSelection == 4) {
        GuiSlider({ 440, 80, 100, 25 }, "Sides", TextFormat("%d", (int)s.polygonSidesF),
                  &s.polygonSidesF, 3.0f, 12.0f);
    }

    if (s.shapeSelection == 3) {
        GuiSlider({ 120, 115, 140, 20 }, "Width", TextFormat("%.0f", s.rectWidthSlider),
                  &s.rectWidthSlider, 20.0f, 150.0f);
        GuiSlider({ 300, 115, 140, 20 }, "Height", TextFormat("%.0f", s.rectHeightSlider),
                  &s.rectHeightSlider, 10.0f, 100.0f);
    } else {
        GuiSlider({ 120, 115, 200, 20 }, "Size", TextFormat("%.0f", s.sizeSlider),
                  &s.sizeSlider, 8.0f, 60.0f);
    }

    DrawText(TextFormat("Bodies: %d", (int)s.bodies.size()), 20, 145, 18, Theme::TEXT);
    DrawText("Click below to spawn", 160, 147, 16, Theme::TEXT_DIM);

    GuiColorPicker({ 600, 15, 100, 100 }, "Spawn Color", &s.spawnColor);
    GuiCheckBox({ 600, 125, 20, 20 }, "Random Color", &s.randomColor);

    if (GuiButton({ (float)W - 210, 15, 190, 30 }, "Open Sandbox Editor")) {
        mode = AppMode::EDITOR;
        accumulator = 0.0f;
        SetMouseCursor(MOUSE_CURSOR_DEFAULT);
    }
    DrawText("F11: fullscreen", W - 130, 55, 14, Theme::TEXT_DIM);

    EndDrawing();
}

// =========================================================
// Sandbox editor
//   Left:   "Add" (drag shapes in) + "Scene" (list of everything in the world)
//   Middle: the work area, with the Assests shelf underneath
//   Right:  Properties of whatever is selected (an object, or the Environment)
// =========================================================

constexpr float LIST_ROW_H = 26.0f;
constexpr float UI_LABEL_W = 96.0f;
constexpr float UI_ROW_H = 26.0f;
static const char* PALETTE_NAMES[PALETTE_COUNT] = { "Circle", "Triangle", "Square", "Rectangle", "Polygon" };

float luminance(Color c) { return 0.299f * c.r + 0.587f * c.g + 0.114f * c.b; }

// Text color that is readable on top of `bg`
Color textOn(Color bg, float alpha) { return luminance(bg) < 140.0f ? Fade(WHITE, alpha) : Fade(BLACK, alpha); }

// A slightly lighter (dark colors) or darker (light colors) version of c, for edges and grid lines
Color edgeOf(Color c) {
    float t = 0.25f;
    if (luminance(c) < 128.0f) return Color{ (unsigned char)(c.r + (255 - c.r) * t), (unsigned char)(c.g + (255 - c.g) * t),
                                              (unsigned char)(c.b + (255 - c.b) * t), 255 };
    return Color{ (unsigned char)(c.r * (1 - t)), (unsigned char)(c.g * (1 - t)), (unsigned char)(c.b * (1 - t)), 255 };
}

// ---- Adding / copying objects ----
void addObject(Editor& ed, Body b, const std::string& base) {
    b.gravityEnabled = false;                          // gravity stays off until enabled
    b.name = base + " " + std::to_string(ed.nextObjectId++);
    ed.bodies.push_back(b);
    ed.selected = (int)ed.bodies.size() - 1;
}

void duplicateObject(Editor& ed, int index) {
    Body c = ed.bodies[index];
    c.position += Vec2{ 30.0f, 30.0f };
    c.velocity = Vec2{ 0, 0 };
    c.name += " copy";
    ed.bodies.push_back(c);
    ed.selected = (int)ed.bodies.size() - 1;
}

// ---- Tiny immediate-mode UI helpers for the Properties panel ----
// The panel scrolls, so interactive widgets are only drawn when fully inside the visible area.
struct Ui {
    float x, w;          // content column
    float top, bottom;   // visible range
    float y;             // current row
    Vector2 mouse;
    bool vis(float h) const { return y >= top && y + h <= bottom; }
};

// Collapsible section header. Returns whether the section is open.
bool uiSection(Ui& ui, const char* title, bool& open) {
    Rectangle r = { ui.x - 8, ui.y, ui.w + 16, 24 };
    bool hovered = CheckCollisionPointRec(ui.mouse, r) && ui.mouse.y >= ui.top && ui.mouse.y <= ui.bottom;
    DrawRectangleRec(r, hovered ? Color{ 58, 64, 84, 255 } : Theme::CELL);
    DrawText(open ? "-" : "+", (int)r.x + 9, (int)r.y + 4, 16, Theme::TEXT_DIM);
    DrawText(title, (int)r.x + 28, (int)r.y + 5, 14, Theme::TEXT);
    if (hovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) open = !open;
    ui.y += 30;
    return open;
}

void uiLabel(Ui& ui, const char* text, Color col = Theme::TEXT_DIM, int size = 12) {
    DrawText(text, (int)ui.x, (int)ui.y, size, col);
    ui.y += size + 6;
}

// Labelled slider. Returns true when the user changed the value.
bool uiSlider(Ui& ui, const char* label, float& value, float lo, float hi, const char* fmt, bool enabled = true) {
    bool changed = false;
    Color tc = enabled ? Theme::TEXT : Theme::TEXT_DIM;
    DrawText(label, (int)ui.x, (int)ui.y + 3, 14, tc);
    float barX = ui.x + UI_LABEL_W;
    float barW = ui.w - UI_LABEL_W - 64.0f;
    if (ui.vis(20)) {
        float v = value;
        if (!enabled) GuiDisable();
        GuiSlider({ barX, ui.y, barW, 20 }, nullptr, nullptr, &v, lo, hi);
        if (!enabled) GuiEnable();
        if (enabled && std::fabs(v - value) > 1e-4f) { value = v; changed = true; }
    }
    DrawText(TextFormat(fmt, value), (int)(barX + barW + 8), (int)ui.y + 3, 14, tc);
    ui.y += UI_ROW_H;
    return changed;
}

bool uiCheck(Ui& ui, const char* label, bool& value, bool enabled = true) {
    bool changed = false;
    if (ui.vis(20)) {
        bool v = value;
        if (!enabled) GuiDisable();
        GuiCheckBox({ ui.x, ui.y, 20, 20 }, nullptr, &v);
        if (!enabled) GuiEnable();
        if (enabled && v != value) { value = v; changed = true; }
    }
    DrawText(label, (int)ui.x + 28, (int)ui.y + 3, 14, enabled ? Theme::TEXT : Theme::TEXT_DIM);
    ui.y += UI_ROW_H;
    return changed;
}

// One button (full width) or two side by side. Returns 0 (none), 1 (first) or 2 (second).
int uiButtons(Ui& ui, const char* a, const char* b = nullptr) {
    int result = 0;
    if (ui.vis(28)) {
        if (b) {
            float bw = (ui.w - 8.0f) / 2.0f;
            if (GuiButton({ ui.x, ui.y, bw, 28 }, a)) result = 1;
            if (GuiButton({ ui.x + bw + 8.0f, ui.y, bw, 28 }, b)) result = 2;
        } else if (GuiButton({ ui.x, ui.y, ui.w, 28 }, a)) {
            result = 1;
        }
    }
    ui.y += 34;
    return result;
}

void uiTextRow(Ui& ui, const char* label, char* buf, int bufSize, bool& edit) {
    DrawText(label, (int)ui.x, (int)ui.y + 6, 14, Theme::TEXT);
    if (ui.vis(26)) {
        if (GuiTextBox({ ui.x + UI_LABEL_W, ui.y, ui.w - UI_LABEL_W, 26 }, buf, bufSize, edit)) edit = !edit;
    } else {
        edit = false; // scrolled out of view: stop typing into it
    }
    ui.y += 32;
}

// Hex box + swatch + color picker for one color. Returns true when the color changed.
bool uiColor(Ui& ui, const char* label, Color& color, HexField& hf, int ownerKey) {
    bool changed = false;
    if (hf.owner != ownerKey) { hf.owner = ownerKey; hf.edit = false; }
    if (!hf.edit) std::snprintf(hf.buf, sizeof(hf.buf), "#%02X%02X%02X", color.r, color.g, color.b);

    DrawText(label, (int)ui.x, (int)ui.y + 6, 14, Theme::TEXT);
    if (ui.vis(26)) {
        if (GuiTextBox({ ui.x + UI_LABEL_W, ui.y, 96, 26 }, hf.buf, 8, hf.edit)) hf.edit = !hf.edit;
        if (hf.edit) {
            Color parsed;
            if (parseHexColor(hf.buf, parsed) && (parsed.r != color.r || parsed.g != color.g || parsed.b != color.b)) {
                color = parsed;
                changed = true;
            }
        }
    } else {
        hf.edit = false;
    }
    DrawRectangle((int)(ui.x + UI_LABEL_W + 104), (int)ui.y, 26, 26, color);
    DrawRectangleLines((int)(ui.x + UI_LABEL_W + 104), (int)ui.y, 26, 26, Theme::BORDER);
    ui.y += 32;

    // Picker: only adopt its value while the user is actually dragging inside it
    Rectangle area = { ui.x - 4, ui.y - 2, ui.w + 8, 156 };
    if (ui.vis(150)) {
        Color picked = color;
        GuiColorPicker({ ui.x + 4, ui.y, 150, 150 }, "", &picked);
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && CheckCollisionPointRec(ui.mouse, area) &&
            (picked.r != color.r || picked.g != color.g || picked.b != color.b)) {
            color = Color{ picked.r, picked.g, picked.b, 255 };
            changed = true;
        }
    }
    ui.y += 158;
    return changed;
}

// ---- Properties: the selected object ----
void objectInspector(Editor& ed, Ui& ui, const Bounds& bd, bool& deleteSelected, bool& duplicateSelected) {
    Body& s = ed.bodies[ed.selected];
    bool isCircle = s.shape == ShapeType::CIRCLE;
    bool isImage = s.assetIndex >= 0;

    // Name
    if (ed.nameOwner != ed.selected) { ed.nameOwner = ed.selected; ed.nameEdit = false; }
    if (!ed.nameEdit) std::snprintf(ed.nameBuf, sizeof(ed.nameBuf), "%s", s.name.c_str());
    uiTextRow(ui, "Name", ed.nameBuf, 40, ed.nameEdit);
    if (ed.nameEdit) s.name = ed.nameBuf;
    uiLabel(ui, shapeName(s));
    int act = uiButtons(ui, "Duplicate (Ctrl+D)", "Delete (Del)");
    if (act == 1) duplicateSelected = true;
    if (act == 2) deleteSelected = true;

    if (uiSection(ui, "Transform", ed.objSection[0])) {
        float px = s.position.x;
        if (uiSlider(ui, "Position X", px, bd.left, bd.right, "%.0f")) s.position.x = px;
        float py = s.position.y;
        if (uiSlider(ui, "Position Y", py, bd.top, bd.bottom, "%.0f")) s.position.y = py;
        float deg = s.rotation * RAD2DEG;
        if (uiSlider(ui, "Rotation", deg, -180.0f, 180.0f, "%.0f deg", !isCircle)) s.rotation = deg * DEG2RAD;
        float size = s.radius;
        if (uiSlider(ui, "Size", size, 8.0f, 300.0f, "%.0f")) s.scaleBy(size / s.radius);
        ui.y += 4;
    }

    if (uiSection(ui, "Physics", ed.objSection[1])) {
        bool st = s.isStatic;
        if (uiCheck(ui, "Static (immovable)", st)) {
            s.isStatic = st;
            s.refreshInvMass();
            if (st) s.velocity = Vec2{ 0, 0 };
        }
        bool dyn = !s.isStatic;
        float mass = s.mass;
        if (uiSlider(ui, "Mass", mass, 0.1f, 200.0f, "%.1f", dyn)) { s.mass = mass; s.refreshInvMass(); }
        uiSlider(ui, "Bounciness", s.restitution, 0.0f, 1.0f, "%.2f");
        uiSlider(ui, "Friction", s.friction, 0.0f, 1.0f, "%.2f");

        bool ge = s.gravityEnabled;
        if (uiCheck(ui, "Enable gravity", ge, dyn)) {
            if (ge && s.gravityScale <= 0.0f) s.gravityScale = 1.0f;
            s.gravityEnabled = ge;
        }
        uiSlider(ui, "Gravity scale", s.gravityScale, 0.0f, 3.0f, "%.2fx", dyn && s.gravityEnabled);
        float effective = (dyn && s.gravityEnabled) ? s.gravityScale * ed.gravityMs2 : 0.0f;
        uiLabel(ui, TextFormat("= %.1f m/s^2 on this object", effective));
        ui.y += 4;
    }

    if (uiSection(ui, "Appearance", ed.objSection[2])) {
        uiColor(ui, isImage ? "Tint" : "Color", s.color, ed.hex, ed.selected);
    }

}

// ---- Properties: the Environment (nothing selected) ----
void environmentInspector(Editor& ed, Ui& ui, float maxGroundOffset) {
    uiLabel(ui, "Settings for the whole scene.");
    ui.y += 2;

    if (uiSection(ui, "Gravity & motion", ed.envSection[0])) {
        uiSlider(ui, "Gravity", ed.gravityMs2, 0.0f, 30.0f, "%.1f");
        uiLabel(ui, "m/s^2 (Earth = 9.8). Objects opt in.");
        uiSlider(ui, "Air drag", ed.params.airDrag, 0.0f, 2.0f, "%.2f");
        uiSlider(ui, "Time scale", ed.timeScale, 0.1f, 2.0f, "%.2fx");
        int act = uiButtons(ui, "All gravity on", "All gravity off");
        if (act == 1) for (auto& b : ed.bodies) b.gravityEnabled = true;
        if (act == 2) for (auto& b : ed.bodies) b.gravityEnabled = false;
        ui.y += 2;
    }

    if (uiSection(ui, "Ground", ed.envSection[1])) {
        uiSlider(ui, "Height", ed.groundOffset, 10.0f, maxGroundOffset, "%.0f");
        uiSlider(ui, "Bounciness", ed.params.groundRestitution, 0.0f, 1.0f, "%.2f");
        uiSlider(ui, "Friction", ed.params.groundFriction, 0.0f, 1.0f, "%.2f");
        uiLabel(ui, "Tip: drag the ground line in the world.");
        ui.y += 2;
    }

    if (uiSection(ui, "Colors", ed.envSection[2])) {
        if (ui.vis(26)) GuiToggleGroup({ ui.x, ui.y, (ui.w - 4.0f) / 2.0f, 26 }, "Ground;Background", &ed.envColorTarget);
        ui.y += 32;
        Color& c = (ed.envColorTarget == 0) ? ed.groundColor : ed.backgroundColor;
        uiColor(ui, ed.envColorTarget == 0 ? "Ground" : "Background", c, ed.hex, 1000 + ed.envColorTarget);
    }

    if (uiSection(ui, "View", ed.envSection[3])) {
        uiCheck(ui, "Show grid", ed.showGrid);
        uiLabel(ui, "1 square = 1 m, measured up from ground.");
        ui.y += 2;
    }

    uiLabel(ui, TextFormat("%d object%s in the scene", (int)ed.bodies.size(), ed.bodies.size() == 1 ? "" : "s"));
}

// =========================================================
// The editor frame
// =========================================================
void editorFrame(Editor& ed, float& accumulator, AppMode& mode) {
    const int W = GetScreenWidth();
    const int H = GetScreenHeight();
    const int insX = W - INS_W;
    const int worldW = W - LEFT_W - INS_W;
    const int worldBottom = H - ASSET_H;   // the world ends where the Assests shelf begins

    // Keep the ground inside the world (also after the window is resized)
    float maxOffset = std::max(60.0f, (float)(worldBottom - TOP_H - 100));
    ed.groundOffset = std::clamp(ed.groundOffset, 10.0f, maxOffset);
    float groundY = (float)worldBottom - ed.groundOffset;
    ed.params.gravity = ed.gravityMs2 * PIXELS_PER_METER;

    Vector2 mouse = GetMousePosition();
    Vec2 m{ mouse.x, mouse.y };
    Rectangle worldRect = { (float)LEFT_W, (float)TOP_H, (float)worldW, (float)(worldBottom - TOP_H) };
    Rectangle shelfRect = { (float)LEFT_W, (float)worldBottom, (float)worldW, (float)ASSET_H };
    Rectangle insRect = { (float)insX, (float)TOP_H, (float)INS_W, (float)(H - TOP_H) };
    const float listTop = (float)(TOP_H + 250);
    Rectangle listRect = { 6.0f, listTop, (float)(LEFT_W - 12), std::max(40.0f, (float)H - listTop - 8.0f) };
    const float insViewTop = (float)(TOP_H + 40);
    const float insViewBottom = (float)H - 8.0f;
    bool mouseInWorld = CheckCollisionPointRec(mouse, worldRect);
    bool mouseInShelf = CheckCollisionPointRec(mouse, shelfRect);
    bool mouseInList = CheckCollisionPointRec(mouse, listRect);
    bool mouseInInspector = CheckCollisionPointRec(mouse, insRect);
    Bounds bounds{ (float)LEFT_W, (float)TOP_H, (float)insX, groundY };

    int objCount = (int)ed.bodies.size();
    if (ed.selected >= objCount) ed.selected = -1;
    if (ed.dragging >= objCount) ed.dragging = -1;
    if (ed.assetDrag >= (int)g_assets.size()) ed.assetDrag = -1;
    g_textEditing = ed.hex.edit || ed.nameEdit || ed.pathEdit;

    // ---- Scrolling (mouse wheel over a panel) ----
    float wheel = GetMouseWheelMove();
    float listContentH = (float)(objCount + 1) * LIST_ROW_H;
    if (mouseInList) ed.listScroll -= wheel * 40.0f;
    ed.listScroll = std::clamp(ed.listScroll, 0.0f, std::max(0.0f, listContentH - listRect.height));

    float insViewH = insViewBottom - insViewTop;
    if (mouseInInspector) ed.inspectorScroll -= wheel * 40.0f;
    ed.inspectorScroll = std::clamp(ed.inspectorScroll, 0.0f, std::max(0.0f, ed.inspectorContentH - insViewH));

    float shelfContentW = (float)g_assets.size() * 110.0f + 10.0f;
    if (mouseInShelf) ed.assetScroll -= wheel * 60.0f;
    ed.assetScroll = std::clamp(ed.assetScroll, 0.0f, std::max(0.0f, shelfContentW - (float)worldW));

    // ---- What is the mouse over? ----
    int hoverBody = -1;
    if (mouseInWorld && ed.paletteDrag < 0 && ed.assetDrag < 0) {
        for (int i = objCount - 1; i >= 0; --i) { // topmost first
            if (containsPoint(ed.bodies[i], m)) { hoverBody = i; break; }
        }
    }
    bool nearGround = mouseInWorld && hoverBody < 0 && std::fabs(m.y - groundY) <= 10.0f;

    int cursor = MOUSE_CURSOR_DEFAULT;
    if (ed.draggingGround || nearGround) cursor = MOUSE_CURSOR_RESIZE_NS;
    else if (ed.dragging >= 0) cursor = MOUSE_CURSOR_RESIZE_ALL;
    else if (hoverBody >= 0) cursor = MOUSE_CURSOR_POINTING_HAND;
    SetMouseCursor(cursor);

    // ---- Mouse press ----
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        // Start dragging a shape out of the "Add" grid
        for (int i = 0; i < PALETTE_COUNT; ++i) {
            if (CheckCollisionPointRec(mouse, paletteCell(i))) {
                ed.paletteDrag = i;
                ed.paletteColor = randomBrightColor();
            }
        }
        // Start dragging an image out of the Assests shelf
        if (mouseInShelf && mouse.y > shelfRect.y + 34.0f) {
            for (int i = 0; i < (int)g_assets.size(); ++i) {
                if (CheckCollisionPointRec(mouse, assetCell(i, shelfRect, ed.assetScroll))) ed.assetDrag = i;
            }
        }
        // Pick something from the Scene list (row 0 is the Environment)
        if (mouseInList) {
            for (int r = 0; r <= objCount; ++r) {
                Rectangle row = { listRect.x, listRect.y + r * LIST_ROW_H - ed.listScroll, listRect.width, LIST_ROW_H - 2.0f };
                if (CheckCollisionPointRec(mouse, row)) ed.selected = r - 1;
            }
        }
        // Work area: pick a shape, grab the ground line, or click empty space (= Environment)
        if (mouseInWorld) {
            if (hoverBody >= 0) {
                ed.selected = hoverBody;
                ed.dragging = hoverBody;
                ed.dragOffset = ed.bodies[hoverBody].position - m;
                ed.dragVelocity = Vec2{ 0, 0 };
            } else if (nearGround) {
                ed.draggingGround = true;
            } else {
                ed.selected = -1;
            }
        }
    }

    // ---- Drop: create the item where the mouse was released (gravity starts OFF) ----
    if (ed.paletteDrag >= 0 && !IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        if (mouseInWorld) {
            addObject(ed, makeShape(ed.paletteDrag, m, 28.0f, (int)ed.paletteSides, 0.0f, ed.paletteColor),
                      PALETTE_NAMES[ed.paletteDrag]);
        }
        ed.paletteDrag = -1;
    }
    if (ed.assetDrag >= 0 && !IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        if (mouseInWorld) {
            std::string stem = g_assets[ed.assetDrag].name;
            size_t dot = stem.rfind('.');
            if (dot != std::string::npos && dot > 0) stem = stem.substr(0, dot);
            addObject(ed, makeImageBody(ed.assetDrag, m), stem);
        }
        ed.assetDrag = -1;
    }

    // ---- Dragging a body (follows the mouse; lets go with a throw) ----
    if (ed.dragging >= 0) {
        Body& d = ed.bodies[ed.dragging];
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            Vec2 target = m + ed.dragOffset;
            float dt = GetFrameTime();
            if (dt > 0.0f) ed.dragVelocity = ed.dragVelocity * 0.6f + ((target - d.position) * (1.0f / dt)) * 0.4f;
            d.position = target;
            d.velocity = Vec2{ 0, 0 };
        } else {
            float speed = ed.dragVelocity.length();
            if (speed > 1500.0f) ed.dragVelocity = ed.dragVelocity * (1500.0f / speed);
            d.velocity = d.isStatic ? Vec2{ 0, 0 } : ed.dragVelocity;
            ed.dragging = -1;
        }
    }

    // ---- Dragging the ground line ----
    if (ed.draggingGround) {
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            ed.groundOffset = std::clamp((float)worldBottom - m.y, 10.0f, maxOffset);
            groundY = (float)worldBottom - ed.groundOffset;
            bounds.bottom = groundY;
        } else {
            ed.draggingGround = false;
        }
    }

    // ---- Keyboard shortcuts (not while typing in a text box) ----
    bool deleteSelected = false, duplicateSelected = false;
    if (!g_textEditing && ed.selected >= 0) {
        if (IsKeyPressed(KEY_DELETE)) deleteSelected = true;
        if (IsKeyPressed(KEY_D) && (IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL))) duplicateSelected = true;
    }

    // ---- When the selection changes: reset the properties scroll, keep the list row in view ----
    if (ed.selected != ed.lastSelected) {
        ed.lastSelected = ed.selected;
        ed.inspectorScroll = 0.0f;
        float contentH = (float)((int)ed.bodies.size() + 1) * LIST_ROW_H;
        float rowTop = (float)(ed.selected + 1) * LIST_ROW_H;
        if (rowTop < ed.listScroll) ed.listScroll = rowTop;
        else if (rowTop + LIST_ROW_H > ed.listScroll + listRect.height) ed.listScroll = rowTop + LIST_ROW_H - listRect.height;
        ed.listScroll = std::clamp(ed.listScroll, 0.0f, std::max(0.0f, contentH - listRect.height));
    }

    // ---- Physics ----
    // A body being dragged is made temporarily immovable so it pushes others out of the way.
    float savedInvMass = 0.0f;
    bool pinned = ed.dragging >= 0;
    if (pinned) {
        savedInvMass = ed.bodies[ed.dragging].invMass;
        ed.bodies[ed.dragging].invMass = 0.0f;
    }
    if (!ed.paused) {
        accumulator = std::min(accumulator + GetFrameTime(), 0.25f);
        while (accumulator >= FIXED_DT) {
            stepWorld(ed.bodies, bounds, ed.params, FIXED_DT * ed.timeScale);
            accumulator -= FIXED_DT;
        }
    } else {
        accumulator = 0.0f;
        resolveWallCollisions(ed.bodies, bounds, ed.params); // still keep things inside the area
    }
    if (pinned) ed.bodies[ed.dragging].invMass = savedInvMass;

    // =====================================================
    // Render
    // =====================================================
    BeginDrawing();
    ClearBackground(Theme::BG);

    // ---- Work area: background, grid, ground ----
    DrawRectangleRec(worldRect, ed.backgroundColor);
    if (ed.showGrid) {
        Color gc = luminance(ed.backgroundColor) < 128.0f ? Fade(WHITE, 0.06f) : Fade(BLACK, 0.08f);
        BeginScissorMode((int)worldRect.x, (int)worldRect.y, (int)worldRect.width, (int)worldRect.height);
        for (float gx = (float)LEFT_W; gx <= (float)(LEFT_W + worldW); gx += PIXELS_PER_METER)
            DrawLine((int)gx, TOP_H, (int)gx, worldBottom, gc);
        for (float gy = groundY; gy >= (float)TOP_H; gy -= PIXELS_PER_METER)
            DrawLine(LEFT_W, (int)gy, LEFT_W + worldW, (int)gy, gc);
        EndScissorMode();
    }
    DrawRectangle(LEFT_W, (int)groundY, worldW, worldBottom - (int)groundY, ed.groundColor);
    DrawRectangle(LEFT_W, (int)groundY, worldW, 4, edgeOf(ed.groundColor));
    {
        float gx = LEFT_W + worldW / 2.0f;
        bool hot = nearGround || ed.draggingGround;
        DrawRectangleRounded({ gx - 26, groundY - 5, 52, 10 }, 0.8f, 8, hot ? ORANGE : edgeOf(ed.groundColor));
        if (groundY + 30 < worldBottom) {
            DrawText("Ground - drag the line to move it", LEFT_W + 12, (int)groundY + 14, 14, textOn(ed.groundColor, 0.55f));
        }
    }

    for (int i = 0; i < (int)ed.bodies.size(); ++i) drawBody(ed.bodies[i], i == ed.selected);

    // ---- Assests shelf (bottom of the work area) ----
    DrawRectangleRec(shelfRect, Theme::PANEL);
    DrawLine(LEFT_W, worldBottom, LEFT_W + worldW, worldBottom, Theme::BORDER);
    BeginScissorMode((int)shelfRect.x, (int)shelfRect.y, (int)shelfRect.width, (int)shelfRect.height);
    {
        float sx = shelfRect.x, sy = shelfRect.y;
        DrawText(ASSETS_TITLE, (int)sx + 12, (int)sy + 8, 18, Theme::TEXT);

        if (GuiButton({ sx + 100, sy + 5, 100, 24 }, "Browse...")) ed.browseRequested = true;

        // Optional: type or paste a file path instead
        float pathW = std::clamp((float)worldW - 600.0f, 100.0f, 300.0f);
        float px = sx + 210;
        bool loadNow = false;
        if (GuiTextBox({ px, sy + 5, pathW, 24 }, ed.pathBuf, 500, ed.pathEdit)) {
            if (ed.pathEdit && IsKeyPressed(KEY_ENTER)) loadNow = true;
            ed.pathEdit = !ed.pathEdit;
        }
        if (GuiButton({ px + pathW + 6, sy + 5, 60, 24 }, "Load")) loadNow = true;
        if (loadNow && loadAsset(ed.pathBuf)) ed.pathBuf[0] = '\0';
        DrawText(g_assetMsg.c_str(), (int)(px + pathW + 76), (int)sy + 10, 14, Theme::TEXT_DIM);

        if (g_assets.empty()) {
            DrawText("Click Browse... to pick images from your computer,", (int)sx + 20, (int)sy + 55, 16, Theme::TEXT_DIM);
            DrawText("or drag image files onto this window.", (int)sx + 20, (int)sy + 78, 16, Theme::TEXT_DIM);
            DrawText("Then drag a thumbnail up into the world.", (int)sx + 20, (int)sy + 101, 16, Theme::TEXT_DIM);
        }
        for (int i = 0; i < (int)g_assets.size(); ++i) {
            Rectangle cell = assetCell(i, shelfRect, ed.assetScroll);
            if (cell.x + cell.width < sx || cell.x > sx + shelfRect.width) continue; // off-screen
            bool hovered = mouseInShelf && CheckCollisionPointRec(mouse, cell);
            DrawRectangleRec(cell, hovered ? Color{ 58, 64, 84, 255 } : Theme::CELL);
            DrawRectangleLinesEx(cell, 1.0f, Theme::BORDER);

            const Texture2D& tex = g_assets[i].tex;
            float fit = std::min(92.0f / (float)tex.width, 68.0f / (float)tex.height);
            float tw = tex.width * fit, th = tex.height * fit;
            Rectangle src = { 0, 0, (float)tex.width, (float)tex.height };
            Rectangle dst = { cell.x + (cell.width - tw) / 2.0f, cell.y + 6.0f + (68.0f - th) / 2.0f, tw, th };
            DrawTexturePro(tex, src, dst, Vector2{ 0, 0 }, 0.0f, WHITE);

            std::string label = g_assets[i].name;
            if (label.size() > 14) label = label.substr(0, 12) + "..";
            DrawText(label.c_str(), (int)cell.x + 6, (int)cell.y + 80, 12, Theme::TEXT);
        }
    }
    EndScissorMode();

    // ---- Top bar ----
    DrawRectangle(0, 0, W, TOP_H, Theme::PANEL);
    DrawLine(0, TOP_H, W, TOP_H, Theme::BORDER);
    DrawText("Sandbox Editor", 20, 13, 22, Theme::TEXT);
    if (GuiButton({ 260, 10, 150, 30 }, "Back to Spawner")) {
        mode = AppMode::SPAWNER;
        accumulator = 0.0f;
        ed.dragging = -1;
        ed.paletteDrag = -1;
        ed.assetDrag = -1;
        ed.draggingGround = false;
        SetMouseCursor(MOUSE_CURSOR_DEFAULT);
    }
    GuiCheckBox({ 440, 15, 20, 20 }, "Paused", &ed.paused);
    if (GuiButton({ 560, 10, 100, 30 }, "Clear all")) {
        ed.bodies.clear();
        ed.selected = -1;
        ed.dragging = -1;
    }
    DrawText(TextFormat("Objects: %d", (int)ed.bodies.size()), 690, 16, 18, Theme::TEXT);
    DrawText("F11: fullscreen", W - 130, 18, 14, Theme::TEXT_DIM);

    // ---- Left panel: Add + Scene list ----
    DrawRectangle(0, TOP_H, LEFT_W, H - TOP_H, Theme::PANEL);
    DrawLine(LEFT_W, TOP_H, LEFT_W, H, Theme::BORDER);

    DrawText("Add", 14, TOP_H + 10, 18, Theme::TEXT);
    DrawText("drag into the work area", 58, TOP_H + 14, 12, Theme::TEXT_DIM);
    for (int i = 0; i < PALETTE_COUNT; ++i) {
        Rectangle cell = paletteCell(i);
        bool hovered = CheckCollisionPointRec(mouse, cell);
        DrawRectangleRec(cell, hovered ? Color{ 58, 64, 84, 255 } : Theme::CELL);
        DrawRectangleLinesEx(cell, 1.0f, Theme::BORDER);
        Body icon = makeShape(i, Vec2{ cell.x + cell.width / 2.0f, cell.y + 28.0f }, 16.0f,
                              (int)ed.paletteSides, 0.0f, SKYBLUE);
        drawBody(icon);
        int tw = MeasureText(PALETTE_NAMES[i], 12);
        DrawText(PALETTE_NAMES[i], (int)(cell.x + (cell.width - tw) / 2.0f), (int)cell.y + 53, 12, Theme::TEXT);
    }
    // 6th grid slot: polygon side count
    DrawText("Polygon", 172, TOP_H + 124, 12, Theme::TEXT_DIM);
    DrawText(TextFormat("sides: %d", (int)ed.paletteSides), 172, TOP_H + 140, 12, Theme::TEXT);
    GuiSlider({ 170, (float)(TOP_H + 160), 66, 18 }, nullptr, nullptr, &ed.paletteSides, 3.0f, 12.0f);

    DrawText("Scene", 14, TOP_H + 222, 18, Theme::TEXT);
    DrawText(TextFormat("%d object%s", (int)ed.bodies.size(), ed.bodies.size() == 1 ? "" : "s"), 78, TOP_H + 227, 12, Theme::TEXT_DIM);

    BeginScissorMode((int)listRect.x, (int)listRect.y, (int)listRect.width, (int)listRect.height);
    for (int r = 0; r <= (int)ed.bodies.size(); ++r) {
        float ry = listRect.y + r * LIST_ROW_H - ed.listScroll;
        if (ry + LIST_ROW_H < listRect.y || ry > listRect.y + listRect.height) continue;
        Rectangle row = { listRect.x, ry, listRect.width, LIST_ROW_H - 2.0f };
        bool sel = (r - 1 == ed.selected);
        bool hov = mouseInList && CheckCollisionPointRec(mouse, row);
        DrawRectangleRec(row, sel ? Color{ 54, 104, 184, 255 } : (hov ? Color{ 58, 64, 84, 255 } : Theme::CELL));

        if (r == 0) {
            DrawCircleLines((int)row.x + 16, (int)(row.y + row.height / 2), 7.0f, Theme::TEXT);
            DrawLine((int)row.x + 9, (int)(row.y + row.height / 2), (int)row.x + 23, (int)(row.y + row.height / 2), Theme::TEXT);
            DrawText("Environment", (int)row.x + 32, (int)row.y + 6, 14, Theme::TEXT);
        } else {
            const Body& b = ed.bodies[r - 1];
            float cx = row.x + 16.0f, cy = row.y + row.height / 2.0f;
            if (b.assetIndex >= 0 && b.assetIndex < (int)g_assets.size()) {
                const Texture2D& tex = g_assets[b.assetIndex].tex;
                float fit = std::min(16.0f / (float)tex.width, 16.0f / (float)tex.height);
                float tw = tex.width * fit, th = tex.height * fit;
                DrawTexturePro(tex, Rectangle{ 0, 0, (float)tex.width, (float)tex.height },
                               Rectangle{ cx - tw / 2.0f, cy - th / 2.0f, tw, th }, Vector2{ 0, 0 }, 0.0f, WHITE);
            } else if (b.shape == ShapeType::CIRCLE) {
                DrawCircle((int)cx, (int)cy, 7.0f, b.color);
            } else {
                DrawRectangle((int)cx - 7, (int)cy - 7, 14, 14, b.color);
            }
            std::string label = b.name.empty() ? "(unnamed)" : b.name;
            if (label.size() > 20) label = label.substr(0, 18) + "..";
            DrawText(label.c_str(), (int)row.x + 32, (int)row.y + 6, 14, Theme::TEXT);
            if (b.isStatic) {
                int sw = MeasureText("static", 11);
                DrawText("static", (int)(row.x + row.width - sw - 6), (int)row.y + 8, 11, Theme::TEXT_DIM);
            }
        }
    }
    EndScissorMode();
    if (listContentH > listRect.height) { // scrollbar
        float barH = listRect.height * listRect.height / listContentH;
        float barY = listRect.y + ed.listScroll / listContentH * listRect.height;
        DrawRectangle((int)(listRect.x + listRect.width - 3), (int)barY, 3, (int)barH, Theme::BORDER);
    }

    // ---- Right panel: Properties of the selection ----
    DrawRectangle(insX, TOP_H, INS_W, H - TOP_H, Theme::PANEL);
    DrawLine(insX, TOP_H, insX, H, Theme::BORDER);
    DrawText("Properties", insX + 14, TOP_H + 10, 18, Theme::TEXT);
    DrawText(ed.selected >= 0 ? "Object" : "Environment", insX + 120, TOP_H + 14, 13, Theme::ACCENT);

    BeginScissorMode((int)insRect.x + 1, (int)insViewTop, (int)insRect.width - 1, (int)(insViewBottom - insViewTop));
    Ui ui{ (float)insX + 16.0f, (float)INS_W - 34.0f, insViewTop, insViewBottom, insViewTop - ed.inspectorScroll + 4.0f, mouse };
    float contentStart = ui.y;
    if (ed.selected >= 0) objectInspector(ed, ui, bounds, deleteSelected, duplicateSelected);
    else environmentInspector(ed, ui, maxOffset);
    EndScissorMode();
    ed.inspectorContentH = (ui.y - contentStart) + 12.0f;
    if (ed.inspectorContentH > insViewH) { // scrollbar
        float barH = insViewH * insViewH / ed.inspectorContentH;
        float barY = insViewTop + ed.inspectorScroll / ed.inspectorContentH * insViewH;
        DrawRectangle(W - 4, (int)barY, 3, (int)barH, Theme::BORDER);
    }

    // ---- Drag-and-drop ghost (on top of everything) ----
    if (ed.paletteDrag >= 0) {
        Body ghost = makeShape(ed.paletteDrag, m, 28.0f, (int)ed.paletteSides, 0.0f, ed.paletteColor);
        drawBody(ghost, false, true);
    }
    if (ed.assetDrag >= 0) {
        Body ghost = makeImageBody(ed.assetDrag, m);
        drawBody(ghost, false, true);
    }

    EndDrawing();

    // The OS dialog blocks until it closes, so it runs after the frame is drawn.
    if (ed.browseRequested) {
        ed.browseRequested = false;
        std::string note;
        std::vector<std::string> picked = openImageFilesDialog(note);
        if (!picked.empty()) loadAssets(picked);
        else g_assetMsg = note;
    }

    if (duplicateSelected && ed.selected >= 0 && ed.selected < (int)ed.bodies.size()) {
        duplicateObject(ed, ed.selected);
    }
    if (deleteSelected && ed.selected >= 0 && ed.selected < (int)ed.bodies.size()) {
        ed.bodies.erase(ed.bodies.begin() + ed.selected);
        ed.selected = -1;
        ed.dragging = -1;
    }
}

int main() {
    // Resizable window that starts maximized ("full screen kinda window").
    // F11 toggles true borderless fullscreen; ESC leaves fullscreen, or quits from the window.
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(1280, 800, "C++ Physics Engine - raylib + raygui");
    SetWindowMinSize(1100, 720);
    MaximizeWindow();
    SetExitKey(KEY_NULL);
    SetTargetFPS(60);

    applyDarkTheme();

    AppMode mode = AppMode::SPAWNER;
    SpawnerState spawner;
    Editor editor;
    editor.params.useBodyMaterials = true; // each object uses its own bounciness / friction

    float accumulator = 0.0f;
    bool quit = false;

    while (!WindowShouldClose() && !quit) {
        if (IsKeyPressed(KEY_F11)) ToggleBorderlessWindowed();
        if (IsKeyPressed(KEY_ESCAPE) && !g_textEditing) {
            if (IsWindowState(FLAG_BORDERLESS_WINDOWED_MODE)) ToggleBorderlessWindowed();
            else quit = true;
        }

        handleDroppedFiles();

        if (mode == AppMode::SPAWNER) spawnerFrame(spawner, accumulator, mode);
        else editorFrame(editor, accumulator, mode);
    }

    for (auto& a : g_assets) UnloadTexture(a.tex);
    CloseWindow();
    return 0;
}