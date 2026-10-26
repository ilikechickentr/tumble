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
std::vector<Vec2> makeRegularPolygon(int sides, float circumradius) {
    std::vector<Vec2> verts;
    verts.reserve(sides);
    for (int i = 0; i < sides; ++i) {
        float angle = -PI / 2.0f + (2.0f * PI * i) / sides; // start pointing up
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
        invMass = (mass <= 0.0f) ? 0.0f : 1.0f / mass;
    }
};

// ---------------------------------------------------------
// Simulation parameters (tweakable live via raygui)
// ---------------------------------------------------------
struct SimParams {
    float gravity = 500.0f;      // pixels/sec^2
    float restitution = 0.8f;    // 0 = no bounce, 1 = perfectly elastic
    bool paused = false;
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
            if (b.velocity.x < 0.0f) b.velocity.x = -b.velocity.x * params.restitution;
        } else if (maxX > bd.right) {
            b.position.x -= maxX - bd.right;
            if (b.velocity.x > 0.0f) b.velocity.x = -b.velocity.x * params.restitution;
        }

        // Ceiling / floor
        if (minY < bd.top) {
            b.position.y += bd.top - minY;
            if (b.velocity.y < 0.0f) b.velocity.y = -b.velocity.y * params.restitution;
        } else if (maxY > bd.bottom) {
            b.position.y -= maxY - bd.bottom;
            if (b.velocity.y > 0.0f) {
                b.velocity.y = (b.velocity.y < restSpeed) ? 0.0f : -b.velocity.y * params.restitution;
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
            if (totalInvMass <= 0.0f) continue; // both static

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
            float restVel = std::max(30.0f, params.gravity * (2.0f / 60.0f));
            float e = (-velAlongNormal < restVel) ? 0.0f : params.restitution;

            float impulseMag = -(1.0f + e) * velAlongNormal / totalInvMass;
            Vec2 impulse = m.normal * impulseMag;

            a.velocity -= impulse * a.invMass;
            b.velocity += impulse * b.invMass;
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

int loadAssets(const std::vector<std::string>& paths) {
    int ok = 0;
    for (const auto& p : paths) {
        if (loadAsset(p)) ++ok;
    }
    if (paths.size() > 1) g_assetMsg = TextFormat("Loaded %d of %d files", ok, (int)paths.size());
    return ok;
}

#ifdef _WIN32

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
        return Body(pos, makeRegularPolygon(4, size), size, rot + PI / 4.0f, size * size * 0.08f, c);
    case 3: {
        float w = size * 1.8f, h = size * 0.9f;
        float bounding = std::sqrt((w * 0.5f) * (w * 0.5f) + (h * 0.5f) * (h * 0.5f));
        return Body(pos, makeRectangle(w, h), bounding, rot, w * h * 0.002f, c);
    }
    default: {
        float extra = (sides % 2 == 0) ? PI / sides : 0.0f; // even N: flat edge on the bottom
        return Body(pos, makeRegularPolygon(sides, size), size, rot + extra, size * size * 0.05f, c);
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
constexpr int   PAL_W = 160;             // editor: palette width (left)
constexpr int   INS_W = 270;             // editor: inspector width (right)
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

struct Editor {
    std::vector<Body> bodies;
    SimParams params;             // gravity is fixed at 1g; each shape opts in and scales it
    bool paused = false;

    int selected = -1;            // index of the selected body, -1 = none
    int dragging = -1;            // index of the body being dragged, -1 = none
    Vec2 dragOffset{ 0, 0 };
    Vec2 dragVelocity{ 0, 0 };    // smoothed mouse velocity, applied when you let go (throw)

    int paletteDrag = -1;         // palette item being dragged into the world, -1 = none
    Color paletteColor = SKYBLUE;
    float paletteSides = 5.0f;

    int assetDrag = -1;           // asset being dragged into the world, -1 = none
    float assetScroll = 0.0f;
    char pathBuf[512] = "";
    bool pathEdit = false;
    bool browseRequested = false; // set by the Browse button, handled after the frame is drawn

    float groundOffset = 110.0f;  // how far the ground surface sits above the bottom of the world
    bool draggingGround = false;

    char hexBuf[16] = "#FFFFFF";
    bool hexEdit = false;
    int hexOwner = -2;            // which selection hexBuf currently belongs to
};

Rectangle paletteCell(int i) {
    return { 20.0f, (float)(TOP_H + 55 + i * 85), 120.0f, 75.0f };
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
// Sandbox editor mode (drag & drop, per-shape settings, images, movable ground)
// =========================================================
void editorFrame(Editor& ed, float& accumulator, AppMode& mode) {
    const int W = GetScreenWidth();
    const int H = GetScreenHeight();
    const int insX = W - INS_W;
    const int worldW = W - PAL_W - INS_W;
    const int worldBottom = H - ASSET_H;   // the world ends where the Assests shelf begins

    // Keep the ground inside the world (also after the window is resized)
    float maxOffset = std::max(60.0f, (float)(worldBottom - TOP_H - 100));
    ed.groundOffset = std::clamp(ed.groundOffset, 10.0f, maxOffset);
    float groundY = (float)worldBottom - ed.groundOffset;

    Vector2 mouse = GetMousePosition();
    Vec2 m{ mouse.x, mouse.y };
    Rectangle worldRect = { (float)PAL_W, (float)TOP_H, (float)worldW, (float)(worldBottom - TOP_H) };
    Rectangle shelfRect = { (float)PAL_W, (float)worldBottom, (float)worldW, (float)ASSET_H };
    bool mouseInWorld = CheckCollisionPointRec(mouse, worldRect);
    bool mouseInShelf = CheckCollisionPointRec(mouse, shelfRect);
    Bounds bounds{ (float)PAL_W, (float)TOP_H, (float)insX, groundY };

    if (ed.selected >= (int)ed.bodies.size()) ed.selected = -1;
    if (ed.dragging >= (int)ed.bodies.size()) ed.dragging = -1;
    if (ed.assetDrag >= (int)g_assets.size()) ed.assetDrag = -1;

    g_textEditing = ed.hexEdit || ed.pathEdit;

    // ---- Assests shelf scrolling ----
    float contentW = (float)g_assets.size() * 110.0f + 10.0f;
    float maxScroll = std::max(0.0f, contentW - (float)worldW);
    if (mouseInShelf) ed.assetScroll -= GetMouseWheelMove() * 60.0f;
    ed.assetScroll = std::clamp(ed.assetScroll, 0.0f, maxScroll);

    // ---- What is the mouse over? ----
    int hoverBody = -1;
    if (mouseInWorld && ed.paletteDrag < 0 && ed.assetDrag < 0) {
        for (int i = (int)ed.bodies.size() - 1; i >= 0; --i) { // topmost first
            if (containsPoint(ed.bodies[i], m)) { hoverBody = i; break; }
        }
    }
    bool nearGround = mouseInWorld && hoverBody < 0 && std::fabs(m.y - groundY) <= 10.0f;

    int cursor = MOUSE_CURSOR_DEFAULT;
    if (ed.draggingGround || nearGround) cursor = MOUSE_CURSOR_RESIZE_NS;
    else if (ed.dragging >= 0) cursor = MOUSE_CURSOR_RESIZE_ALL;
    else if (hoverBody >= 0) cursor = MOUSE_CURSOR_POINTING_HAND;
    SetMouseCursor(cursor);

    // ---- Mouse press: start a palette/asset drag, pick a body, or grab the ground ----
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        for (int i = 0; i < PALETTE_COUNT; ++i) {
            if (CheckCollisionPointRec(mouse, paletteCell(i))) {
                ed.paletteDrag = i;
                ed.paletteColor = randomBrightColor();
            }
        }
        if (mouseInShelf && mouse.y > shelfRect.y + 34.0f) {
            for (int i = 0; i < (int)g_assets.size(); ++i) {
                if (CheckCollisionPointRec(mouse, assetCell(i, shelfRect, ed.assetScroll))) ed.assetDrag = i;
            }
        }
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

    // ---- Palette / asset drop: create the item where the mouse was released ----
    // New shapes start with gravity OFF; turn it on per shape in the inspector.
    if (ed.paletteDrag >= 0 && !IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        if (mouseInWorld) {
            Body nb = makeShape(ed.paletteDrag, m, 28.0f, (int)ed.paletteSides, 0.0f, ed.paletteColor);
            nb.gravityEnabled = false;
            ed.bodies.push_back(nb);
            ed.selected = (int)ed.bodies.size() - 1;
        }
        ed.paletteDrag = -1;
    }
    if (ed.assetDrag >= 0 && !IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        if (mouseInWorld) {
            Body nb = makeImageBody(ed.assetDrag, m);
            nb.gravityEnabled = false;
            ed.bodies.push_back(nb);
            ed.selected = (int)ed.bodies.size() - 1;
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
            d.velocity = ed.dragVelocity;
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

    // ---- Delete key (not while typing in a text box) ----
    bool deleteSelected = false;
    if (IsKeyPressed(KEY_DELETE) && !g_textEditing && ed.selected >= 0) deleteSelected = true;

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
            stepWorld(ed.bodies, bounds, ed.params, FIXED_DT);
            accumulator -= FIXED_DT;
        }
    } else {
        accumulator = 0.0f;
        resolveWallCollisions(ed.bodies, bounds, ed.params); // still keep things inside the area
    }
    if (pinned) ed.bodies[ed.dragging].invMass = savedInvMass;

    // The hex box must restart whenever the selection changes
    if (ed.selected != ed.hexOwner) {
        ed.hexOwner = ed.selected;
        ed.hexEdit = false;
    }

    // =====================================================
    // Render
    // =====================================================
    BeginDrawing();
    ClearBackground(Theme::BG);

    // World + ground
    DrawRectangleRec(worldRect, Theme::WORLD);
    DrawRectangle(PAL_W, (int)groundY, worldW, worldBottom - (int)groundY, Theme::GROUND);
    DrawRectangle(PAL_W, (int)groundY, worldW, 4, Theme::GROUND_TOP);
    {
        float gx = PAL_W + worldW / 2.0f;
        bool hot = nearGround || ed.draggingGround;
        DrawRectangleRounded({ gx - 26, groundY - 5, 52, 10 }, 0.8f, 8, hot ? ORANGE : Theme::GROUND_TOP);
        if (groundY + 30 < worldBottom) {
            DrawText("Ground - drag the line to move it", PAL_W + 12, (int)groundY + 14, 14, Fade(WHITE, 0.55f));
        }
    }

    for (int i = 0; i < (int)ed.bodies.size(); ++i) drawBody(ed.bodies[i], i == ed.selected);

    // ---- Assests shelf (bottom) ----
    DrawRectangleRec(shelfRect, Theme::PANEL);
    DrawLine(PAL_W, worldBottom, PAL_W + worldW, worldBottom, Theme::BORDER);
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
    DrawText(TextFormat("Bodies: %d", (int)ed.bodies.size()), 690, 16, 18, Theme::TEXT);
    DrawText("F11: fullscreen", W - 130, 18, 14, Theme::TEXT_DIM);

    // ---- Palette (left) ----
    DrawRectangle(0, TOP_H, PAL_W, H - TOP_H, Theme::PANEL);
    DrawLine(PAL_W, TOP_H, PAL_W, H, Theme::BORDER);
    DrawText("Palette", 20, TOP_H + 10, 18, Theme::TEXT);
    DrawText("drag into the world", 20, TOP_H + 32, 12, Theme::TEXT_DIM);
    static const char* paletteNames[PALETTE_COUNT] = { "Circle", "Triangle", "Square", "Rectangle", "Polygon" };
    for (int i = 0; i < PALETTE_COUNT; ++i) {
        Rectangle cell = paletteCell(i);
        bool hovered = CheckCollisionPointRec(mouse, cell);
        DrawRectangleRec(cell, hovered ? Color{ 58, 64, 84, 255 } : Theme::CELL);
        DrawRectangleLinesEx(cell, 1.0f, Theme::BORDER);
        Body icon = makeShape(i, Vec2{ cell.x + cell.width / 2.0f, cell.y + 30.0f }, 20.0f,
                              (int)ed.paletteSides, 0.0f, SKYBLUE);
        drawBody(icon);
        DrawText(paletteNames[i], (int)cell.x + 8, (int)cell.y + 58, 14, Theme::TEXT);
    }
    GuiSlider({ 55, (float)(TOP_H + 55 + PALETTE_COUNT * 85 + 10), 70, 20 }, "Sides",
              TextFormat("%d", (int)ed.paletteSides), &ed.paletteSides, 3.0f, 12.0f);

    // ---- Inspector (right) ----
    DrawRectangle(insX, TOP_H, INS_W, H - TOP_H, Theme::PANEL);
    DrawLine(insX, TOP_H, insX, H, Theme::BORDER);
    DrawText("Inspector", insX + 15, TOP_H + 10, 18, Theme::TEXT);

    if (ed.selected >= 0) {
        Body& s = ed.bodies[ed.selected];
        bool isImage = s.assetIndex >= 0;
        DrawText(shapeName(s), insX + 15, 95, 14, Theme::TEXT_DIM);

        // Size (uniform scale of the whole shape)
        float newSize = s.radius;
        GuiSlider({ (float)insX + 70, 125, 100, 20 }, "Size", TextFormat("%.0f", newSize), &newSize, 8.0f, 300.0f);
        if (std::fabs(newSize - s.radius) > 0.01f) s.scaleBy(newSize / s.radius);

        // Gravity: off until enabled for this shape
        bool gEnabled = s.gravityEnabled;
        GuiCheckBox({ (float)insX + 15, 157, 20, 20 }, "Enable gravity", &gEnabled);
        if (gEnabled && !s.gravityEnabled && s.gravityScale <= 0.0f) s.gravityScale = 1.0f;
        s.gravityEnabled = gEnabled;

        float gMs2 = s.gravityScale * EARTH_G;
        float newG = gMs2;
        if (!s.gravityEnabled) GuiDisable();
        GuiSlider({ (float)insX + 70, 187, 100, 20 }, "Gravity", TextFormat("%.1f m/s^2", newG), &newG, 0.0f, 30.0f);
        if (!s.gravityEnabled) GuiEnable();
        if (s.gravityEnabled && std::fabs(newG - gMs2) > 0.001f) s.gravityScale = newG / EARTH_G;

        // Hex color (for images: a tint): text box + live swatch
        if (!ed.hexEdit) {
            std::snprintf(ed.hexBuf, sizeof(ed.hexBuf), "#%02X%02X%02X", s.color.r, s.color.g, s.color.b);
        }
        DrawText(isImage ? "Tint" : "Hex", insX + 15, 228, 14, Theme::TEXT);
        if (GuiTextBox({ (float)insX + 50, 222, 100, 26 }, ed.hexBuf, 8, ed.hexEdit)) ed.hexEdit = !ed.hexEdit;
        if (ed.hexEdit) {
            Color parsed;
            if (parseHexColor(ed.hexBuf, parsed)) s.color = parsed;
        }
        DrawRectangle(insX + 160, 222, 26, 26, s.color);
        DrawRectangleLines(insX + 160, 222, 26, 26, Theme::BORDER);

        // Color picker (only adopt its value while the user is actually dragging in it)
        Rectangle pickerArea = { (float)insX + 10, 262, 215, 160 };
        Color picked = s.color;
        GuiColorPicker({ (float)insX + 15, 267, 150, 150 }, "", &picked);
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && CheckCollisionPointRec(mouse, pickerArea) &&
            (picked.r != s.color.r || picked.g != s.color.g || picked.b != s.color.b)) {
            s.color = Color{ picked.r, picked.g, picked.b, 255 };
        }

        if (GuiButton({ (float)insX + 15, 437, 130, 30 }, "Delete shape")) deleteSelected = true;
        DrawText("or press Delete", insX + 155, 445, 12, Theme::TEXT_DIM);
    } else {
        DrawText("Click a shape to edit it,", insX + 15, 95, 16, Theme::TEXT);
        DrawText("or drag one in from the left.", insX + 15, 115, 16, Theme::TEXT);
        DrawText("Size, color and gravity are set", insX + 15, 150, 14, Theme::TEXT_DIM);
        DrawText("per shape. Gravity is off until", insX + 15, 168, 14, Theme::TEXT_DIM);
        DrawText("you enable it.", insX + 15, 186, 14, Theme::TEXT_DIM);
    }

    // World settings
    DrawText("World", insX + 15, 495, 18, Theme::TEXT);
    GuiSlider({ (float)insX + 70, 525, 100, 20 }, "Ground", TextFormat("%.0f px", ed.groundOffset),
              &ed.groundOffset, 10.0f, maxOffset);
    GuiSlider({ (float)insX + 70, 555, 100, 20 }, "Bounce", TextFormat("%.2f", ed.params.restitution),
              &ed.params.restitution, 0.0f, 1.0f);
    if (GuiButton({ (float)insX + 15, 590, 115, 26 }, "Gravity: all on")) {
        for (auto& b : ed.bodies) b.gravityEnabled = true;
    }
    if (GuiButton({ (float)insX + 140, 590, 115, 26 }, "Gravity: all off")) {
        for (auto& b : ed.bodies) b.gravityEnabled = false;
    }

    // Drag-and-drop ghost (on top of everything)
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
    SetWindowMinSize(1000, 720);
    MaximizeWindow();
    SetExitKey(KEY_NULL);
    SetTargetFPS(60);

    applyDarkTheme();

    AppMode mode = AppMode::SPAWNER;
    SpawnerState spawner;
    Editor editor;
    editor.params.gravity = EARTH_G * PIXELS_PER_METER; // 1g; each shape opts in and scales it
    editor.params.restitution = 0.4f;

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