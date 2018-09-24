/*
 * $Id$
 * Copyright (C) 2009 Lucid Fusion Labs

 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <d3dx9core.h>

#ifdef LFL_GDDEBUG
#define GDDebug(...) { \
  CheckForError(__FILE__, __LINE__); \
  if (FLAGS_gd_debug) DebugPrintf("%s", StrCat(__VA_ARGS__).c_str()); }
#else 
#define GDDebug(...)
#endif

#if defined(LFL_GDDEBUG) || defined(LFL_GDLOGREF)
#define GDLogRef(...) { \
  if (app->focused) app->focused->gd->CheckForError(__FILE__, __LINE__); \
  if (FLAGS_gd_debug) DebugPrintf("%s", StrCat(__VA_ARGS__).c_str()); }
#else 
#define GDLogRef(...)
#endif

namespace LFL {
struct DirectX9GraphicsDeviceConstants : public GraphicsDevice::Constants {
  DirectX9GraphicsDeviceConstants() {
    Float = 0; // GL_FLOAT;
    Points = D3DPT_POINTLIST;
    Lines = D3DPT_LINELIST;
    LineLoop = D3DPT_LINESTRIP;
    Triangles = D3DPT_TRIANGLELIST;
    TriangleStrip = D3DPT_TRIANGLESTRIP;
    Texture2D = D3DSTT_2D;
    TextureCubeMap = D3DSTT_CUBE;
    UnsignedByte = 0; //  GL_UNSIGNED_BYTE;
    UnsignedInt = 0; // GL_UNSIGNED_INT;
    FramebufferComplete = 0; // GL_FRAMEBUFFER_COMPLETE;
    Ambient = 0; //  GL_AMBIENT;
    Diffuse = 0; // GL_DIFFUSE;
    Specular = 0; // GL_SPECULAR;
    Position = 0; // GL_POSITION;
    Emission = 0; // GL_EMISSION;
    One = 0; // GL_ONE;
    SrcAlpha = 0; // GL_SRC_ALPHA;
    OneMinusSrcAlpha = 0; // GL_ONE_MINUS_SRC_ALPHA;
    OneMinusDstColor = 0; // GL_ONE_MINUS_DST_COLOR;
    TextureWrapS = 0; // GL_TEXTURE_WRAP_S;
    TextureWrapT = 0; // GL_TEXTURE_WRAP_T;
    ClampToEdge = 0; // GL_CLAMP_TO_EDGE;
    VertexShader = 0; // GL_VERTEX_SHADER;
    FragmentShader = 0; // GL_FRAGMENT_SHADER;
    ShaderVersion = 0; // GL_SHADING_LANGUAGE_VERSION;
    Extensions = 0; // GL_EXTENSIONS;
    GLEWVersion = 0;
    Version = D3D_SDK_VERSION;
    Vendor = 0; //  "Microsoft";
    DepthBits = 0; // GL_DEPTH_BITS;
    ScissorTest = 0; // GL_SCISSOR_TEST;
    ActiveUniforms = 0; // GL_ACTIVE_UNIFORMS;
    ActiveAttributes = 0; //  GL_ACTIVE_ATTRIBUTES;
    MaxVertexAttributes = 0; // MAXD3DDECLLENGTH;
    MaxViewportDims = 0; // GL_MAX_VIEWPORT_DIMS;
    ViewportBox = 0; // GL_VIEWPORT;
    ScissorBox = 0; // GL_SCISSOR_BOX;
    Fill = 0; // D3DFILL_FORCE_DWORD;
    Line = 0; // D3DFILL_WIREFRAME;
    Point = 0; // D3DFILL_POINT;
    Polygon = 0; // D3DFILL_SOLID;
    GLPreferredBuffer = 0; // GL_UNSIGNED_BYTE;
    GLInternalFormat = 0; // GL_RGBA;
    MaxVertexUniformComp = 0; // GL_MAX_VERTEX_UNIFORM_COMPONENTS;
    FramebufferBinding = 0; // GL_FRAMEBUFFER_BINDING;
    FramebufferUndefined = 0; // GL_FRAMEBUFFER_UNDEFINED;
  }
};

struct DirectX9GraphicsDevice : public GraphicsDevice {
  int target_matrix = -1;
  DirectX9GraphicsDevice(Window *P, LFL::Shaders *S) :
    GraphicsDevice(GraphicsDevice::Type::DIRECTX, DirectX9GraphicsDeviceConstants(), P, 9, S) {
    default_color.push_back(Color(1.0, 1.0, 1.0, 1.0));
  }

  void Init(AssetLoading *loader, const Box &b) {
    done_init = true;
    GDDebug("Init");
    shader = &shaders->shader_default;

    ViewPort(b);
    DrawMode(default_draw_mode);
    InitDefaultLight();
    INFO("DirectXGraphicsDevice::Init width=", b.w, ", height=", b.h, ", shaders=", ShaderSupport());
    LogVersion();
  }

  void UpdateColor() { const Color &c = default_color.back();  }
  bool ShaderSupport() {
    return true;
  }

  void MarkDirty() {}
  void  EnableTexture() { GDDebug("Texture=1"); }
  void DisableTexture() { GDDebug("Texture=0"); }
  void  EnableLighting() {  GDDebug("Lighting=1"); }
  void DisableLighting() { GDDebug("Lighting=0"); }
  void  EnableVertexColor() {  GDDebug("VertexColor=1"); }
  void DisableVertexColor() { GDDebug("VertexColor=0"); }
  void  EnableNormals() {  GDDebug("Normals=1"); }
  void DisableNormals() {  GDDebug("Normals=0"); }
  //void TextureEnvReplace()  { glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);  GDDebug("TextureEnv=R"); }
  //void TextureEnvModulate() { glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE); GDDebug("TextureEnv=M"); }
  void  EnableLight(int n) {  GDDebug("Light", n, "=1"); }
  void DisableLight(int n) {  GDDebug("Light", n, "=0"); }
  void Material(int t, float *color) {  }
  void Light(int n, int t, float *color) {  }

#ifdef LFL_MOBILE
  void TextureGenLinear() {}
  void TextureGenReflection() {}
  void DisableTextureGen() {}
#else
  void  EnableTextureGen() { GDDebug("TextureGen=1"); }
  void DisableTextureGen() { GDDebug("TextureGen=0"); }
  void TextureGenLinear() {
    static float X[4] = { -1,0,0,0 }, Y[4] = { 0,-1,0,0 }, Z[4] = { 0,0,-1,0 };
    EnableTextureGen();
    GDDebug("TextureGen=L");
  }
  void TextureGenReflection() {
    EnableTextureGen();
    GDDebug("TextureGen=R");
  }
#endif

  void DisableCubeMap() { GDDebug("CubeMap=", 0); }
  void BindCubeMap(int n) {  GDDebug("CubeMap=", n); }

  void ActiveTexture(int n) {

    GDDebug("ActiveTexture=", n);
  }

  void BindTexture(int t, int n) { EnableTexture(); GDDebug("BindTexture=", t, ",", n); }
  bool VertexPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool ud, int) {  GDDebug("VertexPointer"); return true; }
  void TexPointer(int m, int t, int w, int o, float *tex, int l, int *out, bool ud) {  GDDebug("TexPointer"); }
  void ColorPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool ud) {  GDDebug("ColorPointer"); }
  void NormalPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool ud) {  GDDebug("NormalPointer"); }
  void Color4f(float r, float g, float b, float a) { default_color.back() = Color(r, g, b, a); UpdateColor(); }
  void MatrixProjection() { target_matrix = 2;  }
  void MatrixModelview() { target_matrix = 1;  }
  void LoadIdentity() {  }
  void PushMatrix() {  }
  void PopMatrix() { }
  void GetMatrix(m44 *out) {  }
  void PrintMatrix() {}
  void Scalef(float x, float y, float z) {  }
  void Rotatef(float angle, float x, float y, float z) {  }
  void Ortho(float l, float r, float b, float t, float nv, float fv) { }
  void Frustum(float l, float r, float b, float t, float nv, float fv) {  }
  void Mult(const float *m) { }
  void Translate(float x, float y, float z) {  }

  void DrawElements(int pt, int np, int it, int o, void *index, int l, int *out, bool dirty) {
    GDDebug("DrawElements(", pt, ", ", np, ", ", it, ", ", o, ", ", index, ", ", l, ", ", dirty, ")");
  }

  void DrawArrays(int type, int o, int n) {
    GDDebug("DrawArrays(", type, ", ", o, ", ", n, ")");
  }

  void DeferDrawArrays(int type, int o, int n) {
    GDDebug("DeferDrawArrays(", type, ", ", o, ", ", n, ") deferred 0");
  }

  void UseShader(Shader *S) {
    shader = X_or_Y(S, &shaders->shader_default);
    GDDebug("Shader=", shader->name);
  }

  void Finish() { ClearDeferred(); GDDebug("Finish"); }
  void Flush() { ClearDeferred();  GDDebug("Flush"); }
  void Clear() {  }
  void ClearDepth() {  }
  void ClearColor(const Color &c) { clear_color = c; GDDebug("ClearColor=", c.DebugString()); }
  void PointSize(float n) {  }
  void LineWidth(float n) { }
  void DelTextures(int n, const unsigned *id) {
    ClearDeferred();
    if (FLAGS_gd_debug) for (int i = 0; i < n; i++) GDLogRef("DelTexture ", id[i]);
  }

  void TexImage2D(int targ, int l, int fi, int w, int h, int b, int f, int t, const void *data) {  }
  void TexSubImage2D(int targ, int l, int xo, int yo, int w, int h, int f, int t, const void *data) {  }
  void CopyTexImage2D(int targ, int l, int fi, int x, int y, int w, int h, int b) { }
  void CopyTexSubImage2D(int targ, int l, int xo, int yo, int x, int y, int w, int h) {  }

  void GenTextures(int t, int n, unsigned *out) {
    ClearDeferred();
    for (int i = 0; i < n; i++) CHECK_EQ(0, out[i]);
  
    for (int i = 0; i < n; i++) {

      GDLogRef("GenTexture ", t, ",", out[i]);
    }
  }

  void CheckForError(const char *file, int line) {
  }

  bool GetEnabled(int v) { return false; }
  void EnableScissor() { }
  void DisableScissor() {  }
  void EnableDepthTest() {  GDDebug("DepthTest=1"); }
  void DisableDepthTest() {  GDDebug("DepthTest=0"); }
  void DisableBlend() { if (Changed(&blend_enabled, false)) { ClearDeferred(); GDDebug("Blend=0"); } }
  void EnableBlend() { if (Changed(&blend_enabled, true)) { ClearDeferred();  GDDebug("Blend=1"); } }
  void BlendMode(int sm, int dm) { ClearDeferred();  GDDebug("BlendMode=", sm, ",", dm); }

  void ViewPort(Box w) {
    if (FLAGS_swap_axis) w.swapaxis(parent->gl_w, parent->gl_h);
    ClearDeferred();
    GDDebug("Viewport(", w.x, ", ", w.y, ", ", w.w, ", ", w.h, ")");
  }

  void Scissor(Box w) {
    if (FLAGS_swap_axis) w.swapaxis(parent->gl_w, parent->gl_h);
    ClearDeferred();
    EnableScissor();
    GDDebug("Scissor(", w.x, ", ", w.y, ", ", w.w, ", ", w.h, ")");
  }

  void TexParameter(int t, int p, int v) {
    GDDebug("TexParameter ", t, " ", p, " ", v);
  }

  void GenRenderBuffers(int n, unsigned *out) {
    if (FLAGS_gd_debug) for (int i = 0; i < n; i++) GDDebug("GenRenderBuffer ", out[i]);
  }

  void DelRenderBuffers(int n, const unsigned *id) {
    if (FLAGS_gd_debug) for (int i = 0; i < n; i++) GDDebug("DelRenderBuffer ", id[i]);
  }

  void BindRenderBuffer(int id) {  }
  void RenderBufferStorage(int d, int w, int h) {  }
  void GenFrameBuffers(int n, unsigned *out) {
    if (FLAGS_gd_debug) for (int i = 0; i < n; i++) GDLogRef("GenFrameBuffer ", out[i]);
  }

  void DelFrameBuffers(int n, const unsigned *id) {
    if (FLAGS_gd_debug) for (int i = 0; i < n; i++) GDLogRef("DelFrameBuffer ", id[i]);
  }

  void BindFrameBuffer(int id) { ClearDeferred(); GDDebug("BindFrameBuffer ", id, " default_fb=", default_framebuffer); }
  void FrameBufferTexture(int id) {  }
  void FrameBufferDepthTexture(int id) {  }
  int CheckFrameBufferStatus() { return 0; }

  void Screenshot(Texture *out) { ScreenshotBox(out, Box(parent->gl_w, parent->gl_h), Texture::Flag::FlipY); }
  void ScreenshotBox(Texture *out, const Box &b, int flag) {
    ClearDeferred();
    out->Resize(b.w, b.h, Texture::preferred_pf, Texture::Flag::CreateBuf);
    auto pixels = out->NewBuffer();
   
    GDDebug("ScreenshotBox");
  }

  void DumpTexture(Texture *out, unsigned tex_id) {
#if !defined(LFL_MOBILE) && !defined(LFL_EMSCRIPTEN)
    if (tex_id) {

      // BindTexture(gl_tt, tex_id);

      //CHECK_GT((out->width = tex_w), 0);
      //CHECK_GT((out->height = tex_h), 0);
    }
    out->RenewBuffer();
#endif
  }

  const char *GetString(int t) { return ""; }
  const char *GetGLEWString(int t) {
    return "";
  }

  int CreateProgram() { int p = 0; GDDebug("CreateProgram ", p); return p; }
  void DelProgram(int p) {  GDDebug("DelProgram ", p); }
  int CreateShader(int t) { return 0; }
  void CompileShader(int shader, vector<const char*> source) {
   
  }
  void AttachShader(int prog, int shader) {  }
  void DelShader(int shader) { }
  void BindAttribLocation(int prog, int loc, const string &name) { }
  bool LinkProgram(int prog) {
    return true;
  }
  void GetProgramiv(int p, int t, int *out) {  }
  void GetIntegerv(int t, int *out) {  }
  int GetAttribLocation(int prog, const string &name) { return 0; }
  int GetUniformLocation(int prog, const string &name) { return 0; }
  void Uniform1i(int u, int v) { }
  void Uniform1f(int u, float v) { }
  void Uniform2f(int u, float v1, float v2) {  }
  void Uniform3f(int u, float v1, float v2, float v3) {}
  void Uniform4f(int u, float v1, float v2, float v3, float v4) {  }
  void Uniform3fv(int u, int n, const float *v) {  }

  void InitDefaultLight() {
    float pos[] = { -.5,1,-.3f,0 }, grey20[] = { .2f,.2f,.2f,1 }, white[] = { 1,1,1,1 }, black[] = { 0,0,0,1 };
    EnableLight(0);
    Light(0, c.Position, pos);
    Light(0, c.Ambient, grey20);
    Light(0, c.Diffuse, white);
    Light(0, c.Specular, white);
    Material(c.Emission, black);
    Material(c.Specular, grey20);
  }

  void LogVersion() {
    INFO("DirectX Version: ", GetString(c.Version));
    have_cubemap = 0;

    int depth_bits = 0;
    GetIntegerv(c.DepthBits, &depth_bits);
    INFO("opengles_version = ", version, ", depth_bits = ", depth_bits);
    INFO("have_cubemap = ", have_cubemap ? "true" : "false");

#if 0
    int dim[2] = { 0, 0 };
    screen->gd->GetIntegerv(GraphicsDevice::MaxViewportDims, dim);
    INFO("max_viewport_dims = ", dim[0], ", ", dim[1]);
#endif
  }

  int GetPixel(int p) {
    switch (p) {
    case Pixel::RGBA:   case Pixel::RGB32: return D3DFMT_A8R8G8B8;
    case Pixel::RGB24:                     return D3DFMT_R8G8B8;
    case Pixel::BGRA:   case Pixel::BGR32: return D3DFMT_A8B8G8R8;
    case Pixel::BGR24:                     return -1;
    case Pixel::GRAYA8:                    return D3DFMT_A8L8;
    case Pixel::GRAY8:                     return D3DFMT_L8;
    default:                               return -1;
    }
  }

  int GetDepth(int id) {
    switch (id) {
    case Depth::_16: return 0; // GL_DEPTH_COMPONENT16;
    default: return 0;
    }
  }

  int GetCubeMap(int id) {
    switch (id) {
    case CubeMap::NX: return D3DCUBEMAP_FACE_NEGATIVE_X;
    case CubeMap::PX: return D3DCUBEMAP_FACE_POSITIVE_X;
    case CubeMap::NY: return D3DCUBEMAP_FACE_NEGATIVE_Y;
    case CubeMap::PY: return D3DCUBEMAP_FACE_POSITIVE_Y;
    case CubeMap::NZ: return D3DCUBEMAP_FACE_NEGATIVE_Z;
    case CubeMap::PZ: return D3DCUBEMAP_FACE_POSITIVE_Z;
    default: return 0; //  GL_TEXTURE_2D;
    } 
  }
};

unique_ptr<GraphicsDevice> CreateDirectXGraphicsDevice(Window *w, Shaders *s) {
  return make_unique<DirectX9GraphicsDevice>(w, s);
}

}; // namespace LFL
