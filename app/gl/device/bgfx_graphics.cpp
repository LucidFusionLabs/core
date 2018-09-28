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

#include "bgfx/bgfx.h"
#include "gd_common.h"

namespace LFL {
struct BGFXGraphicsDeviceConstants : public GraphicsDevice::Constants {
  BGFXGraphicsDeviceConstants() {
    Float = bgfx::AttribType::Float;
    Points = bgfx::Topology::PointList;
    Lines = bgfx::Topology::LineList;
    LineLoop = 0;
    Triangles = bgfx::Topology::TriList;
    TriangleStrip = bgfx::Topology::TriStrip;
    Texture2D = 0;
    TextureCubeMap = 0;
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
    Version = 0; // D3D_SDK_VERSION;
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

struct BGFXGraphicsDevice : public ShaderBasedGraphicsDevice {
  int target_matrix = -1;
  BGFXGraphicsDevice(Window *P, LFL::Shaders *S) :
    ShaderBasedGraphicsDevice(GraphicsDevice::Type::BGFX, BGFXGraphicsDeviceConstants(), P, 1, S) {
    default_color.push_back(Color(1.0, 1.0, 1.0, 1.0));
  }

  void Init(AssetLoading *loader, const Box &b) {
    done_init = true;
    GDDebug("Init");
    shader = &shaders->shader_default;

    bgfx::Init init;
    //init.type     = args.m_type;
    //init.vendorId = args.m_pciId;
    init.resolution.width  = b.w;
    init.resolution.height = b.h;
    init.resolution.reset  = BGFX_RESET_VSYNC;
    bgfx::init(init);

    bgfx::setViewClear(0
                       , BGFX_CLEAR_COLOR|BGFX_CLEAR_DEPTH
                       , 0x303030ff
                       , 1.0f
                       , 0
                      );

    ViewPort(b);
    DrawMode(default_draw_mode);
    InitDefaultLight();
    INFO("BGFX::Init width=", b.w, ", height=", b.h, ", shaders=", GetShaderSupport());
    INFO("BGFX Version: ", GetString(c.Version));
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

  void Finish() { ClearDeferred(); GDDebug("Finish"); }
  void Clear() {  }
  void ClearDepth() {  }
  void Flush() { ClearDeferred();  GDDebug("Flush"); }

  bool GetEnabled(int v) { return false; }
  void GetIntegerv(int t, int *out) {  }
  const char *GetString(int t) { return ""; }
  const char *GetGLEWString(int t) { return ""; }
  void CheckForError(const char *file, int line) {}

  void EnableScissor() { }
  void DisableScissor() {  }
  void EnableDepthTest() {  GDDebug("DepthTest=1"); }
  void DisableDepthTest() {  GDDebug("DepthTest=0"); }
  void EnableBlend() { if (Changed(&blend_enabled, true)) { ClearDeferred();  GDDebug("Blend=1"); } }
  void DisableBlend() { if (Changed(&blend_enabled, false)) { ClearDeferred(); GDDebug("Blend=0"); } }
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

  void ClearColor(const Color &c) { clear_color = c; GDDebug("ClearColor=", c.DebugString()); }
  void PointSize(float n) {  }
  void LineWidth(float n) { }

  TextureRef CreateTexture(int t, unsigned w, unsigned h, int f) {
    bgfx::TextureHandle handle = bgfx::createTexture2D(w, h, false, 1, GetPixel(f));
    TextureRef ret = { unsigned(bgfx::isValid(handle) ? handle.idx : 0), w, h, unsigned(t), unsigned(f) };
    GDLogRef("GenTexture ", t, ",", ret);
    return ret;
  }

  void DelTexture(TextureRef id) {
    bgfx::TextureHandle handle = { uint16_t(id.v) };
    bgfx::destroy(handle);
    GDLogRef("DelTexture ", id);
  }

  void UpdateTexture(const TextureRef&, int targ, int l, int fi, int b, int t, const void *data) {  }
  void UpdateSubTexture(const TextureRef&, int targ, int l, int xo, int yo, int w, int h, int t, const void *data) {  }

  void CopyTexImage2D(int targ, int l, int fi, int x, int y, int w, int h, int b) { }
  void CopyTexSubImage2D(int targ, int l, int xo, int yo, int x, int y, int w, int h) {  }
  void BindTexture(const TextureRef &id) { EnableTexture(); GDDebug("BindTexture=", id); }
  void BindCubeMap(const TextureRef&) {  GDDebug("CubeMap=", n); }

  void ActiveTexture(int n) {
    GDDebug("ActiveTexture=", n);
  }
  
  void TextureGenLinear() {
    static float X[4] = { -1,0,0,0 }, Y[4] = { 0,-1,0,0 }, Z[4] = { 0,0,-1,0 };
    GDDebug("TextureGen=L");
  
  }
  void TextureGenReflection() {
    GDDebug("TextureGen=R");
  }

  void TexParameter(int t, int p, int v) {
    GDDebug("TexParameter ", t, " ", p, " ", v);
  }

  FrameBufRef CreateFrameBuffer(unsigned w, unsigned h) {
    FrameBufRef ret;
    if (FLAGS_gd_debug) GDLogRef("GenFrameBuffer ", ret);
    return ret;
  }

  void DelFrameBuffer(FrameBufRef id) {
    if (FLAGS_gd_debug) GDLogRef("DelFrameBuffer ", id);
  }

  void BindFrameBuffer(const FrameBufRef &id) { ClearDeferred(); GDDebug("BindFrameBuffer ", id, " default_fb=", default_framebuffer); }
  void FrameBufferTexture(const TextureRef &id) {  }
  void FrameBufferDepthTexture(const DepthRef &in) {  }
  int CheckFrameBufferStatus() { return 0; }

  DepthRef CreateDepthBuffer(unsigned w, unsigned h, int f) {
    DepthRef ret;
    if (FLAGS_gd_debug) GDDebug("GenRenderBuffer ", ret);
    return ret;
  }

  void DelDepthBuffer(DepthRef id) {
    if (FLAGS_gd_debug) GDDebug("DelRenderBuffer ", id);
  }

  void BindDepthBuffer(const DepthRef &id) {  }

  ProgramRef CreateProgram() { ProgramRef p; GDDebug("CreateProgram ", p); return p; }
  void DelProgram(ProgramRef p) {  GDDebug("DelProgram ", p); }
  ShaderRef CreateShader(int t) { ShaderRef ret; return ret; }
  void CompileShader(const ShaderRef &shader, vector<const char*> source) {
  }
  void AttachShader(const ProgramRef &prog, const ShaderRef &shader) {  }
  void DelShader(const ShaderRef &shader) { }
  void BindAttribLocation(const ProgramRef &prog, int loc, const string &name) { }
  bool LinkProgram(const ProgramRef &prog) {
    return true;
  }
  void GetProgramiv(const ProgramRef &p, int t, int *out) {  }
  AttribRef GetAttribLocation(const ProgramRef &prog, const string &name) { AttribRef ret; return ret; }
  UniformRef GetUniformLocation(const ProgramRef &prog, const string &name) { UniformRef ret; return ret; }
  void Uniform1i(const UniformRef &u, int v) { }
  void Uniform1f(const UniformRef &u, float v) { }
  void Uniform2f(const UniformRef &u, float v1, float v2) {  }
  void Uniform3f(const UniformRef &u, float v1, float v2, float v3) {}
  void Uniform4f(const UniformRef &u, float v1, float v2, float v3, float v4) {  }
  void Uniform3fv(const UniformRef &u, int n, const float *v) {  }
  void Uniform4fv(const UniformRef &u, int n, const float *v) {  }
  void UniformMatrix4fv(const UniformRef &u, int n, const float *v) {  }

  void UseShader(Shader *S) {
    shader = X_or_Y(S, &shaders->shader_default);
    GDDebug("Shader=", shader->name);
  }

  void Screenshot(Texture *out) { ScreenshotBox(out, Box(parent->gl_w, parent->gl_h), Texture::Flag::FlipY); }
  void ScreenshotBox(Texture *out, const Box &b, int flag) {
    ClearDeferred();
    out->Resize(b.w, b.h, Texture::preferred_pf, Texture::Flag::CreateBuf);
    auto pixels = out->NewBuffer();
   
    GDDebug("ScreenshotBox");
  }

  void DumpTexture(Texture *out, const TextureRef &tex_id) {
#if !defined(LFL_MOBILE) && !defined(LFL_EMSCRIPTEN)
    if (tex_id) {

      // BindTexture(gl_tt, tex_id);

      //CHECK_GT((out->width = tex_w), 0);
      //CHECK_GT((out->height = tex_h), 0);
    }
    out->RenewBuffer();
#endif
  }

  bool VertexPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool ud, int) {  GDDebug("VertexPointer"); return true; }
  void TexPointer(int m, int t, int w, int o, float *tex, int l, int *out, bool ud) {  GDDebug("TexPointer"); }
  void ColorPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool ud) {  GDDebug("ColorPointer"); }
  void NormalPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool ud) {  GDDebug("NormalPointer"); }

  void DrawElements(int pt, int np, int it, int o, void *index, int l, int *out, bool dirty) {
    GDDebug("DrawElements(", pt, ", ", np, ", ", it, ", ", o, ", ", index, ", ", l, ", ", dirty, ")");
  }

  void DrawArrays(int type, int o, int n) {
    GDDebug("DrawArrays(", type, ", ", o, ", ", n, ")");
  }

  void DeferDrawArrays(int type, int o, int n) {
    GDDebug("DeferDrawArrays(", type, ", ", o, ", ", n, ") deferred 0");
  }

  void UpdateTexture() {}
  void UpdateNormals() {}
  void UpdateColorVerts() {}

  bgfx::TextureFormat::Enum GetPixel(int p) {
    switch (p) {
      case Pixel::RGBA:   case Pixel::RGB32: return bgfx::TextureFormat::RGBA8;
      case Pixel::RGB24:                     return bgfx::TextureFormat::RGB8;
      case Pixel::BGRA:   case Pixel::BGR32: return bgfx::TextureFormat::BGRA8;
      case Pixel::BGR24:                     return bgfx::TextureFormat::Unknown;
      case Pixel::GRAYA8:                    return bgfx::TextureFormat::Unknown;
      case Pixel::GRAY8:                     return bgfx::TextureFormat::Unknown;
      default:                               return bgfx::TextureFormat::Unknown;
    }
  }

  int GetDepth(int id) {
    switch (id) {
      case Depth::_16: return bgfx::TextureFormat::D16;
      default: return 0;
    }
  }

  int GetCubeMap(int id) {
    switch (id) {
      case CubeMap::NX: return 1;
      case CubeMap::PX: return 0;
      case CubeMap::NY: return 3;
      case CubeMap::PY: return 2;
      case CubeMap::NZ: return 5;
      case CubeMap::PZ: return 4;
      default: return -1;
    } 
  }
};

unique_ptr<GraphicsDevice> GraphicsDevice::Create(Window *w, Shaders *s) {
  return make_unique<BGFXGraphicsDevice>(w, s);
}

}; // namespace LFL
