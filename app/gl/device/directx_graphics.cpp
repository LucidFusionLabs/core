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
struct DirectXGraphicsDeviceConstants : public GraphicsDevice::Constants {
  DirectXGraphicsDeviceConstants() {
    Float = 0; // GL_FLOAT;
    Points = D3DPT_POINTLIST;
    Lines = D3DPT_LINELIST;
    LineLoop = D3DPT_LINESTRIP;
    Triangles = D3DPT_TRIANGLELIST;
    TriangleStrip = D3DPT_TRIANGLESTRIP;
    Texture2D = D3DSTT_2D;
    TextureCubeMap = D3DSTT_CUBE;
#if 0
    const int GraphicsDevice::UnsignedByte = GL_UNSIGNED_BYTE;
    const int GraphicsDevice::UnsignedInt = GL_UNSIGNED_INT;
    const int GraphicsDevice::FramebufferComplete = GL_FRAMEBUFFER_COMPLETE;
    const int GraphicsDevice::Ambient = GL_AMBIENT;
    const int GraphicsDevice::Diffuse = GL_DIFFUSE;
    const int GraphicsDevice::Specular = GL_SPECULAR;
    const int GraphicsDevice::Position = GL_POSITION;
    const int GraphicsDevice::Emission = GL_EMISSION;
    const int GraphicsDevice::One = GL_ONE;
    const int GraphicsDevice::SrcAlpha = GL_SRC_ALPHA;
    const int GraphicsDevice::OneMinusSrcAlpha = GL_ONE_MINUS_SRC_ALPHA;
    const int GraphicsDevice::OneMinusDstColor = GL_ONE_MINUS_DST_COLOR;
    const int GraphicsDevice::TextureWrapS = GL_TEXTURE_WRAP_S;
    const int GraphicsDevice::TextureWrapT = GL_TEXTURE_WRAP_T;
    const int GraphicsDevice::ClampToEdge = GL_CLAMP_TO_EDGE;
    const int GraphicsDevice::VertexShader = GL_VERTEX_SHADER;
    const int GraphicsDevice::FragmentShader = GL_FRAGMENT_SHADER;
    const int GraphicsDevice::ShaderVersion = GL_SHADING_LANGUAGE_VERSION;
    const int GraphicsDevice::Extensions = GL_EXTENSIONS;
    const int GraphicsDevice::GLEWVersion = 0;
    const int GraphicsDevice::Version = D3D_SDK_VERSION;
    const int GraphicsDevice::Vendor = "Microsoft";
    const int GraphicsDevice::DepthBits = GL_DEPTH_BITS;
    const int GraphicsDevice::ScissorTest = GL_SCISSOR_TEST;
    const int GraphicsDevice::ActiveUniforms = GL_ACTIVE_UNIFORMS;
    const int GraphicsDevice::ActiveAttributes = GL_ACTIVE_ATTRIBUTES;
    const int GraphicsDevice::MaxVertexAttributes = MAXD3DDECLLENGTH;
    const int GraphicsDevice::MaxViewportDims = GL_MAX_VIEWPORT_DIMS;
    const int GraphicsDevice::ViewportBox = GL_VIEWPORT;
    const int GraphicsDevice::ScissorBox = GL_SCISSOR_BOX;
    const int GraphicsDevice::Fill = D3DFILL_FORCE_DWORD;
    const int GraphicsDevice::Line = D3DFILL_WIREFRAME;
    const int GraphicsDevice::Point = D3DFILL_POINT;
    const int GraphicsDevice::Polygon = D3DFILL_SOLID;
    const int GraphicsDevice::GLPreferredBuffer = GL_UNSIGNED_BYTE;
    const int GraphicsDevice::GLInternalFormat = GL_RGBA;
    const int GraphicsDevice::MaxVertexUniformComp = GL_MAX_VERTEX_UNIFORM_COMPONENTS;
    const int GraphicsDevice::FramebufferBinding = GL_FRAMEBUFFER_BINDING;
    const int GraphicsDevice::FramebufferUndefined = GL_FRAMEBUFFER_UNDEFINED;
#endif
  }
};

#if 0
struct DirectXGraphicsDevice : public GraphicsDevice {
  int target_matrix = -1;
  DirectXGraphicsDevice(Window *P, LFL::Shaders *S) : GraphicsDevice(P, 1, S) {
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

  void UpdateColor() { const Color &c = default_color.back(); glColor4f(c.r(), c.g(), c.b(), c.a()); }
  bool ShaderSupport() {
#ifdef LFL_MOBILE
    return false;
#endif
    const char *ver = GetString(GL_VERSION);
    return ver && *ver == '2';
  }

  void MarkDirty() {}
  void  EnableTexture() { glEnable(GL_TEXTURE_2D);  glEnableClientState(GL_TEXTURE_COORD_ARRAY); GDDebug("Texture=1"); }
  void DisableTexture() { glDisable(GL_TEXTURE_2D); glDisableClientState(GL_TEXTURE_COORD_ARRAY); GDDebug("Texture=0"); }
  void  EnableLighting() { glEnable(GL_LIGHTING);  glEnable(GL_COLOR_MATERIAL); GDDebug("Lighting=1"); }
  void DisableLighting() { glDisable(GL_LIGHTING); glDisable(GL_COLOR_MATERIAL); GDDebug("Lighting=0"); }
  void  EnableVertexColor() { glEnableClientState(GL_COLOR_ARRAY); GDDebug("VertexColor=1"); }
  void DisableVertexColor() { glDisableClientState(GL_COLOR_ARRAY); GDDebug("VertexColor=0"); }
  void  EnableNormals() { glEnableClientState(GL_NORMAL_ARRAY); GDDebug("Normals=1"); }
  void DisableNormals() { glDisableClientState(GL_NORMAL_ARRAY); GDDebug("Normals=0"); }
  //void TextureEnvReplace()  { glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);  GDDebug("TextureEnv=R"); }
  //void TextureEnvModulate() { glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE); GDDebug("TextureEnv=M"); }
  void  EnableLight(int n) { if (n)  glEnable(GL_LIGHT1); else  glEnable(GL_LIGHT0); GDDebug("Light", n, "=1"); }
  void DisableLight(int n) { if (n) glDisable(GL_LIGHT1); else glDisable(GL_LIGHT0); GDDebug("Light", n, "=0"); }
  void Material(int t, float *color) { glMaterialfv(GL_FRONT_AND_BACK, t, color); }
  void Light(int n, int t, float *color) { glLightfv(((n) ? GL_LIGHT1 : GL_LIGHT0), t, color); }

#ifdef LFL_MOBILE
  void TextureGenLinear() {}
  void TextureGenReflection() {}
  void DisableTextureGen() {}
#else
  void  EnableTextureGen() { glEnable(GL_TEXTURE_GEN_S);  glEnable(GL_TEXTURE_GEN_T);  glEnable(GL_TEXTURE_GEN_R); GDDebug("TextureGen=1"); }
  void DisableTextureGen() { glDisable(GL_TEXTURE_GEN_S); glDisable(GL_TEXTURE_GEN_T); glDisable(GL_TEXTURE_GEN_R); GDDebug("TextureGen=0"); }
  void TextureGenLinear() {
    static float X[4] = { -1,0,0,0 }, Y[4] = { 0,-1,0,0 }, Z[4] = { 0,0,-1,0 };
    EnableTextureGen();
    glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
    glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
    glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
    glTexGenfv(GL_S, GL_OBJECT_PLANE, X);
    glTexGenfv(GL_T, GL_OBJECT_PLANE, Y);
    glTexGenfv(GL_R, GL_OBJECT_PLANE, Z);
    GDDebug("TextureGen=L");
  }
  void TextureGenReflection() {
    EnableTextureGen();
    glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_REFLECTION_MAP);
    glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_REFLECTION_MAP);
    glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_REFLECTION_MAP);
    GDDebug("TextureGen=R");
  }
#endif

  void DisableCubeMap() { glDisable(GL_TEXTURE_CUBE_MAP); DisableTextureGen();                   GDDebug("CubeMap=", 0); }
  void BindCubeMap(int n) { glEnable(GL_TEXTURE_CUBE_MAP); glBindTexture(GL_TEXTURE_CUBE_MAP, n); GDDebug("CubeMap=", n); }

  void ActiveTexture(int n) {
    glClientActiveTexture(GL_TEXTURE0 + n);
    glActiveTexture(GL_TEXTURE0 + n);
    // glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_COMBINE);
    // glTexEnvf (GL_TEXTURE_ENV, GL_COMBINE_RGB_EXT, GL_MODULATE);
    GDDebug("ActiveTexture=", n);
  }

  void BindTexture(int t, int n) { EnableTexture(); glBindTexture(t, n); GDDebug("BindTexture=", t, ",", n); }
  bool VertexPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool ud, int) { glVertexPointer(m, t, w, verts + o / sizeof(float)); GDDebug("VertexPointer"); return true; }
  void TexPointer(int m, int t, int w, int o, float *tex, int l, int *out, bool ud) { glTexCoordPointer(m, t, w, tex + o / sizeof(float)); GDDebug("TexPointer"); }
  void ColorPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool ud) { glColorPointer(m, t, w, verts + o / sizeof(float)); GDDebug("ColorPointer"); }
  void NormalPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool ud) { glNormalPointer(t, w, verts + o / sizeof(float)); GDDebug("NormalPointer"); }
  void Color4f(float r, float g, float b, float a) { default_color.back() = Color(r, g, b, a); UpdateColor(); }
  void MatrixProjection() { target_matrix = 2; glMatrixMode(GL_PROJECTION); }
  void MatrixModelview() { target_matrix = 1; glMatrixMode(GL_MODELVIEW); }
  void LoadIdentity() { glLoadIdentity(); }
  void PushMatrix() { glPushMatrix(); }
  void PopMatrix() { glPopMatrix(); }
  void GetMatrix(m44 *out) { glGetFloatv(target_matrix == 2 ? GL_PROJECTION_MATRIX : GL_MODELVIEW_MATRIX, &(*out)[0][0]); }
  void PrintMatrix() {}
  void Scalef(float x, float y, float z) { glScalef(x, y, z); }
  void Rotatef(float angle, float x, float y, float z) { glRotatef(angle, x, y, z); }
  void Ortho(float l, float r, float b, float t, float nv, float fv) { glOrtho(l, r, b, t, nv, fv); }
  void Frustum(float l, float r, float b, float t, float nv, float fv) { glFrustum(l, r, b, t, nv, fv); }
  void Mult(const float *m) { glMultMatrixf(m); }
  void Translate(float x, float y, float z) { glTranslatef(x, y, z); }

  void DrawElements(int pt, int np, int it, int o, void *index, int l, int *out, bool dirty) {
    glDrawElements(pt, np, it, static_cast<char*>(index) + o);
    GDDebug("DrawElements(", pt, ", ", np, ", ", it, ", ", o, ", ", index, ", ", l, ", ", dirty, ")");
  }

  void DrawArrays(int type, int o, int n) {
    glDrawArrays(type, o, n);
    GDDebug("DrawArrays(", type, ", ", o, ", ", n, ")");
  }

  void DeferDrawArrays(int type, int o, int n) {
    glDrawArrays(type, o, n);
    GDDebug("DeferDrawArrays(", type, ", ", o, ", ", n, ") deferred 0");
  }

  void UseShader(Shader *S) {
    shader = X_or_Y(S, &shaders->shader_default);
    glUseProgram(shader->ID);
    GDDebug("Shader=", shader->name);
  }

  void Finish() { ClearDeferred(); glFinish(); GDDebug("Finish"); }
  void Flush() { ClearDeferred(); glFlush(); GDDebug("Flush"); }
  void Clear() { glClear(GL_COLOR_BUFFER_BIT | (draw_mode == DrawMode::_3D ? GL_DEPTH_BUFFER_BIT : 0)); }
  void ClearDepth() { glClear(GL_DEPTH_BUFFER_BIT); }
  void ClearColor(const Color &c) { clear_color = c; glClearColor(c.r(), c.g(), c.b(), c.a()); GDDebug("ClearColor=", c.DebugString()); }
  void PointSize(float n) { glPointSize(n); }
  void LineWidth(float n) { glLineWidth(n); }
  void DelTextures(int n, const unsigned *id) {
    ClearDeferred();
    if (FLAGS_gd_debug) for (int i = 0; i < n; i++) GDLogRef("DelTexture ", id[i]);
    glDeleteTextures(n, id);
  }

  void TexImage2D(int targ, int l, int fi, int w, int h, int b, int f, int t, const void *data) { glTexImage2D(targ, l, fi, w, h, b, f, t, data); }
  void TexSubImage2D(int targ, int l, int xo, int yo, int w, int h, int f, int t, const void *data) { glTexSubImage2D(targ, l, xo, yo, w, h, f, t, data); }
  void CopyTexImage2D(int targ, int l, int fi, int x, int y, int w, int h, int b) { glCopyTexImage2D(targ, l, fi, x, y, w, h, b); }
  void CopyTexSubImage2D(int targ, int l, int xo, int yo, int x, int y, int w, int h) { glCopyTexSubImage2D(targ, l, xo, yo, x, y, w, h); }

  void GenTextures(int t, int n, unsigned *out) {
    ClearDeferred();
    for (int i = 0; i < n; i++) CHECK_EQ(0, out[i]);
    if (t == GL_TEXTURE_CUBE_MAP) glEnable(GL_TEXTURE_CUBE_MAP);
    glGenTextures(n, out);
    for (int i = 0; i < n; i++) {
      glBindTexture(t, out[i]);
      glTexParameteri(t, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(t, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      GDLogRef("GenTexture ", t, ",", out[i]);
    }
  }

  void CheckForError(const char *file, int line) {
  }

  bool GetEnabled(int v) { return glIsEnabled(v); }
  void EnableScissor() { glEnable(GL_SCISSOR_TEST); }
  void DisableScissor() { glDisable(GL_SCISSOR_TEST); }
  void EnableDepthTest() { glEnable(GL_DEPTH_TEST); glDepthMask(GL_TRUE);  GDDebug("DepthTest=1"); }
  void DisableDepthTest() { glDisable(GL_DEPTH_TEST); glDepthMask(GL_FALSE); GDDebug("DepthTest=0"); }
  void DisableBlend() { if (Changed(&blend_enabled, false)) { ClearDeferred(); glDisable(GL_BLEND);                                                    GDDebug("Blend=0"); } }
  void EnableBlend() { if (Changed(&blend_enabled, true)) { ClearDeferred();  glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); GDDebug("Blend=1"); } }
  void BlendMode(int sm, int dm) { ClearDeferred(); glBlendFunc(sm, dm); GDDebug("BlendMode=", sm, ",", dm); }

  void ViewPort(Box w) {
    if (FLAGS_swap_axis) w.swapaxis(parent->gl_w, parent->gl_h);
    ClearDeferred();
    glViewport(w.x, w.y, w.w, w.h);
    GDDebug("Viewport(", w.x, ", ", w.y, ", ", w.w, ", ", w.h, ")");
  }

  void Scissor(Box w) {
    if (FLAGS_swap_axis) w.swapaxis(parent->gl_w, parent->gl_h);
    ClearDeferred();
    EnableScissor();
    glScissor(w.x, w.y, w.w, w.h);
    GDDebug("Scissor(", w.x, ", ", w.y, ", ", w.w, ", ", w.h, ")");
  }

  void TexParameter(int t, int p, int v) {
    glTexParameteri(t, p, v);
    GDDebug("TexParameter ", t, " ", p, " ", v);
  }

  void GenRenderBuffers(int n, unsigned *out) {
    glGenRenderbuffersEXT(n, out);
    if (FLAGS_gd_debug) for (int i = 0; i < n; i++) GDDebug("GenRenderBuffer ", out[i]);
  }

  void DelRenderBuffers(int n, const unsigned *id) {
    if (FLAGS_gd_debug) for (int i = 0; i < n; i++) GDDebug("DelRenderBuffer ", id[i]);
    glDeleteRenderbuffersEXT(n, id);
  }

  void BindRenderBuffer(int id) { glBindRenderbufferEXT(GL_RENDERBUFFER, id); }
  void RenderBufferStorage(int d, int w, int h) { glRenderbufferStorageEXT(GL_RENDERBUFFER, d, w, h); }
  void GenFrameBuffers(int n, unsigned *out) {
    glGenFramebuffersEXT(n, out);
    if (FLAGS_gd_debug) for (int i = 0; i < n; i++) GDLogRef("GenFrameBuffer ", out[i]);
  }

  void DelFrameBuffers(int n, const unsigned *id) {
    if (FLAGS_gd_debug) for (int i = 0; i < n; i++) GDLogRef("DelFrameBuffer ", id[i]);
    glDeleteFramebuffersEXT(n, id);
  }

  void BindFrameBuffer(int id) { ClearDeferred(); glBindFramebufferEXT(GL_FRAMEBUFFER, id); GDDebug("BindFrameBuffer ", id, " default_fb=", default_framebuffer); }
  void FrameBufferTexture(int id) { glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, id, 0); }
  void FrameBufferDepthTexture(int id) { glFramebufferRenderbufferEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, id); }
  int CheckFrameBufferStatus() { return glCheckFramebufferStatusEXT(GL_FRAMEBUFFER); }

  void Screenshot(Texture *out) { ScreenshotBox(out, Box(parent->gl_w, parent->gl_h), Texture::Flag::FlipY); }
  void ScreenshotBox(Texture *out, const Box &b, int flag) {
    ClearDeferred();
    out->Resize(b.w, b.h, Texture::preferred_pf, Texture::Flag::CreateBuf);
    auto pixels = out->NewBuffer();
    glReadPixels(b.x, b.y, b.w, b.h, out->GLPixelType(), out->GLBufferType(), pixels.get());
    out->UpdateBuffer(pixels.get(), point(b.w, b.h), out->pf, b.w * 4, flag);
    GDDebug("ScreenshotBox");
  }

  void DumpTexture(Texture *out, unsigned tex_id) {
#if !defined(LFL_MOBILE) && !defined(LFL_EMSCRIPTEN)
    if (tex_id) {
      GLint gl_tt = out->GLTexType(), tex_w = 0, tex_h = 0;
      BindTexture(gl_tt, tex_id);
      glGetTexLevelParameteriv(gl_tt, 0, GL_TEXTURE_WIDTH, &tex_w);
      glGetTexLevelParameteriv(gl_tt, 0, GL_TEXTURE_WIDTH, &tex_h);
      CHECK_GT((out->width = tex_w), 0);
      CHECK_GT((out->height = tex_h), 0);
    }
    out->RenewBuffer();
    glGetTexImage(out->GLTexType(), 0, out->GLPixelType(), out->GLBufferType(), out->buf);
#endif
  }

  const char *GetString(int t) { return SpellNull(MakeSigned(glGetString(t))); }
  const char *GetGLEWString(int t) {
#ifdef LFL_GLEW
    return SpellNull(MakeSigned(glewGetString(t)));
#else
    return "";
#endif
  }

  int CreateProgram() { int p = glCreateProgram(); GDDebug("CreateProgram ", p); return p; }
  void DelProgram(int p) { glDeleteProgram(p); GDDebug("DelProgram ", p); }
  int CreateShader(int t) { return glCreateShader(t); }
  void CompileShader(int shader, vector<const char*> source) {
    GLint success = 0;
    glShaderSource(shader, source.size(), &source[0], nullptr);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (success == GL_FALSE) {
      int l = 0;
      char buf[1024] = { 0 };
      glGetShaderInfoLog(shader, sizeof(buf), &l, buf);
      INFO("CompileShader failed: ", buf);
      INFO("Source:");
      int line_no = 1;
      for (auto &s : source) {
        StringLineIter lines(s, StringLineIter::Flag::BlankLines);
        for (const char *line = lines.Next(); line; line = lines.Next())
          INFO(line_no++, ": ", string(line, lines.CurrentLength()));
      }
    }
  }
  void AttachShader(int prog, int shader) { glAttachShader(prog, shader); }
  void DelShader(int shader) { glDeleteShader(shader); }
  void BindAttribLocation(int prog, int loc, const string &name) { glBindAttribLocation(prog, loc, name.c_str()); }
  bool LinkProgram(int prog) {
    char buf[1024] = { 0 }; int l = 0;
    glLinkProgram(prog);
    glGetProgramInfoLog(prog, sizeof(buf), &l, buf);
    if (l) INFO(buf);
    GLint link_status;
    glGetProgramiv(prog, GL_LINK_STATUS, &link_status);
    return link_status == GL_TRUE;
  }
  void GetProgramiv(int p, int t, int *out) { glGetProgramiv(p, t, out); }
  void GetIntegerv(int t, int *out) { glGetIntegerv(t, out); }
  int GetAttribLocation(int prog, const string &name) { return glGetAttribLocation(prog, name.c_str()); }
  int GetUniformLocation(int prog, const string &name) { return glGetUniformLocation(prog, name.c_str()); }
  void Uniform1i(int u, int v) { glUniform1i(u, v); }
  void Uniform1f(int u, float v) { glUniform1f(u, v); }
  void Uniform2f(int u, float v1, float v2) { glUniform2f(u, v1, v2); }
  void Uniform3f(int u, float v1, float v2, float v3) { glUniform3f(u, v1, v2, v3); }
  void Uniform4f(int u, float v1, float v2, float v3, float v4) { glUniform4f(u, v1, v2, v3, v4); }
  void Uniform3fv(int u, int n, const float *v) { glUniform3fv(u, n, v); }

  void InitDefaultLight() {
    float pos[] = { -.5,1,-.3f,0 }, grey20[] = { .2f,.2f,.2f,1 }, white[] = { 1,1,1,1 }, black[] = { 0,0,0,1 };
    EnableLight(0);
    Light(0, GraphicsDevice::Position, pos);
    Light(0, GraphicsDevice::Ambient, grey20);
    Light(0, GraphicsDevice::Diffuse, white);
    Light(0, GraphicsDevice::Specular, white);
    Material(GraphicsDevice::Emission, black);
    Material(GraphicsDevice::Specular, grey20);
  }

  void LogVersion() {
    const char *glslver = GetString(ShaderVersion);
    const char *glexts = GetString(Extensions);
    INFO("OpenGL Version: ", GetString(Version));
    INFO("OpenGL Vendor: ", GetString(Vendor));
    INFO("GLEW Version: ", GetGLEWString(GLEWVersion));
    INFO("GL_SHADING_LANGUAGE_VERSION: ", glslver);
    INFO("GL_EXTENSIONS: ", glexts);

#ifdef LFL_MOBILE
    have_cubemap = strstr(glexts, "GL_EXT_texture_cube_map") != 0;
#else
    have_cubemap = strstr(glexts, "GL_ARB_texture_cube_map") != 0;
#endif
    int depth_bits = 0;
    GetIntegerv(DepthBits, &depth_bits);
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
    case RGBA:   case RGB32: return D3DFMT_A8R8G8B8;
    case RGB24:              return D3DFMT_R8G8B8;
    case BGRA:   case BGR32: return D3DFMT_A8B8G8R8;
    case BGR24:              return D3DFMT_B8G8R8;
    case GRAYA8:             return D3DFMT_A8L8;
    case GRAY8:              return D3DFMT_L8;
    default:                 return -1;
    }
  }

  int GetDepth(int id) {
    switch (id) {
    case _16: return 0; // GL_DEPTH_COMPONENT16;
    default: return 0;
    }
  }

  int GetCubeMap(int id) {
    switch (id) {
    case NX: return D3DCUBEMAP_FACE_NEGATIVE_X;
    case PX: return D3DCUBEMAP_FACE_POSITIVE_X;
    case NY: return D3DCUBEMAP_FACE_NEGATIVE_Y;
    case PY: return D3DCUBEMAP_FACE_POSITIVE_Y;
    case NZ: return D3DCUBEMAP_FACE_NEGATIVE_Z;
    case PZ: return D3DCUBEMAP_FACE_POSITIVE_Z;
    default: return 0; //  GL_TEXTURE_2D;
    } 
  }
};
#endif // 0

unique_ptr<GraphicsDevice> CreateDirectXGraphicsDevice(Window *w, Shaders *s) {
  unique_ptr<GraphicsDevice> gd;
  return gd;
}

}; // namespace LFL
