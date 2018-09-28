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

#if defined(LFL_GLEW)
#define glewGetContext() static_cast<GLEWContext*>(glew_context)
#include <GL/glew.h>
#ifdef LFL_WINDOWS
#include <GL/wglew.h>
#endif
#endif

#if defined(LFL_IOS)
#ifdef LFL_GLES2
#include <OpenGLES/ES2/gl.h>
#include <OpenGLES/ES2/glext.h>
#endif
#include <OpenGLES/ES1/gl.h>
#include <OpenGLES/ES1/glext.h>
#define glOrtho glOrthof 
#define glFrustum glFrustumf 

#elif defined(LFL_ANDROID) || defined(LFL_EMSCRIPTEN)
#ifdef LFL_GLES2
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#endif
#include <GLES/gl.h>
#include <GLES/glext.h>
#define glOrtho glOrthof 
#define glFrustum glFrustumf 

#elif defined(LFL_QTGL)
#include <QtOpenGL>

#elif defined(__APPLE__)
#include <OpenGL/glu.h>

#else
#include <GL/glu.h>
#endif

#if defined(LFL_MOBILE) || defined(LFL_QTGL) || defined(LFL_EMSCRIPTEN)
#define glGenRenderbuffersEXT(a,b) glGenRenderbuffers(a,b)
#define glBindRenderbufferEXT(a,b) glBindRenderbuffer(a,b)
#define glDeleteRenderbuffersEXT(a,b) glDeleteRenderbuffers(a,b)
#define glRenderbufferStorageEXT(a,b,c,d) glRenderbufferStorage(a,b,c,d)
#define glGenFramebuffersEXT(a,b) glGenFramebuffers(a,b)
#define glBindFramebufferEXT(a,b) glBindFramebuffer(a,b)
#define glDeleteFramebuffersEXT(a,b) glDeleteFramebuffers(a,b)
#define glFramebufferTexture2DEXT(a,b,c,d,e) glFramebufferTexture2D(a,b,c,d,e)
#define glFramebufferRenderbufferEXT(a,b,c,d) glFramebufferRenderbuffer(a,b,c,d)
#define glCheckFramebufferStatusEXT(a) glCheckFramebufferStatus(a)
#endif

#define LFL_GLSL_SHADERS
#if defined(LFL_MOBILE) && !defined(LFL_GLES2)
#undef LFL_GLSL_SHADERS
#endif

#include "gd_common.h"

namespace LFL {
struct OpenGLGraphicsDeviceConstants : public GraphicsDevice::Constants {
  OpenGLGraphicsDeviceConstants() {
    Float = GL_FLOAT;
    Points = GL_POINTS;
    Lines = GL_LINES;
    LineLoop = GL_LINE_LOOP;
    Triangles = GL_TRIANGLES;
    TriangleStrip = GL_TRIANGLE_STRIP;
    Texture2D = GL_TEXTURE_2D;
    TextureCubeMap = GL_TEXTURE_CUBE_MAP;
    UnsignedByte = GL_UNSIGNED_BYTE;
    UnsignedInt = GL_UNSIGNED_INT;
    FramebufferComplete = GL_FRAMEBUFFER_COMPLETE;
    Ambient = GL_AMBIENT;
    Diffuse = GL_DIFFUSE;
    Specular = GL_SPECULAR;
    Position = GL_POSITION;
    Emission = GL_EMISSION;
    AmbientAndDiffuse = GL_AMBIENT_AND_DIFFUSE;
    One = GL_ONE;
    SrcAlpha = GL_SRC_ALPHA;
    OneMinusSrcAlpha = GL_ONE_MINUS_SRC_ALPHA;
    OneMinusDstColor = GL_ONE_MINUS_DST_COLOR;
    TextureWrapS = GL_TEXTURE_WRAP_S;
    TextureWrapT = GL_TEXTURE_WRAP_T;
    ClampToEdge = GL_CLAMP_TO_EDGE;
    VertexShader = GL_VERTEX_SHADER;
    FragmentShader = GL_FRAGMENT_SHADER;
    ShaderVersion = GL_SHADING_LANGUAGE_VERSION;
    Extensions = GL_EXTENSIONS;
#ifdef LFL_GLEW
    GLEWVersion = GLEW_VERSION;
#else
    GLEWVersion = 0;
#endif
    Version = GL_VERSION;
    Vendor = GL_VENDOR;
    DepthBits = GL_DEPTH_BITS;
    ScissorTest = GL_SCISSOR_TEST;
    ActiveUniforms = GL_ACTIVE_UNIFORMS;
    ActiveAttributes = GL_ACTIVE_ATTRIBUTES;
    MaxVertexAttributes = GL_MAX_VERTEX_ATTRIBS;
    MaxViewportDims = GL_MAX_VIEWPORT_DIMS;
    ViewportBox = GL_VIEWPORT;
    ScissorBox = GL_SCISSOR_BOX;
#if defined(LFL_MOBILE) || defined(LFL_EMSCRIPTEN)
    Fill = 0;
    Line = 0;
    Point = 0;
    Polygon = 0;
    GLPreferredBuffer = GL_UNSIGNED_BYTE;
    GLInternalFormat = GL_RGBA;
    MaxVertexUniformComp = 0;
    FramebufferBinding = GL_FRAMEBUFFER_BINDING_OES;
    FramebufferUndefined = 0;
#else                                         
    Fill = GL_FILL;
    Line = GL_LINE;
    Point = GL_POINT;
    Polygon = GL_POLYGON;
#ifdef __APPLE__                               
    GLPreferredBuffer = GL_UNSIGNED_INT_8_8_8_8_REV;
#else                                          
    GLPreferredBuffer = GL_UNSIGNED_BYTE;
#endif                                         
    GLInternalFormat = GL_RGBA;
    MaxVertexUniformComp = GL_MAX_VERTEX_UNIFORM_COMPONENTS;
    FramebufferBinding = GL_FRAMEBUFFER_BINDING;
    FramebufferUndefined = GL_FRAMEBUFFER_UNDEFINED;
#endif
  }
};

#ifdef LFL_GLES1
#ifndef LFL_QTGL
struct OpenGLES1 : public GraphicsDevice {
#else
struct OpenGLES1 : public GraphicsDevice, public QOpenGLFunctions {
#endif
#include "core/app/gl/device/opengl_common.h"
  int target_matrix=-1;
  OpenGLES1(Window *P, LFL::Shaders *S) : GraphicsDevice(GraphicsDevice::Type::OPENGL, OpenGLGraphicsDeviceConstants(), P, 1, S) {
    default_color.push_back(Color(1.0, 1.0, 1.0, 1.0));
  }

  void Init(AssetLoading *loader, const Box &b) {
    done_init = true;
    GDDebug("Init");
    shader = &shaders->shader_default; 
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glEnableClientState(GL_VERTEX_ARRAY);
#ifndef LFL_MOBILE
    float black[]={0,0,0,1};
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);
    glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, 1);
    // glLightModelf(GL_LIGHT_MODEL_COLOR_CONTROL, GL_SEPARATE_SPECULAR_COLOR);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
#endif
    ViewPort(b);
    DrawMode(default_draw_mode);
    InitDefaultLight();
    INFO("OpenGLES1::Init width=", b.w, ", height=", b.h, ", shaders=", GetShaderSupport());
    LogVersion();
  }

  void MarkDirty() {}

  bool GetShaderSupport() {
#ifdef LFL_MOBILE
    return false;
#endif
    const char *ver = GetString(GL_VERSION);
    return ver && *ver == '2';
  }

  void  EnableTexture() {  glEnable(GL_TEXTURE_2D);  glEnableClientState(GL_TEXTURE_COORD_ARRAY); GDDebug("Texture=1"); }
  void DisableTexture() { glDisable(GL_TEXTURE_2D); glDisableClientState(GL_TEXTURE_COORD_ARRAY); GDDebug("Texture=0"); }
  void  EnableLighting() {  glEnable(GL_LIGHTING);  glEnable(GL_COLOR_MATERIAL); GDDebug("Lighting=1"); }
  void DisableLighting() { glDisable(GL_LIGHTING); glDisable(GL_COLOR_MATERIAL); GDDebug("Lighting=0"); }
  void  EnableNormals() {  glEnableClientState(GL_NORMAL_ARRAY); GDDebug("Normals=1"); }
  void DisableNormals() { glDisableClientState(GL_NORMAL_ARRAY); GDDebug("Normals=0"); }
  void  EnableVertexColor() {  glEnableClientState(GL_COLOR_ARRAY); GDDebug("VertexColor=1"); }
  void DisableVertexColor() { glDisableClientState(GL_COLOR_ARRAY); GDDebug("VertexColor=0"); }
  //void TextureEnvReplace()  { glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);  GDDebug("TextureEnv=R"); }
  //void TextureEnvModulate() { glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE); GDDebug("TextureEnv=M"); }
  void  EnableLight(int n) { if (n)  glEnable(GL_LIGHT1); else  glEnable(GL_LIGHT0); GDDebug("Light", n, "=1"); }
  void DisableLight(int n) { if (n) glDisable(GL_LIGHT1); else glDisable(GL_LIGHT0); GDDebug("Light", n, "=0"); }
  void DisableCubeMap() { glDisable(GL_TEXTURE_CUBE_MAP); DisableTextureGen(); GDDebug("CubeMap=", 0); }

  void MatrixProjection() { target_matrix=2; glMatrixMode(GL_PROJECTION); }
  void MatrixModelview() { target_matrix=1; glMatrixMode(GL_MODELVIEW); }
  void LoadIdentity() { glLoadIdentity(); }
  void PushMatrix() { glPushMatrix(); }
  void PopMatrix() { glPopMatrix(); }
  void GetMatrix(m44 *out) { glGetFloatv(target_matrix == 2 ? GL_PROJECTION_MATRIX : GL_MODELVIEW_MATRIX, &(*out)[0][0]); }
  void PrintMatrix() {}
  void Scalef(float x, float y, float z) { glScalef(x, y, z); }
  void Rotatef(float angle, float x, float y, float z) { glRotatef(angle, x, y, z); }
  void Ortho(float l, float r, float b, float t, float nv, float fv) { glOrtho(l,r, b,t, nv,fv); }
  void Frustum(float l, float r, float b, float t, float nv, float fv) { glFrustum(l,r, b,t, nv,fv); }
  void Mult(const float *m) { glMultMatrixf(m); }
  void Translate(float x, float y, float z) { glTranslatef(x, y, z); }

  void Color4f(float r, float g, float b, float a) { default_color.back() = Color(r,g,b,a); glColor4f(r,g,b,a); }
  void Light(int n, int t, float *color) { glLightfv(((n) ? GL_LIGHT1 : GL_LIGHT0), t, color); }
  void Material(int t, float *color) { glMaterialfv(GL_FRONT_AND_BACK, t, color); }

  void BindTexture(const TextureRef &n) { EnableTexture(); glBindTexture(n.t, n); GDDebug("BindTexture=", t, ",", n); }
  void BindCubeMap(const TextureRef &n) {  glEnable(GL_TEXTURE_CUBE_MAP); glBindTexture(GL_TEXTURE_CUBE_MAP, n); GDDebug("CubeMap=", n); }

  void ActiveTexture(int n) {
    glClientActiveTexture(GL_TEXTURE0 + n);
    glActiveTexture(GL_TEXTURE0 + n);
    // glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_COMBINE);
    // glTexEnvf (GL_TEXTURE_ENV, GL_COMBINE_RGB_EXT, GL_MODULATE);
    GDDebug("ActiveTexture=", n);
  }

#ifdef LFL_MOBILE
  void TextureGenLinear() {}
  void TextureGenReflection() {}
  void DisableTextureGen() {}
#else
  void  EnableTextureGen() {  glEnable(GL_TEXTURE_GEN_S);  glEnable(GL_TEXTURE_GEN_T);  glEnable(GL_TEXTURE_GEN_R); GDDebug("TextureGen=1"); }
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

  void UseShader(Shader *S) {
    shader = X_or_Y(S, &shaders->shader_default); 
    glUseProgram(shader->ID);
    GDDebug("Shader=", shader->name);
  }

  bool VertexPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool ud, int) { glVertexPointer  (m, t, w, verts + o/sizeof(float)); GDDebug("VertexPointer"); return true; }
  void TexPointer   (int m, int t, int w, int o, float *tex,   int l, int *out, bool ud)      { glTexCoordPointer(m, t, w, tex   + o/sizeof(float)); GDDebug("TexPointer"); }
  void ColorPointer (int m, int t, int w, int o, float *verts, int l, int *out, bool ud)      { glColorPointer   (m, t, w, verts + o/sizeof(float)); GDDebug("ColorPointer"); }
  void NormalPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool ud)      { glNormalPointer  (   t, w, verts + o/sizeof(float)); GDDebug("NormalPointer"); }

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
};
#endif // LFL_GLES1

#ifdef LFL_GLES2
#ifndef LFL_QTGL
struct OpenGLES2 : public ShaderBasedGraphicsDevice {
#else
struct OpenGLES2 : public ShaderBasedGraphicsDevice, public QOpenGLFunctions {
#endif
#include "core/app/gl/device/opengl_common.h"

  OpenGLES2(Window *P, LFL::Shaders *S) :
    ShaderBasedGraphicsDevice(GraphicsDevice::Type::OPENGL, OpenGLGraphicsDeviceConstants(), P, 2, S) {}

  void Init(AssetLoading *loader, const Box &b) {
    done_init = true;
    GDDebug("Init");
    deferred.prim_type = deferred.vertex_size = deferred.vertexbuffer_len = deferred.draw_calls = 0;
    deferred.vertexbuffer = -1;
    modelview_matrix.clear();
    modelview_matrix.push_back(m44::Identity());
    projection_matrix.clear();
    projection_matrix.push_back(m44::Identity());
    default_color.clear();
    default_color.push_back(Color(1.0, 1.0, 1.0, 1.0));
    scissor_stack.clear();
    scissor_stack.push_back(vector<Box>());
    if (vertex_shader.empty()) vertex_shader = loader->FileContents("default.vert");
    if ( pixel_shader.empty()) pixel_shader  = loader->FileContents("default.frag");
    LogVersion();
    Shader::Create(parent->parent, "app",          vertex_shader, pixel_shader, ShaderDefines(1,0,1,0), &shaders->shader_default);
    Shader::Create(parent->parent, "app_cubemap",  vertex_shader, pixel_shader, ShaderDefines(1,0,0,1), &shaders->shader_cubemap);
    Shader::Create(parent->parent, "app_normals",  vertex_shader, pixel_shader, ShaderDefines(0,1,1,0), &shaders->shader_normals);
    Shader::Create(parent->parent, "app_cubenorm", vertex_shader, pixel_shader, ShaderDefines(0,1,0,1), &shaders->shader_cubenorm);
    UseShader((shader = 0));
    VertexPointer(0, 0, 0, 0, NULL, deferred.vertexbuffer_size, NULL, true, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    ViewPort(b);
    DrawMode(default_draw_mode);
    InitDefaultLight();
    INFO("OpenGLES2::Init width=", b.w, ", height=", b.h);
  }

  void BindTexture(const TextureRef &n) {
    if (!Changed(&bound_texture, BoundTexture{ n.t, n, 0 })) return;
    ClearDeferred();
    if (!texture_on) EnableTexture();
    glActiveTexture(GL_TEXTURE0); 
    glBindTexture(n.t, n);
    glUniform1i(shader->uniform_tex, 0);
    GDDebug("BindTexture=", t, ",", n);
  }

  void BindCubeMap(const TextureRef &n) { if (Changed(&cubemap_on, true)) { UpdateShader(); } glUniform1i(shader->uniform_cubetex, 0); glBindTexture(GL_TEXTURE_CUBE_MAP, n); GDDebug("CubeMap=", n); }
  void ActiveTexture(int n) { if (Changed(&bound_texture.l, n)) { ClearDeferred(); glActiveTexture(n ? GL_TEXTURE1 : GL_TEXTURE0); } GDDebug("ActivteTexture=", n); }

  void TextureGenLinear() {}
  void TextureGenReflection() {}

  void UseShader(Shader *S) {
    if (!S) return UpdateShader();
    if (shader == S || !S->ID) return;
    ClearDeferred();
    glUseProgram((shader = S)->ID);
    GDDebug("Shader=", shader->name);
    dirty_matrix = dirty_color = true;
    for (int i=0, s; i<shader->MaxVertexAttrib; i++) {
      if ((s = shader->unused_attrib_slot[i]) < 0) break;
      glDisableVertexAttribArray(s);
    }
    UpdateVertex();
    UpdateNormals();
    UpdateColorVerts();
    UpdateTexture();
  }

  bool VertexPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty, int prim_type) {
    bool defer = !out, input_dirty = dirty;
    if (defer) { out = &deferred.vertexbuffer; deferred.vertexbuffer_appended = l; }
    bool first = (*out == -1), changed = first || *out != bound_vertexbuffer;
    if (first) { glGenBuffers(1, MakeUnsigned(out)); dirty = true; }
    if (changed) {
      CHECK(shader);
      CHECK((!o && !w) || o < w);
      ClearDeferred();
      if (defer) deferred.prim_type = prim_type;
      glBindBuffer(GL_ARRAY_BUFFER, (bound_vertexbuffer = *out));
      vertex_attr = { m, t, w, o };
      UpdateVertex();
    } else if (defer) {
      if (deferred.prim_type != prim_type) { ClearDeferred(); deferred.prim_type = prim_type; changed = true; }
      if (Changed(&vertex_attr, VertexAttribute{ m, t, w, o })) { ClearDeferred(); UpdateVertex(); changed = true; }
    }
    if (first || dirty) {
      int vbo_offset = (defer && !first) ? AddDeferredVertexSpace(l) : 0;
      if (first) glBufferData(GL_ARRAY_BUFFER, l, verts, input_dirty ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
      else       glBufferSubData(GL_ARRAY_BUFFER, vbo_offset, l, verts);
    }
    GDDebug("VertexPointer changed=", changed);
    return changed;
  }

  void TexPointer(int m, int t, int w, int o, float *tex, int l, int *out, bool dirty) {
    if (!out) out = &deferred.vertexbuffer;
    CHECK(*out == bound_vertexbuffer);
    CHECK_LT(o, w);
    tex_attr = VertexAttribute{ m, t, w, o };
    if (shader->slot_tex >= 0) VertexAttribPointer(shader->slot_tex, tex_attr);
    GDDebug("TexPointer");
  }

  void ColorPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {
    if (!out) out = &deferred.vertexbuffer;
    CHECK(*out == bound_vertexbuffer);
    CHECK_LT(o, w);
    color_attr = VertexAttribute{ m, t, w, o };
    if (shader->slot_color >= 0) VertexAttribPointer(shader->slot_color, color_attr);
    GDDebug("ColorPointer");
  }

  void NormalPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {
    if (!out) out = &deferred.vertexbuffer;
    CHECK(*out == bound_vertexbuffer);
    CHECK_LT(o, w);
    normal_attr = VertexAttribute{ m, t, w, o };
    if (shader->slot_normal >= 0) VertexAttribPointer(shader->slot_normal, normal_attr);
    GDDebug("NormalPointer");
  }

  void DrawElements(int pt, int np, int it, int o, void *index, int l, int *out, bool dirty) {
    bool input_dirty = dirty;
    if (*out == -1) { glGenBuffers(1, MakeUnsigned(out)); dirty = true; }
    if (*out != bound_indexbuffer) { 
      bound_indexbuffer = *out;
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *out);
    }
    if (dirty) glBufferData(GL_ELEMENT_ARRAY_BUFFER, l, index, input_dirty ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);

    GDDebug("DrawElements-Pre(", pt, ", ", np, ", ", it, ", ", o, ", ", index, ", ", l, ", ", dirty, ")");
    PushDirtyState();
    glDrawElements(pt, np, it, Void(long(o)));
    GDDebug("DrawElements-Post(", pt, ", ", np, ", ", it, ", ", o, ", ", index, ", ", l, ", ", dirty, ")");
  }

  void DrawArrays(int type, int o, int n) {
    GDDebug("DrawArrays-Pre(", type, ", ", o, ", ", n, ")");
    PushDirtyState();
    glDrawArrays(type, o, n);
    GDDebug("DrawArrays-Post(", type, ", ", o, ", ", n, ")");
  }

  void DeferDrawArrays(int type, int o, int n) {
    CHECK_EQ(deferred.prim_type, type);
    bool first = !(deferred.vertexbuffer_len - deferred.vertexbuffer_appended);
    if (first) { PushDirtyState(); deferred.vertex_size = vertex_attr.w; }
#if 1
    if (type == c.Triangles && o == 0 && n == deferred.vertexbuffer_appended / deferred.vertex_size) {
      deferred.draw_calls++;
    } else  {
      glDrawArrays(type, o, n);
      deferred.vertexbuffer_len = 0;
    }
#else
    if (o || type == LineLoop) {
      glDrawArrays(type, deferred.vertexbuffer_len /*+ deferred.last_append_len*/, n);
    } else if (type == Points || type == Lines || type == Triangles) {

    } else if (type == TriangleStrip) {
    } else FATAL("unknown type ", type);
#endif
    GDDebug("DeferDrawArrays(", type, ", ", o, ", ", n, ") deferred ", deferred.draw_calls);
  }

  void ClearDeferred() {
    if (!deferred.vertexbuffer_len) return;
    if (dont_clear_deferred) { GDDebug("Suppressed ClearDeferred"); return; }
    // INFOf("merged %d %d (type = %d)\n", deferred.draw_calls, deferred.vertexbuffer_len / deferred.vertex_size, deferred.prim_type);
    glDrawArrays(deferred.prim_type, 0, deferred.vertexbuffer_len / deferred.vertex_size);
    deferred.vertexbuffer_len = deferred.draw_calls = 0;
    GDDebug("ClearDeferred");
  }

  void VertexAttribPointer(int slot, const VertexAttribute &attr) { 
    glVertexAttribPointer(slot, attr.m, attr.t, GL_FALSE, attr.w, Void(long(attr.o)));
  }

  void UpdateVertex() {
    glEnableVertexAttribArray(shader->slot_position);
    VertexAttribPointer(shader->slot_position, vertex_attr);
  }

  void UpdateNormals() {
    bool supports = shader->slot_normal >= 0;
    if (supports) {
      if (normals_on) {  glEnableVertexAttribArray(shader->slot_normal); VertexAttribPointer(shader->slot_normal, normal_attr); }
      else            { glDisableVertexAttribArray(shader->slot_normal); }
    } else if (normals_on) ERROR("shader doesnt support normals");
  }

  void UpdateColorVerts() {
    bool supports = shader->slot_color >= 0;
    glUniform1i(shader->uniform_coloron, colorverts_on && supports);
    if (supports) {
      if (colorverts_on) {  glEnableVertexAttribArray(shader->slot_color); VertexAttribPointer(shader->slot_color, color_attr); }
      else               { glDisableVertexAttribArray(shader->slot_color); }
    } else if (colorverts_on) ERROR("shader doesnt support vertex color");
  }

  void UpdateTexture() {
    bool supports = shader->slot_tex >= 0;
    glUniform1i(shader->uniform_texon, texture_on && supports);
    if (supports) {
      if (texture_on) {  glEnableVertexAttribArray(shader->slot_tex); VertexAttribPointer(shader->slot_tex, tex_attr); }
      else            { glDisableVertexAttribArray(shader->slot_tex); }
    } else if (texture_on && FLAGS_gd_debug) ERROR("shader doesnt support texture");
  }
};
#endif // LFL_GLES2

unique_ptr<GraphicsDevice> GraphicsDevice::Create(Window *w, Shaders *s) {
  unique_ptr<GraphicsDevice> gd;
#ifdef LFL_GLEW
  glewExperimental = GL_TRUE;
#ifdef GLEW_MX
  auto glew_context = new GLEWContext();
#endif
  ONCE({
    GLenum glew_err;
    if ((glew_err = glewInit()) != GLEW_OK) return ERRORv(nullptr, "glewInit: ", glewGetErrorString(glew_err));
  });
#endif

  int opengles_version = 1 + (glGetString(GL_SHADING_LANGUAGE_VERSION) != nullptr);

#ifdef LFL_GLES2
  if (opengles_version == 2) gd = make_unique<OpenGLES2>(w, s);
#ifdef LFL_GLES1
  else
#endif
#endif

#ifdef LFL_GLES1  
  if (opengles_version == 1) gd = make_unique<OpenGLES1>(w, s);
#endif

#ifdef LFL_GLEW
#ifdef GLEW_MX
  gd->glew_context = glew_context;
#endif
  gd->have_framebuffer = GLEW_EXT_framebuffer_object;
#endif

  return gd;
}

}; // namespace LFL
