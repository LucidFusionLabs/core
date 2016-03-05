/*
 * $Id: video.cpp 1336 2014-12-08 09:29:59Z justin $
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

#include "core/app/app.h"

#if defined(LFL_GLEW)
#define glewGetContext() static_cast<GLEWContext*>(screen->glew_context)
#include <GL/glew.h>
#ifdef WIN32
#include <GL/wglew.h>
#endif
#endif

#if defined(LFL_IPHONE)
#ifdef LFL_GLES2
#include <OpenGLES/ES2/gl.h>
#include <OpenGLES/ES2/glext.h>
#endif
#include <OpenGLES/ES1/gl.h>
#include <OpenGLES/ES1/glext.h>
#define glOrtho glOrthof 
#define glFrustum glFrustumf 

#elif defined(LFL_ANDROID)
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

#if defined(LFL_MOBILE) || defined(LFL_QTGL)
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

#ifdef LFL_GDDEBUG
#define GDDebug(...) { \
  if (screen) screen->gd->CheckForError(__FILE__, __LINE__); \
  if (FLAGS_gd_debug) printf("%s\n", StrCat(__VA_ARGS__).c_str()); }
#else 
#define GDDebug(...)
#endif

namespace LFL {
const int GraphicsDevice::Float                = GL_FLOAT;
const int GraphicsDevice::Points               = GL_POINTS;
const int GraphicsDevice::Lines                = GL_LINES;
const int GraphicsDevice::LineLoop             = GL_LINE_LOOP;
const int GraphicsDevice::Triangles            = GL_TRIANGLES;
const int GraphicsDevice::TriangleStrip        = GL_TRIANGLE_STRIP;
const int GraphicsDevice::Texture2D            = GL_TEXTURE_2D;
const int GraphicsDevice::TextureCubeMap       = GL_TEXTURE_CUBE_MAP;
const int GraphicsDevice::UnsignedByte         = GL_UNSIGNED_BYTE;
const int GraphicsDevice::UnsignedInt          = GL_UNSIGNED_INT;
const int GraphicsDevice::FramebufferComplete  = GL_FRAMEBUFFER_COMPLETE;
const int GraphicsDevice::Ambient              = GL_AMBIENT;
const int GraphicsDevice::Diffuse              = GL_DIFFUSE;
const int GraphicsDevice::Specular             = GL_SPECULAR;
const int GraphicsDevice::Position             = GL_POSITION;
const int GraphicsDevice::Emission             = GL_EMISSION;
const int GraphicsDevice::One                  = GL_ONE;
const int GraphicsDevice::SrcAlpha             = GL_SRC_ALPHA;
const int GraphicsDevice::OneMinusSrcAlpha     = GL_ONE_MINUS_SRC_ALPHA;
const int GraphicsDevice::OneMinusDstColor     = GL_ONE_MINUS_DST_COLOR;
const int GraphicsDevice::TextureWrapS         = GL_TEXTURE_WRAP_S;
const int GraphicsDevice::TextureWrapT         = GL_TEXTURE_WRAP_T;
const int GraphicsDevice::ClampToEdge          = GL_CLAMP_TO_EDGE;
const int GraphicsDevice::VertexShader         = GL_VERTEX_SHADER;
const int GraphicsDevice::FragmentShader       = GL_FRAGMENT_SHADER;
const int GraphicsDevice::ShaderVersion        = GL_SHADING_LANGUAGE_VERSION;
const int GraphicsDevice::Extensions           = GL_EXTENSIONS;
#ifdef LFL_GLEW
const int GraphicsDevice::GLEWVersion          = GLEW_VERSION;
#else
const int GraphicsDevice::GLEWVersion          = 0;
#endif
const int GraphicsDevice::Version              = GL_VERSION;
const int GraphicsDevice::Vendor               = GL_VENDOR;
const int GraphicsDevice::DepthBits            = GL_DEPTH_BITS;
const int GraphicsDevice::ActiveUniforms       = GL_ACTIVE_UNIFORMS;
const int GraphicsDevice::ActiveAttributes     = GL_ACTIVE_ATTRIBUTES;
const int GraphicsDevice::MaxVertexAttributes  = GL_MAX_VERTEX_ATTRIBS;
const int GraphicsDevice::MaxVertexUniformComp = GL_MAX_VERTEX_UNIFORM_COMPONENTS;
#ifdef LFL_MOBILE
const int GraphicsDevice::Fill                = 0;
const int GraphicsDevice::Line                = 0;
const int GraphicsDevice::Point               = 0;
const int GraphicsDevice::Polygon             = 0;
const int GraphicsDevice::GLPreferredBuffer   = GL_UNSIGNED_BYTE;
const int GraphicsDevice::GLInternalFormat    = GL_RGBA;
#else                                         
const int GraphicsDevice::Fill                = GL_FILL;
const int GraphicsDevice::Line                = GL_LINE;
const int GraphicsDevice::Point               = GL_POINT;
const int GraphicsDevice::Polygon             = GL_POLYGON;
#ifdef __APPLE__                              
const int GraphicsDevice::GLPreferredBuffer   = GL_UNSIGNED_INT_8_8_8_8_REV;
#else                                         
const int GraphicsDevice::GLPreferredBuffer   = GL_UNSIGNED_BYTE;
#endif                                        
const int GraphicsDevice::GLInternalFormat    = GL_RGBA;
#endif

int Depth::OpenGLID(int id) {
  switch(id) {
    case _16: return GL_DEPTH_COMPONENT16;
  } return 0;
}

int CubeMap::OpenGLID(int id) {
  switch (id) {
    case NX: return GL_TEXTURE_CUBE_MAP_NEGATIVE_X;    case PX: return GL_TEXTURE_CUBE_MAP_POSITIVE_X;
    case NY: return GL_TEXTURE_CUBE_MAP_NEGATIVE_Y;    case PY: return GL_TEXTURE_CUBE_MAP_POSITIVE_Y;
    case NZ: return GL_TEXTURE_CUBE_MAP_NEGATIVE_Z;    case PZ: return GL_TEXTURE_CUBE_MAP_POSITIVE_Z;
  } return GL_TEXTURE_2D;
}

int Pixel::OpenGLID(int p) {
  switch (p) {
    case RGBA:   case RGB32: return GL_RGBA;
    case RGB24:              return GL_RGB;
#ifndef LFL_MOBILE
    case BGRA:   case BGR32: return GL_BGRA;
    case BGR24:              return GL_BGR;
#endif
    case GRAYA8:             return GL_LUMINANCE_ALPHA;
    case GRAY8:              return GL_LUMINANCE;
    default:                 return -1;
  }
}

#ifndef LFL_QTGL
struct OpenGLES1 : public GraphicsDevice {
#else
struct OpenGLES1 : public GraphicsDevice, public QOpenGLFunctions {
#endif
#include "core/app/graphics/opengl_common.h"
  int target_matrix=-1;
  OpenGLES1() { default_color.push_back(Color(1.0, 1.0, 1.0, 1.0)); }
  void Init(const Box &b) {
    GDDebug("Init");
    shader = &app->shaders->shader_default; 
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
    INFO("OpenGLES1::Init width=", b.w, ", height=", b.h);
    LogVersion();
  }
  void UpdateColor() { const Color &c = default_color.back(); glColor4f(c.r(), c.g(), c.b(), c.a()); }
  bool ShaderSupport() {
#ifdef LFL_MOBILE
    return false;
#endif
    const char *ver = MakeUnbounded<char>(glGetString(GL_VERSION)).data();
    return ver && *ver == '2';
  }
  void  EnableTexture() {  glEnable(GL_TEXTURE_2D);  glEnableClientState(GL_TEXTURE_COORD_ARRAY); GDDebug("Texture=1"); }
  void DisableTexture() { glDisable(GL_TEXTURE_2D); glDisableClientState(GL_TEXTURE_COORD_ARRAY); GDDebug("Texture=0"); }
  void  EnableLighting() {  glEnable(GL_LIGHTING);  glEnable(GL_COLOR_MATERIAL); GDDebug("Lighting=1"); }
  void DisableLighting() { glDisable(GL_LIGHTING); glDisable(GL_COLOR_MATERIAL); GDDebug("Lighting=0"); }
  void  EnableVertexColor() {  glEnableClientState(GL_COLOR_ARRAY); GDDebug("VertexColor=1"); }
  void DisableVertexColor() { glDisableClientState(GL_COLOR_ARRAY); GDDebug("VertexColor=0"); }
  void  EnableNormals() {  glEnableClientState(GL_NORMAL_ARRAY); GDDebug("Normals=1"); }
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
  void DisableCubeMap()   { glDisable(GL_TEXTURE_CUBE_MAP); DisableTextureGen();                   GDDebug("CubeMap=", 0); }
  void BindCubeMap(int n) {  glEnable(GL_TEXTURE_CUBE_MAP); glBindTexture(GL_TEXTURE_CUBE_MAP, n); GDDebug("CubeMap=", n); }
  void ActiveTexture(int n) {
    glClientActiveTexture(GL_TEXTURE0 + n);
    glActiveTexture(GL_TEXTURE0 + n);
    // glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_COMBINE);
    // glTexEnvf (GL_TEXTURE_ENV, GL_COMBINE_RGB_EXT, GL_MODULATE);
    GDDebug("ActiveTexture=", n);
  }
  void BindTexture(int t, int n) { glBindTexture(t, n); GDDebug("BindTexture=", t, ",", n); }
  bool VertexPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool ud, int) { glVertexPointer  (m, t, w, verts + o/sizeof(float)); GDDebug("VertexPointer"); return true; }
  void TexPointer   (int m, int t, int w, int o, float *tex,   int l, int *out, bool ud)      { glTexCoordPointer(m, t, w, tex   + o/sizeof(float)); GDDebug("TexPointer"); }
  void ColorPointer (int m, int t, int w, int o, float *verts, int l, int *out, bool ud)      { glColorPointer   (m, t, w, verts + o/sizeof(float)); GDDebug("ColorPointer"); }
  void NormalPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool ud)      { glNormalPointer  (   t, w, verts + o/sizeof(float)); GDDebug("NormalPointer"); }
  void Color4f(float r, float g, float b, float a) { default_color.back() = Color(r,g,b,a); UpdateColor(); }
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
    shader = X_or_Y(S, &app->shaders->shader_default); 
    glUseProgram(shader->ID);
    GDDebug("Shader=", shader->name);
  }
};

#ifdef LFL_GLES2
#ifndef LFL_QTGL
struct OpenGLES2 : public GraphicsDevice {
#else
struct OpenGLES2 : public GraphicsDevice, public QOpenGLFunctions {
#endif
#include "core/app/graphics/opengl_common.h"
  struct BoundTexture {
    int t, n, l;
    bool operator!=(const BoundTexture &x) const { return t != x.t || n != x.n || l != x.l; };
  };
  struct VertexAttribute {
    int m, t, w, o;
    bool operator!=(const VertexAttribute &x) const { return m != x.m || t != x.t || w != x.w || o != x.o; };
  };
  struct Deferred {
    int prim_type=0, vertex_size=0, vertexbuffer=-1, vertexbuffer_size=1024*4*4, vertexbuffer_len=0, vertexbuffer_appended=0, draw_calls=0;
  };

  vector<m44> modelview_matrix, projection_matrix;
  bool dirty_matrix=1, dirty_color=1, cubemap_on=0, normals_on=0, texture_on=0, colorverts_on=0, lighting_on=0;
  int matrix_target=-1, bound_vertexbuffer=-1, bound_indexbuffer=-1;
  VertexAttribute vertex_attr, tex_attr, color_attr, normal_attr;
  BoundTexture bound_texture;
  LFL::Material material;
  LFL::Light light[4];
  Deferred deferred;

  void Init(const Box &b) {
    GDDebug("Init");
    memzero(vertex_attr); memzero(tex_attr); memzero(color_attr); memzero(normal_attr); memzero(bound_texture);
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
    if (vertex_shader.empty()) vertex_shader = Asset::FileContents("lfapp_vertex.glsl");
    if ( pixel_shader.empty()) pixel_shader  = Asset::FileContents("lfapp_pixel.glsl");
    Shader::Create("lfapp",          vertex_shader, pixel_shader, ShaderDefines(1,0,1,0), &app->shaders->shader_default);
    Shader::Create("lfapp_cubemap",  vertex_shader, pixel_shader, ShaderDefines(1,0,0,1), &app->shaders->shader_cubemap);
    Shader::Create("lfapp_normals",  vertex_shader, pixel_shader, ShaderDefines(0,1,1,0), &app->shaders->shader_normals);
    Shader::Create("lfapp_cubenorm", vertex_shader, pixel_shader, ShaderDefines(0,1,0,1), &app->shaders->shader_cubenorm);
    UseShader((shader = 0));
    VertexPointer(0, 0, 0, 0, NULL, deferred.vertexbuffer_size, NULL, true, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    ViewPort(b);
    DrawMode(default_draw_mode);
    InitDefaultLight();
    INFO("OpenGLES2::Init width=", b.w, ", height=", b.h);
    LogVersion();
  }

  bool ShaderSupport()      { return true; }
  void EnableLighting()     { lighting_on=1; GDDebug("Lighting=1"); }
  void DisableLighting()    { lighting_on=0; GDDebug("Lighting=0"); }
  void EnableTexture()      { if (Changed(&texture_on,    true))  { ClearDeferred(); UpdateTexture();    } GDDebug("Texture=1"); }
  void DisableTexture()     { if (Changed(&texture_on,    false)) { ClearDeferred(); UpdateTexture();    } GDDebug("Texture=0"); }
  void EnableVertexColor()  { if (Changed(&colorverts_on, true))  { ClearDeferred(); UpdateColorVerts(); } GDDebug("VertexColor=1"); }
  void DisableVertexColor() { if (Changed(&colorverts_on, false)) { ClearDeferred(); UpdateColorVerts(); } GDDebug("VertexColor=0"); }
  void EnableNormals()      { if (Changed(&normals_on,    true))  { UpdateShader();  UpdateNormals();    } GDDebug("Normals=1"); }
  void DisableNormals()     { if (Changed(&normals_on,    false)) { UpdateShader();  UpdateNormals();    } GDDebug("Normals=0"); }
  void DisableCubeMap()     { if (Changed(&cubemap_on,    false)) { UpdateShader(); }                                                                                 GDDebug("CubeMap=", 0); }
  void BindCubeMap(int n)   { if (Changed(&cubemap_on,    true))  { UpdateShader(); } glUniform1i(shader->uniform_cubetex, 0); glBindTexture(GL_TEXTURE_CUBE_MAP, n); GDDebug("CubeMap=", n); }
  void ActiveTexture(int n) { if (Changed(&bound_texture.l, n))   { ClearDeferred(); glActiveTexture(n ? GL_TEXTURE1 : GL_TEXTURE0); } GDDebug("ActivteTexture=", n); }
  void EnableLight(int n) {}
  void DisableLight(int n) {}
  void TextureGenLinear() {}
  void TextureGenReflection() {}
  void BindTexture(int t, int n) {
    if (!Changed(&bound_texture, BoundTexture{ t, n, 0 })) return;
    ClearDeferred();
    if (!texture_on) EnableTexture();
    glActiveTexture(GL_TEXTURE0); 
    glBindTexture(t, n);
    glUniform1i(shader->uniform_tex, 0);
    GDDebug("BindTexture=", t, ",", n);
  }
  void Color4f(float r, float g, float b, float a) {
    if (lighting_on) {
      float c[] = { r, g, b, a };
      Material(GL_AMBIENT_AND_DIFFUSE, c);
    } else if (Changed(&default_color.back(), Color(r,g,b,a))) UpdateColor();
  }
  void Material(int t, float *v) {
    if      (t == GL_AMBIENT)             material.ambient  = Color(v);
    else if (t == GL_DIFFUSE)             material.diffuse  = Color(v);
    else if (t == GL_SPECULAR)            material.specular = Color(v);
    else if (t == GL_EMISSION)            material.emissive = Color(v);
    else if (t == GL_AMBIENT_AND_DIFFUSE) material.ambient = material.diffuse = Color(v);
    UpdateMaterial();
  }
  void Light(int n, int t, float *v) {
    bool light_pos = 0, light_color = 0;
    if (n != 0) return ERROR("ignoring Light(", n, ")");

    if      (t == GL_POSITION) { light_pos=1;   light[n].pos = modelview_matrix.back().Transform(v4(v)); }
    else if (t == GL_AMBIENT)  { light_color=1; light[n].color.ambient  = Color(v); }
    else if (t == GL_DIFFUSE)  { light_color=1; light[n].color.diffuse  = Color(v); }
    else if (t == GL_SPECULAR) { light_color=1; light[n].color.specular = Color(v); }

    if (light_pos)   { shader->dirty_light_pos  [n] = app->shaders->shader_cubenorm.dirty_light_pos  [n] = app->shaders->shader_normals.dirty_light_pos  [n] = 1; }
    if (light_color) { shader->dirty_light_color[n] = app->shaders->shader_cubenorm.dirty_light_color[n] = app->shaders->shader_normals.dirty_light_color[n] = 1; }
  }

  void Scalef(float x, float y, float z) {
    m44 &m = TargetMatrix()->back();
    m[0].x *= x; m[0].y *= x; m[0].z *= x;
    m[1].x *= y; m[1].y *= y; m[1].z *= y;
    m[2].x *= z; m[2].y *= z; m[2].z *= z;
    UpdateMatrix();
  }
  void Translate(float x, float y, float z) { 
    m44 &m = TargetMatrix()->back();
    m[3].x += m[0].x * x + m[1].x * y + m[2].x * z;
    m[3].y += m[0].y * x + m[1].y * y + m[2].y * z;
    m[3].z += m[0].z * x + m[1].z * y + m[2].z * z;
    m[3].w += m[0].w * x + m[1].w * y + m[2].w * z;
    UpdateMatrix();
  }
  void Rotatef(float angle, float x, float y, float z) { TargetMatrix()->back().Mult(m44::Rotate(DegreeToRadian(angle), x, y, z)); UpdateMatrix(); }
  void Ortho  (float l, float r, float b, float t, float nv, float fv) { TargetMatrix()->back().Mult(m44::Ortho  (l, r, b, t, nv, fv)); UpdateMatrix(); }
  void Frustum(float l, float r, float b, float t, float nv, float fv) { TargetMatrix()->back().Mult(m44::Frustum(l, r, b, t, nv, fv)); UpdateMatrix(); }
  
  void MatrixModelview()  { matrix_target=1; }
  void MatrixProjection() { matrix_target=2; }
  void PopMatrix() {
    vector<m44> *target = TargetMatrix();
    if      (target->size() >= 1) target->pop_back();
    else if (target->size() == 1) target->back().Assign(m44::Identity());
    UpdateMatrix();
  }
  void PushMatrix()         { TargetMatrix()->push_back(TargetMatrix()->back()); UpdateMatrix(); }
  void LoadIdentity()       { TargetMatrix()->back().Assign(m44::Identity());    UpdateMatrix(); }
  void Mult(const float *m) { TargetMatrix()->back().Mult(m44(m));               UpdateMatrix(); }
  void PrintMatrix()        { TargetMatrix()->back().Print(StrCat("mt", matrix_target)); }
  void GetMatrix(m44 *out)  { *out = TargetMatrix()->back(); }
  vector<m44> *TargetMatrix() {
    if      (matrix_target == 1) return &modelview_matrix;
    else if (matrix_target == 2) return &projection_matrix;
    else FATAL("uknown matrix ", matrix_target);
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
  void VertexAttribPointer(int slot, const VertexAttribute &attr) { 
    glVertexAttribPointer(slot, attr.m, attr.t, GL_FALSE, attr.w, Void(long(attr.o)));
  }

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

  void UpdateShader() {
    if (cubemap_on && normals_on) UseShader(&app->shaders->shader_cubenorm);
    else if          (cubemap_on) UseShader(&app->shaders->shader_cubemap);
    else if          (normals_on) UseShader(&app->shaders->shader_normals);
    else                          UseShader(&app->shaders->shader_default);
  }
  void UpdateColor()  { ClearDeferred(); dirty_color = true; }
  void UpdateMatrix() { ClearDeferred(); dirty_matrix = true; }
  void UpdateMaterial() {
    ClearDeferred();
    shader->dirty_material = app->shaders->shader_cubenorm.dirty_material = app->shaders->shader_normals.dirty_material = true;
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
    if (type == Triangles && o == 0 && n == deferred.vertexbuffer_appended / deferred.vertex_size) {
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
    // INFOf("merged %d %d (type = %d)\n", deferred.draw_calls, deferred.vertexbuffer_len / deferred.vertex_size, deferred.prim_type);
    glDrawArrays(deferred.prim_type, 0, deferred.vertexbuffer_len / deferred.vertex_size);
    deferred.vertexbuffer_len = deferred.draw_calls = 0;
    GDDebug("ClearDeferred");
  }
  int AddDeferredVertexSpace(int l) {
    if (l + deferred.vertexbuffer_len > deferred.vertexbuffer_size) ClearDeferred();
    int ret = deferred.vertexbuffer_len;
    deferred.vertexbuffer_len += l;
    CHECK_LE(deferred.vertexbuffer_len, deferred.vertexbuffer_size);
    return ret;
  }

  void PushDirtyState() {
    if (dirty_matrix) {
      dirty_matrix = false;
      m44 m = projection_matrix.back();
      m.Mult(modelview_matrix.back());
      if (1)                  glUniformMatrix4fv(shader->uniform_modelviewproj, 1, 0, m[0]);
      if (1)                  glUniformMatrix4fv(shader->uniform_modelview,     1, 0, modelview_matrix.back()[0]);
      if (1)                  glUniform3fv      (shader->uniform_campos,        1,    camera_pos);
      if (invert_view_matrix) glUniformMatrix4fv(shader->uniform_invview,       1, 0, invview_matrix[0]);
      if (track_model_matrix) glUniformMatrix4fv(shader->uniform_model,         1, 0, model_matrix[0]);
    }
    if (dirty_color && shader->uniform_colordefault >= 0) {
      dirty_color = false;
      glUniform4fv(shader->uniform_colordefault, 1, default_color.back().x);
    }
    if (shader->dirty_material) {
      glUniform4fv(shader->uniform_material_ambient,  1, material.ambient.x);
      glUniform4fv(shader->uniform_material_diffuse,  1, material.diffuse.x);
      glUniform4fv(shader->uniform_material_specular, 1, material.specular.x);
      glUniform4fv(shader->uniform_material_emission, 1, material.emissive.x);
    }
    for (int i=0; i<sizeofarray(light) && i<sizeofarray(shader->dirty_light_pos); i++) {
      if (shader->dirty_light_pos[i]) {
        shader->dirty_light_pos[i] = 0;
        glUniform4fv(shader->uniform_light0_pos, 1, light[i].pos);
      }
      if (shader->dirty_light_color[i]) {
        shader->dirty_light_color[i] = 0;
        glUniform4fv(shader->uniform_light0_ambient,  1, light[i].color.ambient.x);
        glUniform4fv(shader->uniform_light0_diffuse,  1, light[i].color.diffuse.x);
        glUniform4fv(shader->uniform_light0_specular, 1, light[i].color.specular.x);
      }
    }
  }
};
#endif // LFL_GLES2

unique_ptr<GraphicsDevice> CreateGraphicsDevice(int opengles_version) {
  ONCE({
#ifdef LFL_GLEW
#ifdef GLEW_MX
    screen->glew_context = new GLEWContext();
#endif
    GLenum glew_err;
    if ((glew_err = glewInit()) != GLEW_OK) return ERRORv(nullptr, "glewInit: ", glewGetErrorString(glew_err));
#endif
  });

  unique_ptr<GraphicsDevice> gd;
#ifdef LFL_GLES2
  if (opengles_version == 2) gd = make_unique<OpenGLES2>();
#endif
  else gd = make_unique<OpenGLES1>();
#ifdef LFL_GLEW
  gd->have_framebuffer = GLEW_EXT_framebuffer_object;
#endif
  return gd;
}

}; // namespace LFL
