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

#ifdef LFL_OPENCV
#include "opencv/cxcore.h"
#endif

#include "lfapp/lfapp.h"
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/flow.h"
#include "lfapp/gui.h"

#if defined(LFL_GLEW) && !defined(LFL_HEADLESS)
#define glewGetContext() ((GLEWContext*)screen->glew_context)
#include <GL/glew.h>
#ifdef WIN32
#include <GL/wglew.h>
#endif
#endif

#define LFL_GLSL_SHADERS
#if (defined(LFL_ANDROID) || defined(LFL_IPHONE)) && !defined(LFL_GLES2)
#undef LFL_GLSL_SHADERS
#endif

#if defined(LFL_HEADLESS)
#define GLint int
#define GLuint unsigned 
#define glFlush(x)
#define glGetError() 0
#define glGetString(a) ""
#define glGetIntegerv(a,b)
#define glReadPixels(a,b,c,d,e,f,g)
#define glTexImage2D(a,b,c,d,e,f,g,h,i)
#define glTexSubImage2D(a,b,c,d,e,f,g,h,i)
#define glTexParameteri(a,b,c)
#define glGetTexImage(a,b,c,d,e)
#define glGetTexLevelParameteriv(a,b,c,d)
#define glGenRenderbuffersEXT(a,b)
#define glGenFramebuffersEXT(a,b)
#define glBindRenderbufferEXT(a,b)
#define glBindFramebufferEXT(a,b)
#define glRenderbufferStorageEXT(a,b,c,d)
#define glFramebufferRenderbufferEXT(a,b,c,d)
#define glFramebufferTexture2DEXT(a,b,c,d,e)
#define glCheckFramebufferStatusEXT(a) 0
#define glewGetString(a) 0
#define GL_FRAMEBUFFER 0
#define GL_FRAMEBUFFER_BINDING_OES 0
#define GL_LUMINANCE 0
#define GL_LUMINANCE_ALPHA 0
#define GL_RGB 0
#define GL_BGR 0
#define GL_RGBA 0
#define GL_BGRA 0
#define GL_TEXTURE_2D 0
#define GL_TEXTURE_WIDTH 0
#define GL_TEXTURE_CUBE_MAP 0
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_X 0
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Y 0
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Z 0
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X 0
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Y 0
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Z 0
#define GL_UNSIGNED_BYTE 0
#define GL_FRAGMENT_SHADER 0
#define GL_ACTIVE_UNIFORMS 0
#define GL_ACTIVE_ATTRIBUTES 0
#define GL_FRAMEBUFFER_COMPLETE 0
#define GL_VERTEX_SHADER 0
#define GL_MAX_VERTEX_UNIFORM_COMPONENTS 0
#define GL_MAX_VERTEX_ATTRIBS 0
#define GL_DEPTH_COMPONENT16 0
#define GLEW_EXT_framebuffer_object 0

#elif defined(LFL_IPHONE)
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

#elif defined(__APPLE__)
#include <OpenGL/glu.h>

#else
#include <GL/glu.h>
#endif

#if !defined(LFL_HEADLESS) && (defined(LFL_MOBILE) || defined(LFL_QT))
#define glGenRenderbuffersEXT(a,b) glGenRenderbuffers(a,b)
#define glBindRenderbufferEXT(a,b) glBindRenderbuffer(a,b)
#define glRenderbufferStorageEXT(a,b,c,d) glRenderbufferStorage(a,b,c,d)
#define glGenFramebuffersEXT(a,b) glGenFramebuffers(a,b)
#define glBindFramebufferEXT(a,b) glBindFramebuffer(a,b)
#define glFramebufferTexture2DEXT(a,b,c,d,e) glFramebufferTexture2D(a,b,c,d,e)
#define glFramebufferRenderbufferEXT(a,b,c,d) glFramebufferRenderbuffer(a,b,c,d)
#define glCheckFramebufferStatusEXT(a) glCheckFramebufferStatus(a)
#endif

#ifdef LFL_X11VIDEO
#include <X11/Xlib.h>
#include <GL/glx.h>
#endif

#ifdef LFL_XTVIDEO
#include <X11/IntrinsicP.h>
#include <X11/ShellP.h>
#include <GL/glx.h>
#endif

extern "C" {
#ifdef LFL_FFMPEG
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#endif
};

#ifdef LFL_QT
#include <QApplication>
extern QApplication *lfl_qapp;
extern "C" int LFLQTInit();
#endif // LFL_QT

#ifdef LFL_WXWIDGETS
#include <wx/wx.h>
#include <wx/glcanvas.h>
#endif // LFL_WXWIDGETS

#ifdef LFL_GLFWVIDEO
#include "GLFW/glfw3.h"
#endif

#ifdef LFL_SDLVIDEO
#include "SDL.h"
#if defined(LFL_ANDROID)
extern "C" {
#include "SDL_androidvideo.h"
};
#endif
#endif

namespace LFL {
DEFINE_bool(gd_debug, false, "Debug graphics device");
DEFINE_float(rotate_view, 0, "Rotate view by angle");
DEFINE_float(field_of_view, 45, "Field of view");
DEFINE_float(near_plane, 1, "Near clipping plane");
DEFINE_float(far_plane, 100, "Far clipping plane");
DEFINE_int(dots_per_inch, 75, "Screen DPI");
DEFINE_bool(swap_axis, false," Swap x,y axis");

Color Color::white (1.0, 1.0, 1.0);
Color Color::black (0.0, 0.0, 0.0);
Color Color::red   (1.0, 0.0, 0.0);
Color Color::green (0.0, 1.0, 0.0);
Color Color::blue  (0.0, 0.0, 1.0);
Color Color::cyan  (0.0, 1.0, 1.0);
Color Color::yellow(1.0, 1.0, 0.0);
Color Color::magenta(1.0, 0.0, 1.0);
Color Color::grey90(.9, .9, .9);
Color Color::grey80(.8, .8, .8);
Color Color::grey70(.7, .7, .7);
Color Color::grey60(.6, .6, .6);
Color Color::grey50(.5, .5, .5);
Color Color::grey40(.4, .4, .4);
Color Color::grey30(.3, .3, .3);
Color Color::grey20(.2, .2, .2);
Color Color::grey10(.1, .1, .1);
Color Color::clear(0.0, 0.0, 0.0, 0.0);

const int Texture::CoordMinX = 0;
const int Texture::CoordMinY = 1;
const int Texture::CoordMaxX = 2;
const int Texture::CoordMaxY = 3;
#ifdef LFL_MOBILE
const int Texture::preferred_pf = Pixel::RGBA;
#else
const int Texture::preferred_pf = Pixel::BGRA;
#endif

Window::WindowMap Window::active;

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

int ColorChannel::PixelOffset(int c) {
  switch (c) {
    case Red:  return 0;    case Green: return 1;
    case Blue: return 2;    case Alpha: return 3;
  }
  return 0;
}

const char *Pixel::Name(int p) {
  switch (p) {
    case RGB32:   return "RGB32";      case RGB555:  return "RGB555";
    case BGR32:   return "BGR32";      case BGR555:  return "BGR555";
    case RGB24:   return "RGB24";      case RGB565:  return "RGB565";
    case BGR24:   return "BGR24";      case BGR565:  return "BGR565";
    case RGBA:    return "RGBA";       case BGRA:    return "BGRA";
    case YUV420P: return "YUV420P";    case YUYV422: return "YUYV422";
    case GRAY8:   return "GRAY8";      case GRAYA8:  return "GRAYA8";
    case LCD:     return "LCD";
  }; return 0; 
}

int Pixel::size(int p) {
  switch (p) {
    case RGB32:   case BGR32:  case RGBA:   case BGRA:                return 4;
    case RGB24:   case BGR24:  case LCD:                              return 3;
    case RGB555:  case BGR555: case RGB565: case BGR565: case GRAYA8: return 2;
    case YUYV422: case GRAY8:                                         return 1;
    default:                                                          return 0;
  }
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

#ifdef LFL_FFMPEG
int Pixel::FromFFMpegId(int fmt) {
  switch (fmt) {
    case AV_PIX_FMT_RGB32:    return Pixel::RGB32;
    case AV_PIX_FMT_BGR32:    return Pixel::BGR32;
    case AV_PIX_FMT_RGB24:    return Pixel::RGB24;
    case AV_PIX_FMT_BGR24:    return Pixel::BGR24;
    case AV_PIX_FMT_GRAY8:    return Pixel::GRAY8;
    case AV_PIX_FMT_YUV410P:  return Pixel::YUV410P;
    case AV_PIX_FMT_YUV420P:  return Pixel::YUV420P;
    case AV_PIX_FMT_YUYV422:  return Pixel::YUYV422;
    case AV_PIX_FMT_YUVJ420P: return Pixel::YUVJ420P;
    case AV_PIX_FMT_YUVJ422P: return Pixel::YUVJ422P;
    case AV_PIX_FMT_YUVJ444P: return Pixel::YUVJ444P;
    default: ERROR("unknown pixel fmt: ", fmt); return 0;
  }
}
int Pixel::ToFFMpegId(int fmt) {
  switch (fmt) {
    case Pixel::RGB32:    return AV_PIX_FMT_RGB32;
    case Pixel::BGR32:    return AV_PIX_FMT_BGR32;
    case Pixel::RGB24:    return AV_PIX_FMT_RGB24;
    case Pixel::BGR24:    return AV_PIX_FMT_BGR24;
    case Pixel::RGBA:     return AV_PIX_FMT_RGBA;
    case Pixel::BGRA:     return AV_PIX_FMT_BGRA;
    case Pixel::GRAY8:    return AV_PIX_FMT_GRAY8;
    case Pixel::YUV410P:  return AV_PIX_FMT_YUV410P;
    case Pixel::YUV420P:  return AV_PIX_FMT_YUV420P;
    case Pixel::YUYV422:  return AV_PIX_FMT_YUYV422;
    case Pixel::YUVJ420P: return AV_PIX_FMT_YUVJ420P;
    case Pixel::YUVJ422P: return AV_PIX_FMT_YUVJ422P;
    case Pixel::YUVJ444P: return AV_PIX_FMT_YUVJ444P;
    default: ERROR("unknown pixel fmt: ", fmt); return 0;
  }
}
#endif // LFL_FFMPEG

void Color::ToHSV(float *h, float *s, float *v) const {
  float M = max(r(), max(g(), b()));
  float m = min(r(), min(g(), b()));
  float C = M - m;
  if (!C) { *h = *s = 0; *v = m; return; }

  *v = M;
  *s = C / M;

  if      (r() == m) *h = 3 - (g() - b()) / C;
  else if (g() == m) *h = 5 - (b() - r()) / C;
  else               *h = 1 - (r() - g()) / C;

  *h *= 60;
  if (*h < 0) *h += 360;
}

Color Color::FromHSV(float h, float s, float v) {
  if (s == 0) return Color(v, v, v);
  while (h >= 360) h -= 360;
  while (h <    0) h += 360;

  float hf = Decimals(h / 60);
  float p = v * (1 - s);
  float q = v * (1 - s * hf);
  float t = v * (1 - s * (1 - hf));

  if      (h < 60)  return Color(v, t, p);
  else if (h < 120) return Color(q, v, p);
  else if (h < 180) return Color(p, v, t);
  else if (h < 240) return Color(p, q, v);
  else if (h < 300) return Color(t, p, v);
  else              return Color(v, p, q);
}

Color Color::fade(float v) {
  Color color;
  if      (v < 0.166667f) color = Color(1.0, v * 6.0, 0.0);
  else if (v < 0.333333f) color = Color((1/3.0 - v) * 6.0, 1.0, 0.0);
  else if (v < 0.5f)      color = Color(0.0, 1.0, (v - 1/3.0) * 6);
  else if (v < 0.666667f) color = Color(0.0, (2/3.0 - v) * 6, 1.0);
  else if (v < 0.833333f) color = Color((v - 2/3.0) * 6, 0.0, 1.0);
  else                    color = Color(1.0, 0.0, (1 - v)*6.0);
  for (int i = 0; i < 4; i++) color.x[i] = min(1.0f, max(color.x[i], 0.0f));
  return color;
}

#if 0
void Material::SetLightColor(const Color &color) {
  diffuse = specular = ambient = color;
  ambient.scale(.2);
  emissive = Color::black;
}
void Material::SetMaterialColor(const Color &color) {
  diffuse = ambient = color;
  specular = Color::white;
  emissive = Color::black;
}
#else
void Material::SetLightColor(const Color &color) {
  diffuse = ambient = color;
  diffuse.scale(.9);
  ambient.scale(.5);
  specular = emissive = Color::black;
}
void Material::SetMaterialColor(const Color &color) {
  diffuse = ambient = color;
  specular = emissive = Color::black;
}
#endif

Box::Box(float X, float Y, float W, float H, bool round) {
  if (round) { x=RoundF(X); y=RoundF(Y); w=RoundF(W); h=RoundF(H); }
  else       { x= (int)(X); y= (int)(Y); w= (int)(W); h= (int)(H); }
}

Box::Box(const float *v4, bool round) {
  if (round) { x=RoundF(v4[0]); y=RoundF(v4[1]); w=RoundF(v4[2]); h=RoundF(v4[3]); }
  else       { x= (int)(v4[0]); y= (int)(v4[1]); w= (int)(v4[2]); h= (int)(v4[3]); }
}

string Box::DebugString() const { return StringPrintf("Box = { %d, %d, %d, %d }", x, y, w, h); }

void Box::Draw(const float *texcoord) const {
  static const float default_texcoord[4] = {0, 0, 1, 1};
  const float *tc = X_or_Y(texcoord, default_texcoord);
#if 1
  float verts[] = { (float)x,   (float)y,   tc[Texture::CoordMinX], tc[Texture::CoordMinY],
                    (float)x,   (float)y+h, tc[Texture::CoordMinX], tc[Texture::CoordMaxY],
                    (float)x+w, (float)y,   tc[Texture::CoordMaxX], tc[Texture::CoordMinY],
                    (float)x,   (float)y+h, tc[Texture::CoordMinX], tc[Texture::CoordMaxY],
                    (float)x+w, (float)y,   tc[Texture::CoordMaxX], tc[Texture::CoordMinY],
                    (float)x+w, (float)y+h, tc[Texture::CoordMaxX], tc[Texture::CoordMaxY] };
  bool changed = screen->gd->VertexPointer(2, GraphicsDevice::Float, sizeof(float)*4, 0,               verts, sizeof(verts), NULL, true, GraphicsDevice::Triangles);
  if (changed)   screen->gd->TexPointer   (2, GraphicsDevice::Float, sizeof(float)*4, sizeof(float)*2, verts, sizeof(verts), NULL, false);
  if (1)         screen->gd->DeferDrawArrays(GraphicsDevice::Triangles, 0, 6);
#else
  float verts[] = { (float)x,   (float)y,   tc[Texture::CoordMinX], tc[Texture::CoordMinY],
                    (float)x,   (float)y+h, tc[Texture::CoordMinX], tc[Texture::CoordMaxY],
                    (float)x+w, (float)y,   tc[Texture::CoordMaxX], tc[Texture::CoordMinY],
                    (float)x+w, (float)y+h, tc[Texture::CoordMaxX], tc[Texture::CoordMaxY] };
  bool changed = screen->gd->VertexPointer(2, GraphicsDevice::Float, sizeof(float)*4, 0,               verts, sizeof(verts), NULL, true, GraphicsDevice::TriangleStrip);
  if  (changed)  screen->gd->TexPointer   (2, GraphicsDevice::Float, sizeof(float)*4, sizeof(float)*2, verts, sizeof(verts), NULL, false);
  if (1)         screen->gd->DeferDrawArrays(GraphicsDevice::TriangleStrip, 0, 4);
#endif
}

void Box::DrawCrimped(const float *texcoord, int orientation, float scrollX, float scrollY) const {
  float left=x, right=x+w, top=y, bottom=y+h;
  float texMinX, texMinY, texMaxX, texMaxY, texMidX1, texMidX2, texMidY1, texMidY2;

  scrollX *= (texcoord[2] - texcoord[0]);
  scrollY *= (texcoord[3] - texcoord[1]);
  scrollX = ScrollCrimped(texcoord[0], texcoord[2], scrollX, &texMinX, &texMidX1, &texMidX2, &texMaxX);
  scrollY = ScrollCrimped(texcoord[1], texcoord[3], scrollY, &texMinY, &texMidY1, &texMidY2, &texMaxY);

#define DrawCrimpedBoxTriangleStrip() \
  screen->gd->VertexPointer(2, GraphicsDevice::Float, 4*sizeof(float), 0,               verts, sizeof(verts), NULL, true, GraphicsDevice::TriangleStrip); \
  screen->gd->TexPointer   (2, GraphicsDevice::Float, 4*sizeof(float), 2*sizeof(float), verts, sizeof(verts), NULL, false); \
  screen->gd->DeferDrawArrays(GraphicsDevice::TriangleStrip, 0, 4); \
  screen->gd->DeferDrawArrays(GraphicsDevice::TriangleStrip, 4, 4); \
  screen->gd->DeferDrawArrays(GraphicsDevice::TriangleStrip, 8, 4); \
  screen->gd->DeferDrawArrays(GraphicsDevice::TriangleStrip, 12, 4);

  switch (orientation) {
    case 0: {
      float xmid = x + w * scrollX, ymid = y + h * scrollY, verts[] = {
        /*02*/ xmid,  top,  texMidX1, texMaxY,  /*01*/ left, top,  texMinX,  texMaxY,  /*03*/ xmid,  ymid,   texMidX1, texMidY1, /*04*/ left, ymid,   texMinX,  texMidY1,
        /*06*/ right, top,  texMaxX,  texMaxY,  /*05*/ xmid, top,  texMidX2, texMaxY,  /*07*/ right, ymid,   texMaxX,  texMidY1, /*08*/ xmid, ymid,   texMidX2, texMidY1,
        /*10*/ right, ymid, texMaxX,  texMidY2, /*09*/ xmid, ymid, texMidX2, texMidY2, /*11*/ right, bottom, texMaxX,  texMinY,  /*12*/ xmid, bottom, texMidX2, texMinY,
        /*14*/ xmid,  ymid, texMidX1, texMidY2, /*13*/ left, ymid, texMinX,  texMidY2, /*15*/ xmid,  bottom, texMidX1, texMinY,  /*16*/ left, bottom, texMinX,  texMinY 
      };
      DrawCrimpedBoxTriangleStrip();
    } break;
    case 1: {
      float xmid = x + w * scrollX, ymid = y + h * (1-scrollY), verts[] = {
        /*02*/ xmid,  top,  texMidX1, texMinY,  /*01*/ left,  top,  texMinX,  texMinY,  /*03*/ xmid, ymid,    texMidX1, texMidY2, /*04*/ left, ymid,   texMinX,  texMidY2,
        /*06*/ right, top,  texMaxX,  texMinY,  /*05*/ xmid,  top,  texMidX2, texMinY,  /*07*/ right, ymid,   texMaxX,  texMidY2, /*08*/ xmid, ymid,   texMidX2, texMidY2,
        /*10*/ right, ymid, texMaxX,  texMidY1, /*09*/ xmid,  ymid, texMidX2, texMidY1, /*11*/ right, bottom, texMaxX,  texMaxY,  /*12*/ xmid, bottom, texMidX2, texMaxY,
        /*14*/ xmid,  ymid, texMidX1, texMidY1, /*13*/ left,  ymid, texMinX,  texMidY1, /*15*/ xmid, bottom,  texMidX1, texMaxY,  /*16*/ left, bottom, texMinX,  texMaxY 
      };
      DrawCrimpedBoxTriangleStrip();
    } break;
    case 2: {
      float xmid = x + w * (1-scrollX), ymid = y + h * scrollY, verts[] = {
        /*02*/ xmid,  top,  texMidX2, texMaxY,  /*01*/ left,  top,  texMaxX,  texMaxY,  /*03*/ xmid, ymid,    texMidX2, texMidY1, /*04*/ left, ymid,   texMaxX,  texMidY1,
        /*06*/ right, top,  texMinX,  texMaxY,  /*05*/ xmid,  top,  texMidX1, texMaxY,  /*07*/ right, ymid,   texMinX,  texMidY1, /*08*/ xmid, ymid,   texMidX1, texMidY1,
        /*10*/ right, ymid, texMinX,  texMidY2, /*09*/ xmid,  ymid, texMidX1, texMidY2, /*11*/ right, bottom, texMinX,  texMinY,  /*12*/ xmid, bottom, texMidX1, texMinY,
        /*14*/ xmid,  ymid, texMidX2, texMidY2, /*13*/ left,  ymid, texMaxX,  texMidY2, /*15*/ xmid, bottom,  texMidX2, texMinY,  /*16*/ left, bottom, texMaxX,  texMinY 
      };
      DrawCrimpedBoxTriangleStrip();
    } break;
    case 3: {
      float xmid = x + w * (1-scrollX), ymid = y + h * (1-scrollY), verts[] = {
        /*02*/ xmid,  top,  texMidX2, texMinY,  /*01*/ left,  top,   texMaxX,  texMinY,  /*03*/ xmid, ymid,    texMidX2, texMidY2, /*04*/ left, ymid,   texMaxX,  texMidY2,
        /*06*/ right, top,  texMinX,  texMinY,  /*05*/ xmid,  top,   texMidX1, texMinY,  /*07*/ right, ymid,   texMinX,  texMidY2, /*08*/ xmid, ymid,   texMidX1, texMidY2,
        /*10*/ right, ymid, texMinX,  texMidY1, /*09*/ xmid,  ymid,  texMidX1, texMidY1, /*11*/ right, bottom, texMinX,  texMaxY,  /*12*/ xmid, bottom, texMidX1, texMaxY,
        /*14*/ xmid,  ymid, texMidX2, texMidY1, /*13*/ left,  ymid,  texMaxX,  texMidY1, /*15*/ xmid, bottom,  texMidX2, texMaxY,  /*16*/ left, bottom, texMaxX,  texMaxY 
      };
      DrawCrimpedBoxTriangleStrip();
    } break;
    case 4: {
      float xmid = x + w * (1-scrollY), ymid = y + h * scrollX, verts[] = {
        /*13*/ xmid,  top,  texMinX,  texMidY2, /*16*/ left,  top,  texMinX,  texMaxY,  /*14*/ xmid, ymid,    texMidX1, texMidY2, /*15*/ left, ymid,   texMidX1, texMaxY, 
        /*01*/ right, top,  texMinX,  texMinY,  /*04*/ xmid,  top,  texMinX,  texMidY1, /*02*/ right, ymid,   texMidX1, texMinY,  /*03*/ xmid, ymid,   texMidX1, texMidY1,
        /*05*/ right, ymid, texMidX2, texMinY,  /*08*/ xmid,  ymid, texMidX2, texMidY1, /*06*/ right, bottom, texMaxX,  texMinY,  /*07*/ xmid, bottom, texMaxX,  texMidY1,
        /*09*/ xmid,  ymid, texMidX2, texMidY2, /*12*/ left,  ymid, texMidX2, texMaxY,  /*10*/ xmid, bottom,  texMaxX,  texMidY2, /*11*/ left, bottom, texMaxX,  texMaxY 
      };
      DrawCrimpedBoxTriangleStrip();
    } break;
    case 5: {
      float xmid = x + w * scrollY, ymid = y + h * scrollX, verts[] = {
        /*13*/ xmid,  top,  texMinX,  texMidY1, /*16*/ left,  top,  texMinX,  texMinY,  /*14*/ xmid, ymid,    texMidX1, texMidY1, /*15*/ left, ymid,   texMidX1, texMinY, 
        /*01*/ right, top,  texMinX,  texMaxY,  /*04*/ xmid,  top,  texMinX,  texMidY2, /*02*/ right, ymid,   texMidX1, texMaxY,  /*03*/ xmid, ymid,   texMidX1, texMidY2,
        /*05*/ right, ymid, texMidX2, texMaxY,  /*08*/ xmid,  ymid, texMidX2, texMidY2, /*06*/ right, bottom, texMaxX,  texMaxY,  /*07*/ xmid, bottom, texMaxX,  texMidY2,
        /*09*/ xmid,  ymid, texMidX2, texMidY1, /*12*/ left,  ymid, texMidX2, texMinY,  /*10*/ xmid, bottom,  texMaxX,  texMidY1, /*11*/ left, bottom, texMaxX,  texMinY 
      };
      DrawCrimpedBoxTriangleStrip();
    } break;
    case 6: {
      float xmid = x + w * (1-scrollY), ymid = y + h * (1-scrollX), verts[] = {
        /*13*/ xmid,  top,  texMaxX,  texMidY2, /*16*/ left,  top,  texMaxX,  texMaxY,  /*14*/ xmid, ymid,    texMidX2, texMidY2, /*15*/ left, ymid,   texMidX2, texMaxY, 
        /*01*/ right, top,  texMaxX,  texMinY,  /*04*/ xmid,  top,  texMaxX,  texMidY1, /*02*/ right, ymid,   texMidX2, texMinY,  /*03*/ xmid, ymid,   texMidX2, texMidY1,
        /*05*/ right, ymid, texMidX1, texMinY,  /*08*/ xmid,  ymid, texMidX1, texMidY1, /*06*/ right, bottom, texMinX,  texMinY,  /*07*/ xmid, bottom, texMinX,  texMidY1,
        /*09*/ xmid,  ymid, texMidX1, texMidY2, /*12*/ left,  ymid, texMidX1, texMaxY,  /*10*/ xmid, bottom,  texMinX,  texMidY2, /*11*/ left, bottom, texMinX,  texMaxY 
      };
      DrawCrimpedBoxTriangleStrip();
    } break;
    case 7: {
      float xmid = x + w * scrollY, ymid = y + h * (1-scrollX), verts[] = {
        /*13*/ xmid,  top,  texMaxX,  texMidY1, /*16*/ left,  top,  texMaxX,  texMinY,  /*14*/ xmid, ymid,    texMidX2, texMidY1, /*15*/ left, ymid,   texMidX2, texMinY, 
        /*01*/ right, top,  texMaxX,  texMaxY,  /*04*/ xmid,  top,  texMaxX,  texMidY2, /*02*/ right, ymid,   texMidX2, texMaxY,  /*03*/ xmid, ymid,   texMidX2, texMidY2,
        /*05*/ right, ymid, texMidX1, texMaxY,  /*08*/ xmid,  ymid, texMidX1, texMidY2, /*06*/ right, bottom, texMinX,  texMaxY,  /*07*/ xmid, bottom, texMinX,  texMidY2,
        /*09*/ xmid,  ymid, texMidX1, texMidY1, /*12*/ left,  ymid, texMidX1, texMinY,  /*10*/ xmid, bottom,  texMinX,  texMidY1, /*11*/ left, bottom, texMinX,  texMinY 
      };
      DrawCrimpedBoxTriangleStrip();
    } break;
  }
}

float Box::ScrollCrimped(float tex0, float tex1, float scroll, float *min, float *mid1, float *mid2, float *max) {
  if (tex1 < 1.0 && tex0 == 0.0) {
    *mid1=tex1; *mid2=0;
    if (scroll > 0) *min = *max = tex1 - scroll;
    else            *min = *max = tex0 - scroll;
  } else if (tex0 > 0.0 && tex1 == 1.0) {
    *mid1=1; *mid2=tex0;
    if (scroll > 0) *min = *max = tex0 + scroll;
    else            *min = *max = tex1 + scroll;
  } else if (tex0 == 0 && tex1 == 1) {
    *min = *max = 1;
    *mid1 = tex1; *mid2 = tex0;
  } else {
    return 0;
  }
  return (*mid1 - *min) / (tex1 - tex0); 
}

Box3::Box3(const Box &cont, const point &pb, const point &pe, int first_line_height, int last_line_height) {
  if (pb.y == pe.y) {
    v[0] = Box(pb.x, pb.y, pe.x - pb.x, first_line_height);
    v[1] = v[2] = Box();
  } else {
    v[0] = Box(pb.x, pb.y, cont.w - pb.x, first_line_height);
    v[1] = Box(0, pe.y + last_line_height, cont.w, pb.y - pe.y - first_line_height);
    v[2] = Box(0, pe.y, pe.x, last_line_height);
  }
}

void Box3::Draw(const point &p, const Color *c) const {
  if (c) screen->gd->SetColor(*c);
  for (int i=0; i<3; i++) if (v[i].h) (v[i] + p).Draw();
}

Box Box3::BoundingBox() const {
  int min_x = v[0].x, min_y = v[0].y, max_x = v[0].x + v[0].w, max_y = v[0].y + v[0].h;
  if (v[1].h) { min_x = min(min_x, v[1].x); min_y = min(min_y, v[1].y); max_x = max(max_x, v[1].x + v[1].w); max_y = max(max_y, v[1].y + v[1].h); }
  if (v[2].h) { min_x = min(min_x, v[2].x); min_y = min(min_y, v[2].y); max_x = max(max_x, v[2].x + v[2].w); max_y = max(max_y, v[2].y + v[2].h); }
  return Box(min_x, min_y, max_x - min_x, max_y - min_y);
}

void Drawable::AttrVec::Insert(const Drawable::Attr &v) {
  if (v.font) font_refs.Insert(&v.font->ref);
  push_back(v);
}

/* Texture */

int Texture::GLBufferType() const { return pf == preferred_pf ? GraphicsDevice::GLPreferredBuffer : GL_UNSIGNED_BYTE; }

void Texture::Coordinates(float *texcoord, int w, int h, int wd, int hd) {
  texcoord[CoordMinX] = texcoord[CoordMinY] = 0;
  texcoord[CoordMaxX] = (float)w / wd;
  texcoord[CoordMaxY] = (float)h / hd;
}

void Texture::Resize(int W, int H, int PF, int flag) {
  if (PF) pf = PF;
  width=W; height=H;
  if (buf || (flag & Flag::CreateBuf)) RenewBuffer();
  if (!ID && (flag & Flag::CreateGL)) {
    if (!cubemap) {
      screen->gd->DisableCubeMap();
      screen->gd->GenTextures(GL_TEXTURE_2D, 1, &ID);
    } else if (cubemap == CubeMap::PX) {
      screen->gd->ActiveTexture(0);
      screen->gd->GenTextures(GL_TEXTURE_CUBE_MAP, 1, &ID);
      glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    }
  }
  if (ID || cubemap) {
    int opengl_width = NextPowerOfTwo(width), opengl_height = NextPowerOfTwo(height);
    int gl_tt = GLTexType(), gl_pt = GLPixelType(), gl_bt = GLBufferType();
    if (ID) screen->gd->BindTexture(gl_tt, ID);
    glTexImage2D(gl_tt, 0, GraphicsDevice::GLInternalFormat, opengl_width, opengl_height, 0, gl_pt, gl_bt, 0);
    Coordinates(coord, width, height, opengl_width, opengl_height);
  }
}

void Texture::LoadBuffer(const unsigned char *B, const point &dim, int PF, int linesize, int flag) {
  Resize(dim.x, dim.y, pf, Flag::CreateBuf);
  int resample_flag = 0 | ((flag & Flag::FlipY) ? SimpleVideoResampler::Flag::FlipY : 0);
  SimpleVideoResampler::Blit(B, buf, width, height,
                             PF, linesize,   0, 0,
                             pf, LineSize(), 0, 0, resample_flag);
}

void Texture::UpdateBuffer(const unsigned char *B, const point &dim, int PF, int linesize, int flag) {
  bool resample = flag & Flag::Resample;
  VideoResampler conv;
  conv.Open(dim.x, dim.y, PF, resample ? width : dim.x, resample ? height : dim.y, pf);
  conv.Resample(B, linesize, buf, LineSize(), 0, flag & Flag::FlipY);
}

void Texture::UpdateBuffer(const unsigned char *B, const ::LFL::Box &box, int PF, int linesize, int blit_flag) {
  SimpleVideoResampler::Blit(B, buf, box.w, box.h, PF, linesize, 0, 0, pf, LineSize(), box.x, box.y, blit_flag);
}

void Texture::Bind() const { screen->gd->BindTexture(GLTexType(), ID); }
void Texture::ClearGL() { 
  if (!MainThread()) { RunInMainThread(new Callback(bind(&GraphicsDevice::DelTexture, screen->gd, ID))); ID=0; }
  else if (ID) { screen->gd->DelTexture(ID); ID=0; }
}

void Texture::LoadGL(const unsigned char *B, const point &dim, int PF, int linesize, int flag) {
  Texture temp;
  temp .Resize(dim.x, dim.y, preferred_pf, Flag::CreateBuf);
  temp .UpdateBuffer(B, dim, PF, linesize, Flag::FlipY);
  this->Resize(dim.x, dim.y, preferred_pf, Flag::CreateGL);
  this->UpdateGL(temp.buf, LFL::Box(dim), flag);
}

void Texture::UpdateGL(const unsigned char *B, const ::LFL::Box &box, int flag) {
  int gl_tt = GLTexType(), gl_y = (flag & Flag::FlipY) ? (height - box.y - box.h) : box.y;
  screen->gd->BindTexture(gl_tt, ID);
  glTexSubImage2D(gl_tt, 0, box.x, gl_y, box.w, box.h, GLPixelType(), GLBufferType(), B);
}

void Texture::DumpGL(unsigned tex_id) {
#ifndef LFL_MOBILE
  if (tex_id) {
    GLint gl_tt = GLTexType(), tex_w = 0, tex_h = 0;
    screen->gd->BindTexture(gl_tt, tex_id);
    glGetTexLevelParameteriv(gl_tt, 0, GL_TEXTURE_WIDTH, &tex_w);
    glGetTexLevelParameteriv(gl_tt, 0, GL_TEXTURE_WIDTH, &tex_h);
    CHECK_GT((width  = tex_w), 0);
    CHECK_GT((height = tex_h), 0);
  }
  RenewBuffer();
  glGetTexImage(GLTexType(), 0, GLPixelType(), GLBufferType(), buf);
#endif
}

void Texture::ToIplImage(_IplImage *out) {
#ifdef LFL_OPENCV
  memset(out, 0, sizeof(IplImage));
  out->nSize = sizeof(IplImage);
  out->nChannels = Pixel::size(pf);
  out->depth = IPL_DEPTH_8U;
  out->origin = 1;
  out->width = width;
  out->height = height;
  out->widthStep = out->width * out->nChannels;
  out->imageSize = out->widthStep * out->height;
  out->imageData = (char*)buf;
  out->imageDataOrigin = out->imageData;
#else
  ERROR("ToIplImage not implemented");
#endif
}

#ifdef __APPLE__
#import <CoreGraphics/CGBitmapContext.h> 
CGContextRef Texture::CGBitMap() { return CGBitMap(0, 0, width, height); }
CGContextRef Texture::CGBitMap(int X, int Y, int W, int H) {
  int linesize = LineSize(), alpha_info = 0;
  if      (pf == Pixel::RGBA)  alpha_info = kCGImageAlphaPremultipliedLast;
  else if (pf == Pixel::BGRA)  alpha_info = kCGBitmapByteOrder32Host | kCGImageAlphaPremultipliedFirst;
  else if (pf == Pixel::RGB32) alpha_info = kCGImageAlphaNoneSkipLast;
  else if (pf == Pixel::BGR32) alpha_info = kCGBitmapByteOrder32Host | kCGImageAlphaNoneSkipFirst;
  else { ERROR("unsupported pixel format: ", pf, " = ", Pixel::Name(pf)); return 0; }
  CGColorSpaceRef colors = CGColorSpaceCreateDeviceRGB();
  // CGColorSpaceRef colors = CGColorSpaceCreateWithName(kCGColorSpaceGenericRGB);
  CGContextRef ret = CGBitmapContextCreate(buf + Y*linesize + X*PixelSize(), W, H, 8, linesize, colors, alpha_info);
  CGColorSpaceRelease(colors);
  return ret;
}
#endif

#ifdef WIN32
HBITMAP Texture::CreateGDIBitMap(HDC dc) {
  ClearBuffer();
  buf_owner = false;
  pf = Pixel::BGR32;
  BITMAPINFO bmi;
  memzero(bmi.bmiHeader);
  bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
  bmi.bmiHeader.biWidth = width;
  bmi.bmiHeader.biHeight = -height;
  bmi.bmiHeader.biPlanes = 1;
  bmi.bmiHeader.biCompression = BI_RGB;
  bmi.bmiHeader.biBitCount = 32;
  HBITMAP ret = CreateDIBSection(dc, &bmi, DIB_RGB_COLORS, (void**)&buf, NULL, 0);
  return ret;
}
#endif

void Texture::Screenshot() { ScreenshotBox(Box(screen->width, screen->height), Flag::FlipY); }
void Texture::ScreenshotBox(const Box &b, int flag) {
  Resize(b.w, b.h, preferred_pf, Flag::CreateBuf);
  unsigned char *pixels = NewBuffer();
  glReadPixels(b.x, b.y, b.w, b.h, GLPixelType(), GLBufferType(), pixels);
  UpdateBuffer(pixels, point(b.w, b.h), pf, b.w*4, flag);
  delete [] pixels;
}

/* DepthTexture */

void DepthTexture::Resize(int W, int H, int DF, int flag) {
  if (DF) df = DF;
  width=W; height=H;
  if (!ID && (flag & Flag::CreateGL)) screen->gd->GenRenderBuffers(1, &ID);
  int opengl_width = NextPowerOfTwo(width), opengl_height = NextPowerOfTwo(height);
  if (ID) {
    screen->gd->BindRenderBuffer(ID);
    screen->gd->RenderBufferStorage(Depth::OpenGLID(df), opengl_width, opengl_height);
  }
}

/* FrameBuffer */

void FrameBuffer::Resize(int W, int H, int flag) {
  width=W; height=H;
  if (!ID && (flag & Flag::CreateGL)) {
    screen->gd->GenFrameBuffers(1, &ID);
    if (flag & Flag::CreateTexture)      AllocTexture(&tex, !(flag & Flag::NoClampToEdge));
    if (flag & Flag::CreateDepthTexture) AllocDepthTexture(&depth);
  } else {
    tex.Resize(width, height);
    depth.Resize(width, height);
  }
  Attach(tex.ID, depth.ID);
  int status = screen->gd->CheckFrameBufferStatus();
  if (status != GL_FRAMEBUFFER_COMPLETE) ERROR("FrameBuffer status ", status);
  if (flag & Flag::ReleaseFB) Release();
}

void FrameBuffer::AllocDepthTexture(DepthTexture *out) { CHECK_EQ(out->ID, 0); out->Create(width, height); }
void FrameBuffer::AllocTexture(Texture *out, bool clamp_to_edge) {
  CHECK_EQ(out->ID, 0);
  out->Create(width, height); 
  if (clamp_to_edge) {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  }
}

void FrameBuffer::Release() { screen->gd->BindFrameBuffer(screen->gd->default_framebuffer); }
void FrameBuffer::Attach(int ct, int dt) {
  screen->gd->BindFrameBuffer(ID);
  if (ct) screen->gd->FrameBufferTexture     ((tex.ID   = ct));
  if (dt) screen->gd->FrameBufferDepthTexture((depth.ID = dt));
}

/* Shader */

void Shader::SetGlobalUniform1f(const string &name, float v) {
  screen->gd->UseShader(&app->video->shader_default);  app->video->shader_default .SetUniform1f(name, v);
  screen->gd->UseShader(&app->video->shader_normals);  app->video->shader_normals .SetUniform1f(name, v);
  screen->gd->UseShader(&app->video->shader_cubemap);  app->video->shader_cubemap .SetUniform1f(name, v);
  screen->gd->UseShader(&app->video->shader_cubenorm); app->video->shader_cubenorm.SetUniform1f(name, v);
}

void Shader::SetGlobalUniform2f(const string &name, float v1, float v2){ 
  screen->gd->UseShader(&app->video->shader_default);  app->video->shader_default .SetUniform2f(name, v1, v2);
  screen->gd->UseShader(&app->video->shader_normals);  app->video->shader_normals .SetUniform2f(name, v1, v2);
  screen->gd->UseShader(&app->video->shader_cubemap);  app->video->shader_cubemap .SetUniform2f(name, v1, v2);
  screen->gd->UseShader(&app->video->shader_cubenorm); app->video->shader_cubenorm.SetUniform2f(name, v1, v2);
}

#ifdef LFL_GLSL_SHADERS
int Shader::Create(const string &name, const string &vertex_shader, const string &fragment_shader, const ShaderDefines &defines, Shader *out) {
  INFO("Shader::Create ", name);
  GLuint p = screen->gd->CreateProgram();

  string hdr =
    "#ifdef GL_ES\r\n"
    "precision highp float;\r\n"
    "#else\r\n"
    "#define lowp\r\n"
    "#define highp\r\n"
    "#endif\r\n";
#ifdef LFL_GLES2
  if (app->video->opengles_version == 2) hdr += "#define LFL_GLES2\r\n";
#endif
  hdr += defines.text + string("\r\n");

  if (vertex_shader.size()) {
    GLuint vs = screen->gd->CreateShader(GL_VERTEX_SHADER);
    const char *vss[] = { hdr.c_str(), vertex_shader.c_str(), 0 };
    screen->gd->ShaderSource(vs, 2, vss, 0);
    screen->gd->CompileShader(vs);
    screen->gd->AttachShader(p, vs);
  }

  if (fragment_shader.size()) {
    GLuint fs = screen->gd->CreateShader(GL_FRAGMENT_SHADER);
    const char *fss[] = { hdr.c_str(), fragment_shader.c_str(), 0 };
    screen->gd->ShaderSource(fs, 2, fss, 0);
    screen->gd->CompileShader(fs);
    screen->gd->AttachShader(p, fs);
  }

  if (1)                    screen->gd->BindAttribLocation(p, 0, "Position"   );
  if (defines.normals)      screen->gd->BindAttribLocation(p, 1, "Normal"     );
  if (defines.vertex_color) screen->gd->BindAttribLocation(p, 2, "VertexColor");
  if (defines.tex_2d)       screen->gd->BindAttribLocation(p, 3, "TexCoordIn" );

  screen->gd->LinkProgram(p);

  int active_uniforms=0, max_uniform_components=0, active_attributes=0, max_attributes=0;
  screen->gd->GetProgramiv(p, GL_ACTIVE_UNIFORMS, &active_uniforms);
  screen->gd->GetProgramiv(p, GL_ACTIVE_ATTRIBUTES, &active_attributes);
#if !defined(LFL_ANDROID) && !defined(LFL_IPHONE)
  screen->gd->GetIntegerv(GL_MAX_VERTEX_UNIFORM_COMPONENTS, &max_uniform_components);
#endif
  screen->gd->GetIntegerv(GL_MAX_VERTEX_ATTRIBS, &max_attributes);
  INFO("shader=", name, ", mu=", active_uniforms, " avg_comps/", max_uniform_components, ", ma=", active_attributes, "/", max_attributes);

  bool log_missing_attrib = false;
  if (out) {
    *out = Shader();
    out->ID = p;
    out->name = name;
    if ((out->slot_position             = screen->gd->GetAttribLocation (p, "Position"))            < 0 && log_missing_attrib) INFO("shader ", name, " missing Position");
    if ((out->slot_normal               = screen->gd->GetAttribLocation (p, "Normal"))              < 0 && log_missing_attrib) INFO("shader ", name, " missing Normal");
    if ((out->slot_color                = screen->gd->GetAttribLocation (p, "VertexColor"))         < 0 && log_missing_attrib) INFO("shader ", name, " missing VertexColor");
    if ((out->slot_tex                  = screen->gd->GetAttribLocation (p, "TexCoordIn"))          < 0 && log_missing_attrib) INFO("shader ", name, " missing TexCoordIn");
    if ((out->uniform_modelview         = screen->gd->GetUniformLocation(p, "Modelview"))           < 0 && log_missing_attrib) INFO("shader ", name, " missing Modelview");
    if ((out->uniform_modelviewproj     = screen->gd->GetUniformLocation(p, "ModelviewProjection")) < 0 && log_missing_attrib) INFO("shader ", name, " missing ModelviewProjection");
    if ((out->uniform_tex               = screen->gd->GetUniformLocation(p, "iChannel0"))           < 0 && log_missing_attrib) INFO("shader ", name, " missing Texture");
    if ((out->uniform_cubetex           = screen->gd->GetUniformLocation(p, "CubeTexture"))         < 0 && log_missing_attrib) INFO("shader ", name, " missing CubeTexture");
    if ((out->uniform_normalon          = screen->gd->GetUniformLocation(p, "NormalEnabled"))       < 0 && log_missing_attrib) INFO("shader ", name, " missing NormalEnabled");
    if ((out->uniform_texon             = screen->gd->GetUniformLocation(p, "TexCoordEnabled"))     < 0 && log_missing_attrib) INFO("shader ", name, " missing TexCoordEnabled");
    if ((out->uniform_coloron           = screen->gd->GetUniformLocation(p, "VertexColorEnabled"))  < 0 && log_missing_attrib) INFO("shader ", name, " missing VertexColorEnabled");
    if ((out->uniform_cubeon            = screen->gd->GetUniformLocation(p, "CubeMapEnabled"))      < 0 && log_missing_attrib) INFO("shader ", name, " missing CubeMapEnabled");
    if ((out->uniform_colordefault      = screen->gd->GetUniformLocation(p, "DefaultColor"))        < 0 && log_missing_attrib) INFO("shader ", name, " missing DefaultColor");
    if ((out->uniform_material_ambient  = screen->gd->GetUniformLocation(p, "MaterialAmbient"))     < 0 && log_missing_attrib) INFO("shader ", name, " missing MaterialAmbient");
    if ((out->uniform_material_diffuse  = screen->gd->GetUniformLocation(p, "MaterialDiffuse"))     < 0 && log_missing_attrib) INFO("shader ", name, " missing MaterialDiffuse");
    if ((out->uniform_material_specular = screen->gd->GetUniformLocation(p, "MaterialSpecular"))    < 0 && log_missing_attrib) INFO("shader ", name, " missing MaterialSpecular");
    if ((out->uniform_material_emission = screen->gd->GetUniformLocation(p, "MaterialEmission"))    < 0 && log_missing_attrib) INFO("shader ", name, " missing MaterialEmission");
    if ((out->uniform_light0_pos        = screen->gd->GetUniformLocation(p, "LightZeroPosition"))   < 0 && log_missing_attrib) INFO("shader ", name, " missing LightZeroPosition");
    if ((out->uniform_light0_ambient    = screen->gd->GetUniformLocation(p, "LightZeroAmbient"))    < 0 && log_missing_attrib) INFO("shader ", name, " missing LightZeroAmbient");
    if ((out->uniform_light0_diffuse    = screen->gd->GetUniformLocation(p, "LightZeroDiffuse"))    < 0 && log_missing_attrib) INFO("shader ", name, " missing LightZeroDiffuse");
    if ((out->uniform_light0_specular   = screen->gd->GetUniformLocation(p, "LightZeroSpecular"))   < 0 && log_missing_attrib) INFO("shader ", name, " missing LightZeroSpecular");

    int unused_attrib = 0;
    memset(out->unused_attrib_slot, -1, sizeof(out->unused_attrib_slot));
    for (int i=0; i<MaxVertexAttrib; i++) {
      if (out->slot_position == i || out->slot_normal == i || out->slot_color == i || out->slot_tex == i) continue;
      out->unused_attrib_slot[unused_attrib++] = i;
    }
  }

  return p;
}

int Shader::CreateShaderToy(const string &name, const string &pixel_shader, Shader *out) {
  static string header =
    "uniform float iGlobalTime, iBlend;\r\n"
    "uniform vec3 iResolution;\r\n"
    "uniform vec2 iScroll;\r\n"
    "uniform vec4 iMouse;\r\n"
    "uniform sampler2D iChannel0;\r\n"
    "uniform vec3 iChannelResolution[1];\r\n"
    "#define SampleChannelAtPointAndModulus(c, p, m) texture2D(c, mod((p), (m)))\r\n"
    "#define SampleChannelAtPoint(c, p) SampleChannelAtPointAndModulus(c, p, iChannelResolution[0].xy/iResolution.xy)\r\n"
    "#define SamplePoint() ((fragCoord.xy + iScroll)/iResolution.xy)\r\n"
    "#define SamplePointFlipY() vec2((fragCoord.x+iScroll.x)/iResolution.x, (iResolution.y-fragCoord.y-iScroll.y)/iResolution.y)\r\n"
    "#define SampleChannel(c) SampleChannelAtPoint(c, SamplePoint())\r\n"
#ifdef LFL_MOBILE
    "#define BlendChannels(c1,c2) (((c1) + (c2))/2.0)\r\n";
#else
    "#define BlendChannels(c1,c2) ((c1)*iBlend + (c2)*(1.0-iBlend))\r\n";
#endif

  static string footer =
    "void main(void) { mainImage(gl_FragColor, gl_FragCoord.xy); }\r\n";
  return Shader::Create(name, screen->gd->vertex_shader, StrCat(header, pixel_shader, footer), ShaderDefines(1,0,1,0), out);
}

int Shader::GetUniformIndex(const string &name) { return screen->gd->GetUniformLocation(ID, name); }
void Shader::SetUniform1i(const string &name, float v)                                { screen->gd->Uniform1i (GetUniformIndex(name), v); }
void Shader::SetUniform1f(const string &name, float v)                                { screen->gd->Uniform1f (GetUniformIndex(name), v); }
void Shader::SetUniform2f(const string &name, float v1, float v2)                     { screen->gd->Uniform2f (GetUniformIndex(name), v1, v2); }
void Shader::SetUniform3f(const string &name, float v1, float v2, float v3)           { screen->gd->Uniform3f (GetUniformIndex(name), v1, v2, v3); }
void Shader::SetUniform4f(const string &name, float v1, float v2, float v3, float v4) { screen->gd->Uniform4f (GetUniformIndex(name), v1, v2, v3, v4); }
void Shader::SetUniform3fv(const string &name, const float *v)                        { screen->gd->Uniform3fv(GetUniformIndex(name), 1, v); }
void Shader::SetUniform3fv(const string &name, int n, const float *v)                 { screen->gd->Uniform3fv(GetUniformIndex(name), n, v); }

#else /* LFL_GLSL_SHADERS */

int Shader::Create(const string &name, const string &vert, const string &frag, const ShaderDefines &defines, Shader *out) { return -1; }
int Shader::GetUniformIndex(const string &name) { return -1; }
void Shader::SetUniform1i(const string &name, float v) {}
void Shader::SetUniform1f(const string &name, float v) {}
void Shader::SetUniform2f(const string &name, float v1, float v2) {}
void Shader::SetUniform3f(const string &name, float v1, float v2, float v3) {}
void Shader::SetUniform4f(const string &name, float v1, float v2, float v3, float v4) {}
void Shader::SetUniform3fv(const string &name, const float *v) {}
void Shader::SetUniform3fv(const string &name, int n, const float *v) {}
#endif /* LFL_GLSL_SHADERS */

#ifndef LFL_HEADLESS
#ifdef LFL_GDDEBUG
#define GDDebug(...) { screen->gd->CheckForError(__FILE__, __LINE__); if (FLAGS_gd_debug) INFO(__VA_ARGS__); }
#else 
#define GDDebug(...)
#endif
const int GraphicsDevice::Float            = GL_FLOAT;
const int GraphicsDevice::Points           = GL_POINTS;
const int GraphicsDevice::Lines            = GL_LINES;
const int GraphicsDevice::LineLoop         = GL_LINE_LOOP;
const int GraphicsDevice::Triangles        = GL_TRIANGLES;
const int GraphicsDevice::TriangleStrip    = GL_TRIANGLE_STRIP;
const int GraphicsDevice::Texture2D        = GL_TEXTURE_2D;
const int GraphicsDevice::UnsignedInt      = GL_UNSIGNED_INT;
const int GraphicsDevice::Ambient          = GL_AMBIENT;
const int GraphicsDevice::Diffuse          = GL_DIFFUSE;
const int GraphicsDevice::Specular         = GL_SPECULAR;
const int GraphicsDevice::Position         = GL_POSITION;
const int GraphicsDevice::Emission         = GL_EMISSION;
const int GraphicsDevice::One              = GL_ONE;
const int GraphicsDevice::SrcAlpha         = GL_SRC_ALPHA;
const int GraphicsDevice::OneMinusSrcAlpha = GL_ONE_MINUS_SRC_ALPHA;
const int GraphicsDevice::OneMinusDstColor = GL_ONE_MINUS_DST_COLOR;

#ifdef LFL_MOBILE
const int GraphicsDevice::Fill              = 0;
const int GraphicsDevice::Line              = 0;
const int GraphicsDevice::Point             = 0;
const int GraphicsDevice::Polygon           = 0;
const int GraphicsDevice::GLPreferredBuffer = GL_UNSIGNED_BYTE;
const int GraphicsDevice::GLInternalFormat  = GL_RGBA;
#else
const int GraphicsDevice::Fill              = GL_FILL;
const int GraphicsDevice::Line              = GL_LINE;
const int GraphicsDevice::Point             = GL_POINT;
const int GraphicsDevice::Polygon           = GL_POLYGON;
#ifdef __APPLE__
const int GraphicsDevice::GLPreferredBuffer = GL_UNSIGNED_INT_8_8_8_8_REV;
#else
const int GraphicsDevice::GLPreferredBuffer = GL_UNSIGNED_BYTE;
#endif
const int GraphicsDevice::GLInternalFormat  = GL_RGBA;
#endif

#ifdef LFL_QT
static bool lfl_qt_init = false;
class QTWindow : public QWindow {
  Q_OBJECT
  public:
  QSocketNotifier *wait_forever_socket=0;
  bool init=0, grabbed=0, frame_on_mouse_input=0, frame_on_keyboard_input=0, reenable_wait_forever_socket=0;
  LFL::Window *lfl_window=0;
  point mouse_p;

  void MyInit() {
    CHECK(!lfl_window->gl);
    QOpenGLContext *glc = new QOpenGLContext(this);
    if (LFL::screen->gl) glc->setShareContext((QOpenGLContext*)LFL::screen->gl);
    glc->setFormat(requestedFormat());
    glc->create();
    lfl_window->gl = glc;
    Window::MakeCurrent(lfl_window);
    lfl_window->gd->initializeOpenGLFunctions();
    if (lfl_qt_init) LFL::app->StartNewWindow(lfl_window);
  }
  bool event(QEvent *event) {
    if (event->type() != QEvent::UpdateRequest) return QWindow::event(event);
    if (!init && (init=1)) MyInit();
    if (!lfl_qt_init && (lfl_qt_init=true)) if (LFLQTInit() < 0) { app->Free(); lfl_qapp->exit(); return true; }
    if (!LFL::screen || LFL::screen->impl != this) LFL::Window::MakeCurrent(lfl_window);
    app->EventDrivenFrame(true);
    if (!app->run) { app->Free(); lfl_qapp->exit(); return true; }
    if (reenable_wait_forever_socket && !(reenable_wait_forever_socket=0)) wait_forever_socket->setEnabled(true);
    if (LFL::screen->target_fps) RequestRender();
    return QWindow::event(event);
  }
  void resizeEvent(QResizeEvent *ev) {
    QWindow::resizeEvent(ev);
    if (!init) return; 
    LFL::Window::MakeCurrent(lfl_window);
    LFL::screen->Reshaped(ev->size().width(), ev->size().height());
    RequestRender();
  }
  void keyPressEvent  (QKeyEvent *ev) { keyEvent(ev, true); }
  void keyReleaseEvent(QKeyEvent *ev) { keyEvent(ev, false); }
  void keyEvent       (QKeyEvent *ev, bool down) {
    if (!init) return;
    ev->accept();
    int key = GetKeyCode(ev), fired = key ? KeyPress(key, down) : 0;
    if (fired && frame_on_keyboard_input) RequestRender();
  }
  void mouseReleaseEvent(QMouseEvent *ev) { QWindow::mouseReleaseEvent(ev); mouseClickEvent(ev, false); }
  void mousePressEvent  (QMouseEvent *ev) { QWindow::mousePressEvent(ev);   mouseClickEvent(ev, true);
#if 0
    if (ev->button() == Qt::RightButton) {
      QMenu menu;
      QAction* openAct = new QAction("Open...", this);
      menu.addAction(openAct);
      menu.addSeparator();
      menu.exec(mapToGlobal(ev->pos()));
    }
#endif
  }
  void mouseClickEvent  (QMouseEvent *ev, bool down) {
    if (!init) return;
    int fired = LFL::app->input->MouseClick(GetMouseButton(ev), down, GetMousePosition(ev));
    if (fired && frame_on_mouse_input) RequestRender();
  }
  void mouseMoveEvent(QMouseEvent *ev) {
    QWindow::mouseMoveEvent(ev);
    if (!init) return;
    point p = GetMousePosition(ev);
    int fired = LFL::app->input->MouseMove(p, p - mouse_p);
    if (fired && frame_on_mouse_input) RequestRender();
    if (!grabbed) mouse_p = p;
    else        { mouse_p = point(width()/2, height()/2); QCursor::setPos(mapToGlobal(QPoint(mouse_p.x, mouse_p.y))); }
  }
  void RequestRender() { QCoreApplication::postEvent(this, new QEvent(QEvent::UpdateRequest)); }
  void DelWaitForeverSocket(Socket fd) {}
  void AddWaitForeverSocket(Socket fd) {
    CHECK(!wait_forever_socket);
    wait_forever_socket = new QSocketNotifier(fd, QSocketNotifier::Read, this);
    lfl_qapp->connect(wait_forever_socket, SIGNAL(activated(int)), this, SLOT(ReadInputChannel(int)));
  }
  static unsigned GetKeyCode(QKeyEvent *ev) { int k = ev->key(); return k < 256 && isalpha(k) ? ::tolower(k) : k; }
  static point    GetMousePosition(QMouseEvent *ev) { return point(ev->x(), LFL::screen->height - ev->y()); }
  static unsigned GetMouseButton  (QMouseEvent *ev) {
    int b = ev->button();
    if      (b == Qt::LeftButton)  return 1;
    else if (b == Qt::RightButton) return 2;
    return 0;
  }
  public slots:
  void ReadInputChannel(int fd) {
    reenable_wait_forever_socket = true; 
    wait_forever_socket->setEnabled(false);
    RequestRender(); 
  }
};
#include "video.moc"
extern "C" void QTTriggerFrame          (void *w)            { ((QTWindow*)w)->RequestRender(); }
extern "C" void QTSetWaitForeverMouse   (void *w, bool v)    { ((QTWindow*)w)->frame_on_mouse_input    = v; }
extern "C" void QTSetWaitForeverKeyboard(void *w, bool v)    { ((QTWindow*)w)->frame_on_keyboard_input = v; }
extern "C" void QTAddWaitForeverSocket  (void *w, Socket fd) { ((QTWindow*)w)->AddWaitForeverSocket(fd); }
extern "C" void QTDelWaitForeverSocket  (void *w, Socket fd) { ((QTWindow*)w)->DelWaitForeverSocket(fd); }
#endif

#ifndef LFL_QT
struct OpenGLES1 : public GraphicsDevice {
#else
struct OpenGLES1 : public GraphicsDevice, public QTWindow {
#endif
  int target_matrix;
  OpenGLES1() : target_matrix(-1) { default_color.push_back(Color(1.0, 1.0, 1.0, 1.0)); }
  void Init() {
    shader = &app->video->shader_default; 
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glEnableClientState(GL_VERTEX_ARRAY);
#if !defined(LFL_IPHONE) && !defined(LFL_ANDROID)
    float black[]={0,0,0,1};
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);
    glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, 1);
    // glLightModelf(GL_LIGHT_MODEL_COLOR_CONTROL, GL_SEPARATE_SPECULAR_COLOR);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
#endif
    GDDebug("Init");
  }
  void UpdateColor() { const Color &c = default_color.back(); glColor4f(c.r(), c.g(), c.b(), c.a()); }
  bool ShaderSupport() {
#if defined(LFL_ANDROID) || defined(LFL_IPHONE)
    return false;
#endif
    const char *ver = (const char*)glGetString(GL_VERSION);
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
#if defined(LFL_IPHONE) || defined(LFL_ANDROID)
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
    glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
    glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
    glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
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
    glDrawElements(pt, np, it, (char*)index + o);
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
    shader = X_or_Y(S, &app->video->shader_default); 
    glUseProgram(shader->ID);
    GDDebug("Shader=", shader->name);
  }
};

#ifdef LFL_GLES2
#ifndef LFL_QT
struct OpenGLES2 : public GraphicsDevice {
#else
struct OpenGLES2 : public GraphicsDevice, public QTWindow {
#endif
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

  void Init() {
    INFO("GraphicsDevice::Init");
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
    Shader::Create("lfapp",          vertex_shader, pixel_shader, ShaderDefines(1,0,1,0), &app->video->shader_default);
    Shader::Create("lfapp_cubemap",  vertex_shader, pixel_shader, ShaderDefines(1,0,0,1), &app->video->shader_cubemap);
    Shader::Create("lfapp_normals",  vertex_shader, pixel_shader, ShaderDefines(0,1,1,0), &app->video->shader_normals);
    Shader::Create("lfapp_cubenorm", vertex_shader, pixel_shader, ShaderDefines(0,1,0,1), &app->video->shader_cubenorm);
    GDDebug("Init");
    UseShader((shader = 0));
    VertexPointer(0, 0, 0, 0, NULL, deferred.vertexbuffer_size, NULL, true, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
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
  void DisableCubeMap()     { if (Changed(&cubemap_on,    false)) { UpdateShader();  glUniform1i(shader->uniform_cubeon, 0); }                                                                                 GDDebug("CubeMap=", 0); }
  void BindCubeMap(int n)   { if (Changed(&cubemap_on,    true))  { UpdateShader();  glUniform1i(shader->uniform_cubeon, 1); } glUniform1i(shader->uniform_cubetex, 0); glBindTexture(GL_TEXTURE_CUBE_MAP, n); GDDebug("CubeMap=", n); }
  void ActiveTexture(int n) { if (Changed(&bound_texture.l, n))   { ClearDeferred(); glActiveTexture(n ? GL_TEXTURE1 : GL_TEXTURE0); } GDDebug("ActivteTexture=", n); }
  void EnableLight(int n) {}
  void DisableLight(int n) {}
  void TextureGenLinear() {}
  void TextureGenReflection() {}
  void BindTexture(int t, int n) {
    if (!Changed(&bound_texture, BoundTexture{ t, n, 0 })) return;
    if (!texture_on) EnableTexture();
    else ClearDeferred();
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
    if (n != 0) { ERROR("ignoring Light(", n, ")"); return; }

    if      (t == GL_POSITION) { light_pos=1;   light[n].pos = modelview_matrix.back().Transform(v4(v)); }
    else if (t == GL_AMBIENT)  { light_color=1; light[n].color.ambient  = Color(v); }
    else if (t == GL_DIFFUSE)  { light_color=1; light[n].color.diffuse  = Color(v); }
    else if (t == GL_SPECULAR) { light_color=1; light[n].color.specular = Color(v); }

    if (light_pos)   { shader->dirty_light_pos  [n] = app->video->shader_cubenorm.dirty_light_pos  [n] = app->video->shader_normals.dirty_light_pos  [n] = 1; }
    if (light_color) { shader->dirty_light_color[n] = app->video->shader_cubenorm.dirty_light_color[n] = app->video->shader_normals.dirty_light_color[n] = 1; }
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
    if (first) { glGenBuffers(1, (GLuint*)out); dirty = true; }
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
    glVertexAttribPointer(slot, attr.m, attr.t, GL_FALSE, attr.w, (GLvoid*)(long)attr.o);
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
    if (cubemap_on && normals_on) UseShader(&app->video->shader_cubenorm);
    else if          (cubemap_on) UseShader(&app->video->shader_cubemap);
    else if          (normals_on) UseShader(&app->video->shader_normals);
    else                          UseShader(&app->video->shader_default);
  }
  void UpdateColor()  { ClearDeferred(); dirty_color = true; }
  void UpdateMatrix() { ClearDeferred(); dirty_matrix = true; }
  void UpdateMaterial() {
    ClearDeferred();
    shader->dirty_material = app->video->shader_cubenorm.dirty_material = app->video->shader_normals.dirty_material = true;
  }
  void UpdateVertex() {
    glEnableVertexAttribArray(shader->slot_position);
    VertexAttribPointer(shader->slot_position, vertex_attr);
  }
  void UpdateNormals() {
    bool supports = shader->slot_normal >= 0;
    glUniform1i(shader->uniform_normalon, normals_on && supports);
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
    if (*out == -1) { glGenBuffers(1, (GLuint*)out); dirty = true; }
    if (*out != bound_indexbuffer) { 
      bound_indexbuffer = *out;
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *out);
    }
    if (dirty) glBufferData(GL_ELEMENT_ARRAY_BUFFER, l, index, input_dirty ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);

    GDDebug("DrawElements-Pre(", pt, ", ", np, ", ", it, ", ", o, ", ", index, ", ", l, ", ", dirty, ")");
    PushDirtyState();
    glDrawElements(pt, np, it, (GLvoid*)(long)o);
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
      glUniformMatrix4fv(shader->uniform_modelviewproj, 1, 0, m[0]);
      glUniformMatrix4fv(shader->uniform_modelview,     1, 0, modelview_matrix.back()[0]);
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

// Shader interaface
int GraphicsDevice::CreateProgram() { return glCreateProgram(); }
int GraphicsDevice::CreateShader(int t) { return glCreateShader(t); }
void GraphicsDevice::ShaderSource(int shader, int count, const char **source, int *len) { glShaderSource(shader, count, source, len); }
void GraphicsDevice::CompileShader(int shader) {
  char buf[1024] = {0}; int l=0;
  glCompileShader(shader);
  glGetShaderInfoLog(shader, sizeof(buf), &l, buf);
  if (l) INFO(buf);
}
void GraphicsDevice::AttachShader(int prog, int shader) { glAttachShader(prog, shader); }
void GraphicsDevice::BindAttribLocation(int prog, int loc, const string &name) { glBindAttribLocation(prog, loc, name.c_str()); }
void GraphicsDevice::LinkProgram(int prog) {
  char buf[1024] = {0}; int l=0;
  glLinkProgram(prog);
  glGetProgramInfoLog(prog, sizeof(buf), &l, buf);
  if (l) INFO(buf);
  GLint link_status;
  glGetProgramiv(prog, GL_LINK_STATUS, &link_status);
  if (link_status != GL_TRUE) FATAL("link failed");
}
void GraphicsDevice::GetProgramiv(int p, int t, int *out) { glGetProgramiv(p, t, out); }
void GraphicsDevice::GetIntegerv(int t, int *out) { glGetIntegerv(t, out); }
int GraphicsDevice::GetAttribLocation (int prog, const string &name) { return glGetAttribLocation (prog, name.c_str()); }
int GraphicsDevice::GetUniformLocation(int prog, const string &name) { return glGetUniformLocation(prog, name.c_str()); }
void GraphicsDevice::Uniform1i(int u, int v) { glUniform1i(u, v); }
void GraphicsDevice::Uniform1f(int u, float v) { glUniform1f(u, v); }
void GraphicsDevice::Uniform2f(int u, float v1, float v2) { glUniform2f(u, v1, v2); }
void GraphicsDevice::Uniform3f(int u, float v1, float v2, float v3) { glUniform3f(u, v1, v2, v3); }
void GraphicsDevice::Uniform4f(int u, float v1, float v2, float v3, float v4) { glUniform4f(u, v1, v2, v3, v4); }
void GraphicsDevice::Uniform3fv(int u, int n, const float *v) { glUniform3fv(u, n, v); }

// Common layer
void GraphicsDevice::Flush() { ClearDeferred(); glFlush(); }
void GraphicsDevice::Clear() { glClear(GL_COLOR_BUFFER_BIT | (draw_mode == DrawMode::_3D ? GL_DEPTH_BUFFER_BIT : 0)); }
void GraphicsDevice::ClearColor(const Color &c) { glClearColor(c.r(), c.g(), c.b(), c.a()); }
void GraphicsDevice::PushColor() { default_color.push_back(default_color.back()); UpdateColor();  }
void GraphicsDevice::PopColor() {
  if      (default_color.size() >= 1) default_color.pop_back();
  else if (default_color.size() == 1) default_color.back() = Color(1.0, 1.0, 1.0, 1.0);
  UpdateColor();
}
void GraphicsDevice::PointSize(float n) { glPointSize(n); }
void GraphicsDevice::LineWidth(float n) { glLineWidth(n); }
void GraphicsDevice::DelTextures(int n, const unsigned *id) { glDeleteTextures(n, id); }
void GraphicsDevice::GenTextures(int t, int n, unsigned *out) {
  ClearDeferred();
  for (int i=0; i<n; i++) CHECK_EQ(0, out[i]);
  if (t == GL_TEXTURE_CUBE_MAP) glEnable(GL_TEXTURE_CUBE_MAP);
  glGenTextures(n, out);
  for (int i=0; i<n; i++) {
    glBindTexture(t, out[i]);
    glTexParameteri(t, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(t, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  }
}

void GraphicsDevice::CheckForError(const char *file, int line) {
  GLint gl_error=0, gl_validate_status=0;
  if ((gl_error = glGetError())) {
    ERROR(file, ":", line, " gl error: ", gl_error);
    BreakHook();
#ifdef LFL_GLES2
    if (app->video->opengles_version == 2) {
      Shader *shader = ((OpenGLES2*)screen->gd)->shader;
      glValidateProgram(shader->ID);
      glGetProgramiv(shader->ID, GL_VALIDATE_STATUS, &gl_validate_status);
      if (gl_validate_status != GL_TRUE) ERROR(shader->name, ": gl validate status ", gl_validate_status);

      char buf[1024]; int len;
      glGetProgramInfoLog(shader->ID, sizeof(buf), &len, buf);
      if (len) INFO(buf);
    }
#endif
  }
}

void GraphicsDevice::EnableDepthTest()  {  glEnable(GL_DEPTH_TEST); glDepthMask(GL_TRUE);  GDDebug("DepthTest=1"); }
void GraphicsDevice::DisableDepthTest() { glDisable(GL_DEPTH_TEST); glDepthMask(GL_FALSE); GDDebug("DepthTest=0"); }
void GraphicsDevice::DisableBlend() { if (Changed(&blend_enabled, false)) { ClearDeferred(); glDisable(GL_BLEND);                                                    GDDebug("Blend=0"); } }
void GraphicsDevice::EnableBlend()  { if (Changed(&blend_enabled, true )) { ClearDeferred();  glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); GDDebug("Blend=1"); } }
void GraphicsDevice::BlendMode(int sm, int dm) { glBlendFunc(sm, dm); GDDebug("BlendMode=", sm, ",", dm); }
void GraphicsDevice::RestoreViewport(int dm) { ViewPort(screen->Box()); DrawMode(dm); }
void GraphicsDevice::DrawMode(int dm, bool flush) { return DrawMode(dm, screen->width, screen->height, flush); }
void GraphicsDevice::DrawMode(int dm, int W, int H, bool flush) {
  if (draw_mode == dm && !flush) return;
  bool _2D = (draw_mode = dm) == DrawMode::_2D;
  Color4f(1,1,1,1);
  MatrixProjection();
  LoadIdentity();
  if (FLAGS_rotate_view) Rotatef(FLAGS_rotate_view,0,0,1);

  if (_2D) Ortho(0, W, 0, H, 0, 100);
  else {
    float aspect=(float)W/H;
    double top = tan(FLAGS_field_of_view * M_PI/360.0) * FLAGS_near_plane;
    screen->gd->Frustum(aspect*-top, aspect*top, -top, top, FLAGS_near_plane, FLAGS_far_plane);
  }

  if (_2D) DisableDepthTest();
  else     EnableDepthTest();

  MatrixModelview();
  LoadIdentity();
  Scene::Select();
  if (_2D) EnableLayering();
}

void GraphicsDevice::LookAt(const v3 &pos, const v3 &targ, const v3 &up) {
  v3 Z = pos - targ;       Z.Norm();
  v3 X = v3::Cross(up, Z); X.Norm();
  v3 Y = v3::Cross(Z,  X); Y.Norm();
  float m[16] = {
    X.x, Y.x, Z.x, 0.0,
    X.y, Y.y, Z.y, 0.0,
    X.z, Y.z, Z.z, 0.0,
    0.0, 0.0, 0.0, 1.0
  };
  Mult(m);
  Translate(-pos.x, -pos.y, -pos.z);
}

void GraphicsDevice::ViewPort(Box w) {
  if (FLAGS_swap_axis) w.swapaxis(screen->width, screen->height);
  ClearDeferred();
  glViewport(w.x, w.y, w.w, w.h);
}

void GraphicsDevice::Scissor(Box w) {
  if (FLAGS_swap_axis) w.swapaxis(screen->width, screen->height);
  ClearDeferred();
  glEnable(GL_SCISSOR_TEST);
  glScissor(w.x, w.y, w.w, w.h);
}

void GraphicsDevice::PushScissor(Box w) {
  auto &ss = scissor_stack.back();
  if (ss.empty()) ss.push_back(w);
  else ss.push_back(w.Intersect(ss.back()));
  screen->gd->Scissor(ss.back());
}

void GraphicsDevice::PopScissor() {
  auto &ss = scissor_stack.back();
  if (ss.size()) ss.pop_back();
  if (ss.size()) screen->gd->Scissor(ss.back());
  else { ClearDeferred(); glDisable(GL_SCISSOR_TEST); }
}

void GraphicsDevice::PushScissorStack() {
  scissor_stack.push_back(vector<Box>());
  ClearDeferred();
  glDisable(GL_SCISSOR_TEST);
}

void GraphicsDevice::PopScissorStack() {
  CHECK_GT(scissor_stack.size(), 1);
  scissor_stack.pop_back();
  screen->gd->Scissor(scissor_stack.back().back());
}

void GraphicsDevice::DrawPixels(const Box &b, const Texture &tex) {
  Texture temp;
  temp.Resize(tex.width, tex.height, tex.pf, Texture::Flag::CreateGL);
  temp.UpdateGL(tex.buf, LFL::Box(tex.width, tex.height), Texture::Flag::FlipY); 
  b.Draw(temp.coord);
  temp.ClearGL();
}

void GraphicsDevice::GenRenderBuffers(int n, unsigned *out) { glGenRenderbuffersEXT(n, out); }
void GraphicsDevice::BindRenderBuffer(int id) { glBindRenderbufferEXT(GL_RENDERBUFFER, id); }
void GraphicsDevice::RenderBufferStorage(int d, int w, int h) { glRenderbufferStorageEXT(GL_RENDERBUFFER, d, w, h); }
void GraphicsDevice::GenFrameBuffers(int n, unsigned *out) { glGenFramebuffersEXT(n, out); }
void GraphicsDevice::BindFrameBuffer(int id) { ClearDeferred(); glBindFramebufferEXT(GL_FRAMEBUFFER, id); }
void GraphicsDevice::FrameBufferTexture(int id) { glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, id, 0); }
void GraphicsDevice::FrameBufferDepthTexture(int id) { glFramebufferRenderbufferEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, id); }
int GraphicsDevice::CheckFrameBufferStatus() { return glCheckFramebufferStatusEXT(GL_FRAMEBUFFER); }

int GraphicsDevice::VertsPerPrimitive(int primtype) {
  if (primtype == GL_TRIANGLES) return 3;
  return 0;
}

#else // LFL_HEADLESS
struct FakeGraphicsDevice : public GraphicsDevice {
  virtual void Init() {}
  virtual bool ShaderSupport() { return 0; }
  virtual void EnableTexture() {}
  virtual void DisableTexture() {}
  virtual void EnableLighting() {}
  virtual void DisableLighting() {}
  virtual void EnableNormals() {}
  virtual void DisableNormals() {}
  virtual void EnableVertexColor() {}
  virtual void DisableVertexColor() {}
  virtual void EnableLight(int n) {}
  virtual void DisableLight(int n) {}
  virtual void DisableCubeMap() {}
  virtual void BindCubeMap(int n) {}
  virtual void TextureGenLinear() {}
  virtual void TextureGenReflection() {}
  virtual void Material(int t, float *color) {}
  virtual void Light(int n, int t, float *color) {}
  virtual void BindTexture(int t, int n) {}
  virtual void ActiveTexture(int n) {}
  virtual void TexPointer(int m, int t, int w, int o, float *tex, int l, int *out, bool dirty) {}
  virtual bool VertexPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty, int) { return true; }
  virtual void ColorPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {}
  virtual void NormalPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {}
  virtual void Color4f(float r, float g, float b, float a) {}
  virtual void UpdateColor() {}
  virtual void MatrixProjection() {}
  virtual void MatrixModelview() {}
  virtual void LoadIdentity() {}
  virtual void PushMatrix() {}
  virtual void PopMatrix() {}
  virtual void GetMatrix(m44 *out) {}
  virtual void PrintMatrix() {}
  virtual void Scalef(float x, float y, float z) {}
  virtual void Rotatef(float angle, float x, float y, float z) {}
  virtual void Ortho(float l, float r, float b, float t, float nv, float fv) {}
  virtual void Frustum(float l, float r, float b, float t, float nv, float fv) {}
  virtual void Mult(const float *m) {}
  virtual void Translate(float x, float y, float z) {}
  virtual void UseShader(Shader *shader) {}
  virtual void DrawElements(int pt, int np, int it, int o, void *index, int l, int *out, bool dirty) {}
  virtual void DrawArrays(int t, int o, int n) {}
  virtual void DeferDrawArrays(int t, int o, int n) {}
};

const int GraphicsDevice::Float = 0;
const int GraphicsDevice::Points = 0;
const int GraphicsDevice::Lines = 0;
const int GraphicsDevice::LineLoop = 0;
const int GraphicsDevice::Triangles = 0;
const int GraphicsDevice::Polygon = 0;
const int GraphicsDevice::TriangleStrip = 0;
const int GraphicsDevice::Texture2D = 0;
const int GraphicsDevice::UnsignedInt = 0;
const int GraphicsDevice::Ambient = 0;
const int GraphicsDevice::Diffuse = 0;
const int GraphicsDevice::Specular = 0;
const int GraphicsDevice::Position = 0;
const int GraphicsDevice::Emission = 0;
const int GraphicsDevice::One = 0;
const int GraphicsDevice::SrcAlpha = 0;
const int GraphicsDevice::OneMinusSrcAlpha = 0;
const int GraphicsDevice::OneMinusDstColor = 0;
const int GraphicsDevice::Fill = 0;
const int GraphicsDevice::Line = 0;
const int GraphicsDevice::Point = 0;
const int GraphicsDevice::GLPreferredBuffer = 0;
const int GraphicsDevice::GLInternalFormat = 0;

int GraphicsDevice::CreateProgram() { return 0; }
int GraphicsDevice::CreateShader(int t) { return 0; }
int GraphicsDevice::GetAttribLocation (int prog, const string &name) { return 0; }
int GraphicsDevice::GetUniformLocation(int prog, const string &name) { return 0; }
void GraphicsDevice::ShaderSource(int shader, int count, const char **source, int *len) {}
void GraphicsDevice::CompileShader(int shader) {}
void GraphicsDevice::AttachShader(int prog, int shader) {}
void GraphicsDevice::BindAttribLocation(int prog, int loc, const string &name) {}
void GraphicsDevice::LinkProgram(int prog) {}
void GraphicsDevice::GetProgramiv(int p, int t, int *out) {}
void GraphicsDevice::GetIntegerv(int t, int *out) {}
void GraphicsDevice::Uniform1i(int u, int v) {}
void GraphicsDevice::Uniform1f(int u, float v) {}
void GraphicsDevice::Uniform2f(int u, float v1, float v2) {}
void GraphicsDevice::Uniform3f(int u, float v1, float v2, float v3) {}
void GraphicsDevice::Uniform4f(int u, float v1, float v2, float v3, float v4) {}
void GraphicsDevice::Uniform3fv(int u, int n, const float *v) {}
void GraphicsDevice::Flush() {}
void GraphicsDevice::Clear() {}
void GraphicsDevice::ClearColor(const Color &c) {}
void GraphicsDevice::PushColor() {}
void GraphicsDevice::PopColor() {}
void GraphicsDevice::PointSize(float n) {}
void GraphicsDevice::LineWidth(float n) {}
void GraphicsDevice::DelTextures(int n, const unsigned *id) {}
void GraphicsDevice::GenTextures(int t, int n, unsigned *out) {}
void GraphicsDevice::CheckForError(const char *file, int line) {}
void GraphicsDevice::EnableDepthTest() {}
void GraphicsDevice::DisableDepthTest() {}
void GraphicsDevice::DisableBlend() {}
void GraphicsDevice::EnableBlend() {}
void GraphicsDevice::BlendMode(int sm, int dm) {}
void GraphicsDevice::RestoreViewport(int dm) {}
void GraphicsDevice::DrawMode(int dm, bool flush) {}
void GraphicsDevice::DrawMode(int dm, int W, int H, bool flush) {}
void GraphicsDevice::LookAt(const v3 &pos, const v3 &targ, const v3 &up) {}
void GraphicsDevice::ViewPort(Box w) {}
void GraphicsDevice::Scissor(Box w) {}
void GraphicsDevice::PushScissor(Box w) {}
void GraphicsDevice::PopScissor() {}
void GraphicsDevice::PushScissorStack() {}
void GraphicsDevice::PopScissorStack() {}
void GraphicsDevice::DrawPixels(const Box&, const Texture&) {}
void GraphicsDevice::GenRenderBuffers(int n, unsigned *out) {}
void GraphicsDevice::BindRenderBuffer(int id) {}
void GraphicsDevice::RenderBufferStorage(int d, int w, int h) {}
void GraphicsDevice::GenFrameBuffers(int n, unsigned *out) {}
void GraphicsDevice::BindFrameBuffer(int id) {}
void GraphicsDevice::FrameBufferTexture(int id) {}
void GraphicsDevice::FrameBufferDepthTexture(int id) {}
int GraphicsDevice::CheckFrameBufferStatus() { return 0; }
int GraphicsDevice::GraphicsDevice::VertsPerPrimitive(int primtype) { return 0; }

bool Window::Create(Window *W) { screen->gd = new FakeGraphicsDevice(); Window::active[W->id] = W; return true; }
void Window::Close(Window *W) {}
void Window::MakeCurrent(Window *W) {}
#endif // LFL_HEADLESS

#ifdef LFL_ANDROIDVIDEO
struct AndroidVideoModule : public Module {
  int Init() {
    INFO("AndroidVideoModule::Init()");
    if (AndroidVideoInit(&app->video.opengles_version)) return -1;
    CHECK(!screen->id);
    screen->id = screen;
    Window::active[screen->id] = screen;
    return 0;
  }
};
bool Window::Create(Window *W) { return true; }
void Window::Close(Window *W) {}
void Window::MakeCurrent(Window *W) {}
#endif

#ifdef LFL_IPHONEVIDEO
extern "C" void iPhoneVideoSwap();
struct IPhoneVideoModule : public Module {
  int Init() {
    INFO("IPhoneVideoModule::Init()");
    CHECK(!screen->id);
    NativeWindowInit();
    NativeWindowSize(&screen->width, &screen->height);
    CHECK(screen->id);
    Window::active[screen->id] = screen;
    return 0;
  }
};
bool Window::Create(Window *W) { return false; }
void Window::Close(Window *W) {}
void Window::MakeCurrent(Window *W) {}
#endif

#ifdef LFL_OSXVIDEO
extern "C" void OSXVideoSwap(void*);
extern "C" void *OSXCreateWindow(int W, int H, struct NativeWindow *nw);
extern "C" void OSXDestroyWindow(void *O);
extern "C" void OSXMakeWindowCurrent(void *O);
extern "C" void OSXSetWindowSize(void*, int W, int H);
extern "C" void OSXSetWindowTitle(void *O, const char *v);
extern "C" void OSXSetWindowResizeIncrements(void *O, float x, float y);
extern "C" void OSXSetWindowTransparency(void *O, float v);
extern "C" void *OSXCreateGLContext(void *O);
struct OSXVideoModule : public Module {
  int Init() {
    INFO("OSXVideoModule::Init()");
    NativeWindowInit();
    NativeWindowSize(&screen->width, &screen->height);
    CHECK(Window::Create(screen));
    return 0;
  }
};
bool Window::Create(Window *W) { 
  W->id = OSXCreateWindow(W->width, W->height, W);
  if (W->id) Window::active[W->id] = W;
  OSXSetWindowTitle(W->id, W->caption.c_str());
  return true; 
}
void Window::MakeCurrent(Window *W) { 
  if (W) OSXMakeWindowCurrent((screen = W)->id);
}
void Window::Close(Window *W) {
  Window::active.erase(W->id);
  if (Window::active.empty()) app->run = false;
  if (app->window_closed_cb) app->window_closed_cb(W);
  // OSXDestroyWindow(W->id);
  screen = 0;
}
#endif

#ifdef LFL_WINVIDEO
struct WinVideoModule : public Module {
  int Init() {
    INFO("WinVideoModule::Init()");
    CHECK(Window::Create(screen));
    return 0;
  }
};

bool Window::Create(Window *W) {
  static WinApp *winapp = Singleton<WinApp>::Get();
  ONCE({ winapp->CreateClass(); });
  RECT r = { 0, 0, W->width, W->height };
  DWORD dwStyle = WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS;
  if (!AdjustWindowRect(&r, dwStyle, 0)) return ERRORv(false, "AdjustWindowRect");
  HWND hWnd = CreateWindow(app->name.c_str(), W->caption.c_str(), dwStyle, 0, 0, r.right-r.left, r.bottom-r.top, NULL, NULL, winapp->hInst, NULL);
  if (!hWnd) return ERRORv(false, "CreateWindow: ", GetLastError());
  HDC hDC = GetDC(hWnd);
  PIXELFORMATDESCRIPTOR pfd = { sizeof(PIXELFORMATDESCRIPTOR), 1, PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW | PFD_DOUBLEBUFFER,
    PFD_TYPE_RGBA, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, PFD_MAIN_PLANE, 0, 0, 0, 0, };
  int pf = ChoosePixelFormat(hDC, &pfd);
  if (!pf) return ERRORv(false, "ChoosePixelFormat: ", GetLastError());
  if (SetPixelFormat(hDC, pf, &pfd) != TRUE) return ERRORv(false, "SetPixelFormat: ", GetLastError());
  if (!(W->gl = wglCreateContext(hDC))) return ERRORv(false, "wglCreateContext: ", GetLastError());
  W->surface = hDC;
  W->impl = new WinWindow();
  Window::active[(W->id = hWnd)] = W;
  INFOf("Window::Create %p %p %p (%p)", W->id, W->surface, W->gl, W);
  MakeCurrent(W);
  ShowWindow(hWnd, winapp->nCmdShow);
  app->scheduler.Wakeup(0);
  return true;
}
void Window::MakeCurrent(Window *W) { if (W) wglMakeCurrent((HDC)W->surface, (HGLRC)W->gl); }
void Window::Close(Window *W) {
  delete (WinWindow*)W->impl;
  Window::active.erase(W->id);
  if (Window::active.empty()) app->run = false;
  if (app->window_closed_cb) app->window_closed_cb(W);
  screen = 0;
}
#endif

#ifdef LFL_X11VIDEO
struct X11VideoModule : public Module {
  Display *display = 0;
  XVisualInfo *vi = 0;
  int Init() {
    GLint att[] = { GLX_RGBA, GLX_DEPTH_SIZE, 16, GLX_DOUBLEBUFFER, None };
    if (!(display = XOpenDisplay(NULL))) return ERRORv(-1, "XOpenDisplay");
    if (!(vi = glXChooseVisual(display, 0, att))) return ERRORv(-1, "glXChooseVisual");
    app->scheduler.system_event_socket = ConnectionNumber(display);
    app->scheduler.AddWaitForeverSocket(app->scheduler.system_event_socket, SocketSet::READABLE, 0);
    SystemNetwork::SetSocketCloseOnExec(app->scheduler.system_event_socket, true);
    INFO("X11VideoModule::Init()");
    return Window::Create(screen) ? 0 : -1;
  }
  int Free() {
    XCloseDisplay(display);
    return 0;
  }
};
bool Window::Create(Window *W) {
  X11VideoModule *video = dynamic_cast<X11VideoModule*>(app->video.impl);
  ::Window root = DefaultRootWindow(video->display);
  XSetWindowAttributes swa;
  swa.colormap = XCreateColormap(video->display, root, video->vi->visual, AllocNone);
  swa.event_mask = ExposureMask | KeyPressMask | KeyReleaseMask | PointerMotionMask |
    ButtonPressMask | ButtonReleaseMask | StructureNotifyMask;
  ::Window win = XCreateWindow(video->display, root, 0, 0, W->width, W->height, 0, video->vi->depth,
                               InputOutput, video->vi->visual, CWColormap | CWEventMask, &swa);
  Atom protocols[] = { XInternAtom(video->display, "WM_DELETE_WINDOW", 0) };
  XSetWMProtocols(video->display, win, protocols, sizeofarray(protocols));
  if (!(W->id = (void*)(win))) return ERRORv(false, "XCreateWindow");
  XMapWindow(video->display, win);
  XStoreName(video->display, win, W->caption.c_str());
  if (!(W->gl = glXCreateContext(video->display, video->vi, NULL, GL_TRUE))) return ERRORv(false, "glXCreateContext");
  W->surface = video->display;
  Window::active[W->id] = W;
  MakeCurrent(W);
  return true;
}
void Window::Close(Window *W) {
  Display *display = static_cast<Display*>(W->surface);
  glXMakeCurrent(display, None, NULL);
  glXDestroyContext(display, static_cast<GLXContext>(W->gl));
  XDestroyWindow(display, reinterpret_cast<::Window>(W->id));
  Window::active.erase(W->id);
  if (Window::active.empty()) app->run = false;
  if (app->window_closed_cb) app->window_closed_cb(W);
  screen = 0;
}
void Window::MakeCurrent(Window *W) {
  glXMakeCurrent(static_cast<Display*>(W->surface), (::Window)(W->id), static_cast<GLXContext>(W->gl));
}
#endif // LFL_X11VIDEO

#ifdef LFL_XTVIDEO
struct XTVideoModule : public Module {
  Widget toplevel = 0;
  char *argv[2] = { 0, 0 };
  int argc = 1;

  int Init() {
    XtAppContext xt_app;
    argv[0] = &app->progname[0];
    toplevel = XtOpenApplication(&xt_app, screen->caption.c_str(), NULL, 0, &argc, argv,
                                 NULL, applicationShellWidgetClass, NULL, 0);
    INFO("XTideoModule::Init()");
    return Window::Create(screen) ? 0 : -1;
  }
};
bool Window::Create(Window *W) {
  XTVideoModule *video = dynamic_cast<XTVideoModule*>(app->video.impl);
  W->surface = XtDisplay((::Widget)W->impl);
  W->id = XmCreateFrame(video->toplevel, "frame", NULL, 0);
  W->impl = video->toplevel;
  return true;
}
void Window::Close(Window *W) {}
void Window::MakeCurrent(Window *W) {}
#endif // LFL_XTVIDEO

#ifdef LFL_QT
struct QTVideoModule : public Module {
  int Init() {
    INFO("QTVideoModule::Init()");
    return 0;
  }
};
bool Window::Create(Window *W) {
  CHECK(!W->id && !W->gd);
  OpenGLES2 *gd = new OpenGLES2();
  QWindow *qwin = (QWindow*)gd;
  QTWindow *my_qwin = (QTWindow*)gd;
  my_qwin->lfl_window = W;

  QSurfaceFormat format;
  format.setSamples(16);
  qwin->setFormat(format);
  qwin->setSurfaceType(QWindow::OpenGLSurface);
  qwin->setWidth(W->width);
  qwin->setHeight(W->height);
  qwin->show();
  gd->RequestRender();

  W->id = qwin;
  W->gd = gd;
  W->impl = my_qwin;
  Window::active[W->id] = W;
  return true;
}
void Window::Close(Window *W) {
  Window::active.erase(W->id);
  if (Window::active.empty()) app->run = false;
  if (app->window_closed_cb) app->window_closed_cb(W);
  screen = 0;
}
void Window::MakeCurrent(Window *W) {
  screen = W; 
  ((QOpenGLContext*)screen->gl)->makeCurrent((QWindow*)screen->id);
}
void Mouse::GrabFocus()    { ((QTWindow*)screen->impl)->grabbed=1; ((QWindow*)screen->id)->setCursor(Qt::BlankCursor); app->grab_mode.On();  screen->cursor_grabbed=true;  }
void Mouse::ReleaseFocus() { ((QTWindow*)screen->impl)->grabbed=0; ((QWindow*)screen->id)->unsetCursor();              app->grab_mode.Off(); screen->cursor_grabbed=false; }
#endif // LFL_QT

#ifdef LFL_WXWIDGETS
}; // namespace LFL
struct LFLWxWidgetCanvas : public wxGLCanvas {
  wxGLContext *context=0;
  NativeWindow *screen=0;
  bool frame_on_keyboard_input=0, frame_on_mouse_input=0;
  virtual ~LFLWxWidgetCanvas() { delete context; }
  LFLWxWidgetCanvas(NativeWindow *s, wxFrame *parent, int *args) :
    wxGLCanvas(parent, wxID_ANY, args, wxDefaultPosition, wxDefaultSize, wxFULL_REPAINT_ON_RESIZE),
    context((wxGLContext*)s->gl), screen(s) {}
  void OnPaint(wxPaintEvent& event) {
    wxPaintDC(this);
    SetCurrent(*context);
    if (LFL::app->run) LFAppFrame();
    else exit(0);
  }
  void OnMouseMove(wxMouseEvent& event) {
    SetNativeWindow(screen);
    LFL::point p = GetMousePosition(event);
    int fired = LFL::app->input.MouseMove(p, p - LFL::screen->mouse);
    if (fired && frame_on_mouse_input) Refresh();
  }
  void OnMouseDown(wxMouseEvent& event) { OnMouseClick(1, true,  GetMousePosition(event)); }
  void OnMouseUp  (wxMouseEvent& event) { OnMouseClick(1, false, GetMousePosition(event)); }
  void OnMouseClick(int button, bool down, const LFL::point &p) {
    SetNativeWindow(screen);
    int fired = LFL::app->input.MouseClick(button, down, p);
    if (fired && frame_on_mouse_input) Refresh();
  }
  void OnKeyDown(wxKeyEvent& event) { OnKeyEvent(GetKeyCode(event), true); }
  void OnKeyUp  (wxKeyEvent& event) { OnKeyEvent(GetKeyCode(event), false); }
  void OnKeyEvent(int key, bool down) {
    SetNativeWindow(screen);
    int fired = key ? KeyPress(key, down) : 0;
    if (fired && frame_on_keyboard_input) Refresh();
  }
  static int GetKeyCode(wxKeyEvent& event) {
    int key = event.GetUnicodeKey();
    if (key == WXK_NONE) key = event.GetKeyCode();
    return key < 256 && isalpha(key) ? ::tolower(key) : key;
  }
  static LFL::point GetMousePosition(wxMouseEvent& event) {
    return LFL::Input::TransformMouseCoordinate(LFL::point(event.GetX(), event.GetY()));
  }
  DECLARE_EVENT_TABLE()
};
BEGIN_EVENT_TABLE(LFLWxWidgetCanvas, wxGLCanvas)
  EVT_PAINT    (LFLWxWidgetCanvas::OnPaint)
  EVT_KEY_DOWN (LFLWxWidgetCanvas::OnKeyDown)
  EVT_KEY_UP   (LFLWxWidgetCanvas::OnKeyUp)
  EVT_LEFT_DOWN(LFLWxWidgetCanvas::OnMouseDown)
  EVT_LEFT_UP  (LFLWxWidgetCanvas::OnMouseUp)
  EVT_MOTION   (LFLWxWidgetCanvas::OnMouseMove)
END_EVENT_TABLE()

struct LFLWxWidgetFrame : public wxFrame {
  LFLWxWidgetCanvas *canvas=0;
  LFLWxWidgetFrame(LFL::Window *w) : wxFrame(NULL, wxID_ANY, wxString::FromUTF8(w->caption.c_str())) {
    int args[] = { WX_GL_RGBA, WX_GL_DOUBLEBUFFER, WX_GL_DEPTH_SIZE, 16, 0 };
    canvas = new LFLWxWidgetCanvas(w, this, args);
    SetClientSize(w->width, w->height);
    wxMenu *menu = new wxMenu;
    menu->Append(wxID_NEW);
    menu->Append(wxID_CLOSE);
    wxMenuBar *menuBar = new wxMenuBar;
    menuBar->Append(menu, wxT("&Window"));
    SetMenuBar(menuBar);
    if (w->gl) Show();
  }
  void OnClose(wxCommandEvent& event) { Close(true); }
  void OnNewWindow(wxCommandEvent& event) { LFL::app->create_win_f(); }
  wxDECLARE_EVENT_TABLE();
};
wxBEGIN_EVENT_TABLE(LFLWxWidgetFrame, wxFrame)
  EVT_MENU(wxID_NEW, LFLWxWidgetFrame::OnNewWindow)
  EVT_MENU(wxID_CLOSE, LFLWxWidgetFrame::OnClose)
wxEND_EVENT_TABLE()

struct LFLWxWidgetApp : public wxApp {
  virtual bool OnInit() override {
    if (!wxApp::OnInit()) return false;
    vector<string> ab;
    vector<const char *> av;
    for (int i=0; i<argc; i++) {
      ab.push_back(argv[i].utf8_str().data());
      av.push_back(ab.back().c_str());
    }
    av.push_back(0);
    INFOf("WxWidgetsModule::Main argc=%d\n", argc);
    int ret = LFLWxWidgetsMain(argc, &av[0]);
    if (ret) exit(ret);
    INFOf("%s", "WxWidgetsModule::Main done");
    ((wxGLCanvas*)LFL::screen->id)->GetParent()->Show();
    return TRUE;
  }
  int OnExit() override {
    return wxApp::OnExit();
  }
};
#undef main
IMPLEMENT_APP(LFLWxWidgetApp)

namespace LFL {
struct WxWidgetsVideoModule : public Module {
  int Init() {
    INFOf("WxWidgetsVideoModule::Init() %p", screen);
    CHECK(Window::Create(screen));
    return 0;
  }
};
bool Window::Create(Window *W) {
  if (!Window::active.empty()) W->gl = Window::active.begin()->second->gl;
  LFLWxWidgetCanvas *canvas = (new LFLWxWidgetFrame(W))->canvas;
  if ((W->id = canvas)) Window::active[W->id] = W;
  if (!W->gl) W->gl = canvas->context = new wxGLContext(canvas);
  MakeCurrent(W);
  return true; 
}
void Window::MakeCurrent(Window *W) { 
  LFLWxWidgetCanvas *canvas = (LFLWxWidgetCanvas*)W->id;
  canvas->SetCurrent(*canvas->context);
}
void Window::Close(Window *W) {
  Window::active.erase(W->id);
  if (Window::active.empty()) app->run = false;
  if (app->window_closed_cb) app->window_closed_cb(W);
  screen = 0;
}
void Mouse::GrabFocus()    {}
void Mouse::ReleaseFocus() {}
#endif

#ifdef LFL_GLFWVIDEO
/* struct NativeWindow { GLFWwindow *id; }; */
struct GLFWVideoModule : public Module {
  int Init() {
    INFO("GLFWVideoModule::Init");
    CHECK(Window::Create(screen));
    Window::MakeCurrent(screen);
    glfwSwapInterval(1);
    return 0;
  }
  int Free() {
    glfwTerminate();
    return 0;
  }
};
bool Window::Create(Window *W) {
  GLFWwindow *share = Window::active.empty() ? 0 : (GLFWwindow*)Window::active.begin()->second->id;
  if (!(W->id = glfwCreateWindow(W->width, W->height, W->caption.c_str(), 0, share))) { ERROR("glfwCreateWindow"); return false; }
  Window::active[W->id] = W;
  return true;
}
void Window::MakeCurrent(Window *W) {
  glfwMakeContextCurrent((GLFWwindow*)W->id);
  screen = W;
}
void Window::Close(Window *W) {
  Window::active.erase(W->id);
  bool done = Window::active.empty();
  if (done) app->shell.quit(vector<string>());
  if (!done) glfwDestroyWindow((GLFWwindow*)W->id);
  if (app->window_closed_cb) app->window_closed_cb(W);
  screen = 0;
}
#endif

#ifdef LFL_SDLVIDEO
/* struct NativeWindow { SDL_Window* id; SDL_GLContext gl; SDL_Surface *surface; }; */
struct SDLVideoModule : public Module {
  int Init() {
    INFO("SFLVideoModule::Init");
    CHECK(Window::Create(screen));
    Window::MakeCurrent(screen);
    SDL_GL_SetSwapInterval(1);
    return 0;
  }
  int Free() {
    SDL_Quit();
    return 0;
  }
};
bool Window::Create(Window *W) {
  int createFlag = SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN;
#if defined(LFL_IPHONE) || defined(LFL_ANDROID)
  createFlag |= SDL_WINDOW_BORDERLESS;
  int bitdepth[] = { 5, 6, 5 };
#else
  int bitdepth[] = { 8, 8, 8 };
#endif
  SDL_GL_SetAttribute(SDL_GL_RED_SIZE, bitdepth[0]);
  SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, bitdepth[1]);
  SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, bitdepth[2]);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

  if (!(W->id = SDL_CreateWindow(W->caption.c_str(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, W->width, W->height, createFlag)))
  { ERROR("SDL_CreateWindow: ",     SDL_GetError()); return false; }

  if (!Window::active.empty()) W->gl = Window::active.begin()->second->gl;
  else if (!(W->gl = SDL_GL_CreateContext((SDL_Window*)W->id)))
  { ERROR("SDL_GL_CreateContext: ", SDL_GetError()); return false; } 

  SDL_Surface* icon = SDL_LoadBMP(StrCat(app->assetdir, "icon.bmp").c_str());
  SDL_SetWindowIcon((SDL_Window*)W->id, icon);

  Window::active[(void*)(long)SDL_GetWindowID((SDL_Window*)W->id)] = W;
  return true;
}
void Window::MakeCurrent(Window *W) {
  if (SDL_GL_MakeCurrent((SDL_Window*)W->id, W->gl) < 0) ERROR("SDL_GL_MakeCurrent: ", SDL_GetError());
  screen = W; 
}
void Window::Close(Window *W) {
  SDL_GL_MakeCurrent(NULL, NULL);
  Window::active.erase((void*)(long)SDL_GetWindowID((SDL_Window*)W->id));
  if (Window::active.empty()) {
    app->run = false;
    SDL_GL_DeleteContext(W->gl);
  }
  SDL_DestroyWindow((SDL_Window*)W->id);
  if (app->window_closed_cb) app->window_closed_cb(W);
  screen = 0;
}
#endif /* LFL_SDLVIDEO */

Window::Window() : caption("lfapp"), fps(128) {
  id = gl = surface = glew_context = impl = user1 = user2 = user3 = 0;
  minimized = cursor_grabbed = frame_init = 0;
  target_fps = FLAGS_target_fps;
  pow2_width = NextPowerOfTwo((width = 640));
  pow2_height = NextPowerOfTwo((height = 480));
  multitouch_keyboard_x = .93; 
  cam = new Entity(v3(5.54, 1.70, 4.39), v3(-.51, -.03, -.49), v3(-.03, 1, -.03));
  ClearEvents();
  ClearGesture();
}

Window::~Window() {
  if (lfapp_console) {
    lfapp_console->WriteHistory(LFAppDownloadDir(), "console");
    delete lfapp_console;
  }
  delete cam;
}

Window *Window::Get(void *id) { return FindOrNull(Window::active, id); }

Box Window::Box(float xp, float yp, float xs, float ys, float xbl, float ybt, float xbr, float ybb) const {
  if (isinf(xbr)) xbr = xbl;
  if (isinf(ybb)) ybb = ybt;
  return LFL::Box(width  * (xp + xbl),
                  height * (yp + ybb),
                  width  * xs - width  * (xbl + xbr),
                  height * ys - height * (ybt + ybb), false);
}

void Window::ClearEvents() { 
  ClearMouseGUIEvents();
  ClearKeyboardGUIEvents();
  ClearInputBindEvents();
}

void Window::ClearGesture() {
  gesture_swipe_up = gesture_swipe_down = 0;
  gesture_tap[0] = gesture_tap[1] = gesture_dpad_stop[0] = gesture_dpad_stop[1] = 0;
  gesture_dpad_dx[0] = gesture_dpad_dx[1] = gesture_dpad_dy[0] = gesture_dpad_dy[1] = 0;
}

void Window::ClearMouseGUIEvents() {
  for (auto i = mouse_gui.begin(); i != mouse_gui.end(); ++i) (*i)->ClearEvents();
}
void Window::ClearKeyboardGUIEvents() {
  for (auto i = keyboard_gui.begin(); i != keyboard_gui.end(); ++i) (*i)->ClearEvents();
}
void Window::ClearInputBindEvents() {
  for (auto i = input_bind.begin(); i != input_bind.end(); ++i) (*i)->ClearEvents();
}

void Window::InitLFAppConsole() {
  lfapp_console = new Console(screen, Fonts::Get(A_or_B(FLAGS_lfapp_console_font, FLAGS_default_font), "", 9, Color::white, Color::clear, FLAGS_lfapp_console_font_flag));
  lfapp_console->ReadHistory(LFAppDownloadDir(), "console");
  lfapp_console->Write(StrCat(screen->caption, " started"));
  lfapp_console->Write("Try console commands 'cmds' and 'flags'");
}

void Window::DrawDialogs() {
  if (screen->lfapp_console) screen->lfapp_console->Draw();
  if (FLAGS_draw_grid) {
    Color c(.7, .7, .7);
    glIntersect(screen->mouse.x, screen->mouse.y, &c);
    Fonts::Default()->Draw(StrCat("draw_grid ", screen->mouse.x, " , ", screen->mouse.y), point(0,0));
  }
  for (auto i = screen->dialogs.begin(), e = screen->dialogs.end(); i != e; ++i) (*i)->Draw();
}

void Window::SetCaption(const string &v) {
#if defined(LFL_OSXVIDEO)
  OSXSetWindowTitle(id, v.c_str());
#elif defined(LFL_WINVIDEO)
  SetWindowText((HWND)screen->id, v.c_str());
#elif defined(LFL_QT)
  ((QWindow*)screen->id)->setTitle(QString::fromUtf8(v.data(), v.size()));
#endif
}

void Window::SetResizeIncrements(float x, float y) {
#if defined(LFL_OSXVIDEO)
  OSXSetWindowResizeIncrements(id, x, y);
#elif defined(LFL_WINVIDEO)
  WinWindow *win = static_cast<WinWindow*>(screen->impl);
  win->resize_increment = point(x, y);
#elif defined(LFL_QT)
  ((QWindow*)screen->id)->setSizeIncrement(QSize(x, y));
#endif
}

void Window::SetTransparency(float v) {
#if defined(LFL_OSXVIDEO)
  OSXSetWindowTransparency(id, v);
#elif defined(LFL_WINVIDEO)
  if (v <= 0) SetWindowLong((HWND)screen->id, GWL_EXSTYLE, GetWindowLong((HWND)screen->id, GWL_EXSTYLE) & (~WS_EX_LAYERED));
  else {      SetWindowLong((HWND)screen->id, GWL_EXSTYLE, GetWindowLong((HWND)screen->id, GWL_EXSTYLE) | ( WS_EX_LAYERED));
    SetLayeredWindowAttributes((HWND)screen->id, 0, static_cast<BYTE>(max(1.0, (1-v)*255.0)), LWA_ALPHA);
  }
#elif defined(LFL_QT)
  ((QWindow*)screen->id)->setOpacity(1-v);
#endif
}

void Window::Reshape(int w, int h) {
#if defined(LFL_OSXVIDEO)
  OSXSetWindowSize(id, w, h);
#elif defined(LFL_WINVIDEO)
  WinWindow *win = static_cast<WinWindow*>(screen->impl);
  long lStyle = GetWindowLong((HWND)screen->id, GWL_STYLE);
  RECT r = { 0, 0, w, h };
  AdjustWindowRect(&r, lStyle, win->menubar);
  SetWindowPos((HWND)screen->id, 0, 0, 0, r.right-r.left, r.bottom-r.top, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
#elif defined(LFL_X11VIDEO)
  X11VideoModule *video = dynamic_cast<X11VideoModule*>(app->video.impl);
  XWindowChanges resize;
  resize.width = w;
  resize.height = h;
  XConfigureWindow(video->display, (::Window)screen->id, CWWidth|CWHeight, &resize);
#elif defined(LFL_QT)
  ((QWindow*)id)->resize(w, h);
  Window::MakeCurrent(screen);
#elif defined(LFL_WXWIDGETS)
  ((wxGLCanvas*)screen->id)->SetSize(w, h);
#elif defined(LFL_GLFWVIDEO)
  glfwSetWindowSize((GLFWwindow*)id, w, h);
#elif defined(LFL_SDLVIDEO)
  SDL_SetWindowSize((SDL_Window*)id, w, h);
#endif
}

void Window::Reshaped(int w, int h) {
  INFO("Window::Reshaped(", w, ", ", h, ")");
  pow2_width = NextPowerOfTwo((width = w));
  pow2_height = NextPowerOfTwo((height = h));
  if (!gd) return;
  gd->ViewPort(LFL::Box(width, height));
  gd->DrawMode(screen->gd->default_draw_mode);
  for (auto g = screen->mouse_gui.begin(); g != screen->mouse_gui.end(); ++g) (*g)->Layout();
  if (app->reshaped_cb) app->reshaped_cb();
}

void Window::ResetGL() {
  Video::InitGraphicsDevice(this);
  for (auto g : screen->keyboard_gui) g->ResetGL();
  for (auto g : screen->   mouse_gui) g->ResetGL();
  for (auto g : screen->     dialogs) g->ResetGL();
}

void Window::SwapAxis() {
  FLAGS_rotate_view = FLAGS_rotate_view ? 0 : -90;
  FLAGS_swap_axis = FLAGS_rotate_view != 0;
  Reshaped(height, width);
}

int Window::Frame(unsigned clicks, int flag) {
  if (screen != this) Window::MakeCurrent(this);

  if (FLAGS_lfapp_video) {
    if (!frame_init && (frame_init = true))  {
#ifdef LFL_IPHONE
      glGetIntegerv(GL_FRAMEBUFFER_BINDING_OES, &screen->gd->default_framebuffer);
      INFO("default_framebuffer = ", screen->gd->default_framebuffer);
#endif
    }
    gd->DrawMode(gd->default_draw_mode);
    gd->Clear();
    gd->LoadIdentity();
  }

  /* frame */
  int ret = frame_cb ? frame_cb(screen, clicks, flag) : 0;
  ClearEvents();

  /* allow app to skip frame */
  if (ret < 0) return ret;
  fps.Add(clicks);

  if (FLAGS_lfapp_video) {
    app->video->Swap();
  }
  return ret;
}

void Window::RenderToFrameBuffer(FrameBuffer *fb) {
  int dm = screen->gd->draw_mode;
  fb->Attach();
  screen->gd->ViewPort(Box(0, 0, fb->tex.width, fb->tex.height));
  screen->gd->Clear();
  frame_cb(0, 0, 0);
  fb->Release();
  screen->gd->RestoreViewport(dm);
}

/* Video */

int Video::Init() {
  INFO("Video::Init()");
#if defined(LFL_ANDROIDVIDEO)
  impl = new AndroidVideoModule();
#elif defined(LFL_IPHONEVIDEO)
  impl = new IPhoneVideoModule();
#elif defined(LFL_OSXVIDEO)
  impl = new OSXVideoModule();
#elif defined(LFL_WINVIDEO)
  impl = new WinVideoModule();
#elif defined(LFL_X11VIDEO)
  impl = new X11VideoModule();
#elif defined(LFL_XTVIDEO)
  impl = new XTVideoModule();
#elif defined(LFL_QT)
  impl = new QTVideoModule();
#elif defined(LFL_WXWIDGETS)
  impl = new WxWidgetsVideoModule();
#elif defined(LFL_GLFWVIDEO)
  impl = new GLFWVideoModule();
#elif defined(LFL_SDLVIDEO)
  impl = new SDLVideoModule();
#endif
  if (impl) if (impl->Init()) return -1;

#if defined(LFL_GLEW) && !defined(LFL_HEADLESS)
#ifdef GLEW_MX
  screen->glew_context = new GLEWContext();
#endif
  GLenum glew_err;
  if ((glew_err = glewInit()) != GLEW_OK) { ERROR("glewInit: ", glewGetErrorString(glew_err)); return -1; }
  app->video->opengl_framebuffer = GLEW_EXT_framebuffer_object;
#endif

  const char *glslver = (const char *)glGetString(GL_SHADING_LANGUAGE_VERSION);
  const char *glexts = SpellNull((const char *)glGetString(GL_EXTENSIONS));
  INFO("OpenGL Version: ", SpellNull((const char *)glGetString(GL_VERSION)));
  INFO("OpenGL Vendor: ",  SpellNull((const char *)glGetString(GL_VENDOR)));
#ifdef LFL_GLEW
  INFO("GLEW Version: ", SpellNull((const char*)glewGetString(GLEW_VERSION)));
#endif
#ifdef LFL_GLSL_SHADERS
  INFO("GL_SHADING_LANGUAGE_VERSION: ", SpellNull(glslver));
#endif
  INFO("GL_EXTENSIONS: ", glexts);

  opengles_version = 1 + (glslver != NULL);
#if defined(LFL_ANDROID) || defined(LFL_IPHONE)
  opengles_cubemap = strstr(glexts, "GL_EXT_texture_cube_map") != 0;
#else
  opengles_cubemap = strstr(glexts, "GL_ARB_texture_cube_map") != 0;
#endif
  INFO("lfapp_opengles_cubemap = ", opengles_cubemap ? "true" : "false");
  INFO("screen->opengles_version = ", opengles_version);

  if (!screen->gd) CreateGraphicsDevice(screen);
  InitGraphicsDevice(screen);

#ifndef WIN32
  if (app->splash_color) {
    screen->gd->ClearColor(*app->splash_color);
    screen->gd->Clear();
    screen->gd->Flush();
    Swap();
    screen->gd->ClearColor(Color::black);
  }
#endif
  return 0;
}

void *Video::BeginGLContextCreate(Window *W) {
#if defined(LFL_WINVIDEO)
  if (wglewIsSupported("WGL_ARB_create_context")) return wglCreateContextAttribsARB((HDC)W->surface, (HGLRC)W->gl, 0);
  else { HGLRC ret = wglCreateContext((HDC)W->surface); wglShareLists((HGLRC)W->gl, ret); return ret; }
#else
  return 0;
#endif
}

void *Video::CompleteGLContextCreate(Window *W, void *gl_context) {
#if defined(LFL_OSXVIDEO)
  return OSXCreateGLContext(W->id);
#elif defined(LFL_WINVIDEO)
  wglMakeCurrent((HDC)W->surface, (HGLRC)gl_context);
  return gl_context;
#elif defined(LFL_X11VIDEO)
  X11VideoModule *video = dynamic_cast<X11VideoModule*>(app->video.impl);
  GLXContext glc = glXCreateContext(video->display, video->vi, static_cast<GLXContext>(W->gl), GL_TRUE);
  glXMakeCurrent(video->display, (::Window)(W->id), glc);
  return glc;
#else
  return 0;
#endif
}

void Video::CreateGraphicsDevice(Window *W) {
  CHECK(!W->gd);
#if !defined(LFL_HEADLESS) && !defined(LFL_QT)
#ifdef LFL_GLES2
  if (app->video->opengles_version == 2) W->gd = new OpenGLES2();
  else
#endif
    W->gd = new OpenGLES1();
#endif
}

void Video::InitGraphicsDevice(Window *W) {
  W->gd->Init();
  W->gd->ViewPort(W->Box());
  W->gd->DrawMode(W->gd->default_draw_mode);

  float pos[]={-.5,1,-.3f,0}, grey20[]={.2f,.2f,.2f,1}, white[]={1,1,1,1}, black[]={0,0,0,1};
  W->gd->EnableLight(0);
  W->gd->Light(0, GraphicsDevice::Position, pos);
  W->gd->Light(0, GraphicsDevice::Ambient,  grey20);
  W->gd->Light(0, GraphicsDevice::Diffuse,  white);
  W->gd->Light(0, GraphicsDevice::Specular, white);
  W->gd->Material(GraphicsDevice::Emission, black);
  W->gd->Material(GraphicsDevice::Specular, grey20);
  INFO("opengl_init: width=", W->width, ", height=", W->height, ", opengles_version: ", app->video->opengles_version);
}

void Video::InitFonts() {
  FontEngine *font_engine = Fonts::DefaultFontEngine();
  if (!FLAGS_default_font.size()) font_engine->SetDefault();

  vector<string> atlas_font_size;
  Split(FLAGS_atlas_font_sizes, iscomma, &atlas_font_size);
  for (int i=0; i<atlas_font_size.size(); i++) {
    int size = atoi(atlas_font_size[i].c_str());
    font_engine->Init(FontDesc(FLAGS_default_font, FLAGS_default_font_family, size, Color::white, Color::clear, FLAGS_default_font_flag));
  }

  FontEngine *atlas_engine = Singleton<AtlasFontEngine>::Get();
  atlas_engine->Init(FontDesc("MenuAtlas", "", 0, Color::black, Color::clear, 0, 0));

  if (FLAGS_lfapp_console && FLAGS_font_engine != "atlas" && FLAGS_font_engine != "freetype") {
    FLAGS_atlas_font_sizes = "32";
    string console_font = "VeraMoBd.ttf";
    Singleton<AtlasFontEngine>::Get()->Init(FontDesc(console_font, "", 32, Color::white, Color::clear, FLAGS_lfapp_console_font_flag));
    FLAGS_lfapp_console_font = StrCat("atlas://", console_font);
  }
}

int Video::InitFontWidth() {
#if defined(WIN32)
  return 8;
#elif defined(__APPLE__)
  return 9;
#else
  return 10;
#endif
}

int Video::InitFontHeight() {
#if defined(WIN32)
  return 17;
#elif defined(__APPLE__)
  return 20;
#else
  return 18;
#endif
}

int Video::Swap() {
#ifndef LFL_QT
  screen->gd->Flush();
#endif

#if defined(LFL_ANDROIDVIDEO)
  AndroidVideoSwap();
#elif defined(LFL_IPHONEVIDEO)
  iPhoneVideoSwap();
#elif defined(LFL_OSXVIDEO)
  OSXVideoSwap(screen->id);
#elif defined(LFL_WINVIDEO)
  SwapBuffers((HDC)screen->surface);
#elif defined(LFL_X11VIDEO)
  glXSwapBuffers((Display*)screen->surface, (::Window)screen->id);
#elif defined(LFL_QT)
  ((QOpenGLContext*)screen->gl)->swapBuffers((QWindow*)screen->id);
#elif defined(LFL_WXWIDGETS)
  ((wxGLCanvas*)screen->id)->SwapBuffers();
#elif defined(LFL_GLFWVIDEO)
  glfwSwapBuffers((GLFWwindow*)screen->id);
#elif defined(LFL_SDLVIDEO)
  SDL_GL_SwapWindow((SDL_Window*)screen->id);
#endif

  screen->gd->CheckForError(__FILE__, __LINE__);
  return 0;
}

int Video::Free() {
  vector<Window*> close_list;
  for (auto &i : Window::active) close_list.push_back(i.second);
  for (auto &i : close_list)     Window::Close(i);

  if (impl) impl->Free();
  Fonts::DefaultFontEngine()->Shutdown();
  return 0;
}

void SimpleVideoResampler::RGB2BGRCopyPixels(unsigned char *dst, const unsigned char *src, int l, int bpp) {
  for (int k = 0; k < l; k++) for (int i = 0; i < bpp; i++) dst[k*bpp+(!i?2:(i==2?0:i))] = src[k*bpp+i];
}

bool SimpleVideoResampler::Supports(int f) { return f == Pixel::RGB24 || f == Pixel::BGR24 || f == Pixel::RGB32 || f == Pixel::BGR32 || f == Pixel::RGBA; }

bool SimpleVideoResampler::Opened() { return s_fmt && d_fmt && s_width && d_width && s_height && d_height; }

void SimpleVideoResampler::Open(int sw, int sh, int sf, int dw, int dh, int df) {
  s_fmt = sf; s_width = sw; s_height = sh;
  d_fmt = df; d_width = dw; d_height = dh;
  // INFO("resample ", BlankNull(Pixel::Name(s_fmt)), " -> ", BlankNull(Pixel::Name(d_fmt)), " : (", sw, ",", sh, ") -> (", dw, ",", dh, ")");
}

void SimpleVideoResampler::Resample(const unsigned char *sb, int sls, unsigned char *db, int dls, bool flip_x, bool flip_y) {
  if (!Opened()) { ERROR("resample not opened()"); return; }

  int spw = Pixel::size(s_fmt), dpw = Pixel::size(d_fmt);
  if (spw * s_width > sls) { ERROR(spw * s_width, " > ", sls); return; }
  if (dpw * d_width > dls) { ERROR(dpw * d_width, " > ", dls); return; }

  if (s_width == d_width && s_height == d_height) {
    for (int y=0; y<d_height; y++) {
      for (int x=0; x<d_width; x++) {
        const unsigned char *sp = (sb + sls * y                           + x                          * spw);
        /**/  unsigned char *dp = (db + dls * (flip_y ? d_height-1-y : y) + (flip_x ? d_width-1-x : x) * dpw);
        CopyPixel(s_fmt, d_fmt, sp, dp, x == 0, x == d_width-1);
      }
    }
  } else {
    for (int po=0; po<spw && po<dpw; po++) {
      Matrix M(s_height, s_width);
      CopyColorChannelsToMatrix(sb, s_width, s_height, spw, sls, 0, 0, &M, po);
      for (int y=0; y<d_height; y++) {
        for (int x=0; x<d_width; x++) {
          unsigned char *dp = (db + dls * (flip_y ? d_height-1-y : y) + (flip_x ? d_width-1-x : x) * dpw);
          *(dp + po) = MatrixAsFunc(&M, x?(float)x/(d_width-1):0, y?(float)y/(d_height-1):0) * 255;
        }
      }
    }
  }
}

void SimpleVideoResampler::CopyPixel(int s_fmt, int d_fmt, const unsigned char *sp, unsigned char *dp, bool sxb, bool sxe, int f) {
  unsigned char r, g, b, a;
  switch (s_fmt) {
    case Pixel::RGB24: r = *sp++; g = *sp++; b = *sp++; a = ((f & Flag::TransparentBlack) && !r && !g && !b) ? 0 : 255; break;
    case Pixel::BGR24: b = *sp++; g = *sp++; r = *sp++; a = ((f & Flag::TransparentBlack) && !r && !g && !b) ? 0 : 255; break;
    case Pixel::RGB32: r = *sp++; g = *sp++; b = *sp++; a=255; sp++; break;
    case Pixel::BGR32: b = *sp++; g = *sp++; r = *sp++; a=255; sp++; break;
    case Pixel::RGBA:  r = *sp++; g = *sp++; b = *sp++; a=*sp++; break;
    case Pixel::BGRA:  b = *sp++; g = *sp++; r = *sp++; a=*sp++; break;
    case Pixel::GRAY8: r = 255;   g = 255;   b = 255;   a=*sp++; break;
    // case Pixel::GRAY8: r = g = b = a = *sp++; break;
    case Pixel::LCD: 
      r = (sxb ? 0 : *(sp-1)) / 3.0 + *sp / 3.0 + (          *(sp+1)) / 3.0; sp++; 
      g = (          *(sp-1)) / 3.0 + *sp / 3.0 + (          *(sp+1)) / 3.0; sp++; 
      b = (          *(sp-1)) / 3.0 + *sp / 3.0 + (sxe ? 0 : *(sp+1)) / 3.0; sp++;
      a = ((f & Flag::TransparentBlack) && !r && !g && !b) ? 0 : 255;
      break;
    default: ERROR("s_fmt ", s_fmt, " not supported"); return;
  }
  switch (d_fmt) {
    case Pixel::RGB24: *dp++ = r; *dp++ = g; *dp++ = b; break;
    case Pixel::BGR24: *dp++ = b; *dp++ = g; *dp++ = r; break;
    case Pixel::RGB32: *dp++ = r; *dp++ = g; *dp++ = b; *dp++ = a; break;
    case Pixel::BGR32: *dp++ = b; *dp++ = g; *dp++ = r; *dp++ = a; break;
    case Pixel::RGBA:  *dp++ = r; *dp++ = g; *dp++ = b; *dp++ = a; break;
    case Pixel::BGRA:  *dp++ = b; *dp++ = g; *dp++ = r; *dp++ = a; break;
    default: ERROR("d_fmt ", d_fmt, " not supported"); return;
  }
}

void SimpleVideoResampler::Blit(const unsigned char *src, unsigned char *dst, int w, int h,
                                int sf, int sls, int sx, int sy,
                                int df, int dls, int dx, int dy, int flag) {
  bool flip_y = flag & Flag::FlipY;
  int sw = Pixel::size(sf), dw = Pixel::size(df); 
  for (int yi = 0; yi < h; ++yi) {
    for (int xi = 0; xi < w; ++xi) {
      int sind = flip_y ? sy + h - yi - 1 : sy + yi;
      const unsigned char *sp = src + (sls*(sind)    + (sx + xi)*sw);
      unsigned       char *dp = dst + (dls*(dy + yi) + (dx + xi)*dw);
      CopyPixel(sf, df, sp, dp, xi == 0, xi == w-1, flag);
    }
  }
}

void SimpleVideoResampler::Filter(unsigned char *buf, int w, int h,
                                  int pf, int ls, int x, int y,
                                  Matrix *kernel, int channel, int flag) {
  Matrix M(h, w), out(h, w);
  int pw = Pixel::size(pf);
  CopyColorChannelsToMatrix(buf, w, h, pw, ls, x, y, &M, ColorChannel::PixelOffset(channel));
  Matrix::Convolve(&M, kernel, &out, (flag & Flag::ZeroOnly) ? mZeroOnly : 0);
  CopyMatrixToColorChannels(&out, w, h, pw, ls, x, y, buf, ColorChannel::PixelOffset(channel));
}

void SimpleVideoResampler::CopyColorChannelsToMatrix(const unsigned char *buf, int w, int h,
                                                     int pw, int ls, int x, int y,
                                                     Matrix *out, int po) {
  MatrixIter(out) { 
    const unsigned char *p = buf + (ls*(y + i) + (x + j)*pw);
    out->row(i)[j] = *(p + po) / 255.0;
  }
}

void SimpleVideoResampler::CopyMatrixToColorChannels(const Matrix *M, int w, int h,
                                                     int pw, int ls, int x, int y,
                                                     unsigned char *out, int po) {
  MatrixIter(M) { 
    unsigned char *p = out + (ls*(y + i) + (x + j)*pw);
    *(p + po) = M->row(i)[j] * 255.0;
  }
}

#ifdef LFL_FFMPEG
FFMPEGVideoResampler::~FFMPEGVideoResampler() { if (conv) sws_freeContext((SwsContext*)conv); }
bool FFMPEGVideoResampler::Opened() { return conv || simple_resampler_passthru; }

void FFMPEGVideoResampler::Open(int sw, int sh, int sf, int dw, int dh, int df) {
  s_fmt = sf; s_width = sw; s_height = sh;
  d_fmt = df; d_width = dw; d_height = dh;
  // INFO("resample ", BlankNull(Pixel::Name(s_fmt)), " -> ", BlankNull(Pixel::Name(d_fmt)), " : (", sw, ",", sh, ") -> (", dw, ",", dh, ")");

  if (SimpleVideoResampler::Supports(s_fmt) && SimpleVideoResampler::Supports(d_fmt) && sw == dw && sh == dh)
  { simple_resampler_passthru = 1; return; }

  conv = sws_getContext(sw, sh, (PixelFormat)Pixel::ToFFMpegId(sf),
                        dw, dh, (PixelFormat)Pixel::ToFFMpegId(df), SWS_BICUBIC, 0, 0, 0);
}

void FFMPEGVideoResampler::Resample(const unsigned char *s, int sls, unsigned char *d, int dls, bool flip_x, bool flip_y) {
  if (simple_resampler_passthru) return SimpleVideoResampler::Resample(s, sls, d, dls, flip_x, flip_y);
  uint8_t *source  [4] = { (uint8_t*)s, 0, 0, 0 }, *dest[  4] = { (uint8_t*)d, 0, 0, 0 };
  int      sourcels[4] = {         sls, 0, 0, 0 },  destls[4] = {         dls, 0, 0, 0 };
  if (flip_y) {
    source[0] += sls * (s_height - 1);
    sourcels[0] *= -1;
  }
  sws_scale((SwsContext*)conv,
            flip_y ? source   : source,
            flip_y ? sourcels : sourcels, 0, s_height, dest, destls);
}
#endif // LFL_FFMPEG

}; // namespace LFL
