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

#include "core/app/flow.h"
#include "core/app/ipc.h"

namespace LFL {
DEFINE_int(dots_per_inch, 75, "Screen DPI");
DEFINE_unsigned(depth_buffer_bits, 0, "Depth buffer bits");
DEFINE_float(rotate_view, 0, "Rotate view by angle");
DEFINE_float(field_of_view, 45, "Field of view");
DEFINE_float(near_plane, 1, "Near clipping plane");
DEFINE_float(far_plane, 100, "Far clipping plane");
DEFINE_bool(swap_axis, false," Swap x,y axis");
DEFINE_bool(gd_debug, false, "Debug graphics device");

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

const int Texture::minx_coord_ind = 0;
const int Texture::miny_coord_ind = 1;
const int Texture::maxx_coord_ind = 2;
const int Texture::maxy_coord_ind = 3;
const float Texture::unit_texcoord[4] = { 0, 0, 1, 1 };
#if defined(LFL_MOBILE) || defined(LFL_EMSCRIPTEN)
const int Texture::preferred_pf = Pixel::RGBA;
#else
const int Texture::preferred_pf = Pixel::BGRA;
#endif

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
    case ARGB:    return "ARGB";       case LCD:     return "LCD";
    case YUV420P: return "YUV420P";    case YUYV422: return "YUYV422";
    case GRAY8:   return "GRAY8";      case GRAYA8:  return "GRAYA8";
  }; return 0; 
}

int Pixel::Size(int p) {
  switch (p) {
    case RGB32:   case BGR32:  case RGBA:   case BGRA:   case ARGB:   return 4;
    case RGB24:   case BGR24:  case LCD:                              return 3;
    case RGB555:  case BGR555: case RGB565: case BGR565: case GRAYA8: return 2;
    case YUYV422: case GRAY8:                                         return 1;
    default:                                                          return 0;
  }
}

int Pixel::GetNumComponents(int p) {
  switch (p) {
    case RGB32:  case BGR32:  case RGB24:  case BGR24: 
    case RGB555: case BGR555: case RGB565: case BGR565: return 3;
    case RGBA:   case BGRA:   case ARGB:                return 4;
    case GRAY8:                                         return 1;
    case GRAYA8:                                        return 2;
    default:                                            return 0;
  }
}

int Pixel::GetRGBAIndex(int p, int i) {
  switch (p) {
    case RGB32: case RGB24:  case RGB555: case RGB565: switch(i) { case Index::R: return 0; case Index::G: return 1; case Index::B: return 2; default: return -1; } break;
    case BGR32: case BGR24:  case BGR555: case BGR565: switch(i) { case Index::R: return 2; case Index::G: return 1; case Index::B: return 0; default: return -1; } break;
    case RGBA: switch(i) { case Index::R: return 0; case Index::G: return 1; case Index::B: return 2; case Index::A: return 3; default: return -1; } break;
    case BGRA: switch(i) { case Index::R: return 2; case Index::G: return 1; case Index::B: return 0; case Index::A: return 3; default: return -1; } break;
    case ARGB: switch(i) { case Index::R: return 1; case Index::G: return 2; case Index::B: return 3; case Index::A: return 0; default: return -1; } break;
    default: return -1;
  }
}

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

/* GrahicsContext */

void GraphicsContext::DrawTexturedBox1(GraphicsDevice *gd, const Box &b, const float *texcoord, int orientation) {
  static const float default_texcoord[4] = {0, 0, 1, 1};
  const float *tc = X_or_Y(texcoord, default_texcoord);

  if (gd->version == 2) {
#define DrawTexturedBoxTriangles(gd, v) \
    bool changed = gd->VertexPointer(2, gd->c.Float, sizeof(float)*4, 0,               v, sizeof(v), NULL, true, gd->c.Triangles); \
    if (changed)   gd->TexPointer   (2, gd->c.Float, sizeof(float)*4, sizeof(float)*2, v, sizeof(v), NULL, false); \
    if (1)         gd->DeferDrawArrays(gd->c.Triangles, 0, 6);

    if (orientation == 0) {
      float verts[] = { float(b.x),     float(b.y),     tc[Texture::minx_coord_ind], tc[Texture::miny_coord_ind],
        float(b.x),     float(b.y+b.h), tc[Texture::minx_coord_ind], tc[Texture::maxy_coord_ind],
        float(b.x+b.w), float(b.y),     tc[Texture::maxx_coord_ind], tc[Texture::miny_coord_ind],
        float(b.x),     float(b.y+b.h), tc[Texture::minx_coord_ind], tc[Texture::maxy_coord_ind],
        float(b.x+b.w), float(b.y),     tc[Texture::maxx_coord_ind], tc[Texture::miny_coord_ind],
        float(b.x+b.w), float(b.y+b.h), tc[Texture::maxx_coord_ind], tc[Texture::maxy_coord_ind] };
      DrawTexturedBoxTriangles(gd, verts);
    } else if (orientation == 1) {
      float verts[] = { float(b.x),     float(b.y),     tc[Texture::minx_coord_ind], tc[Texture::maxy_coord_ind],
        float(b.x),     float(b.y+b.h), tc[Texture::minx_coord_ind], tc[Texture::miny_coord_ind],
        float(b.x+b.w), float(b.y),     tc[Texture::maxx_coord_ind], tc[Texture::maxy_coord_ind],
        float(b.x),     float(b.y+b.h), tc[Texture::minx_coord_ind], tc[Texture::miny_coord_ind],
        float(b.x+b.w), float(b.y),     tc[Texture::maxx_coord_ind], tc[Texture::maxy_coord_ind],
        float(b.x+b.w), float(b.y+b.h), tc[Texture::maxx_coord_ind], tc[Texture::miny_coord_ind] };
      DrawTexturedBoxTriangles(gd, verts);
    }
  } else {
#define DrawTexturedBoxTriangleStrip(gd, v) \
    bool changed = gd->VertexPointer(2, gd->c.Float, sizeof(float)*4, 0,               verts, sizeof(verts), NULL, true, gd->c.TriangleStrip); \
    if  (changed)  gd->TexPointer   (2, gd->c.Float, sizeof(float)*4, sizeof(float)*2, verts, sizeof(verts), NULL, false); \
    if (1)         gd->DeferDrawArrays(gd->c.TriangleStrip, 0, 4);

    if (orientation == 0) {
      float verts[] = { float(b.x),     float(b.y),     tc[Texture::minx_coord_ind], tc[Texture::miny_coord_ind],
                        float(b.x),     float(b.y+b.h), tc[Texture::minx_coord_ind], tc[Texture::maxy_coord_ind],
                        float(b.x+b.w), float(b.y),     tc[Texture::maxx_coord_ind], tc[Texture::miny_coord_ind],
                        float(b.x+b.w), float(b.y+b.h), tc[Texture::maxx_coord_ind], tc[Texture::maxy_coord_ind] };
      DrawTexturedBoxTriangleStrip(gd, verts);
    } else if (orientation == 1) {
    }
  }
}

void GraphicsContext::DrawGradientBox1(GraphicsDevice *gd, const Box &b, const Color *c) {
#if 0
  float verts[] = { float(x),   float(y),   c[0].r(), c[0].g(), c[0].b(), c[0].a(),
                    float(x),   float(y+h), c[3].r(), c[3].g(), c[3].b(), c[3].a(),
                    float(x+w), float(y),   c[1].r(), c[1].g(), c[1].b(), c[1].a(),
                    float(x),   float(y+h), c[3].r(), c[3].g(), c[3].b(), c[3].a(),
                    float(x+w), float(y),   c[1].r(), c[1].g(), c[1].b(), c[1].a(),
                    float(x+w), float(y+h), c[2].r(), c[2].g(), c[2].b(), c[2].a() };
  bool changed = gd->VertexPointer(2, GraphicsDevice::Float, sizeof(float)*6, 0,               verts, sizeof(verts), NULL, true, GraphicsDevice::Triangles);
  if (changed)   gd->ColorPointer (4, GraphicsDevice::Float, sizeof(float)*6, sizeof(float)*2, verts, sizeof(verts), NULL, false);
  if (1)         gd->DeferDrawArrays(GraphicsDevice::Triangles, 0, 6);
#else
  float verts[] = { float(b.x),     float(b.y),     c[0].r(), c[0].g(), c[0].b(), c[0].a(),
                    float(b.x),     float(b.y+b.h), c[3].r(), c[3].g(), c[3].b(), c[3].a(),
                    float(b.x+b.w), float(b.y),     c[1].r(), c[1].g(), c[1].b(), c[1].a(),
                    float(b.x+b.w), float(b.y+b.h), c[2].r(), c[2].g(), c[2].b(), c[2].a() };
  bool changed = gd->VertexPointer(2, gd->c.Float, sizeof(float)*6, 0,               verts, sizeof(verts), NULL, true, gd->c.TriangleStrip);
  if  (changed)  gd->ColorPointer (4, gd->c.Float, sizeof(float)*6, sizeof(float)*2, verts, sizeof(verts), NULL, false);
  if (1)         gd->DeferDrawArrays(gd->c.TriangleStrip, 0, 4);
#endif
}

void GraphicsContext::DrawCrimpedBox1(GraphicsDevice *gd, const Box &b, const float *texcoord, int orientation, float scrollX, float scrollY) {
  float left=b.x, right=b.x+b.w, top=b.y, bottom=b.y+b.h;
  float texMinX, texMinY, texMaxX, texMaxY, texMidX1, texMidX2, texMidY1, texMidY2;

  scrollX *= (texcoord[2] - texcoord[0]);
  scrollY *= (texcoord[3] - texcoord[1]);
  scrollX = Box::ScrollCrimped(texcoord[0], texcoord[2], scrollX, &texMinX, &texMidX1, &texMidX2, &texMaxX);
  scrollY = Box::ScrollCrimped(texcoord[1], texcoord[3], scrollY, &texMinY, &texMidY1, &texMidY2, &texMaxY);

#define DrawCrimpedBoxTriangleStrip() \
  gd->VertexPointer(2, gd->c.Float, 4*sizeof(float), 0,               verts, sizeof(verts), NULL, true, gd->c.TriangleStrip); \
  gd->TexPointer   (2, gd->c.Float, 4*sizeof(float), 2*sizeof(float), verts, sizeof(verts), NULL, false); \
  gd->DeferDrawArrays(gd->c.TriangleStrip, 0, 4); \
  gd->DeferDrawArrays(gd->c.TriangleStrip, 4, 4); \
  gd->DeferDrawArrays(gd->c.TriangleStrip, 8, 4); \
  gd->DeferDrawArrays(gd->c.TriangleStrip, 12, 4);

  switch (orientation) {
    case 0: {
      float xmid = b.x + b.w * scrollX, ymid = b.y + b.h * scrollY, verts[] = {
        /*02*/ xmid,  top,  texMidX1, texMaxY,  /*01*/ left, top,  texMinX,  texMaxY,  /*03*/ xmid,  ymid,   texMidX1, texMidY1, /*04*/ left, ymid,   texMinX,  texMidY1,
        /*06*/ right, top,  texMaxX,  texMaxY,  /*05*/ xmid, top,  texMidX2, texMaxY,  /*07*/ right, ymid,   texMaxX,  texMidY1, /*08*/ xmid, ymid,   texMidX2, texMidY1,
        /*10*/ right, ymid, texMaxX,  texMidY2, /*09*/ xmid, ymid, texMidX2, texMidY2, /*11*/ right, bottom, texMaxX,  texMinY,  /*12*/ xmid, bottom, texMidX2, texMinY,
        /*14*/ xmid,  ymid, texMidX1, texMidY2, /*13*/ left, ymid, texMinX,  texMidY2, /*15*/ xmid,  bottom, texMidX1, texMinY,  /*16*/ left, bottom, texMinX,  texMinY 
      };
      DrawCrimpedBoxTriangleStrip();
    } break;
    case 1: {
      float xmid = b.x + b.w * scrollX, ymid = b.y + b.h * (1-scrollY), verts[] = {
        /*02*/ xmid,  top,  texMidX1, texMinY,  /*01*/ left,  top,  texMinX,  texMinY,  /*03*/ xmid, ymid,    texMidX1, texMidY2, /*04*/ left, ymid,   texMinX,  texMidY2,
        /*06*/ right, top,  texMaxX,  texMinY,  /*05*/ xmid,  top,  texMidX2, texMinY,  /*07*/ right, ymid,   texMaxX,  texMidY2, /*08*/ xmid, ymid,   texMidX2, texMidY2,
        /*10*/ right, ymid, texMaxX,  texMidY1, /*09*/ xmid,  ymid, texMidX2, texMidY1, /*11*/ right, bottom, texMaxX,  texMaxY,  /*12*/ xmid, bottom, texMidX2, texMaxY,
        /*14*/ xmid,  ymid, texMidX1, texMidY1, /*13*/ left,  ymid, texMinX,  texMidY1, /*15*/ xmid, bottom,  texMidX1, texMaxY,  /*16*/ left, bottom, texMinX,  texMaxY 
      };
      DrawCrimpedBoxTriangleStrip();
    } break;
    case 2: {
      float xmid = b.x + b.w * (1-scrollX), ymid = b.y + b.h * scrollY, verts[] = {
        /*02*/ xmid,  top,  texMidX2, texMaxY,  /*01*/ left,  top,  texMaxX,  texMaxY,  /*03*/ xmid, ymid,    texMidX2, texMidY1, /*04*/ left, ymid,   texMaxX,  texMidY1,
        /*06*/ right, top,  texMinX,  texMaxY,  /*05*/ xmid,  top,  texMidX1, texMaxY,  /*07*/ right, ymid,   texMinX,  texMidY1, /*08*/ xmid, ymid,   texMidX1, texMidY1,
        /*10*/ right, ymid, texMinX,  texMidY2, /*09*/ xmid,  ymid, texMidX1, texMidY2, /*11*/ right, bottom, texMinX,  texMinY,  /*12*/ xmid, bottom, texMidX1, texMinY,
        /*14*/ xmid,  ymid, texMidX2, texMidY2, /*13*/ left,  ymid, texMaxX,  texMidY2, /*15*/ xmid, bottom,  texMidX2, texMinY,  /*16*/ left, bottom, texMaxX,  texMinY 
      };
      DrawCrimpedBoxTriangleStrip();
    } break;
    case 3: {
      float xmid = b.x + b.w * (1-scrollX), ymid = b.y + b.h * (1-scrollY), verts[] = {
        /*02*/ xmid,  top,  texMidX2, texMinY,  /*01*/ left,  top,   texMaxX,  texMinY,  /*03*/ xmid, ymid,    texMidX2, texMidY2, /*04*/ left, ymid,   texMaxX,  texMidY2,
        /*06*/ right, top,  texMinX,  texMinY,  /*05*/ xmid,  top,   texMidX1, texMinY,  /*07*/ right, ymid,   texMinX,  texMidY2, /*08*/ xmid, ymid,   texMidX1, texMidY2,
        /*10*/ right, ymid, texMinX,  texMidY1, /*09*/ xmid,  ymid,  texMidX1, texMidY1, /*11*/ right, bottom, texMinX,  texMaxY,  /*12*/ xmid, bottom, texMidX1, texMaxY,
        /*14*/ xmid,  ymid, texMidX2, texMidY1, /*13*/ left,  ymid,  texMaxX,  texMidY1, /*15*/ xmid, bottom,  texMidX2, texMaxY,  /*16*/ left, bottom, texMaxX,  texMaxY 
      };
      DrawCrimpedBoxTriangleStrip();
    } break;
    case 4: {
      float xmid = b.x + b.w * (1-scrollY), ymid = b.y + b.h * scrollX, verts[] = {
        /*13*/ xmid,  top,  texMinX,  texMidY2, /*16*/ left,  top,  texMinX,  texMaxY,  /*14*/ xmid, ymid,    texMidX1, texMidY2, /*15*/ left, ymid,   texMidX1, texMaxY, 
        /*01*/ right, top,  texMinX,  texMinY,  /*04*/ xmid,  top,  texMinX,  texMidY1, /*02*/ right, ymid,   texMidX1, texMinY,  /*03*/ xmid, ymid,   texMidX1, texMidY1,
        /*05*/ right, ymid, texMidX2, texMinY,  /*08*/ xmid,  ymid, texMidX2, texMidY1, /*06*/ right, bottom, texMaxX,  texMinY,  /*07*/ xmid, bottom, texMaxX,  texMidY1,
        /*09*/ xmid,  ymid, texMidX2, texMidY2, /*12*/ left,  ymid, texMidX2, texMaxY,  /*10*/ xmid, bottom,  texMaxX,  texMidY2, /*11*/ left, bottom, texMaxX,  texMaxY 
      };
      DrawCrimpedBoxTriangleStrip();
    } break;
    case 5: {
      float xmid = b.x + b.w * scrollY, ymid = b.y + b.h * scrollX, verts[] = {
        /*13*/ xmid,  top,  texMinX,  texMidY1, /*16*/ left,  top,  texMinX,  texMinY,  /*14*/ xmid, ymid,    texMidX1, texMidY1, /*15*/ left, ymid,   texMidX1, texMinY, 
        /*01*/ right, top,  texMinX,  texMaxY,  /*04*/ xmid,  top,  texMinX,  texMidY2, /*02*/ right, ymid,   texMidX1, texMaxY,  /*03*/ xmid, ymid,   texMidX1, texMidY2,
        /*05*/ right, ymid, texMidX2, texMaxY,  /*08*/ xmid,  ymid, texMidX2, texMidY2, /*06*/ right, bottom, texMaxX,  texMaxY,  /*07*/ xmid, bottom, texMaxX,  texMidY2,
        /*09*/ xmid,  ymid, texMidX2, texMidY1, /*12*/ left,  ymid, texMidX2, texMinY,  /*10*/ xmid, bottom,  texMaxX,  texMidY1, /*11*/ left, bottom, texMaxX,  texMinY 
      };
      DrawCrimpedBoxTriangleStrip();
    } break;
    case 6: {
      float xmid = b.x + b.w * (1-scrollY), ymid = b.y + b.h * (1-scrollX), verts[] = {
        /*13*/ xmid,  top,  texMaxX,  texMidY2, /*16*/ left,  top,  texMaxX,  texMaxY,  /*14*/ xmid, ymid,    texMidX2, texMidY2, /*15*/ left, ymid,   texMidX2, texMaxY, 
        /*01*/ right, top,  texMaxX,  texMinY,  /*04*/ xmid,  top,  texMaxX,  texMidY1, /*02*/ right, ymid,   texMidX2, texMinY,  /*03*/ xmid, ymid,   texMidX2, texMidY1,
        /*05*/ right, ymid, texMidX1, texMinY,  /*08*/ xmid,  ymid, texMidX1, texMidY1, /*06*/ right, bottom, texMinX,  texMinY,  /*07*/ xmid, bottom, texMinX,  texMidY1,
        /*09*/ xmid,  ymid, texMidX1, texMidY2, /*12*/ left,  ymid, texMidX1, texMaxY,  /*10*/ xmid, bottom,  texMinX,  texMidY2, /*11*/ left, bottom, texMinX,  texMaxY 
      };
      DrawCrimpedBoxTriangleStrip();
    } break;
    case 7: {
      float xmid = b.x + b.w * scrollY, ymid = b.y + b.h * (1-scrollX), verts[] = {
        /*13*/ xmid,  top,  texMaxX,  texMidY1, /*16*/ left,  top,  texMaxX,  texMinY,  /*14*/ xmid, ymid,    texMidX2, texMidY1, /*15*/ left, ymid,   texMidX2, texMinY, 
        /*01*/ right, top,  texMaxX,  texMaxY,  /*04*/ xmid,  top,  texMaxX,  texMidY2, /*02*/ right, ymid,   texMidX2, texMaxY,  /*03*/ xmid, ymid,   texMidX2, texMidY2,
        /*05*/ right, ymid, texMidX1, texMaxY,  /*08*/ xmid,  ymid, texMidX1, texMidY2, /*06*/ right, bottom, texMinX,  texMaxY,  /*07*/ xmid, bottom, texMinX,  texMidY2,
        /*09*/ xmid,  ymid, texMidX1, texMidY1, /*12*/ left,  ymid, texMidX1, texMinY,  /*10*/ xmid, bottom,  texMinX,  texMidY1, /*11*/ left, bottom, texMinX,  texMinY 
      };
      DrawCrimpedBoxTriangleStrip();
    } break;
  }
}

void GraphicsContext::DrawTexturedBox3(GraphicsDevice *gd, const Box3 &b, const point &p, const Color *c) {
  if (c) gd->SetColor(*c);
  for (int i=0; i<3; i++) if (b.v[i].h) GraphicsContext::DrawTexturedBox1(gd, Box(b.v[i] + p));
}

/* Drawable */

string Drawable::Attr::DebugString() const {
  return StrCat("Attr ", Void(this),
                " = { font=", font?CheckPointer(font->desc)->DebugString():string(), ", fg=", fg?fg->DebugString():string(),
                ", bg=", bg?bg->DebugString():string(), ", tex=", tex?tex->ID:0,
                ", scissor=", scissor?StrCat(scissor->x, ",", scissor->y, ",", scissor->w, ",", scissor->h):string(),
                ", blend=", blend, " }");
}

void Drawable::AttrVec::Insert(const Drawable::Attr &v) {
  // if (v.font) font_refs.Insert(&v.font->ref);
  push_back(v);
}

void Drawable::DrawGD(GraphicsDevice *gd, const LFL::Box &b) const { GraphicsContext gc(gd); Draw(&gc, b); }

/* Texture */

int Texture::GDBufferType(GraphicsDevice *gd) const {
  return pf == preferred_pf ? gd->c.GLPreferredBuffer : gd->c.UnsignedByte;
}

void Texture::Coordinates(float *texcoord, int w, int h, int wd, int hd) { return Coordinates(texcoord, Box(w,h), wd, hd); }
void Texture::Coordinates(float *texcoord, const Box &b, int wd, int hd) {
  if (!wd || !hd) {
    texcoord[0] = unit_texcoord[0];
    texcoord[1] = unit_texcoord[1];
    texcoord[2] = unit_texcoord[2];
    texcoord[3] = unit_texcoord[3];
  } else {
    texcoord[minx_coord_ind] = float(b.x)       / wd;
    texcoord[miny_coord_ind] = float(b.y)       / hd;
    texcoord[maxx_coord_ind] = float(b.x + b.w) / wd;
    texcoord[maxy_coord_ind] = float(b.y + b.h) / hd;
  }
}

void Texture::Resize(int W, int H, int PF, int flag) {
  auto gd = parent->GD();
  bool changed_pf = PF && PF != pf;
  if (PF) pf = PF;
  Assign(&width, &height, W, H);
  if (buf || (flag & Flag::CreateBuf)) RenewBuffer();

  int gl_width = gd->TextureDim(width), gl_height = gd->TextureDim(height);
  if (ID && (ID.w != gl_width || ID.h != gl_height || changed_pf)) {
    ClearGL();
    flag |= Flag::CreateGL;
  }
  if (!ID && (flag & Flag::CreateGL)) {
    if (!cubemap) {
      gd->DisableCubeMap();
      ID = gd->CreateTexture(gd->c.Texture2D, gl_width, gl_height, pf);
    } else if (cubemap == CubeMap::PX) {
      gd->ActiveTexture(0);
      ID = gd->CreateTexture(gd->c.TextureCubeMap, gl_width, gl_height, pf);
    }
    if (!(flag & Flag::RepeatGL)) {
      gd->TexParameter(gd->c.Texture2D, gd->c.TextureWrapS, gd->c.ClampToEdge);
      gd->TexParameter(gd->c.Texture2D, gd->c.TextureWrapT, gd->c.ClampToEdge);
    }
  }
  if (ID || cubemap) {
    if (ID) gd->BindTexture(ID);
    gd->UpdateTexture(ID, cubemap, 0, gd->c.GLInternalFormat, 0, GDBufferType(gd), 0);
    Coordinates(coord, width, height, gl_width, gl_height);
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
  unique_ptr<VideoResamplerInterface> conv(CreateVideoResampler());
  conv->Open(dim.x, dim.y, PF, resample ? width : dim.x, resample ? height : dim.y, pf);
  conv->Resample(B, linesize, buf, LineSize(), 0, flag & Flag::FlipY);
}

void Texture::UpdateBuffer(const unsigned char *B, const ::LFL::Box &box, int PF, int linesize, int blit_flag) {
  SimpleVideoResampler::Blit(B, buf, box.w, box.h, PF, linesize, 0, 0, pf, LineSize(), box.x, box.y, blit_flag);
}

void Texture::Bind() const { parent->GD()->BindTexture(ID); }
void Texture::ClearGL() { 
  if (ID) {
    auto gd = parent->GD();
    if (!gd) { if (FLAGS_gd_debug) ERROR("DelTexture no device ", ID); return; }
    auto a = gd->parent->parent;
    if (!a->MainThread()) { a->RunInMainThread(bind(&GraphicsDevice::DelTexture, gd, ID)); }
    else gd->DelTexture(ID);
    ID = TextureRef();
  }
}

void Texture::LoadGL(const MultiProcessTextureResource &t) { return LoadGL(MakeUnsigned(t.buf.data()), point(t.width, t.height), t.pf, t.linesize); }
void Texture::LoadGL(const unsigned char *B, const point &dim, int PF, int linesize, int flag) {
  Texture temp(parent);
  temp .Resize(dim.x, dim.y, preferred_pf, Flag::CreateBuf);
  temp .UpdateBuffer(B, dim, PF, linesize, Flag::FlipY);
  this->Resize(dim.x, dim.y, preferred_pf, Flag::CreateGL | (flag & Flag::RepeatGL));
  this->UpdateGL(temp.buf, LFL::Box(dim), 0, flag);
}

void Texture::UpdateGL(const unsigned char *B, const ::LFL::Box &box, int npf, int flag) {
  if (npf) CHECK_EQ(pf, npf); // else blit
  auto gd = parent->GD();
  int gl_y = (flag & Flag::FlipY) ? (height - box.y - box.h) : box.y;
  gd->BindTexture(ID);
  gd->UpdateSubTexture(ID, cubemap, 0, box.x, gl_y, box.w, box.h, GDBufferType(gd), B);
}

void Texture::ResetGL(int flag) {
  bool reload = flag & ResetGLFlag::Reload, forget = (flag & ResetGLFlag::Delete) == 0;
  if (forget) ID = TextureRef();
  if (reload && width && height) RenewGL();
  else ClearGL();
}

#ifdef LFL_APPLE
#import <CoreGraphics/CGBitmapContext.h> 
CGContextRef Texture::CGBitMap() { return CGBitMap(0, 0, width, height); }
CGContextRef Texture::CGBitMap(int X, int Y, int W, int H) {
  int linesize = LineSize(), alpha_info = 0;
  if      (pf == Pixel::RGBA)  alpha_info = kCGImageAlphaPremultipliedLast;
  else if (pf == Pixel::BGRA)  alpha_info = kCGBitmapByteOrder32Host | kCGImageAlphaPremultipliedFirst;
  else if (pf == Pixel::RGB32) alpha_info = kCGImageAlphaNoneSkipLast;
  else if (pf == Pixel::BGR32) alpha_info = kCGBitmapByteOrder32Host | kCGImageAlphaNoneSkipFirst;
  else return ERRORv(nullptr, "unsupported pixel format: ", pf, " = ", Pixel::Name(pf));
  CGColorSpaceRef colors = CGColorSpaceCreateDeviceRGB();
  // CGColorSpaceRef colors = CGColorSpaceCreateWithName(kCGColorSpaceGenericRGB);
  CGContextRef ret = CGBitmapContextCreate(buf + Y*linesize + X*PixelSize(), W, H, 8, linesize, colors, alpha_info);
  CGColorSpaceRelease(colors);
  return ret;
}
#endif

#ifdef LFL_WINDOWS
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

void TextureArray::DrawSequence(Asset *out, Entity *e, int *ind) {
  *ind = (*ind + 1) % a.size();
  const Texture *in = &a[*ind];
  out->tex.ID = in->ID;
  if (out->geometry) Scene::Draw(in->parent->GD(), out->geometry, e);
}

/* DepthTexture */

void DepthTexture::Resize(int W, int H, int DF, int flag) {
  auto gd = parent->GD();
  if (DF) df = DF;
  Assign(&width, &height, W, H);
  int gl_width = gd->TextureDim(width), gl_height = gd->TextureDim(height);
  if (ID && (ID.w != gl_width || ID.h != gl_height)) {
    ClearGL();
    flag |= Flag::CreateGL;
  }
  if (!ID && (flag & Flag::CreateGL)) ID = gd->CreateDepthBuffer(W, H, DF);
  else if (ID) gd->BindDepthBuffer(ID);
}

void DepthTexture::ClearGL() {
  if (ID) {
    auto gd = parent->GD();
    if (gd) gd->DelDepthBuffer(ID);
    else if (FLAGS_gd_debug) ERROR("DelRenderBuffer no device ", ID);
    ID = DepthRef();
  }
}

void DepthTexture::ResetGL(int flag) {
  bool reload = flag & ResetGLFlag::Reload, forget = (flag & ResetGLFlag::Delete) == 0;
  if (forget) ID = DepthRef();
  ClearGL();
  if (reload && width && height) Create(width, height);
}

/* FrameBuffer */

void FrameBuffer::Resize(int W, int H, int flag) {
  auto gd = parent->GD();
  Assign(&width, &height, W, H);
  if (ID && (ID.w != width || ID.h != height)) {
    ClearGL();
    flag |= Flag::CreateGL;
  }
  if (!ID && (flag & Flag::CreateGL)) {
    ID = gd->CreateFrameBuffer(W, H);
    if (flag & Flag::CreateTexture)      AllocTexture(&tex);
    if (flag & Flag::CreateDepthTexture) AllocDepthTexture(&depth);
  } else {
    if (tex.ID)     tex.Resize(width, height);
    if (depth.ID) depth.Resize(width, height);
  }
  Attach(tex.ID, depth.ID);
  int status = gd->CheckFrameBufferStatus();
  if (status != gd->c.FramebufferComplete) {
#if 0
    ERROR("FrameBuffer Resize(", W, ", ", H, ") status ", status);
#else
    ERROR("FrameBuffer Resize(", W, ", ", H, ") status ", status, " from:\n", gd->parent->parent->PrintCallStack());
#endif
  }
  if (flag & Flag::ReleaseFB) Release();
}

void FrameBuffer::AllocDepthTexture(DepthTexture *out) { CHECK_EQ(out->ID, 0); out->Create(width, height); }
void FrameBuffer::AllocTexture(TextureRef *out) { Texture t(parent); AllocTexture(&t); *out = t.ReleaseGL(); } 
void FrameBuffer::AllocTexture(Texture *out) {
  CHECK_EQ(out->ID, 0);
  out->Create(width, height); 
}

void FrameBuffer::Release(bool update_viewport) {
  auto gd = parent->GD();
  gd->attached_framebuffer = nullptr;
  gd->BindFrameBuffer(gd->default_framebuffer);
  if (update_viewport) gd->RestoreViewport(DrawMode::_2D);
}

void FrameBuffer::Attach(const TextureRef &ct, const DepthRef &dt, bool update_viewport) {
  auto gd = parent->GD();
  gd->attached_framebuffer = this;
  gd->BindFrameBuffer(ID);
  if (ct) { if (tex  .ID != ct) tex.owner   = false; gd->FrameBufferTexture     ((tex.ID   = ct)); }
  if (dt) { if (depth.ID != dt) depth.owner = false; gd->FrameBufferDepthTexture((depth.ID = dt)); }
  if (update_viewport) {
    Box b(width, height);
    gd->ViewPort(b);
    gd->DrawMode(DrawMode::_2D, b, true);
  }
}

void FrameBuffer::ClearGL() {
  if (ID) {
    auto gd = parent->GD();
    if (gd) gd->DelFrameBuffer(ID);
    else if (FLAGS_gd_debug) ERROR("DelFrameBuffer no device ", ID);
    ID = FrameBufRef();
  }
}

void FrameBuffer::ResetGL(int flag) {
  bool reload = flag & ResetGLFlag::Reload, forget = (flag & ResetGLFlag::Delete) == 0;
  if (forget) ID = FrameBufRef();
  ClearGL();
  tex.ResetGL(flag);
  depth.ResetGL(flag);
  if (reload && width && height) Create(width, height);
}

/* Shaders */

void Shaders::SetGlobalUniform1f(GraphicsDevice *gd, const string &name, float v) {
  gd->UseShader(&shader_default);  shader_default .SetUniform1f(name, v);
  gd->UseShader(&shader_normals);  shader_normals .SetUniform1f(name, v);
  gd->UseShader(&shader_cubemap);  shader_cubemap .SetUniform1f(name, v);
  gd->UseShader(&shader_cubenorm); shader_cubenorm.SetUniform1f(name, v);
}

void Shaders::SetGlobalUniform2f(GraphicsDevice *gd, const string &name, float v1, float v2){ 
  gd->UseShader(&shader_default);  shader_default .SetUniform2f(name, v1, v2);
  gd->UseShader(&shader_normals);  shader_normals .SetUniform2f(name, v1, v2);
  gd->UseShader(&shader_cubemap);  shader_cubemap .SetUniform2f(name, v1, v2);
  gd->UseShader(&shader_cubenorm); shader_cubenorm.SetUniform2f(name, v1, v2);
}

/* Shader */

int Shader::Create(GraphicsDeviceHolder *parent, const string &name, const string &vertex_shader, const string &fragment_shader, const ShaderDefines &defines, Shader *out) {
  if (out) *out = Shader(parent);
  auto gd = parent->GD();
  auto p = gd->CreateProgram();

  string hdr =
    "#ifdef GL_ES\r\n"
    "precision highp float;\r\n"
    "#else\r\n"
    "#define lowp\r\n"
    "#define highp\r\n"
    "#endif\r\n";
  if (gd->version == 2) hdr += "#define LFL_GLES2\r\n";
  hdr += defines.text + string("\r\n");

  if (vertex_shader.size()) {
    auto vs = gd->CreateShader(gd->c.VertexShader);
    gd->CompileShader(vs, { hdr.c_str(), vertex_shader.c_str() });
    gd->AttachShader(p, vs);
    gd->DelShader(vs);
  }

  if (fragment_shader.size()) {
    auto fs = gd->CreateShader(gd->c.FragmentShader);
    gd->CompileShader(fs, { hdr.c_str(), fragment_shader.c_str() });
    gd->AttachShader(p, fs);
    gd->DelShader(fs);
  }

  if (1)                    gd->BindAttribLocation(p, 0, "Position"   );
  if (defines.normals)      gd->BindAttribLocation(p, 1, "Normal"     );
  if (defines.vertex_color) gd->BindAttribLocation(p, 2, "VertexColor");
  if (defines.tex_2d)       gd->BindAttribLocation(p, 3, "TexCoordIn" );

  if (!gd->LinkProgram(p)) {
    gd->DelProgram(p);
    return ERRORv(0, "Shader::Create ", name, ": link failed");
  } else INFO("Shader::Create ", name);

  int active_uniforms=0, max_uniform_components=0, active_attributes=0, max_attributes=0;
  gd->GetProgramiv(p, gd->c.ActiveUniforms, &active_uniforms);
  gd->GetProgramiv(p, gd->c.ActiveAttributes, &active_attributes);
#ifndef LFL_MOBILE
  gd->GetIntegerv(gd->c.MaxVertexUniformComp, &max_uniform_components);
#endif
  gd->GetIntegerv(gd->c.MaxVertexAttributes, &max_attributes);
  INFO("shader=", name, ", mu=", active_uniforms, " avg_comps/", max_uniform_components, ", ma=", active_attributes, "/", max_attributes);

  bool log_missing_attrib = false;
  if (out) {
    out->ID = p;
    out->name = name;
    if ((out->slot_position             = gd->GetAttribLocation (p, "Position"))            < 0 && log_missing_attrib) INFO("shader ", name, " missing Position");
    if ((out->slot_normal               = gd->GetAttribLocation (p, "Normal"))              < 0 && log_missing_attrib) INFO("shader ", name, " missing Normal");
    if ((out->slot_color                = gd->GetAttribLocation (p, "VertexColor"))         < 0 && log_missing_attrib) INFO("shader ", name, " missing VertexColor");
    if ((out->slot_tex                  = gd->GetAttribLocation (p, "TexCoordIn"))          < 0 && log_missing_attrib) INFO("shader ", name, " missing TexCoordIn");
    if ((out->uniform_model             = gd->GetUniformLocation(p, "Model"))               < 0 && log_missing_attrib) INFO("shader ", name, " missing Model");
    if ((out->uniform_invview           = gd->GetUniformLocation(p, "InverseView"))         < 0 && log_missing_attrib) INFO("shader ", name, " missing InverseView");
    if ((out->uniform_modelview         = gd->GetUniformLocation(p, "Modelview"))           < 0 && log_missing_attrib) INFO("shader ", name, " missing Modelview");
    if ((out->uniform_modelviewproj     = gd->GetUniformLocation(p, "ModelviewProjection")) < 0 && log_missing_attrib) INFO("shader ", name, " missing ModelviewProjection");
    if ((out->uniform_campos            = gd->GetUniformLocation(p, "CameraPosition"))      < 0 && log_missing_attrib) INFO("shader ", name, " missing CameraPosition");
    if ((out->uniform_tex               = gd->GetUniformLocation(p, "iChannel0"))           < 0 && log_missing_attrib) INFO("shader ", name, " missing Texture");
    if ((out->uniform_cubetex           = gd->GetUniformLocation(p, "CubeTexture"))         < 0 && log_missing_attrib) INFO("shader ", name, " missing CubeTexture");
    if ((out->uniform_texon             = gd->GetUniformLocation(p, "TexCoordEnabled"))     < 0 && log_missing_attrib) INFO("shader ", name, " missing TexCoordEnabled");
    if ((out->uniform_coloron           = gd->GetUniformLocation(p, "VertexColorEnabled"))  < 0 && log_missing_attrib) INFO("shader ", name, " missing VertexColorEnabled");
    if ((out->uniform_colordefault      = gd->GetUniformLocation(p, "DefaultColor"))        < 0 && log_missing_attrib) INFO("shader ", name, " missing DefaultColor");
    if ((out->uniform_material_ambient  = gd->GetUniformLocation(p, "MaterialAmbient"))     < 0 && log_missing_attrib) INFO("shader ", name, " missing MaterialAmbient");
    if ((out->uniform_material_diffuse  = gd->GetUniformLocation(p, "MaterialDiffuse"))     < 0 && log_missing_attrib) INFO("shader ", name, " missing MaterialDiffuse");
    if ((out->uniform_material_specular = gd->GetUniformLocation(p, "MaterialSpecular"))    < 0 && log_missing_attrib) INFO("shader ", name, " missing MaterialSpecular");
    if ((out->uniform_material_emission = gd->GetUniformLocation(p, "MaterialEmission"))    < 0 && log_missing_attrib) INFO("shader ", name, " missing MaterialEmission");
    if ((out->uniform_light0_pos        = gd->GetUniformLocation(p, "LightZeroPosition"))   < 0 && log_missing_attrib) INFO("shader ", name, " missing LightZeroPosition");
    if ((out->uniform_light0_ambient    = gd->GetUniformLocation(p, "LightZeroAmbient"))    < 0 && log_missing_attrib) INFO("shader ", name, " missing LightZeroAmbient");
    if ((out->uniform_light0_diffuse    = gd->GetUniformLocation(p, "LightZeroDiffuse"))    < 0 && log_missing_attrib) INFO("shader ", name, " missing LightZeroDiffuse");
    if ((out->uniform_light0_specular   = gd->GetUniformLocation(p, "LightZeroSpecular"))   < 0 && log_missing_attrib) INFO("shader ", name, " missing LightZeroSpecular");

    int unused_attrib = 0;
    memset(out->unused_attrib_slot, -1, sizeof(out->unused_attrib_slot));
    for (int i=0; i<MaxVertexAttrib; i++) {
      if (out->slot_position == i || out->slot_normal == i || out->slot_color == i || out->slot_tex == i) continue;
      out->unused_attrib_slot[unused_attrib++] = i;
    }
  }

  return p;
}

int Shader::CreateShaderToy(GraphicsDeviceHolder *parent, const string &name, const string &pixel_shader, Shader *out) {
  static string header =
    "uniform float iGlobalTime, iBlend;\r\n"
    "uniform vec3 iResolution;\r\n"
    "uniform vec4 iTargetBox, iMouse;\r\n"
    "uniform sampler2D iChannel0;\r\n"
    "uniform vec3 iChannelResolution[1];\r\n"
    "uniform vec2 iChannelScroll[1], iChannelModulus[1];\r\n"
    "uniform bool iChannelFlip[1];\r\n"
    "#define SampleChannelAtPointAndModulus(c, p, m) texture2D(c, mod((p), (m)))\r\n"
    "#define SampleChannelAtPoint(c, p) SampleChannelAtPointAndModulus(c, p, iChannelModulus[0])\r\n"
    "#define SamplePointXY (((fragCoord.xy - iTargetBox.xy) * iChannelResolution[0].xy / iTargetBox.zw + iChannelScroll[0]) / iChannelResolution[0].xy)\r\n"
    "#define SamplePointFlippedXY vec2(((fragCoord.x - iTargetBox.x)                           * iChannelResolution[0].x / iTargetBox.z + iChannelScroll[0].x) / iChannelResolution[0].x, \\\r\n"
    "                                    (iChannelResolution[0].y - (fragCoord.y - iTargetBox.y) * iChannelResolution[0].y / iTargetBox.w - iChannelScroll[0].y) / iChannelResolution[0].y)\r\n"
    "#define SamplePoint (iChannelFlip[0] ? SamplePointFlippedXY : SamplePointXY)\r\n"
    "#define SamplePointFlipY (iChannelFlip[0] ? SamplePointXY : SamplePointFlippedXY)\r\n"
    "#define SampleChannel(c) SampleChannelAtPoint(c, SamplePoint)\r\n"
    "#define BlendChannels(c1,c2) ((c1)*iBlend + (c2)*(1.0-iBlend))\r\n";

  static string footer =
    "void main(void) { mainImage(gl_FragColor, gl_FragCoord.xy); }\r\n";

  GraphicsDevice *gd = parent->GD();
  return Shader::Create(parent, name, gd->vertex_shader, StrCat(header, pixel_shader, footer), ShaderDefines(1,0,1,0), out);
}

GraphicsDevice::UniformRef Shader::GetUniformIndex(const string &name) { return parent->GD()->GetUniformLocation(ID, name); }
void Shader::SetUniform1i(const string &name, float v)                                { parent->GD()->Uniform1i (GetUniformIndex(name), v); }
void Shader::SetUniform1f(const string &name, float v)                                { parent->GD()->Uniform1f (GetUniformIndex(name), v); }
void Shader::SetUniform2f(const string &name, float v1, float v2)                     { parent->GD()->Uniform2f (GetUniformIndex(name), v1, v2); }
void Shader::SetUniform3f(const string &name, float v1, float v2, float v3)           { parent->GD()->Uniform3f (GetUniformIndex(name), v1, v2, v3); }
void Shader::SetUniform4f(const string &name, float v1, float v2, float v3, float v4) { parent->GD()->Uniform4f (GetUniformIndex(name), v1, v2, v3, v4); }
void Shader::SetUniform3fv(const string &name, const float *v)                        { parent->GD()->Uniform3fv(GetUniformIndex(name), 1, v); }
void Shader::SetUniform3fv(const string &name, int n, const float *v)                 { parent->GD()->Uniform3fv(GetUniformIndex(name), n, v); }
void Shader::ClearGL() {
  if (ID) {
    auto gd = parent->GD();
    if (gd) gd->DelProgram(ID);
    else if (FLAGS_gd_debug) ERROR("DelProgram no device ", ID);
    ID = GraphicsDevice::ProgramRef();
  }
}

void GraphicsDevice::PushColor() { default_color.push_back(default_color.back()); SetColor(default_color.back());  }
void GraphicsDevice::PopColor() {
  if      (default_color.size() >  1) default_color.pop_back();
  else if (default_color.size() == 1) default_color.back() = Color(1.0, 1.0, 1.0, 1.0);
  else FATAL("no color state");
  SetColor(default_color.back());
}

void GraphicsDevice::RestoreViewport(int dm) { ViewPort(Box(parent->gl_x + parent->gl_w, parent->gl_y + parent->gl_h)); DrawMode(dm); }
void GraphicsDevice::TranslateRotateTranslate(float a, const Box &b) { float x=b.x+b.w/2.0, y=b.y+b.h/2.0; Translate(x,y,0); Rotatef(a,0,0,1); Translate(-x,-y,0); }

void GraphicsDevice::DrawMode(int dm, bool flush) { return DrawMode(dm, parent->Box(), flush); }
void GraphicsDevice::DrawMode(int dm, const Box &b, bool flush) {
  if (draw_mode == dm && !flush) return;
  bool _2D = (draw_mode = dm) == DrawMode::_2D;
  Color4f(1, 1, 1, 1);
  MatrixProjection();
  LoadIdentity();
  if (FLAGS_rotate_view) Rotatef(FLAGS_rotate_view, 0, 0, 1);

  if (_2D) Ortho(0, b.right(), 0, b.top(), 0, 100);
  else {
    float aspect = float(b.w) / b.h;
    double top = tan(FLAGS_field_of_view * M_PI/360.0) * FLAGS_near_plane;
    Frustum(aspect*-top, aspect*top, -top, top, FLAGS_near_plane, FLAGS_far_plane);
  }

  if (_2D) DisableDepthTest();
  else     EnableDepthTest();

  MatrixModelview();
  LoadIdentity();
  Scene::Select(this);
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
    -v3::Dot(X, pos), -v3::Dot(Y, pos), -v3::Dot(Z, pos), 1.0
  };

  if (invert_view_matrix) m44::Invert(m44(m), &invview_matrix);
  camera_pos = pos;
  Mult(m);
}

void GraphicsDevice::PushScissor(Box w) {
  auto &ss = scissor_stack.back();
  if (ss.empty()) ss.push_back(w);
  else ss.push_back(w.Intersect(ss.back()));
  Scissor(ss.back());
}

void GraphicsDevice::PopScissor() {
  auto &ss = scissor_stack.back();
  if (ss.size()) ss.pop_back();
  if (ss.size()) Scissor(ss.back());
  else { Scissor(parent->Box()); DisableScissor(); }
}

void GraphicsDevice::PushScissorStack() {
  scissor_stack.push_back(vector<Box>());
  ClearDeferred();
  DisableScissor();
}

void GraphicsDevice::PopScissorStack() {
  CHECK_GT(scissor_stack.size(), 1);
  scissor_stack.pop_back();
  auto &ss = scissor_stack.back();
  if (ss.size()) Scissor(ss.back());
  else { Scissor(parent->Box()); DisableScissor(); }
}

Box GraphicsDevice::GetViewport() { Box vp; GetIntegerv(c.ViewportBox, &vp.x); return vp; }
Box GraphicsDevice::GetScissorBox() {
  auto &ss = scissor_stack.back();
  Box ret = ss.size() ? ss.back() : Box(-1,-1);
#ifdef LFL_DEBUG
  if (GetEnabled(c.ScissorTest)) { Box glb; GetIntegerv(c.ScissorBox, &glb.x); CHECK_EQ(glb,         ret); }
  else                                                                       { CHECK_EQ(Box(-1, -1), ret); }
#endif
  return ret;
}

int GraphicsDevice::GetPrimitive(int geometry_primtype) {
  switch (geometry_primtype) {
  case Geometry::Primitive::Lines:
    return c.Lines;
  case Geometry::Primitive::Triangles:
    return c.Triangles;
  case Geometry::Primitive::TriangleStrip:
    return c.TriangleStrip;
  default:
    FATAL("unknown primitive: ", geometry_primtype);
  }
}
  
void GraphicsDevice::DrawPixels(const Box &b, const Texture &tex) {
  Texture temp(parent->parent);
  temp.Resize(tex.width, tex.height, tex.pf, Texture::Flag::CreateGL);
  temp.UpdateGL(tex.buf, LFL::Box(tex.width, tex.height), 0, Texture::Flag::FlipY); 
  GraphicsContext::DrawTexturedBox1(this, b, temp.coord);
  temp.ClearGL();
}

void GraphicsDevice::InitDefaultLight() {
  float pos[]={-.5,1,-.3f,0}, grey20[]={.2f,.2f,.2f,1}, white[]={1,1,1,1}, black[]={0,0,0,1};
  EnableLight(0);
  Light(0, c.Position, pos);
  Light(0, c.Ambient,  grey20);
  Light(0, c.Diffuse,  white);
  Light(0, c.Specular, white);
  Material(c.Emission, black);
  Material(c.Specular, grey20);
}

Shaders::Shaders(GraphicsDeviceHolder *p) : shader_default(p), shader_normals(p), shader_cubemap(p),
  shader_cubenorm(p) {}

}; // namespace LFL
