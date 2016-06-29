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
    case YUV420P: return "YUV420P";    case YUYV422: return "YUYV422";
    case GRAY8:   return "GRAY8";      case GRAYA8:  return "GRAYA8";
    case LCD:     return "LCD";
  }; return 0; 
}

int Pixel::Size(int p) {
  switch (p) {
    case RGB32:   case BGR32:  case RGBA:   case BGRA:                return 4;
    case RGB24:   case BGR24:  case LCD:                              return 3;
    case RGB555:  case BGR555: case RGB565: case BGR565: case GRAYA8: return 2;
    case YUYV422: case GRAY8:                                         return 1;
    default:                                                          return 0;
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

Box::Box(float X, float Y, float W, float H, bool round) {
  if (round) { x=RoundF(X); y=RoundF(Y); w=RoundF(W); h=RoundF(H); }
  else       { x=   int(X); y=   int(Y); w=   int(W); h=   int(H); }
}

Box::Box(const float *v4, bool round) {
  if (round) { x=RoundF(v4[0]); y=RoundF(v4[1]); w=RoundF(v4[2]); h=RoundF(v4[3]); }
  else       { x=   int(v4[0]); y=   int(v4[1]); w=   int(v4[2]); h=   int(v4[3]); }
}

string Box::DebugString() const { return StringPrintf("Box = { %d, %d, %d, %d }", x, y, w, h); }

void Box::Draw(GraphicsDevice *gd, const float *texcoord) const {
  static const float default_texcoord[4] = {0, 0, 1, 1};
  const float *tc = X_or_Y(texcoord, default_texcoord);
#if 1
  float verts[] = { float(x),   float(y),   tc[Texture::minx_coord_ind], tc[Texture::miny_coord_ind],
                    float(x),   float(y+h), tc[Texture::minx_coord_ind], tc[Texture::maxy_coord_ind],
                    float(x+w), float(y),   tc[Texture::maxx_coord_ind], tc[Texture::miny_coord_ind],
                    float(x),   float(y+h), tc[Texture::minx_coord_ind], tc[Texture::maxy_coord_ind],
                    float(x+w), float(y),   tc[Texture::maxx_coord_ind], tc[Texture::miny_coord_ind],
                    float(x+w), float(y+h), tc[Texture::maxx_coord_ind], tc[Texture::maxy_coord_ind] };
  bool changed = gd->VertexPointer(2, GraphicsDevice::Float, sizeof(float)*4, 0,               verts, sizeof(verts), NULL, true, GraphicsDevice::Triangles);
  if (changed)   gd->TexPointer   (2, GraphicsDevice::Float, sizeof(float)*4, sizeof(float)*2, verts, sizeof(verts), NULL, false);
  if (1)         gd->DeferDrawArrays(GraphicsDevice::Triangles, 0, 6);
#else
  float verts[] = { float(x),   float(y),   tc[Texture::minx_coord_ind], tc[Texture::miny_coord_ind],
                    float(x),   float(y+h), tc[Texture::minx_coord_ind], tc[Texture::maxy_coord_ind],
                    float(x+w), float(y),   tc[Texture::maxx_coord_ind], tc[Texture::miny_coord_ind],
                    float(x+w), float(y+h), tc[Texture::maxx_coord_ind], tc[Texture::maxy_coord_ind] };
  bool changed = gd->VertexPointer(2, GraphicsDevice::Float, sizeof(float)*4, 0,               verts, sizeof(verts), NULL, true, GraphicsDevice::TriangleStrip);
  if  (changed)  gd->TexPointer   (2, GraphicsDevice::Float, sizeof(float)*4, sizeof(float)*2, verts, sizeof(verts), NULL, false);
  if (1)         gd->DeferDrawArrays(GraphicsDevice::TriangleStrip, 0, 4);
#endif
}

void Box::DrawGradient(GraphicsDevice *gd, const Color *c) const {
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
  float verts[] = { float(x),   float(y),   c[0].r(), c[0].g(), c[0].b(), c[0].a(),
                    float(x),   float(y+h), c[3].r(), c[3].g(), c[3].b(), c[3].a(),
                    float(x+w), float(y),   c[1].r(), c[1].g(), c[1].b(), c[1].a(),
                    float(x+w), float(y+h), c[2].r(), c[2].g(), c[2].b(), c[2].a() };
  bool changed = gd->VertexPointer(2, GraphicsDevice::Float, sizeof(float)*6, 0,               verts, sizeof(verts), NULL, true, GraphicsDevice::TriangleStrip);
  if  (changed)  gd->ColorPointer (4, GraphicsDevice::Float, sizeof(float)*6, sizeof(float)*2, verts, sizeof(verts), NULL, false);
  if (1)         gd->DeferDrawArrays(GraphicsDevice::TriangleStrip, 0, 4);
#endif
}

void Box::DrawCrimped(GraphicsDevice *gd, const float *texcoord, int orientation, float scrollX, float scrollY) const {
  float left=x, right=x+w, top=y, bottom=y+h;
  float texMinX, texMinY, texMaxX, texMaxY, texMidX1, texMidX2, texMidY1, texMidY2;

  scrollX *= (texcoord[2] - texcoord[0]);
  scrollY *= (texcoord[3] - texcoord[1]);
  scrollX = ScrollCrimped(texcoord[0], texcoord[2], scrollX, &texMinX, &texMidX1, &texMidX2, &texMaxX);
  scrollY = ScrollCrimped(texcoord[1], texcoord[3], scrollY, &texMinY, &texMidY1, &texMidY2, &texMaxY);

#define DrawCrimpedBoxTriangleStrip() \
  gd->VertexPointer(2, GraphicsDevice::Float, 4*sizeof(float), 0,               verts, sizeof(verts), NULL, true, GraphicsDevice::TriangleStrip); \
  gd->TexPointer   (2, GraphicsDevice::Float, 4*sizeof(float), 2*sizeof(float), verts, sizeof(verts), NULL, false); \
  gd->DeferDrawArrays(GraphicsDevice::TriangleStrip, 0, 4); \
  gd->DeferDrawArrays(GraphicsDevice::TriangleStrip, 4, 4); \
  gd->DeferDrawArrays(GraphicsDevice::TriangleStrip, 8, 4); \
  gd->DeferDrawArrays(GraphicsDevice::TriangleStrip, 12, 4);

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
  if (tex1 <= 1.0 && tex0 == 0.0) {
    *mid1=tex1; *mid2=0;
    if (scroll > 0) *min = *max = tex1 - scroll;
    else            *min = *max = tex0 - scroll;
  } else if (tex0 > 0.0 && tex1 == 1.0) {
    *mid1=1; *mid2=tex0;
    if (scroll > 0) *min = *max = tex0 + scroll;
    else            *min = *max = tex1 + scroll;
  } else { FATAL("invalid tex coords ", tex0, ", ", tex1); }
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

void Box3::Draw(GraphicsDevice *gd, const point &p, const Color *c) const {
  if (c) gd->SetColor(*c);
  for (int i=0; i<3; i++) if (v[i].h) (v[i] + p).Draw(gd);
}

Box Box3::BoundingBox() const {
  int min_x = v[0].x, min_y = v[0].y, max_x = v[0].x + v[0].w, max_y = v[0].y + v[0].h;
  if (v[1].h) { min_x = min(min_x, v[1].x); min_y = min(min_y, v[1].y); max_x = max(max_x, v[1].x + v[1].w); max_y = max(max_y, v[1].y + v[1].h); }
  if (v[2].h) { min_x = min(min_x, v[2].x); min_y = min(min_y, v[2].y); max_x = max(max_x, v[2].x + v[2].w); max_y = max(max_y, v[2].y + v[2].h); }
  return Box(min_x, min_y, max_x - min_x, max_y - min_y);
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

int Texture::GLBufferType() const {
  return pf == preferred_pf ? GraphicsDevice::GLPreferredBuffer : GraphicsDevice::UnsignedByte;
}

void Texture::Coordinates(float *texcoord, int w, int h, int wd, int hd) {
  texcoord[minx_coord_ind] = texcoord[miny_coord_ind] = 0;
  texcoord[maxx_coord_ind] = float(w) / wd;
  texcoord[maxy_coord_ind] = float(h) / hd;
}

void Texture::Resize(int W, int H, int PF, int flag) {
  if (PF) pf = PF;
  width=W; height=H;
  if (buf || (flag & Flag::CreateBuf)) RenewBuffer();
  if (!ID && (flag & Flag::CreateGL)) {
    if (!cubemap) {
      screen->gd->DisableCubeMap();
      screen->gd->GenTextures(GraphicsDevice::Texture2D, 1, &ID);
    } else if (cubemap == CubeMap::PX) {
      screen->gd->ActiveTexture(0);
      screen->gd->GenTextures(GraphicsDevice::TextureCubeMap, 1, &ID);
    }
    if (!(flag & Flag::RepeatGL)) {
      screen->gd->TexParameter(GraphicsDevice::Texture2D, GraphicsDevice::TextureWrapS, GraphicsDevice::ClampToEdge);
      screen->gd->TexParameter(GraphicsDevice::Texture2D, GraphicsDevice::TextureWrapT, GraphicsDevice::ClampToEdge);
    }
  }
  if (ID || cubemap) {
    int opengl_width = screen->gd->TextureDim(width), opengl_height = screen->gd->TextureDim(height);
    int gl_tt = GLTexType(), gl_pt = GLPixelType(), gl_bt = GLBufferType();
    if (ID) screen->gd->BindTexture(gl_tt, ID);
    screen->gd->TexImage2D(gl_tt, 0, GraphicsDevice::GLInternalFormat, opengl_width, opengl_height, 0, gl_pt, gl_bt, 0);
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
  unique_ptr<VideoResamplerInterface> conv(CreateVideoResampler());
  conv->Open(dim.x, dim.y, PF, resample ? width : dim.x, resample ? height : dim.y, pf);
  conv->Resample(B, linesize, buf, LineSize(), 0, flag & Flag::FlipY);
}

void Texture::UpdateBuffer(const unsigned char *B, const ::LFL::Box &box, int PF, int linesize, int blit_flag) {
  SimpleVideoResampler::Blit(B, buf, box.w, box.h, PF, linesize, 0, 0, pf, LineSize(), box.x, box.y, blit_flag);
}

void Texture::Bind() const { screen->gd->BindTexture(GLTexType(), ID); }
void Texture::ClearGL() { 
  if (!app->MainThread()) { app->RunInMainThread(bind(&GraphicsDevice::DelTexture, screen->gd, ID)); ID=0; }
  else if (ID) { 
    if (screen) screen->gd->DelTexture(ID);
    else if (FLAGS_gd_debug) ERROR("DelTexture no screen ", ID);
    ID = 0;
  }
}

void Texture::LoadGL(const MultiProcessTextureResource &t) { return LoadGL(MakeUnsigned(t.buf.data()), point(t.width, t.height), t.pf, t.linesize); }
void Texture::LoadGL(const unsigned char *B, const point &dim, int PF, int linesize, int flag) {
  Texture temp;
  temp .Resize(dim.x, dim.y, preferred_pf, Flag::CreateBuf);
  temp .UpdateBuffer(B, dim, PF, linesize, Flag::FlipY);
  this->Resize(dim.x, dim.y, preferred_pf, Flag::CreateGL | (flag & Flag::RepeatGL));
  this->UpdateGL(temp.buf, LFL::Box(dim), flag);
}

void Texture::UpdateGL(const unsigned char *B, const ::LFL::Box &box, int flag) {
  int gl_tt = GLTexType(), gl_y = (flag & Flag::FlipY) ? (height - box.y - box.h) : box.y;
  screen->gd->BindTexture(gl_tt, ID);
  screen->gd->TexSubImage2D(gl_tt, 0, box.x, gl_y, box.w, box.h, GLPixelType(), GLBufferType(), B);
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
  if (out->geometry) Scene::Draw(screen->gd, out->geometry, e);
}

/* DepthTexture */

void DepthTexture::Resize(int W, int H, int DF, int flag) {
  if (DF) df = DF;
  width=W; height=H;
  if (!ID && (flag & Flag::CreateGL)) screen->gd->GenRenderBuffers(1, &ID);
  int opengl_width = screen->gd->TextureDim(width), opengl_height = screen->gd->TextureDim(height);
  if (ID) {
    screen->gd->BindRenderBuffer(ID);
    screen->gd->RenderBufferStorage(Depth::OpenGLID(df), opengl_width, opengl_height);
  }
}

void DepthTexture::ClearGL() {
  if (ID) {
    if (screen) screen->gd->DelRenderBuffers(1, &ID);
    else if (FLAGS_gd_debug) ERROR("DelRenderBuffer no screen ", ID);
    ID = 0;
  }
}

/* FrameBuffer */

void FrameBuffer::Resize(int W, int H, int flag) {
  width=W; height=H;
  if (!ID && (flag & Flag::CreateGL)) {
    screen->gd->GenFrameBuffers(1, &ID);
    if (flag & Flag::CreateTexture)      AllocTexture(&tex);
    if (flag & Flag::CreateDepthTexture) AllocDepthTexture(&depth);
  } else {
    tex.Resize(width, height);
    depth.Resize(width, height);
  }
  Attach(tex.ID, depth.ID);
  int status = screen->gd->CheckFrameBufferStatus();
  if (status != GraphicsDevice::FramebufferComplete) ERROR("FrameBuffer status ", status);
  if (flag & Flag::ReleaseFB) Release();
}

void FrameBuffer::AllocDepthTexture(DepthTexture *out) { CHECK_EQ(out->ID, 0); out->Create(width, height); }
void FrameBuffer::AllocTexture(unsigned *out) { Texture t; AllocTexture(&t); *out = t.ReleaseGL(); } 
void FrameBuffer::AllocTexture(Texture *out) {
  CHECK_EQ(out->ID, 0);
  out->Create(width, height); 
}

void FrameBuffer::Release(bool update_viewport) {
  screen->gd->BindFrameBuffer(screen->gd->default_framebuffer);
  if (update_viewport) screen->gd->RestoreViewport(DrawMode::_2D);
}

void FrameBuffer::Attach(int ct, int dt, bool update_viewport) {
  screen->gd->BindFrameBuffer(ID);
  if (ct) { if (tex  .ID != ct) tex.owner   = false; screen->gd->FrameBufferTexture     ((tex.ID   = ct)); }
  if (dt) { if (depth.ID != dt) depth.owner = false; screen->gd->FrameBufferDepthTexture((depth.ID = dt)); }
  if (update_viewport) {
    Box b(width, height);
    screen->gd->ViewPort(b);
    screen->gd->DrawMode(DrawMode::_2D, b, true);
  }
}

void FrameBuffer::ClearGL() {
  if (ID) {
    if (screen) screen->gd->DelFrameBuffers(1, &ID);
    else if (FLAGS_gd_debug) ERROR("DelFrameBuffer no screen ", ID);
    ID = 0;
  }
}

/* Shader */

void Shader::SetGlobalUniform1f(const string &name, float v) {
  screen->gd->UseShader(&app->shaders->shader_default);  app->shaders->shader_default .SetUniform1f(name, v);
  screen->gd->UseShader(&app->shaders->shader_normals);  app->shaders->shader_normals .SetUniform1f(name, v);
  screen->gd->UseShader(&app->shaders->shader_cubemap);  app->shaders->shader_cubemap .SetUniform1f(name, v);
  screen->gd->UseShader(&app->shaders->shader_cubenorm); app->shaders->shader_cubenorm.SetUniform1f(name, v);
}

void Shader::SetGlobalUniform2f(const string &name, float v1, float v2){ 
  screen->gd->UseShader(&app->shaders->shader_default);  app->shaders->shader_default .SetUniform2f(name, v1, v2);
  screen->gd->UseShader(&app->shaders->shader_normals);  app->shaders->shader_normals .SetUniform2f(name, v1, v2);
  screen->gd->UseShader(&app->shaders->shader_cubemap);  app->shaders->shader_cubemap .SetUniform2f(name, v1, v2);
  screen->gd->UseShader(&app->shaders->shader_cubenorm); app->shaders->shader_cubenorm.SetUniform2f(name, v1, v2);
}

int Shader::Create(const string &name, const string &vertex_shader, const string &fragment_shader, const ShaderDefines &defines, Shader *out) {
  INFO("Shader::Create ", name);
  int p = screen->gd->CreateProgram();

  string hdr =
    "#ifdef GL_ES\r\n"
    "precision highp float;\r\n"
    "#else\r\n"
    "#define lowp\r\n"
    "#define highp\r\n"
    "#endif\r\n";
  if (app->opengles_version == 2) hdr += "#define LFL_GLES2\r\n";
  hdr += defines.text + string("\r\n");

  if (vertex_shader.size()) {
    int vs = screen->gd->CreateShader(GraphicsDevice::VertexShader);
    screen->gd->CompileShader(vs, { hdr.c_str(), vertex_shader.c_str() });
    screen->gd->AttachShader(p, vs);
    screen->gd->DelShader(vs);
  }

  if (fragment_shader.size()) {
    int fs = screen->gd->CreateShader(GraphicsDevice::FragmentShader);
    screen->gd->CompileShader(fs, { hdr.c_str(), fragment_shader.c_str() });
    screen->gd->AttachShader(p, fs);
    screen->gd->DelShader(fs);
  }

  if (1)                    screen->gd->BindAttribLocation(p, 0, "Position"   );
  if (defines.normals)      screen->gd->BindAttribLocation(p, 1, "Normal"     );
  if (defines.vertex_color) screen->gd->BindAttribLocation(p, 2, "VertexColor");
  if (defines.tex_2d)       screen->gd->BindAttribLocation(p, 3, "TexCoordIn" );

  screen->gd->LinkProgram(p);

  int active_uniforms=0, max_uniform_components=0, active_attributes=0, max_attributes=0;
  screen->gd->GetProgramiv(p, GraphicsDevice::ActiveUniforms, &active_uniforms);
  screen->gd->GetProgramiv(p, GraphicsDevice::ActiveAttributes, &active_attributes);
#ifndef LFL_MOBILE
  screen->gd->GetIntegerv(GraphicsDevice::MaxVertexUniformComp, &max_uniform_components);
#endif
  screen->gd->GetIntegerv(GraphicsDevice::MaxVertexAttributes, &max_attributes);
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
    if ((out->uniform_model             = screen->gd->GetUniformLocation(p, "Model"))               < 0 && log_missing_attrib) INFO("shader ", name, " missing Model");
    if ((out->uniform_invview           = screen->gd->GetUniformLocation(p, "InverseView"))         < 0 && log_missing_attrib) INFO("shader ", name, " missing InverseView");
    if ((out->uniform_modelview         = screen->gd->GetUniformLocation(p, "Modelview"))           < 0 && log_missing_attrib) INFO("shader ", name, " missing Modelview");
    if ((out->uniform_modelviewproj     = screen->gd->GetUniformLocation(p, "ModelviewProjection")) < 0 && log_missing_attrib) INFO("shader ", name, " missing ModelviewProjection");
    if ((out->uniform_campos            = screen->gd->GetUniformLocation(p, "CameraPosition"))      < 0 && log_missing_attrib) INFO("shader ", name, " missing CameraPosition");
    if ((out->uniform_tex               = screen->gd->GetUniformLocation(p, "iChannel0"))           < 0 && log_missing_attrib) INFO("shader ", name, " missing Texture");
    if ((out->uniform_cubetex           = screen->gd->GetUniformLocation(p, "CubeTexture"))         < 0 && log_missing_attrib) INFO("shader ", name, " missing CubeTexture");
    if ((out->uniform_texon             = screen->gd->GetUniformLocation(p, "TexCoordEnabled"))     < 0 && log_missing_attrib) INFO("shader ", name, " missing TexCoordEnabled");
    if ((out->uniform_coloron           = screen->gd->GetUniformLocation(p, "VertexColorEnabled"))  < 0 && log_missing_attrib) INFO("shader ", name, " missing VertexColorEnabled");
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
    "uniform vec4 iMouse;\r\n"
    "uniform sampler2D iChannel0;\r\n"
    "uniform vec3 iChannelResolution[1];\r\n"
    "uniform vec2 iChannelScroll[1], iChannelModulus[1];\r\n"
    "#define SampleChannelAtPointAndModulus(c, p, m) texture2D(c, mod((p), (m)))\r\n"
    "#define SampleChannelAtPoint(c, p) SampleChannelAtPointAndModulus(c, p, iChannelModulus[0])\r\n"
    "#define SamplePoint() ((fragCoord.xy + iChannelScroll[0]) / iChannelResolution[0].xy)\r\n"
    "#define SamplePointFlipY() vec2((fragCoord.x + iChannelScroll[0].x) / iChannelResolution[0].x, \\\r\n"
    "                                (iChannelResolution[0].y - fragCoord.y - iChannelScroll[0].y) / iChannelResolution[0].y)\r\n"
    "#define SampleChannel(c) SampleChannelAtPoint(c, SamplePoint())\r\n"
    "#define BlendChannels(c1,c2) ((c1)*iBlend + (c2)*(1.0-iBlend))\r\n";

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
void Shader::ClearGL() {
  if (ID) {
    if (screen) screen->gd->DelProgram(ID);
    else if (FLAGS_gd_debug) ERROR("DelProgram no screen ", ID);
    ID = 0;
  }
}

int GraphicsDevice::VertsPerPrimitive(int primtype) {
  if (primtype == GraphicsDevice::Triangles) return 3;
  return 0;
}

void GraphicsDevice::PushColor() { default_color.push_back(default_color.back()); UpdateColor();  }
void GraphicsDevice::PopColor() {
  if      (default_color.size() >  1) default_color.pop_back();
  else if (default_color.size() == 1) default_color.back() = Color(1.0, 1.0, 1.0, 1.0);
  else FATAL("no color state");
  UpdateColor();
}

void GraphicsDevice::RestoreViewport(int dm) { ViewPort(Box(screen->x + screen->width, screen->y + screen->height)); DrawMode(dm); }
void GraphicsDevice::TranslateRotateTranslate(float a, const Box &b) { float x=b.x+b.w/2.0, y=b.y+b.h/2.0; Translate(x,y,0); Rotatef(a,0,0,1); Translate(-x,-y,0); }

void GraphicsDevice::DrawMode(int dm, bool flush) { return DrawMode(dm, screen->Box(), flush); }
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
  else { Scissor(screen->Box()); DisableScissor(); }
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
  else { Scissor(screen->Box()); DisableScissor(); }
}

Box GraphicsDevice::GetViewport() const { Box vp; GetIntegerv(ViewportBox, &vp.x); return vp; }
Box GraphicsDevice::GetScissorBox() const {
  auto &ss = scissor_stack.back();
  Box ret = ss.size() ? ss.back() : Box(-1,-1);
#ifdef LFL_DEBUG
  if (GetEnabled(ScissorTest)) { Box glb; GetIntegerv(ScissorBox, &glb.x); CHECK_EQ(glb,         ret); }
  else                                                                   { CHECK_EQ(Box(-1, -1), ret); }
#endif
  return ret;
}
  
void GraphicsDevice::DrawPixels(const Box &b, const Texture &tex) {
  Texture temp;
  temp.Resize(tex.width, tex.height, tex.pf, Texture::Flag::CreateGL);
  temp.UpdateGL(tex.buf, LFL::Box(tex.width, tex.height), Texture::Flag::FlipY); 
  b.Draw(this, temp.coord);
  temp.ClearGL();
}

/* Video */

#if 0
void *Video::BeginGLContextCreate(Window *W) {
#if defined(LFL_WINVIDEO)
  if (wglewIsSupported("WGL_ARB_create_context")) return wglCreateContextAttribsARB((HDC)W->surface, (HGLRC)W->gl, 0);
  else { HGLRC ret = wglCreateContext((HDC)W->surface); wglShareLists((HGLRC)W->gl, ret); return ret; }
#else
  return 0;
#endif
}

void *Video::CompleteGLContextCreate(Window *W, void *gl_context) {
#if defined(LFL_WINVIDEO)
  wglMakeCurrent((HDC)W->surface, (HGLRC)gl_context);
  return gl_context;
#elif defined(LFL_X11VIDEO)
  X11VideoModule *video = dynamic_cast<X11VideoModule*>(app->video->impl.get());
  GLXContext glc = glXCreateContext(video->display, video->vi, GetTyped<GLXContext>(W->gl), GL_TRUE);
  glXMakeCurrent(video->display, (::Window)(W->id), glc);
  return glc;
#else
  return 0;
#endif
}
#endif

void SimpleVideoResampler::RGB2BGRCopyPixels(unsigned char *dst, const unsigned char *src, int l, int bpp) {
  for (int k = 0; k < l; k++) for (int i = 0; i < bpp; i++) dst[k*bpp+(!i?2:(i==2?0:i))] = src[k*bpp+i];
}

bool SimpleVideoResampler::Supports(int f) { return f == Pixel::RGB24 || f == Pixel::BGR24 || f == Pixel::RGB32 || f == Pixel::BGR32 || f == Pixel::RGBA; }

bool SimpleVideoResampler::Opened() const { return s_fmt && d_fmt && s_width && d_width && s_height && d_height; }

void SimpleVideoResampler::Open(int sw, int sh, int sf, int dw, int dh, int df) {
  s_fmt = sf; s_width = sw; s_height = sh;
  d_fmt = df; d_width = dw; d_height = dh;
  // INFO("resample ", BlankNull(Pixel::Name(s_fmt)), " -> ", BlankNull(Pixel::Name(d_fmt)), " : (", sw, ",", sh, ") -> (", dw, ",", dh, ")");
}

void SimpleVideoResampler::Resample(const unsigned char *sb, int sls, unsigned char *db, int dls, bool flip_x, bool flip_y) {
  if (!Opened()) return ERROR("resample not opened()");

  int spw = Pixel::Size(s_fmt), dpw = Pixel::Size(d_fmt);
  if (spw * s_width > sls) return ERROR(spw * s_width, " > ", sls);
  if (dpw * d_width > dls) return ERROR(dpw * d_width, " > ", dls);

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
          *(dp + po) = MatrixAsFunc(&M, x?float(x)/(d_width-1):0, y?float(y)/(d_height-1):0) * 255;
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
    case Pixel::BGR32: r = *sp++; g = *sp++; b = *sp++; a=255; sp++; break;
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
    default: return ERROR("s_fmt ", s_fmt, " not supported");
  }
  switch (d_fmt) {
    case Pixel::RGB24: *dp++ = r; *dp++ = g; *dp++ = b; break;
    case Pixel::BGR24: *dp++ = b; *dp++ = g; *dp++ = r; break;
    case Pixel::RGB32: *dp++ = r; *dp++ = g; *dp++ = b; *dp++ = a; break;
    case Pixel::BGR32: *dp++ = b; *dp++ = g; *dp++ = r; *dp++ = a; break;
    case Pixel::RGBA:  *dp++ = r; *dp++ = g; *dp++ = b; *dp++ = a; break;
    case Pixel::BGRA:  *dp++ = b; *dp++ = g; *dp++ = r; *dp++ = a; break;
    default: return ERROR("d_fmt ", d_fmt, " not supported");
  }
}

void SimpleVideoResampler::Blit(const unsigned char *src, unsigned char *dst, int w, int h,
                                int sf, int sls, int sx, int sy,
                                int df, int dls, int dx, int dy, int flag) {
  bool flip_y = flag & Flag::FlipY;
  int sw = Pixel::Size(sf), dw = Pixel::Size(df); 
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
  int pw = Pixel::Size(pf);
  CopyColorChannelsToMatrix(buf, w, h, pw, ls, x, y, &M, ColorChannel::PixelOffset(channel));
  Matrix::Convolve(&M, kernel, &out, (flag & Flag::ZeroOnly) ? mZeroOnly : 0);
  CopyMatrixToColorChannels(&out, w, h, pw, ls, x, y, buf, ColorChannel::PixelOffset(channel));
}

void SimpleVideoResampler::Fill(unsigned char *dst, int w, int h,
                                int pf, int ls, int x, int y, const Color &c, int flag) {
  int pw = Pixel::Size(pf); 
  unsigned char pixel[4] = { uint8_t(c.R()), uint8_t(c.G()), uint8_t(c.B()), uint8_t(c.A()) };
  for (int yi = 0; yi < h; ++yi) {
    for (int xi = 0; xi < w; ++xi) {
      unsigned char *dp = dst + (ls*(y + yi) + (x + xi)*pw);
      CopyPixel(Pixel::RGBA, pf, pixel, dp, xi == 0, xi == w-1, flag);
    }
  }
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

}; // namespace LFL
