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

#include "core/app/gl/view.h"
#include "core/app/ipc.h"

namespace LFL {
DEFINE_float(shadertoy_blend, 0.5, "Shader blend factor");
DEFINE_int(soundasset_seconds, 10, "Soundasset buffer seconds");

const int SoundAsset::FlagNoRefill = 1;

int Geometry::VertsPerPrimitive(int primtype) {
  if (primtype == Primitive::Triangles) return 3;
  return 0;
}

void Geometry::SetPosition(const float *v) {
  vector<float> delta(vd);
  if (Vec<float>::Equals(v, &last_position[0], vd)) return;
  for (int j = 0; j<vd; j++) delta[j] = v[j] - last_position[j];
  for (int j = 0; j<vd; j++) last_position[j] = v[j];
  for (int i = 0; i<count; i++) for (int j = 0; j<vd; j++) vert[i*width+j] += delta[j];
}

void Geometry::ScrollTexCoord(GraphicsDevice *gd, float dx, float dx_extra, int *subtract_max_int) {
  int vpp = VertsPerPrimitive(primtype), prims = count / vpp, max_int = 0;
  int texcoord_offset = vd * (norm_offset < 0 ? 1 : 2);
  for (int prim = 0; prim < prims; prim++) {
    for (int i = 0; i < vpp; i++) {
      int vert_ind = prim * vpp + i, float_ind = vert_ind * width + texcoord_offset;
      vert[float_ind] = vert[float_ind] + dx + dx_extra - *subtract_max_int;
      int txi = vert[float_ind] - dx_extra;
      if (txi > max_int || !prim) max_int = txi;
    }
  }
  if (subtract_max_int) *subtract_max_int = max_int;
  int width_bytes = width * sizeof(float);
  int vert_size = count * width_bytes;
  gd->VertexPointer(vd, gd->c.Float, width_bytes, 0, &vert[0], vert_size, &vert_ind, true, gd->GetPrimitive(primtype));
}

string AssetLoading::FileName(const string &asset_fn) {
  if (asset_fn.empty()) return "";
  if (asset_fn[0] == '/') return asset_fn;
  return StrCat(appinfo->assetdir, asset_fn);
}

string AssetLoading::FileContents(const string &asset_fn) {
  auto i = asset_cache.find(asset_fn);
  if (i != asset_cache.end()) return string(i->second.data(), i->second.size());
  if (asset_fn[0] == '/') return LocalFile(asset_fn, "r").Contents();
#ifdef LFL_ANDROID
  static JNI *jni = Singleton<LFL::JNI>::Set();
  unique_ptr<BufferFile> f(jni->OpenAsset(asset_fn));
  return f ? string(move(f->buf)) : string();
#endif
  return LocalFile(FileName(asset_fn), "r").Contents();
}

unique_ptr<File> AssetLoading::OpenFile(const string &asset_fn) {
  auto i = asset_cache.find(asset_fn);
  if (i != asset_cache.end()) return make_unique<BufferFile>(StringPiece(i->second), asset_fn.c_str());
  if (asset_fn[0] == '/') return make_unique<LocalFile>(asset_fn, "r");
#ifdef LFL_ANDROID
  static JNI *jni = Singleton<LFL::JNI>::Set();
  return jni->OpenAsset(asset_fn);
#endif
  return make_unique<LocalFile>(FileName(asset_fn), "r");
}

void AssetLoading::LoadTexture(const string &asset_fn, Texture *out, VideoAssetLoader *l, int flag) {
  if (!FLAGS_enable_video) return;
  auto i = asset_cache.find(asset_fn);
  if (i != asset_cache.end()) return LoadTexture(i->second.data(), asset_fn.c_str(), i->second.size(), out);
  if (!l) l = asset_loader->default_video_loader;

  auto handle = l->LoadVideoFileNamed(asset_fn);
  if (!handle) return ERROR("load: ", asset_fn);
  l->LoadVideo(handle, out, flag);
}

void AssetLoading::LoadTexture(const void *FromBuf, const char *filename, int size, Texture *out, int flag) {
  VideoAssetLoader *l = asset_loader->default_loader->GetVideoAssetLoader(filename);
  auto handle = l->LoadVideoFile(make_unique<BufferFile>(string(static_cast<const char*>(FromBuf), size), filename));
  if (!handle) return;
  l->LoadVideo(handle, out, flag);
}

Texture *AssetLoading::LoadTexture(const MultiProcessFileResource &file, int max_image_size) {
  unique_ptr<Texture> tex = make_unique<Texture>(window);
  LoadTexture(file.buf.data(), file.name.data(), file.buf.size(), tex.get(), 0);

  if (tex->BufferSize() >= max_image_size) {
    unique_ptr<Texture> orig_tex = move(tex);
    tex = make_unique<Texture>(window);
    float scale_factor = sqrt(float(max_image_size)/orig_tex->BufferSize());
    tex->Resize(orig_tex->width*scale_factor, orig_tex->height*scale_factor, Pixel::RGB24, Texture::Flag::CreateBuf);

    unique_ptr<VideoResamplerInterface> resampler(CreateVideoResampler());
    resampler->Open(orig_tex->width, orig_tex->height, orig_tex->pf, tex->width, tex->height, tex->pf);
    resampler->Resample(orig_tex->buf, orig_tex->LineSize(), tex->buf, tex->LineSize());
  }

  if (!tex->buf || !tex->width || !tex->height) return 0;
  return tex.release();
}

void AssetLoading::LoadTextureArray(const string &fmt, const string &prefix, const string &suffix, int N, TextureArray*out, int flag) {
  out->a.clear();
  for (int i=0; i != N; i++) out->a.emplace_back(window);
  for (int i=0; i != N; i++)
    LoadTexture(StringPrintf(fmt.c_str(), prefix.c_str(), i, suffix.c_str()), &out->a[i], 0, flag);
}

Asset::Asset(AssetLoading *P)
  : Asset(P, "", "", 0, 0, 0, nullptr, nullptr, 0, 0) {}
Asset::Asset(AssetLoading *P, const string &N, const string &Tex, float S, int T, int R, const char *G, Geometry *H, unsigned CM, const DrawCB &CB)
  : Asset(P, N, Tex, S, T, R, nullptr, H, CM, 0, CB) { geom_fn = BlankNull(G); }
Asset::Asset(AssetLoading *P, const string &N, const string &Tex, float S, int T, int R, Geometry *G, Geometry *H, unsigned CM, unsigned TG, const DrawCB &CB)
  : parent(P), name(N), texture(Tex), cb(CB), scale(S), translate(T), rotate(R), geometry(G), hull(H), tex(P->window), texgen(TG),
  blends(P->window->GD()->c.SrcAlpha), blendt(P->window->GD()->c.OneMinusSrcAlpha) { tex.cubemap=CM; }

void Asset::Unload() {
  if (storage) storage->Unloaded(this);
  if (tex.ID) tex.ClearGL();
  if (geometry) { delete geometry; geometry = 0; }
  if (hull)     { delete hull;     hull     = 0; }
}

void Asset::ResetGL(int flag) {
  bool reload = flag & ResetGLFlag::Reload, forget = (flag & ResetGLFlag::Delete) == 0;
  if (!texture.empty()) {
    if (tex.ID) {
      if (forget) tex.ID = GraphicsDevice::TextureRef();
      else        tex.ClearGL();
    }
    if (reload) parent->LoadTexture(texture, &tex, nullptr);
  } 
}

void Asset::Load(VideoAssetLoader *l) {
  static int next_asset_type_id = 1, next_list_id = 1;
  if (!name.empty()) typeID = next_asset_type_id++;
  if (!geom_fn.empty()) geometry = Geometry::LoadOBJ(unique_ptr<File>(parent->OpenFile(geom_fn)).get()).release();
  if (!texture.empty()) parent->LoadTexture(texture, &tex, l);
}

void SoundAsset::Unload() {
  if (storage) storage->Unloaded(this);
  wav.reset();
  if (handle) ERROR("leak: ", handle);
}

void SoundAsset::Load(AudioAssetLoader::Handle &h, const char *FN, int Secs, int flag) {
  if (!FLAGS_enable_audio) return ERROR("load: ", FN, ": enable_audio = ", FLAGS_enable_audio);
  if (h) parent->asset_loader->default_audio_loader->LoadAudio(h, this, Secs, flag);
}

void SoundAsset::Load(void const *buf, int len, char const *FN, int Secs) {
  handle = parent->asset_loader->default_audio_loader->LoadAudioFile
    (make_unique<BufferFile>(string(static_cast<const char*>(buf), len), FN));
  if (!handle) return;

  Load(handle, FN, Secs, FlagNoRefill);
  if (!refill) handle.reset(0);
}

void SoundAsset::Load(int Secs, bool unload) {
  if (!filename.empty()) {
    if (0) handle = parent->asset_loader->default_audio_loader->LoadAudioFile(parent->OpenFile(filename));
    else   handle = parent->asset_loader->default_audio_loader->LoadAudioFileNamed(filename);
    if (!handle) ERROR("SoundAsset::Load ", filename);
  }

  Load(handle, filename.c_str(), Secs);

#ifndef LFL_MOBILE /* XXX */
  if (!refill && handle && unload) handle.reset(0);
#endif
}

int SoundAsset::Refill(int reset) { return parent->asset_loader->default_audio_loader->RefillAudio(this, reset); }

void MovieAsset::Load(const char *fn) {
  if (!fn || !(handle = parent->asset_loader->default_movie_loader->LoadMovieFile(parent->OpenFile(fn)))) return;
  parent->asset_loader->default_movie_loader->LoadMovie(handle, this);
  audio.Load();
  video.Load();
}

int MovieAsset::Play(int seek) {
  int ret;
  parent->asset_loader->movie_playing = this;
  if ((ret = parent->asset_loader->default_movie_loader->PlayMovie(this, 0)) <= 0) parent->asset_loader->movie_playing = 0;
  return ret;
}

/* asset impls */

void Line2DAsset::Draw(GraphicsDevice *gd) {
  static int verts_ind = gd->RegisterBuffer(&verts_ind);
  gd->DisableTexture();

  float verts[] = { /*1*/ float(p1.x), float(p1.y), /*2*/ float(p2.x), float(p2.y) };
  gd->VertexPointer(2, gd->c.Float, 0, 0, verts, sizeof(verts), &verts_ind, true);

  if (color) gd->Color4f(color->r(), color->g(), color->b(), color->a());
  gd->DrawArrays(gd->c.LineLoop, 0, 2);
}

void Axis3DAsset::Draw(GraphicsDevice *gd, Asset*, Entity*) {
  static int vert_id = gd->RegisterBuffer(&vert_id);
  const float scaleFactor = 1;
  const float range = powf(10, scaleFactor);
  const float step = range/10;

  gd->DisableNormals();
  gd->DisableTexture();

  float vert[] = { /*1*/ 0, 0, 0, /*2*/ step, 0, 0, /*3*/ 0, 0, 0, /*4*/ 0, step, 0, /*5*/ 0, 0, 0, /*6*/ 0, 0, step };
  gd->VertexPointer(3, gd->c.Float, 0, 0, vert, sizeof(vert), &vert_id, false);

  /* origin */
  gd->PointSize(7);
  gd->Color4f(1, 1, 0, 1);
  gd->DrawArrays(gd->c.Points, 0, 1);

  /* R=X */
  gd->PointSize(5);
  gd->Color4f(1, 0, 0, 1);
  gd->DrawArrays(gd->c.Points, 1, 1);

  /* G=Y */
  gd->Color4f(0, 1, 0, 1);
  gd->DrawArrays(gd->c.Points, 3, 1);

  /* B=Z */
  gd->Color4f(0, 0, 1, 1);
  gd->DrawArrays(gd->c.Points, 5, 1);

  /* axis */
  gd->PointSize(1);
  gd->Color4f(.5, .5, .5, 1);
  gd->DrawArrays(gd->c.Lines, 0, 5);
}

void Room3DAsset::Draw(GraphicsDevice *gd, Asset*, Entity*) {
  gd->DisableNormals();
  gd->DisableTexture();

  static int verts_id = gd->RegisterBuffer(&verts_id);
  float verts[] = {
    /*1*/ 0,0,0, /*2*/ 0,2,0, /*3*/ 2,0,0,
    /*4*/ 0,0,2, /*5*/ 0,2,0, /*6*/ 0,0,0,
    /*7*/ 0,0,0, /*8*/ 2,0,0, /*9*/ 0,0,2 
  };
  gd->VertexPointer(3, gd->c.Float, 0, 0, verts, sizeof(verts), &verts_id, false);

  gd->Color4f(.8,.8,.8,1);
  gd->DrawArrays(gd->c.Triangles, 0, 3);

  gd->Color4f(.7,.7,.7,1);
  gd->DrawArrays(gd->c.Triangles, 3, 3);

  gd->Color4f(.6,.6,.6,1);
  gd->DrawArrays(gd->c.Triangles, 6, 3);;
}

void Intersect2DAsset::Draw(GraphicsDevice *gd) {
  unique_ptr<Geometry> geom = make_unique<Geometry>(gd->c.Lines, 4, NullPointer<v2>(), NullPointer<v3>(), NullPointer<v2>(), *color);
  v2 *vert = reinterpret_cast<v2*>(&geom->vert[0]);

  vert[0] = v2(0, y);
  vert[1] = v2(gd->parent->gl_w, y);
  vert[2] = v2(x, 0);
  vert[3] = v2(x, gd->parent->gl_h);

  gd->DisableTexture();
  Scene::Select(gd, geom.get());
  Scene::Draw(gd, geom.get(), 0);
}

void ShaderToyAsset::Draw(GraphicsDevice *gd) {
  Window *screen = gd->parent;
  Application *a = screen->parent;
  float scale = shader->scale;
  gd->UseShader(shader);
  shader->SetUniform1f("iGlobalTime", ToFSeconds(Now() - a->time_started).count());
  shader->SetUniform1f("iBlend", FLAGS_shadertoy_blend);
  shader->SetUniform4f("iMouse", screen->mouse.x, screen->mouse.y, a->input ? a->input->MouseButton1Down() : 0, 0);
  shader->SetUniform3f("iResolution", XY_or_Y(scale, screen->gl_x + screen->gl_w), XY_or_Y(scale, screen->gl_y + screen->gl_h), 0);
  if (tex) {
    shader->SetUniform3f("iChannelResolution", XY_or_Y(scale, tex->width), XY_or_Y(scale, tex->height), 1);
    shader->SetUniform4f("iTargetBox", 0, 0, XY_or_Y(scale, tex->width), XY_or_Y(scale, tex->height));
  }
}

void ShaderToyAsset::DrawWindows(GraphicsDevice *gd, const Box &w) { DrawWindows(gd, vector<const Box*>(1, &w)); }
void ShaderToyAsset::DrawWindows(GraphicsDevice *gd, const vector<const Box*> &wins, point p) {
  if (shader) Draw(gd);
  else gd->SetColor(backup_color);
  if (tex) { gd->EnableLayering(); tex->Bind(); }
  else gd->DisableTexture();
  if (p.x || p.y) { for (auto w : wins) { Box b = *w + p; GraphicsContext::DrawTexturedBox1(gd,  b, tex ? tex->coord : 0); } }
  else            { for (auto w : wins) {                 GraphicsContext::DrawTexturedBox1(gd, *w, tex ? tex->coord : 0); } }
  if (shader) gd->UseShader(0);
}

void BoxFilled::Draw(GraphicsContext *gc, const LFL::Box &b) const { gc->DrawTexturedBox(b); }
void BoxOutline::Draw(GraphicsContext *gc, const LFL::Box &b) const {
  gc->gd->DisableTexture();
  int line_width = gc->attr ? gc->attr->line_width : 1;
  if (line_width <= 1) {
    static int verts_ind = gc->gd->RegisterBuffer(&verts_ind);
    float verts[] = { /*1*/ float(b.x),     float(b.y),     /*2*/ float(b.x),     float(b.y+b.h),
                      /*3*/ float(b.x+b.w), float(b.y+b.h), /*4*/ float(b.x+b.w), float(b.y) };
    gc->gd->VertexPointer(2, gc->gd->c.Float, 0, 0, verts, sizeof(verts), &verts_ind, true);
    gc->gd->DrawArrays(gc->gd->c.LineLoop, 0, 4);
  } else {
    static int verts_ind = gc->gd->RegisterBuffer(&verts_ind);
    int lw = line_width-1;
    float verts[] = {
      /*1.1*/ float(b.x-lw),     float(b.y-lw),     /*1.2*/ float(b.x-lw),     float(b.y+b.h+lw),
      /*1.4*/ float(b.x),        float(b.y),        /*1.3*/ float(b.x),        float(b.y+b.h),
      /*2.1*/ float(b.x),        float(b.y+b.h),    /*2.2*/ float(b.x-lw),     float(b.y+b.h+lw),
      /*2.4*/ float(b.x+b.w),    float(b.y+b.h),    /*2.3*/ float(b.x+b.w+lw), float(b.y+b.h+lw),
      /*3.3*/ float(b.x+b.w+lw), float(b.y+b.h+lw), /*3.4*/ float(b.x+b.w+lw), float(b.y-lw),
      /*3.2*/ float(b.x+b.w),    float(b.y+b.h),    /*3.1*/ float(b.x+b.w),    float(b.y),
      /*4.3*/ float(b.x+b.w),    float(b.y),        /*4.4*/ float(b.x+b.w+lw), float(b.y-lw),
      /*4.2*/ float(b.x),        float(b.y),        /*4.1*/ float(b.x-lw),     float(b.y-lw)
    };
    gc->gd->VertexPointer(2, gc->gd->c.Float, 0, 0, verts, sizeof(verts), &verts_ind, true);
    gc->gd->DrawArrays(gc->gd->c.TriangleStrip, 0, 16);
  }
}

void BoxTopLeftOutline::Draw(GraphicsContext *gc, const LFL::Box &b) const {
  gc->gd->DisableTexture();
  int line_width = gc->attr ? gc->attr->line_width : 1;
  if (line_width <= 1) {
    static int verts_ind = gc->gd->RegisterBuffer(&verts_ind);
    float verts[] = { /*1*/ float(b.x), float(b.y),     /*2*/ float(b.x),     float(b.y+b.h),
                      /*2*/ float(b.x), float(b.y+b.h), /*3*/ float(b.x+b.w), float(b.y+b.h) };
    gc->gd->VertexPointer(2, gc->gd->c.Float, 0, 0, verts, sizeof(verts), &verts_ind, true);
    gc->gd->DrawArrays(gc->gd->c.Lines, 0, 4);
  } else {
    static int verts_ind = gc->gd->RegisterBuffer(&verts_ind);
    int lw = line_width-1;
    float verts[] = {
      /*1.1*/ float(b.x-lw),     float(b.y-lw),     /*1.2*/ float(b.x-lw),     float(b.y+b.h+lw),
      /*1.4*/ float(b.x),        float(b.y),        /*1.3*/ float(b.x),        float(b.y+b.h),
      /*2.1*/ float(b.x),        float(b.y+b.h),    /*2.2*/ float(b.x-lw),     float(b.y+b.h+lw),
      /*2.4*/ float(b.x+b.w),    float(b.y+b.h),    /*2.3*/ float(b.x+b.w+lw), float(b.y+b.h+lw),
    };
    gc->gd->VertexPointer(2, gc->gd->c.Float, 0, 0, verts, sizeof(verts), &verts_ind, true);
    gc->gd->DrawArrays(gc->gd->c.TriangleStrip, 0, 8);
  }
}

void BoxBottomRightOutline::Draw(GraphicsContext *gc, const LFL::Box &b) const {
  gc->gd->DisableTexture();
  int line_width = gc->attr ? gc->attr->line_width : 1;
  if (line_width <= 1) {
    static int verts_ind = gc->gd->RegisterBuffer(&verts_ind);
    float verts[] = { /*1*/ float(b.x),     float(b.y), /*4*/ float(b.x+b.w), float(b.y),
                      /*4*/ float(b.x+b.w), float(b.y), /*3*/ float(b.x+b.w), float(b.y+b.h) };
    gc->gd->VertexPointer(2, gc->gd->c.Float, 0, 0, verts, sizeof(verts), &verts_ind, true);
    gc->gd->DrawArrays(gc->gd->c.Lines, 0, 4);
  } else {
    static int verts_ind = gc->gd->RegisterBuffer(&verts_ind);
    int lw = line_width-1;
    float verts[] = {
      /*3.3*/ float(b.x+b.w+lw), float(b.y+b.h+lw), /*3.4*/ float(b.x+b.w+lw), float(b.y-lw),
      /*3.2*/ float(b.x+b.w),    float(b.y+b.h),    /*3.1*/ float(b.x+b.w),    float(b.y),
      /*4.3*/ float(b.x+b.w),    float(b.y),        /*4.4*/ float(b.x+b.w+lw), float(b.y-lw),
      /*4.2*/ float(b.x),        float(b.y),        /*4.1*/ float(b.x-lw),     float(b.y-lw)
    };
    gc->gd->VertexPointer(2, gc->gd->c.Float, 0, 0, verts, sizeof(verts), &verts_ind, true);
    gc->gd->DrawArrays(gc->gd->c.TriangleStrip, 0, 8);
  }
}

Waveform::Waveform(point dim, const Color *c, const Vec<float> *sbh) : width(dim.x), height(dim.y) {
  float xmax=sbh->Len(), ymin=INFINITY, ymax=-INFINITY;
  for (int i=0; i<xmax; i++) {
    ymin = min(ymin,sbh->Read(i));
    ymax = max(ymax,sbh->Read(i));
  }

  geom = make_unique<Geometry>(Geometry::Primitive::Lines, int(xmax-1)*2, NullPointer<v2>(), NullPointer<v3>(), NullPointer<v2>(), *c);
  v2 *vert = reinterpret_cast<v2*>(&geom->vert[0]);

  for (int i=0; i<geom->count/2; i++) {
    float x1=(i)  *1.0/xmax, y1=(sbh->Read(i)   - ymin)/(ymax - ymin); 
    float x2=(i+1)*1.0/xmax, y2=(sbh->Read(i+1) - ymin)/(ymax - ymin);

    vert[i*2+0] = v2(x1 * dim.x, -dim.y + y1 * dim.y);
    vert[i*2+1] = v2(x2 * dim.x, -dim.y + y2 * dim.y);
  }
}

void Waveform::Draw(GraphicsContext *gc, const LFL::Box &w) const {
  if (!geom) return;
  geom->SetPosition(w.Position());
  gc->gd->DisableTexture();
  Scene::Select(gc->gd, geom.get());
  Scene::Draw(gc->gd, geom.get(), 0);
}

Waveform Waveform::Decimated(point dim, const Color *c, const RingSampler::Handle *sbh, int decimateBy) {
  unique_ptr<RingSampler> dsb;
  RingSampler::Handle dsbh;
  if (decimateBy) {
    dsb = unique_ptr<RingSampler>(Decimate(sbh, decimateBy));
    dsbh = RingSampler::Handle(dsb.get());
    sbh = &dsbh;
  }
  Waveform WF(dim, c, sbh);
  return WF;
}

void SpectogramAsset::Draw(GraphicsDevice *gd, Matrix *m, unsigned char *data, int pf, int width, int height, int hjump, float vmax, float clip, bool interpolate, int pd) {
  int ps = Pixel::Size(pf);
  unsigned char pb[4];
  double v;

  for (int j = 0; j < width; j++) {
    for (int i = 0; i < height; i++) {
      if (!interpolate) v = m->row(j)[i];
      else              v = MatrixAsFunc(m, i?float(i)/(height-1):0, j?float(j)/(width-1):0);

      if      (pd == PowerDomain::dB)   { /**/ }
      else if (pd == PowerDomain::abs)  { v = AmplitudeRatioDecibels(v, 1); }
      else if (pd == PowerDomain::abs2) { v = AmplitudeRatioDecibels(sqrt(v), 1); }

      if (v < clip) v = clip;
      v = (v - clip) / (vmax - clip);

      pb[0] = unsigned(v > 1.0/3.0 ? 255 : v*3*255); v = max(0.0, v-1.0/3.0);
      pb[1] = unsigned(v > 1.0/3.0 ? 255 : v*3*255); v = max(0.0, v-1.0/3.0);
      pb[2] = unsigned(v > 1.0/3.0 ? 255 : v*3*255); v = max(0.0, v-1.0/3.0);
      pb[3] = unsigned(255);

      SimpleVideoResampler::CopyPixel(Pixel::RGBA, pf, pb, data + j*hjump*ps + i*ps, 0, 0);
    }
  }
}

void SpectogramAsset::Draw(GraphicsDevice *gd, Matrix *m, Texture *t, float *max, float clip, int pd) {
  if (!t->ID) t->CreateBacked(m->N, m->M);
  else {
    if (t->width < m->N || t->height < m->M) t->Resize(m->N, m->M);
    else Texture::Coordinates(t->coord, m->N, m->M, gd->TextureDim(t->width), gd->TextureDim(t->height));
  }

  if (clip == -INFINITY) clip = -65;
  if (!t->buf) return;

  float Max = Matrix::Max(m);
  if (max) *max = Max;

  Draw(gd, m, t->buf, t->pf, m->M, m->N, t->width, Max, clip, 0, pd);

  gd->BindTexture(t->ID);
  t->UpdateGL();
}

void SpectogramAsset::Draw(GraphicsDevice *gd, const RingSampler::Handle *in, Texture *t, Matrix *transform, float *max, float clip) {
  /* 20*log10(abs(specgram(y,2048,sr,hamming(512),256))) */
  unique_ptr<Matrix> m(Spectogram(in, 0, 512, 256, 512, vector<double>(), PowerDomain::abs));
  if (transform) m = Matrix::Mult(move(m), transform);
  Draw(gd, m.get(), t, max, clip, PowerDomain::abs);
}

/* Cube */

unique_ptr<Geometry> Cube::Create(v3 v) { return Create(v.x, v.y, v.z); } 
unique_ptr<Geometry> Cube::Create(float rx, float ry, float rz, bool normals) {
  vector<v3> verts, norms;

  PushBack(&verts, &norms, v3(-rx,  ry, -rz), v3(0, 0, 1));
  PushBack(&verts, &norms, v3(-rx, -ry, -rz), v3(0, 0, 1));
  PushBack(&verts, &norms, v3( rx, -ry, -rz), v3(0, 0, 1));
  PushBack(&verts, &norms, v3( rx,  ry, -rz), v3(0, 0, 1));
  PushBack(&verts, &norms, v3(-rx,  ry, -rz), v3(0, 0, 1));
  PushBack(&verts, &norms, v3( rx, -ry, -rz), v3(0, 0, 1));

  PushBack(&verts, &norms, v3( rx,  ry,  rz), v3(0, 0, -1));
  PushBack(&verts, &norms, v3( rx, -ry,  rz), v3(0, 0, -1));
  PushBack(&verts, &norms, v3(-rx, -ry,  rz), v3(0, 0, -1));
  PushBack(&verts, &norms, v3(-rx,  ry,  rz), v3(0, 0, -1));
  PushBack(&verts, &norms, v3( rx,  ry,  rz), v3(0, 0, -1));
  PushBack(&verts, &norms, v3(-rx, -ry,  rz), v3(0, 0, -1));

  PushBack(&verts, &norms, v3(rx,  ry, -rz), v3(-1, 0, 0));
  PushBack(&verts, &norms, v3(rx, -ry, -rz), v3(-1, 0, 0));
  PushBack(&verts, &norms, v3(rx, -ry,  rz), v3(-1, 0, 0));
  PushBack(&verts, &norms, v3(rx,  ry,  rz), v3(-1, 0, 0));
  PushBack(&verts, &norms, v3(rx,  ry, -rz), v3(-1, 0, 0));
  PushBack(&verts, &norms, v3(rx, -ry,  rz), v3(-1, 0, 0));

  PushBack(&verts, &norms, v3(-rx,  ry,  rz), v3(1, 0, 0));
  PushBack(&verts, &norms, v3(-rx, -ry,  rz), v3(1, 0, 0));
  PushBack(&verts, &norms, v3(-rx, -ry, -rz), v3(1, 0, 0));
  PushBack(&verts, &norms, v3(-rx,  ry, -rz), v3(1, 0, 0));
  PushBack(&verts, &norms, v3(-rx,  ry,  rz), v3(1, 0, 0));
  PushBack(&verts, &norms, v3(-rx, -ry, -rz), v3(1, 0, 0));

  PushBack(&verts, &norms, v3( rx,  ry, -rz), v3(0, -1, 0));
  PushBack(&verts, &norms, v3( rx,  ry,  rz), v3(0, -1, 0));
  PushBack(&verts, &norms, v3(-rx,  ry,  rz), v3(0, -1, 0));
  PushBack(&verts, &norms, v3(-rx,  ry, -rz), v3(0, -1, 0));
  PushBack(&verts, &norms, v3( rx,  ry, -rz), v3(0, -1, 0));
  PushBack(&verts, &norms, v3(-rx,  ry,  rz), v3(0, -1, 0));

  PushBack(&verts, &norms, v3( rx, -ry,  rz), v3(0, 1, 0));
  PushBack(&verts, &norms, v3( rx, -ry, -rz), v3(0, 1, 0));
  PushBack(&verts, &norms, v3(-rx, -ry, -rz), v3(0, 1, 0));
  PushBack(&verts, &norms, v3(-rx, -ry,  rz), v3(0, 1, 0));
  PushBack(&verts, &norms, v3( rx, -ry,  rz), v3(0, 1, 0));
  PushBack(&verts, &norms, v3(-rx, -ry, -rz), v3(0, 1, 0));

  return make_unique<Geometry>(Geometry::Primitive::Triangles, verts.size(), &verts[0], normals ? &norms[0] : 0, nullptr, nullptr);
}

unique_ptr<Geometry> Cube::CreateFrontFace(float r) {
  vector<v3> verts;
  vector<v2> tex;
  PushBack(&verts, &tex, v3(-r,  r, -r), v2(1, 1)); 
  PushBack(&verts, &tex, v3(-r, -r, -r), v2(1, 0)); 
  PushBack(&verts, &tex, v3( r, -r, -r), v2(0, 0)); 
  PushBack(&verts, &tex, v3( r,  r, -r), v2(0, 1)); 
  PushBack(&verts, &tex, v3(-r,  r, -r), v2(1, 1)); 
  PushBack(&verts, &tex, v3( r, -r, -r), v2(0, 0)); 
  return make_unique<Geometry>(Geometry::Primitive::Triangles, verts.size(), &verts[0], nullptr, &tex[0]);
}

unique_ptr<Geometry> Cube::CreateBackFace(float r) {
  vector<v3> verts;
  vector<v2> tex;
  PushBack(&verts, &tex, v3( r,  r,  r), v2(1, 1)); 
  PushBack(&verts, &tex, v3( r, -r,  r), v2(1, 0)); 
  PushBack(&verts, &tex, v3(-r, -r,  r), v2(0, 0)); 
  PushBack(&verts, &tex, v3(-r,  r,  r), v2(0, 1)); 
  PushBack(&verts, &tex, v3( r,  r,  r), v2(1, 1)); 
  PushBack(&verts, &tex, v3(-r, -r,  r), v2(0, 0)); 
  return make_unique<Geometry>(Geometry::Primitive::Triangles, verts.size(), &verts[0], nullptr, &tex[0]);
}

unique_ptr<Geometry> Cube::CreateLeftFace(float r) {
  vector<v3> verts;
  vector<v2> tex;
  PushBack(&verts, &tex, v3(r,  r, -r), v2(1, 1));
  PushBack(&verts, &tex, v3(r, -r, -r), v2(1, 0));
  PushBack(&verts, &tex, v3(r, -r,  r), v2(0, 0));
  PushBack(&verts, &tex, v3(r,  r,  r), v2(0, 1));
  PushBack(&verts, &tex, v3(r,  r, -r), v2(1, 1));
  PushBack(&verts, &tex, v3(r, -r,  r), v2(0, 0));
  return make_unique<Geometry>(Geometry::Primitive::Triangles, verts.size(), &verts[0], nullptr, &tex[0]);
}

unique_ptr<Geometry> Cube::CreateRightFace(float r) {
  vector<v3> verts;
  vector<v2> tex;
  PushBack(&verts, &tex, v3(-r,  r,  r), v2(1, 1));
  PushBack(&verts, &tex, v3(-r, -r,  r), v2(1, 0));
  PushBack(&verts, &tex, v3(-r, -r, -r), v2(0, 0));
  PushBack(&verts, &tex, v3(-r,  r, -r), v2(0, 1));
  PushBack(&verts, &tex, v3(-r,  r,  r), v2(1, 1));
  PushBack(&verts, &tex, v3(-r, -r, -r), v2(0, 0));
  return make_unique<Geometry>(Geometry::Primitive::Triangles, verts.size(), &verts[0], nullptr, &tex[0]);
}

unique_ptr<Geometry> Cube::CreateTopFace(float r) {
  vector<v3> verts;
  vector<v2> tex;
  PushBack(&verts, &tex, v3( r,  r, -r), v2(1, 1));
  PushBack(&verts, &tex, v3( r,  r,  r), v2(1, 0));
  PushBack(&verts, &tex, v3(-r,  r,  r), v2(0, 0));
  PushBack(&verts, &tex, v3(-r,  r, -r), v2(0, 1));
  PushBack(&verts, &tex, v3( r,  r, -r), v2(1, 1));
  PushBack(&verts, &tex, v3(-r,  r,  r), v2(0, 0));
  return make_unique<Geometry>(Geometry::Primitive::Triangles, verts.size(), &verts[0], nullptr, &tex[0]);
}

unique_ptr<Geometry> Cube::CreateBottomFace(float r) {
  vector<v3> verts;
  vector<v2> tex;
  PushBack(&verts, &tex, v3( r, -r,  r), v2(1, 1));
  PushBack(&verts, &tex, v3( r, -r, -r), v2(1, 0));
  PushBack(&verts, &tex, v3(-r, -r, -r), v2(0, 0));
  PushBack(&verts, &tex, v3(-r, -r,  r), v2(0, 1));
  PushBack(&verts, &tex, v3( r, -r,  r), v2(1, 1));
  PushBack(&verts, &tex, v3(-r, -r, -r), v2(0, 0));
  return make_unique<Geometry>(Geometry::Primitive::Triangles, verts.size(), &verts[0], nullptr, &tex[0]);
}

unique_ptr<Geometry> Grid::Grid3D() {
  const float scale_factor = 1, range = powf(10, scale_factor), step = range/10;
  vector<v3> verts;
  for (float d = -range ; d <= range; d += step) {
    verts.push_back(v3(range,0,d));
    verts.push_back(v3(-range,0,d));
    verts.push_back(v3(d,0,range));
    verts.push_back(v3(d,0,-range));
  }
  return make_unique<Geometry>(Geometry::Primitive::Lines, verts.size(), &verts[0], nullptr, nullptr, Color(.5,.5,.5));
}

unique_ptr<Geometry> Grid::Grid2D(float x, float y, float range, float step) {
  vector<v2> verts;
  for (float d = 0; d <= range; d += step) {
    verts.push_back(v2(x+range, y+d));
    verts.push_back(v2(x,       y+d));
    verts.push_back(v2(x+d, y+range));
    verts.push_back(v2(x+d, y));
  }
  return make_unique<Geometry>(Geometry::Primitive::Lines, verts.size(), &verts[0], nullptr, nullptr, Color(0,0,0));
}

Skybox::Skybox(AssetLoading *p) :
  a_left  (p, "", "", 1, 0, 0, Cube::Create(500, 500, 500).release(), 0, CubeMap::PX, TexGen::LINEAR),
  a_right (p, "", "", 1, 0, 0, 0,                                     0, CubeMap::NX, 0),
  a_top   (p, "", "", 1, 0, 0, 0,                                     0, CubeMap::PY, 0),
  a_bottom(p, "", "", 1, 0, 0, 0,                                     0, CubeMap::NY, 0),
  a_front (p, "", "", 1, 0, 0, 0,                                     0, CubeMap::PZ, 0),
  a_back  (p, "", "", 1, 0, 0, 0,                                     0, CubeMap::NZ, 0),
  e_left ("sb_left",  &a_left),  e_right ("sb_right",  &a_right),
  e_top  ("sb_top",   &a_top),   e_bottom("sb_bottom", &a_bottom),
  e_front("sb_front", &a_front), e_back  ("sb_back",   &a_back)
{ 
  PushBack(&v_left, &v_right, &v_top, &e_left, &e_right, &e_left);
  PushBack(&v_bottom, &v_front, &v_back, &e_right, &e_left, &e_right);
}

void Skybox::Load(const string &filename_prefix) {
  a_left  .texture = StrCat(filename_prefix,   "_left.png"); a_left  .Load();
  a_right .texture = StrCat(filename_prefix,  "_right.png"); a_right .Load();
  a_top   .texture = StrCat(filename_prefix,    "_top.png"); a_top   .Load();
  a_bottom.texture = StrCat(filename_prefix, "_bottom.png"); a_bottom.Load();
  a_front .texture = StrCat(filename_prefix,  "_front.png"); a_front .Load();
  a_back  .texture = StrCat(filename_prefix,   "_back.png"); a_back  .Load();
}

void Skybox::Draw(GraphicsDevice *gd) {
  gd->DisableNormals();
  gd->DisableVertexColor();
  gd->DisableDepthTest();
  Scene::Draw(gd, &a_left,   0, v_left);
  Scene::Draw(gd, &a_right,  0, v_right);
  Scene::Draw(gd, &a_top,    0, v_top);
  Scene::Draw(gd, &a_bottom, 0, v_bottom);
  Scene::Draw(gd, &a_front,  0, v_front);
  Scene::Draw(gd, &a_back,   0, v_back);
  gd->EnableDepthTest();
}

}; // namespace LFL
