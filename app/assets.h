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

#ifndef LFL_CORE_APP_ASSETS_H__
#define LFL_CORE_APP_ASSETS_H__
namespace LFL {

DECLARE_int(soundasset_seconds);
DECLARE_float(shadertoy_blend);

struct Geometry {
  int vd, td, cd, primtype, count, material, color;
  Material mat;
  Color col;
  vector<float> vert, last_position;
  int width, vert_ind, norm_offset, tex_offset, color_offset;
  Geometry(int N=0, int V=0, int PT=0, int VD=0, int TD=0, int CD=0) : vd(VD), td(TD), cd(CD), primtype(PT), count(N),
  material(0), color(0), vert(V), last_position(VD), width(VD), vert_ind(-1), norm_offset(-1), tex_offset(-1), color_offset(-1) {}

  static const int TD=2, CD=4;
  template <class X> Geometry(int VD, int primtype, int num, X *v, v3 *norm, v2 *tex, const Color *vcol) :
    Geometry(num, num*VD*(1+(norm!=0)) + num*TD*(tex!=0) + num*CD*(vcol!=0) + 256, primtype, VD, TD, CD)
  {
    if (norm) { norm_offset  = width*sizeof(float); width += VD; }
    if (tex)  { tex_offset   = width*sizeof(float); width += TD; }
    if (vcol) { color_offset = width*sizeof(float); width += CD; }
    for (int i = 0; i<count; i++) {
      int k = 0;
      for (int j = 0; j<VD && v;    j++, k++) vert[i*width + k] = v[i][j];
      for (int j = 0; j<VD && norm; j++, k++) vert[i*width + k] = norm[i][j];
      for (int j = 0; j<TD && tex;  j++, k++) vert[i*width + k] = tex[i][j];
      for (int j = 0; j<CD && vcol; j++, k++) vert[i*width + k] = vcol[i].x[j];
    }
  }
  template <class X> Geometry(int VD, int primtype, int num, X *v, v3 *norm, v2 *tex, const Color &vcol) :
    Geometry(VD, primtype, num, v, norm, tex, nullptr) { color=1; col=vcol; }

  Geometry (int pt, int n, v2 *v, v3 *norm, v2 *tex            ) : Geometry(2, pt, n, v, norm, tex, 0  ) {}
  Geometry (int pt, int n, v3 *v, v3 *norm, v2 *tex            ) : Geometry(3, pt, n, v, norm, tex, 0  ) {}
  Geometry (int pt, int n, v2 *v, v3 *norm, v2 *tex, Color  col) : Geometry(2, pt, n, v, norm, tex, col) {}
  Geometry (int pt, int n, v3 *v, v3 *norm, v2 *tex, Color  col) : Geometry(3, pt, n, v, norm, tex, col) {}
  Geometry (int pt, int n, v2 *v, v3 *norm, v2 *tex, Color *col) : Geometry(2, pt, n, v, norm, tex, col) {}
  Geometry (int pt, int n, v3 *v, v3 *norm, v2 *tex, Color *col) : Geometry(3, pt, n, v, norm, tex, col) {}

  void SetPosition(const float *v);
  void SetPosition(const point &p) { v2 v=p; SetPosition(&v[0]); }
  void ScrollTexCoord(GraphicsDevice*, float dx, float dx_extra, int *subtract_max_int);

  static unique_ptr<Geometry> LoadOBJ(File*, const float *map_tex_coord=0);
  static string ExportOBJ(const Geometry *geometry, const set<int> *prim_filter=0, bool prim_filter_invert=0);
};

template <class X> struct AssetMapT {
  bool loaded=0;
  vector<X> vec;
  unordered_map<string, X*> amap;
  virtual ~AssetMapT() {}

  template<typename ...Args>
  void Add(Args&& ...args) { CHECK(!loaded); vec.emplace_back(forward<Args>(args)...); }
  void Unloaded(X *a) { if (!a->name.empty()) amap.erase(a->name); }
  void Load(X *a) { a->storage = this; if (!a->name.empty()) amap[a->name] = a; a->Load(); }
  void Load() { CHECK(!loaded); for (int i=0; i<vec.size(); i++) Load(&vec[i]); loaded=1; } X *operator()(const string &an) { return FindOrNull(amap, an); }
};

typedef AssetMapT<     Asset>      AssetMap;
typedef AssetMapT<SoundAsset> SoundAssetMap;
typedef AssetMapT<MovieAsset> MovieAssetMap;

struct AssetLoading {
  ApplicationInfo *appinfo;
  WindowHolder *window;
  unique_ptr<AssetLoader> asset_loader;
  unordered_map<string, StringPiece> asset_cache;
  AssetLoading(ApplicationInfo *A, WindowHolder *W) : appinfo(A), window(W) {}
  void LoadTexture(         const string &asset_fn, Texture *out, VideoAssetLoader *l=0, int flag=VideoAssetLoader::Flag::Default) { LoadTexture(0, asset_fn, out, l, flag); }
  void LoadTexture(void *h, const string &asset_fn, Texture *out, VideoAssetLoader *l=0, int flag=VideoAssetLoader::Flag::Default);
  void LoadTexture(const void *from_buf, const char *fn, int size, Texture *out, int flag=VideoAssetLoader::Flag::Default);
  void LoadTextureArray(const string &fmt, const string &prefix, const string &suffix, int N, TextureArray*out, int flag=VideoAssetLoader::Flag::Default);
  Texture *LoadTexture(const MultiProcessFileResource &file, int max_image_size = 1000000);
  string FileName(const string &asset_fn);
  string FileContents(const string &asset_fn);
  File *OpenFile(const string &asset_fn);
};

struct Asset {
  typedef function<void(GraphicsDevice*, Asset*, Entity*)> DrawCB;

  AssetLoading *parent;
  AssetMap *storage=0;
  string name, texture, geom_fn;
  DrawCB cb;
  float scale=0;
  int translate=0, rotate=0;
  Geometry *geometry=0, *hull=0;
  Texture tex;
  unsigned texgen=0, typeID=0, particleTexID=0, blends=GraphicsDevice::SrcAlpha, blendt=GraphicsDevice::OneMinusSrcAlpha;
  Color col;
  bool color=0, zsort=0;

  Asset(AssetLoading *P);
  Asset(AssetLoading *P, const string &N, const string &Tex, float S, int T, int R, const char *G, Geometry *H, unsigned CM, const DrawCB &CB=DrawCB());
  Asset(AssetLoading *P, const string &N, const string &Tex, float S, int T, int R, Geometry *G, Geometry *H, unsigned CM, unsigned TG, const DrawCB &CB=DrawCB());

  void Load(void *handle=0, VideoAssetLoader *l=0);
  void Unload();
  void ResetGL(int flag);

  static void Load(vector<Asset> *assets) { for (int i=0; i<assets->size(); ++i) (*assets)[i].Load(); }
  static void Copy(const Asset *in, Asset *out) {
    string name = out->name;
    unsigned typeID = out->typeID;
    *out = *in;
    out->name = name;
    out->typeID = typeID;
  }
};

struct SoundAsset {
  static const int FlagNoRefill, FromBufPad;
  typedef function<int(SoundAsset*, int)> RefillCB;

  AssetLoading *parent;
  SoundAssetMap *storage;
  string name, filename;
  unique_ptr<RingSampler> wav;
  int channels=FLAGS_chans_out, sample_rate=FLAGS_sample_rate, seconds=FLAGS_soundasset_seconds;
  RefillCB refill;
  void *handle=0;
  int handle_arg1=-1;
  float gain=1, max_distance=0, reference_distance=0;
  unique_ptr<AudioResamplerInterface> resampler;

  SoundAsset(AssetLoading *L) : parent(L) {}
  SoundAsset(AssetLoading *L, const string &N, const string &FN, RingSampler *W, int C, int SR, int S) :
    parent(L), name(N), filename(FN), wav(W), channels(C), sample_rate(SR), seconds(S) {}

  void Load(void *handle, const char *FN, int Secs, int flag=0);
  void Load(const void *FromBuf, int size, const char *FileName, int Seconds=FLAGS_soundasset_seconds);
  void Load(int seconds=FLAGS_soundasset_seconds, bool unload=true);
  void Unload();
  int Refill(int reset);

  static void Load(vector<SoundAsset> *assets) { for (auto &a : *assets) a.Load(); }
  static size_t Size(const SoundAsset *sa) { return sa->seconds * sa->sample_rate * sa->channels; }
};

struct MovieAsset {
  AssetLoading *parent;
  MovieAssetMap *storage=0;
  string name, filename;
  SoundAsset audio;
  Asset video;
  void *handle=0;

  MovieAsset(AssetLoading *H) : parent(H), audio(H), video(H) {}
  void Load(const char *fn=0);
  int Play(int seek);
};

struct MapAsset {
  virtual void Draw(GraphicsDevice*, const Entity &camera) = 0;
};

void glLine(GraphicsDevice*, const point &p1, const point &p2, const Color *color);
void glAxis(GraphicsDevice*, Asset*, Entity*);
void glRoom(GraphicsDevice*, Asset*, Entity*);
void glIntersect(GraphicsDevice*, int x, int y, Color *c);
void glShadertoyShader(GraphicsDevice*, Shader *shader, const Texture *tex=0);
void glShadertoyShaderWindows(GraphicsDevice*, Shader *shader, const Color &backup_color, const Box                &win, const Texture *tex=0);
void glShadertoyShaderWindows(GraphicsDevice*, Shader *shader, const Color &backup_color, const vector<const Box*> &win, const Texture *tex=0);
void glSpectogram(GraphicsDevice*, Matrix *m, unsigned char *data, int pf, int width, int height, int hjump, float max, float clip, bool interpolate, int pd=PowerDomain::dB);
void glSpectogram(GraphicsDevice*, Matrix *m, Texture *t, float *max=0, float clip=-INFINITY, int pd=PowerDomain::dB);
void glSpectogram(GraphicsDevice*, const RingSampler::Handle *in, Texture *t, Matrix *transform=0, float *max=0, float clip=-INFINITY);

struct BoxFilled             : public Drawable { void Draw(GraphicsContext*, const LFL::Box &b) const; };
struct BoxOutline            : public Drawable { void Draw(GraphicsContext*, const LFL::Box &b) const; };
struct BoxTopLeftOutline     : public Drawable { void Draw(GraphicsContext*, const LFL::Box &b) const; };
struct BoxBottomRightOutline : public Drawable { void Draw(GraphicsContext*, const LFL::Box &b) const; };

struct Waveform : public Drawable {
  int width=0, height=0;
  unique_ptr<Geometry> geom;
  Waveform() {}
  Waveform(point dim, const Color *c, const Vec<float> *);
  void Draw(GraphicsContext*, const LFL::Box &w) const;
  static Waveform Decimated(point dim, const Color *c, const RingSampler::Handle *, int decimateBy);
};

struct Cube {
  static unique_ptr<Geometry> Create(v3 v);
  static unique_ptr<Geometry> Create(float rx, float ry, float rz, bool normals=false);
  static unique_ptr<Geometry> CreateFrontFace(float r);
  static unique_ptr<Geometry> CreateBackFace(float r);
  static unique_ptr<Geometry> CreateLeftFace(float r);
  static unique_ptr<Geometry> CreateRightFace(float r);
  static unique_ptr<Geometry> CreateTopFace(float r);
  static unique_ptr<Geometry> CreateBottomFace(float r);
};

struct Grid {
  static unique_ptr<Geometry> Grid3D();
  static unique_ptr<Geometry> Grid2D(float x, float y, float range, float step);
};

struct Skybox {
  Asset               a_left, a_right, a_top, a_bottom, a_front, a_back;
  Entity              e_left, e_right, e_top, e_bottom, e_front, e_back;
  Scene::EntityVector v_left, v_right, v_top, v_bottom, v_front, v_back;
  Skybox(AssetLoading*);

  Asset *asset() { return &a_left; }
  void Load(const string &filename_prefix);
  void Draw(GraphicsDevice*);
};

}; // namespace LFL
#endif // LFL_CORE_APP_ASSETS_H__
