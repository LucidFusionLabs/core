/*
 * $Id: assets.h 1336 2014-12-08 09:29:59Z justin $
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

#ifndef LFL_LFAPP_ASSETS_H__
#define LFL_LFAPP_ASSETS_H__
namespace LFL {

DECLARE_int(soundasset_seconds);

struct AudioAssetLoader {
  virtual void *LoadAudioFile(const string &fn) = 0;
  virtual void UnloadAudioFile(void *h) = 0;

  virtual void *LoadAudioBuf(const char *buf, int len, const char *mimetype) = 0;
  virtual void UnloadAudioBuf(void *h) = 0;

  virtual void LoadAudio(void *h, SoundAsset *a, int seconds, int flag) = 0;
  virtual int RefillAudio(SoundAsset *a, int reset) = 0;
};

struct VideoAssetLoader {
  virtual void *LoadVideoFile(const string &fn) = 0;
  virtual void UnloadVideoFile(void *h) = 0;

  virtual void *LoadVideoBuf(const char *buf, int len, const char *mimetype) = 0;
  virtual void UnloadVideoBuf(void *h) = 0;

  struct Flag { enum { LoadGL=1, Clear=2, Default=LoadGL|Clear }; };
  virtual void LoadVideo(void *h, Texture *out, int flag=Flag::Default) = 0;
};

struct MovieAssetLoader {
  virtual void *LoadMovieFile(const string &fn) = 0;
  virtual void UnloadMovieFile(void *h) = 0;

  virtual void *LoadMovieBuf(const char *buf, int len, const char *mimetype) = 0;
  virtual void UnloadMovieBuf(void *h) = 0;

  virtual void LoadMovie(void *h, MovieAsset *a) = 0;
  virtual int PlayMovie(MovieAsset *a, int seek) = 0;
};

template <class X> struct AssetMapT {
  bool loaded;
  vector<X> vec;
  map<string, X*> amap;
  AssetMapT() : loaded(0) {}
  void Add(const X &a) { CHECK(!loaded); vec.push_back(a); }
  void Unloaded(X *a) { if (!a->name.empty()) amap.erase(a->name); }
  void Load(X *a) { a->parent = this; if (!a->name.empty()) amap[a->name] = a; a->Load(); }
  void Load() { CHECK(!loaded); for (int i=0; i<vec.size(); i++) Load(&vec[i]); loaded=1; }
  X *operator()(const string &an) { return FindOrNull(amap, an); }
};
typedef AssetMapT<     Asset>      AssetMap;
typedef AssetMapT<SoundAsset> SoundAssetMap;
typedef AssetMapT<MovieAsset> MovieAssetMap;

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
    Geometry(VD, primtype, num, v, norm, tex, (const Color*)0) { color=1; col=vcol; }

  Geometry (int pt, int n, v2 *v, v3 *norm, v2 *tex            ) : Geometry(2, pt, n, v, norm, tex, 0  ) {}
  Geometry (int pt, int n, v3 *v, v3 *norm, v2 *tex            ) : Geometry(3, pt, n, v, norm, tex, 0  ) {}
  Geometry (int pt, int n, v2 *v, v3 *norm, v2 *tex, Color  col) : Geometry(2, pt, n, v, norm, tex, col) {}
  Geometry (int pt, int n, v3 *v, v3 *norm, v2 *tex, Color  col) : Geometry(3, pt, n, v, norm, tex, col) {}
  Geometry (int pt, int n, v2 *v, v3 *norm, v2 *tex, Color *col) : Geometry(2, pt, n, v, norm, tex, col) {}
  Geometry (int pt, int n, v3 *v, v3 *norm, v2 *tex, Color *col) : Geometry(3, pt, n, v, norm, tex, col) {}

  void SetPosition(const float *v);
  void SetPosition(const point &p) { v2 v=p; SetPosition(&v[0]); }
  void ScrollTexCoord(float dx, float dx_extra, int *subtract_max_int);

  static Geometry *LoadOBJ(const string &filename, const float *map_tex_coord=0);
  static string ExportOBJ(const Geometry *geometry, const set<int> *prim_filter=0, bool prim_filter_invert=0);
};

struct Asset {
  typedef function<void(Asset*, Entity*)> DrawCB;

  AssetMap *parent;
  string name, texture, geom_fn;
  DrawCB cb;
  float scale;
  int translate, rotate;
  Geometry *geometry, *hull;
  Texture tex;
  unsigned texgen, typeID, particleTexID, blends, blendt;
  Color col;
  bool color, zsort;

  Asset() : parent(0), scale(0), translate(0), rotate(0), geometry(0), hull(0), texgen(0), typeID(0), particleTexID(0),
  blends(GraphicsDevice::SrcAlpha), blendt(GraphicsDevice::OneMinusSrcAlpha), color(0), zsort(0) {}

  Asset(const string &N, const string &Tex, float S, int T, int R, const char *G, Geometry *H, unsigned CM, DrawCB CB=DrawCB())
    : parent(0), name(N), texture(Tex), geom_fn(BlankNull(G)), cb(CB), scale(S), translate(T), rotate(R), geometry(0), hull(H), texgen(0),
    typeID(0), particleTexID(0), blends(GraphicsDevice::SrcAlpha), blendt(GraphicsDevice::OneMinusSrcAlpha), color(0), zsort(0) { tex.cubemap=CM; }

  Asset(const string &N, const string &Tex, float S, int T, int R, Geometry *G, Geometry *H, unsigned CM, unsigned TG, DrawCB CB=DrawCB())
    : parent(0), name(N), texture(Tex), cb(CB), scale(S), translate(T), rotate(R), geometry(G), hull(H), texgen(TG),
    typeID(0), particleTexID(0), blends(GraphicsDevice::SrcAlpha), blendt(GraphicsDevice::OneMinusSrcAlpha), color(0), zsort(0) { tex.cubemap=CM; }

  void Load(void *handle=0, VideoAssetLoader *l=0);
  void Unload();

  static void Load(vector<Asset> *assets) { for (int i=0; i<assets->size(); ++i) (*assets)[i].Load(); }
  static void LoadTexture(         const string &asset_fn, Texture *out, VideoAssetLoader *l=0) { LoadTexture(0, asset_fn, out, l); }
  static void LoadTexture(void *h, const string &asset_fn, Texture *out, VideoAssetLoader *l=0);
  static void LoadTexture(const void *from_buf, const char *fn, int size, Texture *out, int flag=VideoAssetLoader::Flag::Default);
  static Texture *LoadTexture(const MultiProcessFileResource &file, int max_image_size = 1000000);
  static string FileContents(const string &asset_fn);
  static File *OpenFile(const string &asset_fn);
  static unordered_map<string, StringPiece> cache;

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
# define SoundAssetSize(sa) ((sa)->seconds * FLAGS_sample_rate * FLAGS_chans_out)

  SoundAssetMap *parent;
  string name, filename;
  RingBuf *wav;
  int channels, sample_rate, seconds;
  RefillCB refill;
  void *handle;
  int handle_arg1;
  AudioResampler resampler;

  SoundAsset() : parent(0), wav(0), channels(0), sample_rate(0), seconds(0), handle(0), handle_arg1(-1) {}
  SoundAsset(const string &N, const string &FN, RingBuf *W, int C, int SR, int S) : name(N), filename(FN), wav(W), channels(C), sample_rate(SR), seconds(S), handle(0), handle_arg1(-1) {}

  static void Load(vector<SoundAsset> *assets) { for (int i=0; i<assets->size(); ++i) (*assets)[i].Load(); }
  void Load(void *handle, const char *FN, int Secs, int flag=0);
  void Load(const void *FromBuf, int size, const char *FileName, int Seconds=10);
  void Load(int seconds=10, bool unload=true);
  void Unload();
  int Refill(int reset);
};

struct MovieAsset {
  MovieAssetMap *parent;
  string name, filename;
  SoundAsset audio;
  Asset video;
  void *handle;

  MovieAsset() : parent(0), handle(0) {}
  void Load(const char *fn=0);
  int Play(int seek);
};

struct MapAsset {
  virtual void Draw(const Entity &camera) = 0;
};

struct JpegReader {
  static int Read(File *f,            Texture *out);
  static int Read(const string &data, Texture *out);
};

struct GIFReader {
  static int Read(File *f,            Texture *out);
  static int Read(const string &data, Texture *out);
};

struct PngReader {
  static int Read(File *f, Texture *out);
};

struct PngWriter {
  static int Write(File *f,          const Texture &tex);
  static int Write(const string &fn, const Texture &tex);
};

struct WavReader {
  File *f; int last;
  ~WavReader() { Close(); }
  WavReader(File *F=0) { Open(F); }
  void Open(File *F);
  void Close() { if (f) f->Close(); }
  int Read(RingBuf::Handle *, int offset, int size);
};

struct WavWriter {
  File *f; int wrote;
  ~WavWriter() { Flush(); }
  WavWriter(File *F=0) { Open(F); }
  void Open(File *F);
  int Write(const RingBuf::Handle *, bool flush=true);
  int Flush();
};

struct Assets : public Module {
  AudioAssetLoader *default_audio_loader;
  VideoAssetLoader *default_video_loader;
  MovieAssetLoader *default_movie_loader;
  MovieAsset       *movie_playing;
  Assets() : default_audio_loader(0), default_video_loader(0), default_movie_loader(0), movie_playing(0) {}
  int Init();
};

void glLine(const point &p1, const point &p2, const Color *color);
void glAxis(Asset*, Entity*);
void glRoom(Asset*, Entity*);
void glIntersect(int x, int y, Color *c);
void glShadertoyShader(Shader *shader, const Texture *tex=0);
void glShadertoyShaderWindows(Shader *shader, const Color &backup_color, const Box          &win, const Texture *tex=0);
void glShadertoyShaderWindows(Shader *shader, const Color &backup_color, const vector<Box*> &win, const Texture *tex=0);
void glSpectogram(Matrix *m, unsigned char *data, int width, int height, int hjump, float max, float clip, bool interpolate, int pd=PowerDomain::dB);
void glSpectogram(Matrix *m, Asset *a, float *max=0, float clip=-INFINITY, int pd=PowerDomain::dB);
void glSpectogram(SoundAsset *sa, Asset *a, Matrix *transform=0, float *max=0, float clip=-INFINITY);

struct BoxFilled : public Drawable { void Draw(const LFL::Box &b, const Drawable::Attr *a=0) const { b.Draw(); } };
struct BoxOutline : public Drawable {
  int line_width;
  BoxOutline(int LW=1) : line_width(LW) {}
  void Draw(const LFL::Box &b, const Drawable::Attr *a=0) const;
};

struct Waveform : public Drawable {
  Geometry *geom; int width, height;
  virtual ~Waveform() { delete geom; }
  Waveform() : width(0), height(0), geom(0) {}
  Waveform(point dim, const Color *c, const Vec<float> *);
  static Waveform Decimated(point dim, const Color *c, const RingBuf::Handle *, int decimateBy);
  void Draw(const LFL::Box &w, const Drawable::Attr *a=0) const {
    if (!geom) return;
    geom->SetPosition(w.Position());
    screen->gd->DisableTexture();
    Scene::Select(geom);
    Scene::Draw(geom, 0);
  }
};

struct Cube {
  static Geometry *Create(v3 v);
  static Geometry *Create(float rx, float ry, float rz, bool normals=false);
  static Geometry *CreateFrontFace(float r);
  static Geometry *CreateBackFace(float r);
  static Geometry *CreateLeftFace(float r);
  static Geometry *CreateRightFace(float r);
  static Geometry *CreateTopFace(float r);
  static Geometry *CreateBottomFace(float r);
};

struct Grid {
  static Geometry *Grid3D();
  static Geometry *Grid2D(float x, float y, float range, float step);
};

struct TextureArray {
  vector<Texture> a; int ind=0;
  void ClearGL() { for (auto &i : a) i.ClearGL(); }
  void Load(const string &fmt, const string &prefix, const string &suffix, int N);
  void DrawSequence(Asset *out, Entity *e);
};

struct Skybox {
  Asset               a_left, a_right, a_top, a_bottom, a_front, a_back;
  Entity              e_left, e_right, e_top, e_bottom, e_front, e_back;
  Scene::EntityVector v_left, v_right, v_top, v_bottom, v_front, v_back;
  Skybox() : a_left  ("", "", 1, 0, 0, Cube::Create(500, 500, 500), 0, CubeMap::PX, TexGen::LINEAR),
  a_right ("", "", 1, 0, 0, 0,                           0, CubeMap::NX, 0),
  a_top   ("", "", 1, 0, 0, 0,                           0, CubeMap::PY, 0),
  a_bottom("", "", 1, 0, 0, 0,                           0, CubeMap::NY, 0),
  a_front ("", "", 1, 0, 0, 0,                           0, CubeMap::PZ, 0),
  a_back  ("", "", 1, 0, 0, 0,                           0, CubeMap::NZ, 0),
  e_left ("sb_left",  &a_left),  e_right ("sb_right",  &a_right),
  e_top  ("sb_top",   &a_top),   e_bottom("sb_bottom", &a_bottom),
  e_front("sb_front", &a_front), e_back  ("sb_back",   &a_back)
  { 
    v_left .push_back(&e_left); v_right .push_back(&e_right);
    v_top  .push_back(&e_left); v_bottom.push_back(&e_right);
    v_front.push_back(&e_left); v_back  .push_back(&e_right);
  }
  void Load(const string &filename_prefix) {
    a_left  .texture = StrCat(filename_prefix,   "_left.png"); a_left  .Load();
    a_right .texture = StrCat(filename_prefix,  "_right.png"); a_right .Load();
    a_top   .texture = StrCat(filename_prefix,    "_top.png"); a_top   .Load();
    a_bottom.texture = StrCat(filename_prefix, "_bottom.png"); a_bottom.Load();
    a_front .texture = StrCat(filename_prefix,  "_front.png"); a_front .Load();
    a_back  .texture = StrCat(filename_prefix,   "_back.png"); a_back  .Load();
  }
  void Draw() {
    screen->gd->DisableNormals();
    screen->gd->DisableVertexColor();
    screen->gd->DisableDepthTest();
    Scene::Draw(&a_left,  0, v_left ); Scene::Draw(&a_right,  0, v_right);
    Scene::Draw(&a_top,   0, v_top  ); Scene::Draw(&a_bottom, 0, v_bottom);
    Scene::Draw(&a_front, 0, v_front); Scene::Draw(&a_back,   0, v_back);
    screen->gd->EnableDepthTest();
  }
  Asset *asset() { return &a_left; }
};

struct ParticleSystem {
  string name;
  Color color;
  v3 pos, vel, ort, updir;
  vector<v3> *pos_transform;
  int pos_transform_index;
  ParticleSystem(const string &n) : name(n), ort(0,0,1), updir(0,1,0), pos_transform(0), pos_transform_index(0) {}
  virtual void Update(unsigned dt, int mx, int my, int mdown) = 0;
  virtual void Draw() = 0;
};

template <int MP, int MH, bool PerParticleColor> struct Particles : public ParticleSystem {
  typedef Particles<MP, MH, PerParticleColor> ParticlesType;
  static const int MaxParticles=MP, MaxHistory=MH, VertFloats=(PerParticleColor ? 9 : 5), VertSize=VertFloats*sizeof(float);
  static const int ParticleVerts=6, ParticleSize=ParticleVerts*VertSize, NumFloats=MaxParticles*ParticleVerts*VertFloats;
  static const int Trails=MaxHistory>2, TrailVertFloats=(PerParticleColor ? 7 : 3), TrailVertSize=TrailVertFloats*sizeof(float);
  static const int MaxTrailVerts=6*(MaxHistory-2), NumTrailFloats=(Trails ? MaxParticles*MaxTrailVerts*TrailVertFloats : 1);
  int num_particles, nops, texture, verts_id, trailverts_id, num_trailverts, emitter_type, blend_mode_s, blend_mode_t, burst;
  float floorval, gravity, radius_min, radius_max, age_min, age_max, rand_initpos, rand_initvel, emitter_angle, color_fade;
  long long ticks_seen, ticks_processed, ticks_step;
  float verts[NumFloats], trailverts[NumTrailFloats];
  bool trails, floor, always_on, per_particle_color, radius_decay, billboard, move_with_pos, blend, rand_color, draw_each;
  Color rand_color_min, rand_color_max;

  struct Emitter { enum { None=0, Mouse=1, Sprinkler=2, RainbowFade=4, GlowFade=8, FadeFromWhite=16 }; };

  struct Particle {
    ParticlesType *config;
    v3 history[MH], vel;
    int history_len, bounceage;
    float radius, age, maxage, remaining;
    Color color, start_color;
    bool dead;

    void InitColor() {
      if (config->rand_color) {
        color = Color(Rand(config->rand_color_min.r(), config->rand_color_max.r()),
                      Rand(config->rand_color_min.g(), config->rand_color_max.g()),
                      Rand(config->rand_color_min.b(), config->rand_color_max.b()),
                      Rand(config->rand_color_min.a(), config->rand_color_max.a()));
      }
      else if (config->emitter_type & Emitter::RainbowFade) {
        color = Color::fade(config->color_fade);
      } else {
        color = config->color;
      }
      start_color = color;
    }
    void Init() {
      InitColor();
      radius = Rand(config->radius_min, config->radius_max);
      history_len = Trails ? (int)Rand(max(3.0f, config->radius_min), static_cast<float>(MaxHistory)) : 1;

      v3 start;
      if (!config->move_with_pos) start = config->pos;
      if (config->pos_transform) {
        const v3 &tf = (*config->pos_transform)[config->pos_transform_index++];
        v3 right = v3::Cross(config->ort, config->updir);
        start.Add(right * tf.x + config->updir * tf.y + config->ort * tf.z);
        if (config->pos_transform_index >= config->pos_transform->size()) config->pos_transform_index = 0;
      }
      start.Add(v3::Rand() * Rand(0.0f, config->rand_initpos));

      for (int i=0; i<history_len; i++) history[i] = start;

      if (config->emitter_type & Emitter::Sprinkler) {
        if (1) vel  = v3(2.0*cos(config->emitter_angle), 2.0,                   2.0*sin(config->emitter_angle));
        if (0) vel += v3(0.5*Rand(1.0,2.0)-.25,          0.5*Rand(1.0,2.0)-.25, 0.5*Rand(1.0,2.0)-.25);
      } else { 
        vel = config->vel*25.0 + v3::Rand()*Rand(0.0f, config->rand_initvel);
      }

      remaining = 1;
      bounceage = 2;
      maxage = Rand(config->age_min, config->age_max);
      dead = false;
      age = 0; 
    }
    void Update(float secs) {
      float bounced = false;
      if (config->gravity) vel += v3(0, config->gravity * secs, 0);
      if (config->floor && history[0].y + vel.y < config->floorval) {
        bounced = true;
        vel.Scale(0.75);
        vel.y *= -0.5f;
      }

      if (config->trails) for (int i=history_len-1; i>0; i--) history[i] = history[i-1];
      history[0] += vel * secs;

      age += secs * (!config->floor ? 1 : (bounced ? bounceage++ : 0.25));
      if (age < maxage) remaining = 1 - age / maxage;
      else dead = true;
    }
  };
  Particle particles[MP], *free_list[MP];

  float       *particle_verts(int n)       { return &verts[n * ParticleVerts * VertFloats]; }
  const float *particle_verts(int n) const { return &verts[n * ParticleVerts * VertFloats]; }

  static void AssignTex(float *out, float tx, float ty) { out[3]=tx; out[4]=ty; }
  static void AssignPosColor(float *out, const v3 &v, const Color *c, int tex_size) {
    if (1) { out[0]=v.x; out[1]=v.y; out[2]=v.z; }
    if (c) { int oi=3+tex_size; out[oi++]=c->r(); out[oi++]=c->g(); out[oi++]=c->b(); out[oi++]=c->a(); }
  }

  Particles(const string &n, bool AlwaysOn=false, float RadiusMin=10, float RadiusMax=40, float RandInitPos=5, float RandInitVel=500) : ParticleSystem(n), num_particles(AlwaysOn ? MaxParticles : 0),
    nops(0), texture(0), verts_id(-1), trailverts_id(-1), emitter_type(0), blend_mode_s(GraphicsDevice::SrcAlpha), blend_mode_t(GraphicsDevice::One), burst(0), floorval(0), gravity(0),
    age_min(.05), age_max(1), radius_min(RadiusMin), radius_max(RadiusMax), rand_initpos(RandInitPos), rand_initvel(RandInitVel), emitter_angle(0), color_fade(0),
    ticks_seen(0), ticks_processed(0), ticks_step(0), trails(Trails), floor(0), always_on(AlwaysOn), per_particle_color(PerParticleColor), radius_decay(true), billboard(0), move_with_pos(0), blend(true), rand_color(0), draw_each(0)
    {
      for (int i=0; i<MP; i++) {
        Particle *particle = &particles[i];
        particle->dead = true;
        particle->config = this;
        free_list[i] = particle;
        if (always_on) particle->Init();

        float *v = particle_verts(i);
        AssignTex(v, 0, 0); v += VertFloats;
        AssignTex(v, 0, 1); v += VertFloats;
        AssignTex(v, 1, 0); v += VertFloats;

        AssignTex(v, 0, 1); v += VertFloats;
        AssignTex(v, 1, 0); v += VertFloats;
        AssignTex(v, 1, 1); v += VertFloats;
      }
    }
  Particle *AddParticle() {
    if (num_particles == MP) { nops++; return 0; }
    CHECK(num_particles < MP);
    Particle *particle = free_list[num_particles++];
    particle->Init();
    return particle;
  }
  void DelParticle(Particle *particle) {
    CHECK(num_particles > 0);
    free_list[--num_particles] = particle;
  }
  void Update(unsigned dt, int mx, int my, int mdown) {
    if (!dt) return;
    ticks_seen += dt;
    float secs = dt / 1000.0;

    if (emitter_type & Emitter::Mouse) {
      if (mdown) for(int i=0; i<100; i++) AddParticle();
      v3 mouse_delta = v3(mx - pos.x, my - pos.y, 0);
      vel += (mouse_delta - vel) * 0.25;
    }
    if (emitter_type & Emitter::Sprinkler) {
      emitter_angle += 0.5 * secs;
      while (emitter_angle > M_TAU) emitter_angle -= M_TAU;
    }
    if (emitter_type & Emitter::RainbowFade) {
      color_fade += secs / 10;
      while (color_fade >= 1) color_fade -= 1;
    }
    if (burst) {
      for (int i=0; i<burst; i++) AddParticle();
    }

    pos += vel;
    if (floor && pos.y < floorval) { pos.y = floorval; vel.y = 0; }

    unsigned steps = 0, step = ticks_step ? ticks_step : (ticks_seen - ticks_processed);
    for (/**/; ticks_seen >= ticks_processed + step; ticks_processed += step) steps++;
    if (!steps) return;

    num_trailverts = 0;
    int out_particles = 0;
    float stepsecs = step / 1000.0;
    for (int i=0; i<MP; i++) {
      if (particles[i].dead) continue;
      UpdateParticle(&particles[i], stepsecs, steps, particle_verts(out_particles++), &trailverts[num_trailverts * TrailVertFloats]);
    }
  }
  void UpdateParticle(Particle *particle, float stepsecs, int steps, float *v, float *tv) {
    for (int i=0; i<steps; i++) {
      particle->Update(stepsecs);
      if (particle->dead) {
        if (always_on) particle->Init();
        else { DelParticle(particle); return; }
      }
    }
    if (!draw_each) UpdateVertices(particle, v, tv);
  }
  void UpdateVertices(Particle *particle, float *v, float *tv) {
    float *vin = v, remaining = particle->remaining, size = particle->radius * (radius_decay ? remaining : 1);
    if (emitter_type & Emitter::GlowFade) particle->color = Color(remaining, remaining * 0.75, 1-remaining, 1.0);
    if (emitter_type & Emitter::FadeFromWhite) particle->color = Color::Interpolate(Color::white, particle->start_color, remaining);

    v3 p = particle->history[0];
    if (move_with_pos) p.Add(pos);

    v3 o1=p, o2=p, o3=p, o4=p, right, up;

    if (billboard) { right = v3::Cross(screen->cam->ort, screen->cam->up) * size; up = screen->cam->up * size; }
    else           { right = v3(size, 0, 0);                                      up = v3(0, size, 0); }

    o1.Add(-right + -up);
    o2.Add(-right +  up);
    o3.Add( right + -up);
    o4.Add( right +  up);

    AssignPosColor(v, o1, PerParticleColor ? &particle->color : 0, 2); v += VertFloats;
    AssignPosColor(v, o2, PerParticleColor ? &particle->color : 0, 2); v += VertFloats;
    AssignPosColor(v, o3, PerParticleColor ? &particle->color : 0, 2); v += VertFloats;

    if (!draw_each) {
      AssignPosColor(v, o2, PerParticleColor ? &particle->color : 0, 2); v += VertFloats;
      AssignPosColor(v, o3, PerParticleColor ? &particle->color : 0, 2); v += VertFloats;
      AssignPosColor(v, o4, PerParticleColor ? &particle->color : 0, 2); v += VertFloats;
    } else {
      AssignPosColor(v, o4, PerParticleColor ? &particle->color : 0, 2); v += VertFloats;
      DrawParticles(GraphicsDevice::TriangleStrip, 4, vin, 4*VertSize);
    }

    if (trails) {
      v3 last_v1, last_v2, *history = particle->history;
      int history_len = particle->history_len;
      for (int i = 0; i < history_len - 1; i++) {
        float step = 1.0f - i / (float)(history_len-1);
        v3 dp = history[i] - history[i+1];
        v3 perp1 = v3::Cross(dp, updir);
        v3 perp2 = v3::Cross(dp, perp1);
        perp1 = v3::Cross(dp, perp2);
        perp1.Norm();

        Color trail_color(step, step * 0.25f, 1.0 - step, step * 0.5);
        v3 off = perp1 * (particle->radius * particle->remaining * step * 0.1);
        v3 v1 = history[i] - off, v2 = history[i] + off;
        if (i > 0) {
          AssignPosColor(tv, last_v1, PerParticleColor ? &trail_color : 0, 0); tv += TrailVertFloats;
          AssignPosColor(tv, last_v2, PerParticleColor ? &trail_color : 0, 0); tv += TrailVertFloats;
          AssignPosColor(tv,      v1, PerParticleColor ? &trail_color : 0, 0); tv += TrailVertFloats;
          num_trailverts += 3;
#if 0
          AssignPosColor(tv, last_v2, PerParticleColor ? &trail_color : 0, 0); tv += TrailVertFloats;
          AssignPosColor(tv,      v1, PerParticleColor ? &trail_color : 0, 0); tv += TrailVertFloats;
          AssignPosColor(tv,      v2.x,      v2.y,      v2.z, PerParticleColor ? &trail_color : 0, 0); tv += TrailVertFloats;
          num_trailverts += 3;
#endif
        }
        last_v1 = v1;
        last_v2 = v2;
      }
    }
  }
  void Draw() {
    screen->gd->DisableDepthTest();
    screen->gd->DisableLighting();
    screen->gd->DisableNormals();

    if (blend) {
      screen->gd->EnableBlend();
      screen->gd->BlendMode(blend_mode_s, blend_mode_t);
    }
    if (PerParticleColor) {
      screen->gd->EnableVertexColor();
    }
    if (texture) {
      screen->gd->EnableTexture();
      screen->gd->BindTexture(GraphicsDevice::Texture2D, texture);
    }

    if (draw_each) {
      for (int i=0; i<MP; i++) {
        if (particles[i].dead) continue;
        UpdateVertices(&particles[i], particle_verts(i), &trailverts[i * TrailVertFloats]);
      }
    } else {
      int update_size = verts_id < 0 ? sizeof(verts) : num_particles * ParticleSize;
      DrawParticles(GraphicsDevice::Triangles, num_particles*ParticleVerts, verts, update_size);

      if (trails) {
        int trail_update_size = trailverts_id < 0 ? sizeof(trailverts) : num_trailverts * TrailVertSize;
        DrawTrails(trailverts, trail_update_size);
      }
    }

    if (PerParticleColor) screen->gd->DisableVertexColor();
  } 
  void DrawParticles(int prim_type, int num_verts, float *v, int l) {
    if (1)                screen->gd->VertexPointer(3, GraphicsDevice::Float, VertSize, 0,               v, l, &verts_id, true, prim_type);
    if (1)                screen->gd->TexPointer   (2, GraphicsDevice::Float, VertSize, 3*sizeof(float), v, l, &verts_id, false);
    if (PerParticleColor) screen->gd->ColorPointer (4, GraphicsDevice::Float, VertSize, 5*sizeof(float), v, l, &verts_id, true);

    screen->gd->DrawArrays(prim_type, 0, num_verts);
  }
  void DrawTrails(float *v, int l) {
    screen->gd->DisableTexture();

    if (1)                screen->gd->VertexPointer(3, GraphicsDevice::Float, TrailVertSize, 0,               v, l, &trailverts_id, true, GraphicsDevice::Triangles);
    if (PerParticleColor) screen->gd->ColorPointer (4, GraphicsDevice::Float, TrailVertSize, 3*sizeof(float), v, l, &trailverts_id, true);

    screen->gd->DrawArrays(GraphicsDevice::Triangles, 0, num_trailverts);
  }
  void AssetDrawCB(Asset *out, Entity *e) { pos = e->pos; Draw(); }
};

template <class Line> struct RingFrameBuffer {
  typedef point(*PaintCB)(Line*, point, const Box&);
  FrameBuffer fb;
  v2 scroll;
  point p;
  bool wrap=0;
  int w=0, h=0, font_size=0, font_height=0;

  void Reset() { w=h=0; fb.Reset(); }
  virtual int Width()  const { return w; }
  virtual int Height() const { return h; }
  virtual void SizeChangedDone() { fb.Release(); scroll=v2(); p=point(); }
  virtual bool SizeChanged(int W, int H, Font *font) {
    if (W == w && H == h && font->size == font_size) return false;
    SetDimensions(W, H, font);
    fb.Resize(w, Height(), FrameBuffer::Flag::CreateGL | FrameBuffer::Flag::CreateTexture);
    screen->gd->Clear();
    screen->gd->DrawMode(DrawMode::_2D, false);
    return true;
  }
  virtual void Draw(point pos, point adjust, bool scissor=true) {
    Box box(pos.x, pos.y, w, Height());
    if (scissor) screen->gd->PushScissor(box);
    fb.tex.Bind();
    (box + adjust).DrawCrimped(fb.tex.coord, 0, 0, scroll.y);
    if (scissor) screen->gd->PopScissor();
  }
  void Clear(Line *l, const Box &b, bool vwrap=true) {
    if (1)                         { Scissor s(0, l->p.y - b.h,      b.w, b.h); screen->gd->Clear(); }
    if (l->p.y - b.h < 0 && vwrap) { Scissor s(0, l->p.y + Height(), b.w, b.h); screen->gd->Clear(); }
  }
  void Update(Line *l, const Box &b, const PaintCB &paint, bool vwrap=true) {
    point lp = paint(l, l->p, b);
    if (lp.y < 0 && vwrap) paint(l, point(0, lp.y + Height() + b.h), b);
  }
  int PushFrontAndUpdate(Line *l, const Box &b, const PaintCB &paint, bool vwrap=true) {
    int ht = Height();
    if (b.h >= ht)     p = paint(l,      point(0, ht),         b);
    else                   paint(l, (p = point(0, p.y + b.h)), b);
    if (p.y > ht && vwrap) paint(l, (p = point(0, p.y - ht)),  b);
    ScrollPercent((float)-b.h / ht);
    return b.h;
  }
  int PushBackAndUpdate(Line *l, const Box &b, const PaintCB &paint, bool vwrap=true) {
    int ht = Height();
    if (p.y == 0)         p =          point(0, ht);
    if (b.h >= ht)        p = paint(l, point(0, b.h),            b);
    else                  p = paint(l, point(0, p.y),            b);
    if (p.y < 0 && vwrap) p = paint(l, point(0, p.y + b.h + ht), b);
    ScrollPercent((float)b.h / ht);
    return b.h;
  }
  void SetDimensions(int W, int H, Font *f) { w = W; h = H; font_size = f->size; font_height = f->Height(); }
  void ScrollPercent(float y) { scroll.y = fmod(scroll.y + y, 1.0); }
  void ScrollPixels(int y) { ScrollPercent((float)y / Height()); }
  void AdvancePixels(int y) { ScrollPixels(y); p.y = RingIndex::WrapOver(p.y - y, Height()); }
  point BackPlus(const point &o) { return point(RingIndex::WrapOver(p.x + o.x, Width()),
                                                RingIndex::WrapOver(p.y + o.y, Height())); }
};

struct TilesInterface {
  virtual void AddDrawableBoxArray(const DrawableBoxArray &box, point p);
  virtual void SetAttr            (const Drawable::Attr *a)                                = 0;
  virtual void InitDrawBox        (const point&)                                           = 0;
  virtual void InitDrawBackground (const point&)                                           = 0;
  virtual void DrawBox            (const Drawable*, const Box&, const Drawable::Attr *a=0) = 0;
  virtual void DrawBackground     (const Box&)                                             = 0;
  virtual void AddScissor         (const Box&)                                             = 0;
  virtual void Draw(const Box &viewport, const point &doc_position)                        = 0;
  virtual void ContextOpen()                                                               = 0;
  virtual void ContextClose()                                                              = 0;
  virtual void Run()                                                                       = 0;
};

struct LayersInterface {
  UNALIGNED_struct Node  { static const int Size=40; Box box; point scrolled; int layer_id, child_offset; }; UNALIGNED_END(Node,  Node::Size);
  UNALIGNED_struct Child { static const int Size=8;  int node_id, next_child_offset; };                      UNALIGNED_END(Child, Child::Size);
  vector<Node> node;
  vector<Child> child;
  vector<TilesInterface*> layer;

  LayersInterface() { ClearLayerNodes(); }
  void ClearLayerNodes() { child.clear(); node.clear(); node.push_back({ Box(), point(), 0, 0 }); }
  int AddLayerNode(int parent_node_id, const Box &b, int layer_id) {
    node.push_back({ b, point(), layer_id, 0 });
    return AddChild(parent_node_id, node.size());
  }
  int AddChild(int node_id, int child_node_id) {
    child.push_back({ child_node_id, 0 }); 
    Node *n = &node[node_id-1];
    if (!n->child_offset) return (n->child_offset = child.size());
    Child *c = &child[n->child_offset-1];
    while (c->next_child_offset) c = &child[c->next_child_offset-1];
    return (c->next_child_offset = child.size());
  }
  virtual void Update();
  virtual void Draw(const Box &b, const point &p);
  virtual void Init(int N=1) = 0;
};

#define TilesPreAdd(tiles, ...) CallbackListAdd(&(tiles)->prepend[(tiles)->context_depth]->cb, __VA_ARGS__)
#define TilesPostAdd(tiles, ...) CallbackListAdd(&(tiles)->append[(tiles)->context_depth]->cb, __VA_ARGS__)
#define TilesAdd(tiles, w, ...) (tiles)->AddCallback((w), bind(__VA_ARGS__));
#define TilesMatrixIter(m) MatrixIter(m) if (Tile *tile = (Tile*)(m)->row(i)[j])
template<class CB, class CBL, class CBLI> struct TilesT : public TilesInterface {
  struct Tile {
    CBL cb;
    unsigned id, prepend_depth; bool dirty;
    Tile() : id(0), prepend_depth(0), dirty(0) {}
  };
  int layer, W, H, context_depth=-1;
  bool clear=1, clear_empty=1;
  vector<Tile*> prepend, append;
  matrix<Tile*> mat;
  FrameBuffer fb;
  Box current_tile;
  TilesT(int l, int w=256, int h=256) : layer(l), W(w), H(h), mat(1,1) { CHECK(IsPowerOfTwo(W)); CHECK(IsPowerOfTwo(H)); }

  void PushScissor(const Box &w) const { screen->gd->PushScissorOffset(current_tile, w); }
  void GetSpaceCoords(int i, int j, int *xo, int *yo) const { *xo =  j * W; *yo = (-i-1) * H; }
  void GetTileCoords (int x, int y, int *xo, int *yo) const { *xo =  x / W; *yo = -y     / H; }
  void GetTileCoords(const Box &box, int *x1, int *y1, int *x2, int *y2) const {
    GetTileCoords(box.x,         box.y + box.h, x1, y1);
    GetTileCoords(box.x + box.w, box.y,         x2, y2);
  }

  Tile *GetTile(int x, int y) {
    CHECK_GE(x, 0);
    CHECK_GE(y, 0);
    int add;
    if ((add = x - mat.N + 1) > 0) mat.AddCols(add);
    if ((add = y - mat.M + 1) > 0) mat.AddRows(add);
    Tile **ret = (Tile**)&mat.row(y)[x];
    if (!*ret) *ret = new Tile();
    if (!(*ret)->cb.dirty) {
      for (int i = (*ret)->prepend_depth; i <= context_depth; i++) (*ret)->cb.AddList(prepend[i]->cb);
      (*ret)->prepend_depth = context_depth + 1;
    }
    return *ret;
  }

  void ContextOpen() {
    TilesMatrixIter(&mat) { if (tile->cb.dirty) tile->dirty = 1; tile->cb.dirty = 0; }
    if (++context_depth < prepend.size()) { prepend[context_depth]->cb.Clear(); append[context_depth]->cb.Clear(); }
    else { prepend.push_back(new Tile()); append.push_back(new Tile()); CHECK_LT(context_depth, prepend.size()); }
  }
  void ContextClose() {
    CHECK_GE(context_depth, 0);
    TilesMatrixIter(&mat) {
      if (tile->cb.dirty) tile->dirty = 1;
      if (tile->dirty && tile->prepend_depth > context_depth) {
        tile->cb.AddList(append[context_depth]->cb);
        tile->prepend_depth = context_depth;
      }
      if (!context_depth) tile->dirty = 0;
    }
    context_depth--;
  }
  void AddCallback(const Box *box, const CB &cb) {
    bool added = 0;
    int x1, x2, y1, y2;
    GetTileCoords(*box, &x1, &y1, &x2, &y2);
    for (int y = max(y1, 0); y <= y2; y++)
      for (int x = max(x1, 0); x <= x2; x++, added=1) GetTile(x, y)->cb.Add(cb);
    if (!added) FATAL("AddCallback ", box->DebugString(), " = ", x1, " ", y1, " ", x2, " ", y2);
  }

  void Run() {
    Select();
    TilesMatrixIter(&mat) {
      if (!tile->cb.Count() && !clear_empty) continue;
      RunTile(i, j, tile, tile->cb);
      tile->cb.Clear();
    }
    Release();
  }
  void Select() {
    bool init = !fb.ID;
    if (init) fb.Create(W, H);
    current_tile = Box(0, 0, W, H);
    screen->gd->DrawMode(DrawMode::_2D);
    screen->gd->ViewPort(current_tile);
    screen->gd->EnableLayering();
  }
  void RunTile(int i, int j, Tile *tile, const CBLI &tile_cb) {
    GetSpaceCoords(i, j, &current_tile.x, &current_tile.y);
    if (!tile->id) fb.AllocTexture(&tile->id);
    fb.Attach(tile->id);
    screen->gd->MatrixProjection();
    if (clear) screen->gd->Clear();
    screen->gd->LoadIdentity();
    screen->gd->Ortho(current_tile.x, current_tile.x + W, current_tile.y, current_tile.y + H, 0, 100);
    screen->gd->MatrixModelview();
    tile_cb.Run(current_tile);
  }
  void Release() {
    fb.Release();
    screen->gd->RestoreViewport(DrawMode::_2D);
  }

  void Draw(const Box &viewport, const point &docp) {
    int x1, x2, y1, y2, sx, sy;
    point doc_to_view = docp - viewport.Position();
    GetTileCoords(Box(docp.x, docp.y, viewport.w, viewport.h), &x1, &y1, &x2, &y2);

    Scissor scissor(viewport);
    screen->gd->DisableBlend();
    screen->gd->SetColor(Color::white);
    for (int y = max(y1, 0); y <= y2; y++) {
      for (int x = max(x1, 0); x <= x2; x++) {
        Tile *tile = GetTile(x, y);
        if (!tile || !tile->id) continue;
        GetSpaceCoords(y, x, &sx, &sy);
        Texture(0, 0, Pixel::RGBA, tile->id).Draw(Box(sx - doc_to_view.x, sy - doc_to_view.y, W, H));
      }
    }
  }
};

struct Tiles : public TilesT<Callback, CallbackList, CallbackList> {
  const Drawable::Attr *attr=0;
  Tiles(int l, int w=256, int h=256) : TilesT(l, w, h) {}
  void SetAttr           (const Drawable::Attr *a) { attr=a; }
  void InitDrawBox       (const point&);
  void InitDrawBackground(const point&);
  void DrawBox           (const Drawable*, const Box&, const Drawable::Attr *a=0);
  void DrawBackground    (const Box&);
  void AddScissor        (const Box&);
};

template <class X> struct LayersT : public LayersInterface {
  void Init(int N=1) { CHECK_EQ(this->layer.size(), 0); for (int i=0; i<N; i++) this->layer.push_back(new X(i)); }
};

typedef LayersT<Tiles> Layers;

}; // namespace LFL
#endif // LFL_LFAPP_ASSETS_H__
