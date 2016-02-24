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
DECLARE_float(shadertoy_blend);

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
  bool loaded=0;
  vector<X> vec;
  unordered_map<string, X*> amap;
  template<typename ...Args>
  void Add(Args&& ...args) { CHECK(!loaded); vec.emplace_back(forward<Args>(args)...); }
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
    Geometry(VD, primtype, num, v, norm, tex, nullptr) { color=1; col=vcol; }

  Geometry (int pt, int n, v2 *v, v3 *norm, v2 *tex            ) : Geometry(2, pt, n, v, norm, tex, 0  ) {}
  Geometry (int pt, int n, v3 *v, v3 *norm, v2 *tex            ) : Geometry(3, pt, n, v, norm, tex, 0  ) {}
  Geometry (int pt, int n, v2 *v, v3 *norm, v2 *tex, Color  col) : Geometry(2, pt, n, v, norm, tex, col) {}
  Geometry (int pt, int n, v3 *v, v3 *norm, v2 *tex, Color  col) : Geometry(3, pt, n, v, norm, tex, col) {}
  Geometry (int pt, int n, v2 *v, v3 *norm, v2 *tex, Color *col) : Geometry(2, pt, n, v, norm, tex, col) {}
  Geometry (int pt, int n, v3 *v, v3 *norm, v2 *tex, Color *col) : Geometry(3, pt, n, v, norm, tex, col) {}

  void SetPosition(const float *v);
  void SetPosition(const point &p) { v2 v=p; SetPosition(&v[0]); }
  void ScrollTexCoord(float dx, float dx_extra, int *subtract_max_int);

  static unique_ptr<Geometry> LoadOBJ(const string &filename, const float *map_tex_coord=0);
  static string ExportOBJ(const Geometry *geometry, const set<int> *prim_filter=0, bool prim_filter_invert=0);
};

struct Asset {
  typedef function<void(Asset*, Entity*)> DrawCB;

  AssetMap *parent=0;
  string name, texture, geom_fn;
  DrawCB cb;
  float scale=0;
  int translate=0, rotate=0;
  Geometry *geometry=0, *hull=0;
  Texture tex;
  unsigned texgen=0, typeID=0, particleTexID=0, blends=GraphicsDevice::SrcAlpha, blendt=GraphicsDevice::OneMinusSrcAlpha;
  Color col;
  bool color=0, zsort=0;

  Asset() {}
  Asset(const string &N, const string &Tex, float S, int T, int R, const char *G, Geometry *H, unsigned CM, const DrawCB &CB=DrawCB())
    : name(N), texture(Tex), geom_fn(BlankNull(G)), cb(CB), scale(S), translate(T), rotate(R), hull(H) { tex.cubemap=CM; }
  Asset(const string &N, const string &Tex, float S, int T, int R, Geometry *G, Geometry *H, unsigned CM, unsigned TG, const DrawCB &CB=DrawCB())
    : name(N), texture(Tex), cb(CB), scale(S), translate(T), rotate(R), geometry(G), hull(H), texgen(TG) { tex.cubemap=CM; }

  void Load(void *handle=0, VideoAssetLoader *l=0);
  void Unload();

  static void Load(vector<Asset> *assets) { for (int i=0; i<assets->size(); ++i) (*assets)[i].Load(); }
  static void LoadTexture(         const string &asset_fn, Texture *out, VideoAssetLoader *l=0) { LoadTexture(0, asset_fn, out, l); }
  static void LoadTexture(void *h, const string &asset_fn, Texture *out, VideoAssetLoader *l=0);
  static void LoadTexture(const void *from_buf, const char *fn, int size, Texture *out, int flag=VideoAssetLoader::Flag::Default);
  static Texture *LoadTexture(const MultiProcessFileResource &file, int max_image_size = 1000000);
  static string FileName(const string &asset_fn);
  static string FileContents(const string &asset_fn);
  static File *OpenFile(const string &asset_fn);
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

  SoundAssetMap *parent;
  string name, filename;
  unique_ptr<RingBuf> wav;
  int channels=0, sample_rate=0, seconds=0;
  RefillCB refill;
  void *handle=0;
  int handle_arg1=-1;
  AudioResampler resampler;

  SoundAsset() {}
  SoundAsset(const string &N, const string &FN, RingBuf *W, int C, int SR, int S) :
    name(N), filename(FN), wav(W), channels(C), sample_rate(SR), seconds(S), handle(0), handle_arg1(-1) {}

  void Load(void *handle, const char *FN, int Secs, int flag=0);
  void Load(const void *FromBuf, int size, const char *FileName, int Seconds=10);
  void Load(int seconds=10, bool unload=true);
  void Unload();
  int Refill(int reset);

  static void Load(vector<SoundAsset> *assets) { for (auto &a : *assets) a.Load(); }
  static int Size(const SoundAsset *sa) { return sa->seconds * FLAGS_sample_rate * FLAGS_chans_out; }
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

UNALIGNED_struct WavHeader {
  unsigned chunk_id, chunk_size, format, subchunk_id, subchunk_size;
  unsigned short audio_format, num_channels;
  unsigned sample_rate, byte_rate;
  unsigned short block_align, bits_per_sample;
  unsigned subchunk_id2, subchunk_size2;
  static const int Size=44;
};
UNALIGNED_END(WavHeader, WavHeader::Size);

struct WavReader {
  File *f;
  int last;
  ~WavReader() { Close(); }
  WavReader(File *F=0) { Open(F); }
  bool Open(File *F, WavHeader *H=0);
  void Close() { if (f) f->Close(); }
  int Read(RingBuf::Handle *, int offset, int size);
};

struct WavWriter {
  File *f;
  int wrote;
  ~WavWriter() { Flush(); }
  WavWriter(File *F=0) { Open(F); }
  void Open(File *F);
  int Write(const RingBuf::Handle *, bool flush=true);
  int Flush();
};

struct SimpleAssetLoader;
struct FFMpegAssetLoader;
struct AndroidAudioAssetLoader;
struct iPhoneAudioAssetLoader;

struct AssetLoader : public Module {
  AudioAssetLoader *default_audio_loader=0;
  VideoAssetLoader *default_video_loader=0;
  MovieAssetLoader *default_movie_loader=0;
  MovieAsset       *movie_playing=0;
  unique_ptr<SimpleAssetLoader> simple_loader;
  unique_ptr<FFMpegAssetLoader> ffmpeg_loader;
  unique_ptr<AndroidAudioAssetLoader> android_audio_loader;
  unique_ptr<iPhoneAudioAssetLoader> iphone_audio_loader;
  AssetLoader();
  ~AssetLoader();
  int Init();
};

void glLine(const point &p1, const point &p2, const Color *color);
void glAxis(Asset*, Entity*);
void glRoom(Asset*, Entity*);
void glIntersect(int x, int y, Color *c);
void glShadertoyShader(Shader *shader, const Texture *tex=0);
void glShadertoyShaderWindows(Shader *shader, const Color &backup_color, const Box                &win, const Texture *tex=0);
void glShadertoyShaderWindows(Shader *shader, const Color &backup_color, const vector<const Box*> &win, const Texture *tex=0);
void glSpectogram(Matrix *m, unsigned char *data, int pf, int width, int height, int hjump, float max, float clip, bool interpolate, int pd=PowerDomain::dB);
void glSpectogram(Matrix *m, Texture *t, float *max=0, float clip=-INFINITY, int pd=PowerDomain::dB);
void glSpectogram(const RingBuf::Handle *in, Texture *t, Matrix *transform=0, float *max=0, float clip=-INFINITY);

struct BoxFilled : public Drawable { void Draw(const LFL::Box &b, const Drawable::Attr *a=0) const; };
struct BoxOutline : public Drawable {
  int line_width;
  BoxOutline(int LW=1) : line_width(LW) {}
  void Draw(const LFL::Box &b, const Drawable::Attr *a=0) const;
};
struct BoxTopLeftOutline : public Drawable {
  int line_width;
  BoxTopLeftOutline(int LW=1) : line_width(LW) {}
  void Draw(const LFL::Box &b, const Drawable::Attr *a=0) const;
};
struct BoxBottomRightOutline : public Drawable {
  int line_width;
  BoxBottomRightOutline(int LW=1) : line_width(LW) {}
  void Draw(const LFL::Box &b, const Drawable::Attr *a=0) const;
};

struct Waveform : public Drawable {
  int width=0, height=0;
  unique_ptr<Geometry> geom;
  Waveform() {}
  Waveform(point dim, const Color *c, const Vec<float> *);
  void Draw(const LFL::Box &w, const Drawable::Attr *a=0) const;
  static Waveform Decimated(point dim, const Color *c, const RingBuf::Handle *, int decimateBy);
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

struct TextureArray {
  vector<Texture> a;
  void ClearGL() { for (auto &i : a) i.ClearGL(); }
  void Load(const string &fmt, const string &prefix, const string &suffix, int N);
  void DrawSequence(Asset *out, Entity *e, int *ind);
};

struct Skybox {
  Asset               a_left, a_right, a_top, a_bottom, a_front, a_back;
  Entity              e_left, e_right, e_top, e_bottom, e_front, e_back;
  Scene::EntityVector v_left, v_right, v_top, v_bottom, v_front, v_back;
  Skybox();

  Asset *asset() { return &a_left; }
  void Load(const string &filename_prefix);
  void Draw();
};

struct ParticleSystem {
  string name;
  Color color;
  v3 pos, vel, ort, updir;
  vector<v3> *pos_transform;
  int pos_transform_index;
  ParticleSystem(const string &n) : name(n), ort(0,0,1), updir(0,1,0), pos_transform(0), pos_transform_index(0) {}
  virtual void Update(Entity *cam, unsigned dt, int mx, int my, int mdown) = 0;
  virtual void Draw(GraphicsDevice*) = 0;
};

template <int MP, int MH, bool PerParticleColor> struct Particles : public ParticleSystem {
  typedef Particles<MP, MH, PerParticleColor> ParticlesType;
  static const int MaxParticles=MP, MaxHistory=MH, VertFloats=(PerParticleColor ? 9 : 5), VertSize=VertFloats*sizeof(float);
  static const int ParticleVerts=6, ParticleSize=ParticleVerts*VertSize, NumFloats=MaxParticles*ParticleVerts*VertFloats;
  static const int Trails=MaxHistory>2, TrailVertFloats=(PerParticleColor ? 7 : 3), TrailVertSize=TrailVertFloats*sizeof(float);
  static const int MaxTrailVerts=6*(MaxHistory-2), NumTrailFloats=(Trails ? MaxParticles*MaxTrailVerts*TrailVertFloats : 1);
  struct Emitter { enum { None=0, Mouse=1, Sprinkler=2, RainbowFade=4, GlowFade=8, FadeFromWhite=16 }; };

  struct Particle {
    ParticlesType *config;
    v3 history[MH], vel;
    int history_len, bounceage;
    float radius, age, maxage, remaining;
    Color color, start_color;
    bool dead;

    void InitColor() {
      if (config->rand_color)
        color = Color(Rand(config->rand_color_min.r(), config->rand_color_max.r()),
                      Rand(config->rand_color_min.g(), config->rand_color_max.g()),
                      Rand(config->rand_color_min.b(), config->rand_color_max.b()),
                      Rand(config->rand_color_min.a(), config->rand_color_max.a()));
      else if (config->emitter_type & Emitter::RainbowFade) color = Color::fade(config->color_fade);
      else                                                  color = config->color;
      start_color = color;
    }

    void Init() {
      InitColor();
      radius = Rand(config->radius_min, config->radius_max);
      history_len = Trails ? int(Rand(max(3.0f, config->radius_min), float(MaxHistory))) : 1;

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

  int num_particles, nops=0, texture=0, verts_id=-1, trailverts_id=-1, num_trailverts, emitter_type=0;
  int blend_mode_s=GraphicsDevice::SrcAlpha, blend_mode_t=GraphicsDevice::One, burst=0;
  float floorval=0, gravity=0, radius_min, radius_max, age_min=.05, age_max=1, rand_initpos, rand_initvel, emitter_angle=0, color_fade=0;
  long long ticks_seen=0, ticks_processed=0, ticks_step=0;
  float verts[NumFloats], trailverts[NumTrailFloats];
  bool trails, floor=0, always_on, per_particle_color, radius_decay=1, billboard=0, move_with_pos=0, blend=1, rand_color=0;
  Color rand_color_min, rand_color_max;
  Particle particles[MP], *free_list[MP];
  GraphicsDevice *gd=0;
  Entity *cam=0;

  Particles(const string &n, bool AlwaysOn=false, float RadiusMin=10, float RadiusMax=40, float RandInitPos=5, float RandInitVel=500) :
    ParticleSystem(n), num_particles(AlwaysOn ? MaxParticles : 0), radius_min(RadiusMin), radius_max(RadiusMax), rand_initpos(RandInitPos),
    rand_initvel(RandInitVel), trails(Trails), always_on(AlwaysOn), per_particle_color(PerParticleColor) {
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

  float       *particle_verts(int n)       { return &verts[n * ParticleVerts * VertFloats]; }
  const float *particle_verts(int n) const { return &verts[n * ParticleVerts * VertFloats]; }

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

  void Update(Entity *C, unsigned dt, int mx, int my, int mdown) {
    cam = C;
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
        else return DelParticle(particle);
      }
    }
    UpdateVertices(particle, v, tv);
  }

  void UpdateVertices(Particle *particle, float *v, float *tv) {
    float *vin = v, remaining = particle->remaining, size = particle->radius * (radius_decay ? remaining : 1);
    if (emitter_type & Emitter::GlowFade) particle->color = Color(remaining, remaining * 0.75, 1-remaining, 1.0);
    if (emitter_type & Emitter::FadeFromWhite) particle->color = Color::Interpolate(Color::white, particle->start_color, remaining);

    v3 p = particle->history[0], right, up;
    if (move_with_pos) p.Add(pos);
    if (billboard) { right = v3::Cross(cam->ort, cam->up) * size; up = cam->up * size; }
    else           { right = v3(size, 0, 0);                      up = v3(0, size, 0); }

    v3 o1=p, o2=p, o3=p, o4=p;
    o1.Add(-right + -up);
    o2.Add(-right +  up);
    o3.Add( right + -up);
    o4.Add( right +  up);

    AssignPosColor(v, o1, PerParticleColor ? &particle->color : 0, 2); v += VertFloats;
    AssignPosColor(v, o2, PerParticleColor ? &particle->color : 0, 2); v += VertFloats;
    AssignPosColor(v, o3, PerParticleColor ? &particle->color : 0, 2); v += VertFloats;
    AssignPosColor(v, o2, PerParticleColor ? &particle->color : 0, 2); v += VertFloats;
    AssignPosColor(v, o3, PerParticleColor ? &particle->color : 0, 2); v += VertFloats;
    AssignPosColor(v, o4, PerParticleColor ? &particle->color : 0, 2); v += VertFloats;

    if (trails) {
      v3 last_v1, last_v2, *history = particle->history;
      int history_len = particle->history_len;
      for (int i = 0; i < history_len - 1; i++) {
        float step = 1.0f - i / float(history_len-1);
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
        }
        last_v1 = v1;
        last_v2 = v2;
      }
    }
  }

  void Draw(GraphicsDevice *GD) {
    gd = GD;
    gd->DisableDepthTest();
    gd->DisableLighting();
    gd->DisableNormals();
    if (blend) {
      gd->EnableBlend();
      gd->BlendMode(blend_mode_s, blend_mode_t);
    }
    if (PerParticleColor) gd->EnableVertexColor();
    if (texture) {
      gd->EnableTexture();
      gd->BindTexture(GraphicsDevice::Texture2D, texture);
    }

    int update_size = verts_id < 0 ? sizeof(verts) : num_particles * ParticleSize;
    DrawParticles(GraphicsDevice::Triangles, num_particles*ParticleVerts, verts, update_size);

    if (trails) {
      int trail_update_size = trailverts_id < 0 ? sizeof(trailverts) : num_trailverts * TrailVertSize;
      DrawTrails(trailverts, trail_update_size);
    }

    if (PerParticleColor) gd->DisableVertexColor();
  }

  void DrawParticles(int prim_type, int num_verts, float *v, int l) {
    if (1)                gd->VertexPointer(3, GraphicsDevice::Float, VertSize, 0,               v, l, &verts_id, true, prim_type);
    if (1)                gd->TexPointer   (2, GraphicsDevice::Float, VertSize, 3*sizeof(float), v, l, &verts_id, false);
    if (PerParticleColor) gd->ColorPointer (4, GraphicsDevice::Float, VertSize, 5*sizeof(float), v, l, &verts_id, true);
    gd->DrawArrays(prim_type, 0, num_verts);
  }

  void DrawTrails(float *v, int l) {
    gd->DisableTexture();
    if (1)                gd->VertexPointer(3, GraphicsDevice::Float, TrailVertSize, 0,               v, l, &trailverts_id, true, GraphicsDevice::Triangles);
    if (PerParticleColor) gd->ColorPointer (4, GraphicsDevice::Float, TrailVertSize, 3*sizeof(float), v, l, &trailverts_id, true);
    gd->DrawArrays(GraphicsDevice::Triangles, 0, num_trailverts);
  }

  void AssetDrawCB(GraphicsDevice *d, Asset *out, Entity *e) { pos = e->pos; Draw(d); }

  static void AssignTex(float *out, float tx, float ty) { out[3]=tx; out[4]=ty; }
  static void AssignPosColor(float *out, const v3 &v, const Color *c, int tex_size) {
    if (1) { out[0]=v.x; out[1]=v.y; out[2]=v.z; }
    if (c) { int oi=3+tex_size; out[oi++]=c->r(); out[oi++]=c->g(); out[oi++]=c->b(); out[oi++]=c->a(); }
  }
};

template <class Line> struct RingFrameBuffer {
  typedef function<point(Line*, point, const Box&)> PaintCB;
  FrameBuffer fb;
  v2 scroll;
  point p;
  bool wrap=0;
  int w=0, h=0, font_size=0, font_height=0;
  RingFrameBuffer(GraphicsDevice *d) : fb(d) {}

  void ResetGL() { w=h=0; fb.ResetGL(); }
  virtual int Width()  const { return w; }
  virtual int Height() const { return h; }
  virtual void SizeChangedDone() { fb.Release(); scroll=v2(); p=point(); }
  virtual bool SizeChanged(int W, int H, Font *font, const Color *bgc) {
    if (W == w && H == h && font->size == font_size) return false;
    SetDimensions(W, H, font);
    fb.Resize(w, Height(), FrameBuffer::Flag::CreateGL | FrameBuffer::Flag::CreateTexture);
    ScopedClearColor scc(fb.gd, bgc);
    fb.gd->Clear();
    fb.gd->DrawMode(DrawMode::_2D, false);
    return true;
  }

  virtual void Draw(point pos, point adjust, bool scissor=true) {
    Box box(pos.x, pos.y, w, Height());
    if (scissor) fb.gd->PushScissor(box);
    fb.tex.Bind();
    (box + adjust).DrawCrimped(fb.tex.coord, 0, 0, scroll.y);
    if (scissor) fb.gd->PopScissor();
  }

  void Clear(Line *l, const Box &b, bool vwrap=true) {
    if (1)                         { Scissor s(fb.gd, 0, l->p.y - b.h,      b.w, b.h); fb.gd->Clear(); }
    if (l->p.y - b.h < 0 && vwrap) { Scissor s(fb.gd, 0, l->p.y + Height(), b.w, b.h); fb.gd->Clear(); }
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
    ScrollPercent(float(-b.h) / ht);
    return b.h;
  }

  int PushBackAndUpdate(Line *l, const Box &b, const PaintCB &paint, bool vwrap=true) {
    int ht = Height();
    if (p.y == 0)         p =          point(0, ht);
    if (b.h >= ht)        p = paint(l, point(0, b.h),            b);
    else                  p = paint(l, point(0, p.y),            b);
    if (p.y < 0 && vwrap) p = paint(l, point(0, p.y + b.h + ht), b);
    ScrollPercent(float(b.h) / ht);
    return b.h;
  }

  void SetDimensions(int W, int H, Font *f) { w = W; h = H; font_size = f->size; font_height = f->Height(); }
  void ScrollPercent(float y) { scroll.y = fmod(scroll.y + y, 1.0); }
  void ScrollPixels(int y) { ScrollPercent(float(y) / Height()); }
  void AdvancePixels(int y) { ScrollPixels(y); p.y = RingIndex::WrapOver(p.y - y, Height()); }
  point BackPlus(const point &o) const { return point(RingIndex::WrapOver(p.x + o.x, Width()),
                                                      RingIndex::WrapOver(p.y + o.y, Height())); }
};

struct TilesInterface {
  struct RunFlag { enum { DontClear=1, ClearEmpty=2 }; };
  virtual ~TilesInterface() {}
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
  virtual void Run(int flag)                                                               = 0;
};

struct LayersInterface {
  UNALIGNED_struct Node  { static const int Size=40; Box box; point scrolled; int layer_id, child_offset; }; UNALIGNED_END(Node,  Node::Size);
  UNALIGNED_struct Child { static const int Size=8;  int node_id, next_child_offset; };                      UNALIGNED_END(Child, Child::Size);
  vector<Node> node;
  vector<Child> child;
  vector<unique_ptr<TilesInterface>> layer;
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
  virtual void Init(GraphicsDevice *D, int N=1) = 0;
};

#define TilesMatrixIter(m) MatrixIter(m) if (Tile *tile = static_cast<Tile*>((m)->row(i)[j]))
template<class CB, class CBL, class CBLI> struct TilesT : public TilesInterface {
  struct Tile {
    CBL cb;
    unsigned id=0, prepend_depth=0;
    bool dirty=0;
  };
  int layer, W, H, context_depth=-1;
  vector<Tile*> prepend, append;
  matrix<Tile*> mat;
  FrameBuffer fb;
  Box current_tile;
  TilesT(GraphicsDevice *d, int l, int w=256, int h=256) : layer(l), W(w), H(h), mat(1,1), fb(d) { CHECK(IsPowerOfTwo(W)); CHECK(IsPowerOfTwo(H)); }
  ~TilesT() { TilesMatrixIter(&mat) delete tile; for (auto t : prepend) delete t; for (auto t : append) delete t; }
  
  template <class... Args> void PreAdd(Args&&... args) { prepend[context_depth]->cb.Add(forward<Args>(args)...); }
  template <class... Args> void PostAdd(Args&&... args) { append[context_depth]->cb.Add(forward<Args>(args)...); }
  template <class... Args> void AddCallback(const Box *box, Args&&... args) {
    bool added = 0;
    int x1, x2, y1, y2;
    GetTileCoords(*box, &x1, &y1, &x2, &y2);
    for (int y = max(y1, 0); y <= y2; y++)
      for (int x = max(x1, 0); x <= x2; x++, added=1) GetTile(x, y)->cb.Add(args...);
    if (!added) ERROR("AddCallback zero ", box->DebugString(), " = ", x1, " ", y1, " ", x2, " ", y2);
  }

  void PushScissor(const Box &w) const { fb.gd->PushScissorOffset(current_tile, w); }
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
    Tile **ret = reinterpret_cast<Tile**>(&mat.row(y)[x]);
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

  void Run(int flag) {
    bool clear_empty = (flag & RunFlag::ClearEmpty);
    Select();
    TilesMatrixIter(&mat) {
      if (!tile->cb.Count() && !clear_empty) continue;
      RunTile(i, j, flag, tile, tile->cb);
      tile->cb.Clear();
    }
    Release();
  }

  void Select() {
    bool init = !fb.ID;
    if (init) fb.Create(W, H);
    current_tile = Box(0, 0, W, H);
    fb.gd->DrawMode(DrawMode::_2D);
    fb.gd->ViewPort(current_tile);
    fb.gd->EnableLayering();
  }

  void RunTile(int i, int j, int flag, Tile *tile, const CBLI &tile_cb) {
    GetSpaceCoords(i, j, &current_tile.x, &current_tile.y);
    if (!tile->id) fb.AllocTexture(&tile->id);
    fb.Attach(tile->id);
    fb.gd->MatrixProjection();
    if (!(flag & RunFlag::DontClear)) fb.gd->Clear();
    fb.gd->LoadIdentity();
    fb.gd->Ortho(current_tile.x, current_tile.x + W, current_tile.y, current_tile.y + H, 0, 100);
    fb.gd->MatrixModelview();
    tile_cb.Run(current_tile);
  }

  void Release() {
    fb.Release();
    fb.gd->RestoreViewport(DrawMode::_2D);
  }

  void Draw(const Box &viewport, const point &docp) {
    int x1, x2, y1, y2, sx, sy;
    point doc_to_view = docp - viewport.Position();
    GetTileCoords(Box(docp.x, docp.y, viewport.w, viewport.h), &x1, &y1, &x2, &y2);

    Scissor scissor(fb.gd, viewport);
    fb.gd->DisableBlend();
    fb.gd->SetColor(Color::white);
    for (int y = max(y1, 0); y <= y2; y++) {
      for (int x = max(x1, 0); x <= x2; x++) {
        Tile *tile = GetTile(x, y);
        if (!tile || !tile->id) continue;
        GetSpaceCoords(y, x, &sx, &sy);
        fb.gd->BindTexture(GraphicsDevice::Texture2D, tile->id);
        Box(sx - doc_to_view.x, sy - doc_to_view.y, W, H).Draw(Texture::unit_texcoord);
      }
    }
  }
};

struct Tiles : public TilesT<Callback, CallbackList, CallbackList> {
  const Drawable::Attr *attr=0;
  Tiles(GraphicsDevice *d, int l, int w=256, int h=256) : TilesT(d, l, w, h) {}
  void SetAttr           (const Drawable::Attr *a) { attr=a; }
  void InitDrawBox       (const point&);
  void InitDrawBackground(const point&);
  void DrawBox           (const Drawable*, const Box&, const Drawable::Attr *a=0);
  void DrawBackground    (const Box&);
  void AddScissor        (const Box&);
};

template <class X> struct LayersT : public LayersInterface {
  void Init(GraphicsDevice *D, int N=1) {
    CHECK_EQ(this->layer.size(), 0);
    for (int i=0; i<N; i++) this->layer.emplace_back(make_unique<X>(D, i));
  }
};

typedef LayersT<Tiles> Layers;

}; // namespace LFL
#endif // LFL_LFAPP_ASSETS_H__
