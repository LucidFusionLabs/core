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

#ifndef LFL_CORE_APP_LOADER_H__
#define LFL_CORE_APP_LOADER_H__
namespace LFL {
  
struct AudioAssetLoader {
  struct Handle : public UniqueVoidPtr {
    AudioAssetLoader *parent;
    bool owner;
    ~Handle() { reset(0); }
    Handle(Handle &&x) : UniqueVoidPtr(move(x)), parent(x.parent), owner(x.owner) {}
    Handle(AudioAssetLoader *p=0, void *v=0, bool o=1) : UniqueVoidPtr(v), parent(p), owner(o) {}
    Handle &operator=(Handle &&x) { reset(x.release()); parent=x.parent; owner=x.owner; return *this; }
    void reset(void *x) override { if (owner && v) parent->UnloadAudioFile(*this); v=x; }
  };

  virtual void UnloadAudioFile(Handle&) = 0;
  virtual Handle LoadAudioFile(unique_ptr<File>) = 0;
  virtual Handle LoadAudioFileNamed(const string &fn) = 0;
  virtual void LoadAudio(Handle &h, SoundAsset *a, int seconds, int flag) = 0;
  virtual int RefillAudio(SoundAsset *a, int reset) = 0;
};

struct VideoAssetLoader {
  struct Flag { enum { LoadGL=1, Clear=2, RepeatGL=4, Default=LoadGL|Clear }; };
  struct Handle : public UniqueVoidPtr {
    VideoAssetLoader *parent;
    bool owner;
    ~Handle() { reset(0); }
    Handle(Handle &&x) : UniqueVoidPtr(move(x)), parent(x.parent), owner(x.owner) {}
    Handle(VideoAssetLoader *p=0, void *v=0, bool o=1) : UniqueVoidPtr(v), parent(p), owner(o) {}
    Handle &operator=(Handle &&x) { reset(x.release()); parent=x.parent; owner=x.owner; return *this; }
    void reset(void *x) override { if (v) parent->UnloadVideoFile(*this); v=x; }
  };

  virtual void UnloadVideoFile(Handle&) = 0;
  virtual Handle LoadVideoFile(unique_ptr<File>) = 0;
  virtual Handle LoadVideoFileNamed(const string &fn) = 0;
  virtual void LoadVideo(Handle &h, Texture *out, int flag=Flag::Default) = 0;
};

struct MovieAssetLoader {
  struct Handle : public UniqueVoidPtr {
    MovieAssetLoader *parent;
    bool owner;
    ~Handle() { reset(0); }
    Handle(Handle &&x) : UniqueVoidPtr(move(x)), parent(x.parent), owner(x.owner) {}
    Handle(MovieAssetLoader *p=0, void *v=0, bool o=1) : UniqueVoidPtr(v), parent(p), owner(o) {}
    Handle &operator=(Handle &&x) { reset(x.release()); parent=x.parent; owner=x.owner; return *this; }
    void reset(void *x) override { if (v) parent->UnloadMovieFile(*this); v=x; }
  };

  virtual void UnloadMovieFile(Handle&) = 0;
  virtual Handle LoadMovieFile(unique_ptr<File>) = 0;
  virtual Handle LoadMovieFileNamed(const string &fn) = 0;
  virtual void LoadMovie(Handle &h, MovieAsset *a) = 0;
  virtual int PlayMovie(MovieAsset *a, int seek) = 0;
};

struct AssetLoaderInterface : public AudioAssetLoader, public VideoAssetLoader, public MovieAssetLoader {
  VideoAssetLoader *GetVideoAssetLoader(const char *fn) { return this; }
  AudioAssetLoader *GetAudioAssetLoader(const char *fn) { return this; }
  MovieAssetLoader *GetMovieAssetLoader(const char *fn) { return this; }
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
  optional_ptr<File> f;
  int last;
  ~WavReader() { Close(); }
  WavReader(optional_ptr<File> F=optional_ptr<File>()) { Open(move(F)); }
  bool Open(optional_ptr<File> F, WavHeader *H=0);
  void Close() { if (f) f->Close(); }
  int Read(RingSampler::Handle *, int offset, int size);
};

struct WavWriter {
  optional_ptr<File> f;
  int wrote;
  ~WavWriter() { Flush(); }
  WavWriter(optional_ptr<File> F=optional_ptr<File>()) { Open(move(F)); }
  void Open(optional_ptr<File> F);
  int Write(const RingSampler::Handle *, bool flush=true);
  int Flush();
};

struct ZLibStream : public UniqueVoidPtr {
  ZLibStream();
  ZLibStream(ZLibStream &&x) : UniqueVoidPtr(move(x)) {}
  ~ZLibStream() { reset(0); }
  void reset(void *x) override;
};

struct ZLibReader {
  string out;
  int blocksize;
  ZLibStream stream;
  ~ZLibReader();
  ZLibReader(int blocksize=32768);
  bool Add(const StringPiece &in, bool partial_flush=0, bool final=0, int max_out=0);
  static string Decompress(const StringPiece&);
};

struct ZLibWriter {
  string out;
  int blocksize;
  ZLibStream stream;
  ~ZLibWriter();
  ZLibWriter(int blocksize=32768);
  bool Add(const StringPiece &in, bool partial_flush=0, bool final=0, int max_out=0);
  static string Compress(const StringPiece&);
};

struct OGGReader {
  static AudioAssetLoader::Handle OpenBuffer(AudioAssetLoader *p, const char *buf, size_t len, int *sr_out, int *chans_out, int *total_out);
  static AudioAssetLoader::Handle OpenFile(AudioAssetLoader *p, const string &fn, int *sr_out, int *chans_out, int *total_out);
  static int Read(AudioAssetLoader::Handle &h, int chans, int samples, RingSampler::Handle *out, bool reset);
  static void Close(AudioAssetLoader::Handle &h);
};

struct AssetLoader : public Module {
  unique_ptr<AssetLoaderInterface> default_loader;
  AudioAssetLoader *default_audio_loader=0;
  VideoAssetLoader *default_video_loader=0;
  MovieAssetLoader *default_movie_loader=0;
  MovieAsset       *movie_playing=0;
  AssetLoading     *loading;
  AssetLoader(AssetLoading*);
  ~AssetLoader();
  int Init() override;
};

unique_ptr<AssetLoaderInterface> CreateAssetLoader(AssetLoading*);
unique_ptr<AssetLoaderInterface> CreateSimpleAssetLoader(AssetLoading*);

struct SimpleAssetLoader : public AssetLoaderInterface {
  AssetLoading *parent;
  SimpleAssetLoader(AssetLoading*);
  virtual void *LoadFile(File*);
  virtual void *LoadFileNamed(const string &filename);
  virtual void UnloadFile(void *h);

  void UnloadAudioFile(AudioAssetLoader::Handle &h) override;
  AudioAssetLoader::Handle LoadAudioFile(unique_ptr<File>) override;
  AudioAssetLoader::Handle LoadAudioFileNamed(const string &filename) override;
  void LoadAudio(AudioAssetLoader::Handle &h, SoundAsset *a, int seconds, int flag) override;
  int RefillAudio(SoundAsset *a, int reset) override;

  void UnloadVideoFile(VideoAssetLoader::Handle &h) override;
  VideoAssetLoader::Handle LoadVideoFile(unique_ptr<File>) override;
  VideoAssetLoader::Handle LoadVideoFileNamed(const string &filename) override;
  void LoadVideo(VideoAssetLoader::Handle &h, Texture *out, int load_flag=VideoAssetLoader::Flag::Default) override;

  void UnloadMovieFile(MovieAssetLoader::Handle &h) override;
  MovieAssetLoader::Handle LoadMovieFile(unique_ptr<File>) override;
  MovieAssetLoader::Handle LoadMovieFileNamed(const string &filename) override;
  void LoadMovie(MovieAssetLoader::Handle &h, MovieAsset *a) override;
  int PlayMovie(MovieAsset *a, int seek) override;
};

struct SimpleVideoResampler : public VideoResamplerInterface {
  virtual ~SimpleVideoResampler() {}
  virtual bool Opened() const;
  virtual void Open(int sw, int sh, int sf, int dw, int dh, int df);
  virtual void Resample(const unsigned char *s, int sls, unsigned char *d, int dls, bool flip_x=0, bool flip_y=0);

  static bool Supports(int fmt);
  static void CopyPixel(int s_fmt, int d_fmt, const unsigned char *sp, unsigned char *dp, bool sxb=0, bool sxe=0, int flag=0);
  static void RGB2BGRCopyPixels(unsigned char *dst, const unsigned char *src, int l, int bpp);

  struct Flag { enum { FlipY=1, TransparentBlack=2, ZeroOnly=4 }; };

  static void Blit(const unsigned char *src, unsigned char *dst, int w, int h,
                   int sf, int sls, int sx, int sy,
                   int df, int dls, int dx, int dy, int flag=0);

  static void Filter(unsigned char *dst, int w, int h,
                     int pf, int ls, int x, int y, Matrix *kernel, int channel, int flag=0);

  static void Fill(unsigned char *dst, int l, int pf, const Color &c);
  static void Fill(unsigned char *dst, int w, int h,
                   int pf, int ls, int x, int y, const Color &c);

  static void CopyColorChannelsToMatrix(const unsigned char *buf, int w, int h,
                                        int pw, int ls, int x, int y, Matrix *out, int po);
  static void CopyMatrixToColorChannels(const Matrix *M, int w, int h,
                                        int pw, int ls, int x, int y, unsigned char *out, int po);
};

}; // namespace LFL
#endif // LFL_CORE_APP_LOADER_H__
