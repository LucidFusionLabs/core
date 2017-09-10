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
  virtual void *LoadAudioFile(File*) = 0;
  virtual void *LoadAudioFileNamed(const string &fn) = 0;
  virtual void UnloadAudioFile(void *h) = 0;

  virtual void LoadAudio(void *h, SoundAsset *a, int seconds, int flag) = 0;
  virtual int RefillAudio(SoundAsset *a, int reset) = 0;
};

struct VideoAssetLoader {
  virtual void *LoadVideoFile(File*) = 0;
  virtual void *LoadVideoFileNamed(const string &fn) = 0;
  virtual void UnloadVideoFile(void *h) = 0;

  struct Flag { enum { LoadGL=1, Clear=2, RepeatGL=4, Default=LoadGL|Clear }; };
  virtual void LoadVideo(void *h, Texture *out, int flag=Flag::Default) = 0;
};

struct MovieAssetLoader {
  virtual void *LoadMovieFile(File*) = 0;
  virtual void *LoadMovieFileNamed(const string &fn) = 0;
  virtual void UnloadMovieFile(void *h) = 0;

  virtual void LoadMovie(void *h, MovieAsset *a) = 0;
  virtual int PlayMovie(MovieAsset *a, int seek) = 0;
};

struct AssetLoaderInterface : public AudioAssetLoader, public VideoAssetLoader, public MovieAssetLoader {
  virtual VideoAssetLoader *GetVideoAssetLoader(const char *fn) { return this; }
  virtual AudioAssetLoader *GetAudioAssetLoader(const char *fn) { return this; }
  virtual MovieAssetLoader *GetMovieAssetLoader(const char *fn) { return this; }
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
  int Read(RingSampler::Handle *, int offset, int size);
};

struct WavWriter {
  File *f;
  int wrote;
  ~WavWriter() { Flush(); }
  WavWriter(File *F=0) { Open(F); }
  void Open(File *F);
  int Write(const RingSampler::Handle *, bool flush=true);
  int Flush();
};

struct ZLibReader {
  string out;
  int blocksize;
  VoidPtr stream;
  ~ZLibReader();
  ZLibReader(int blocksize=32768);
  bool Add(const StringPiece &in, bool partial_flush=0, bool final=0, int max_out=0);
  static string Decompress(const StringPiece&);
};

struct ZLibWriter {
  string out;
  int blocksize;
  VoidPtr stream;
  ~ZLibWriter();
  ZLibWriter(int blocksize=32768);
  bool Add(const StringPiece &in, bool partial_flush=0, bool final=0, int max_out=0);
  static string Compress(const StringPiece&);
};

struct OGGReader {
  static void  Close(void *h);
  static void *OpenBuffer(const char *buf, size_t len, int *sr_out, int *chans_out, int *total_out);
  static void *OpenFile(const string &fn, int *sr_out, int *chans_out, int *total_out);
  static int   Read(void *h, int chans, int samples, RingSampler::Handle *out, bool reset);
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
  int Init();
};

unique_ptr<AssetLoaderInterface> CreateAssetLoader(AssetLoading*);
unique_ptr<AssetLoaderInterface> CreateSimpleAssetLoader(AssetLoading*);

struct SimpleAssetLoader : public AssetLoaderInterface {
  AssetLoading *parent;
  SimpleAssetLoader(AssetLoading*);
  virtual void *LoadFile(File*);
  virtual void *LoadFileNamed(const string &filename);
  virtual void UnloadFile(void *h);

  virtual void *LoadAudioFile(File*);
  virtual void *LoadAudioFileNamed(const string &filename);
  virtual void UnloadAudioFile(void *h);

  virtual void *LoadVideoFile(File*);
  virtual void *LoadVideoFileNamed(const string &filename);
  virtual void UnloadVideoFile(void *h);

  virtual void *LoadMovieFile(File*);
  virtual void *LoadMovieFileNamed(const string &filename);
  virtual void UnloadMovieFile(void *h);

  virtual void LoadVideo(void *h, Texture *out, int load_flag=VideoAssetLoader::Flag::Default);
  virtual void LoadAudio(void *h, SoundAsset *a, int seconds, int flag);
  virtual int RefillAudio(SoundAsset *a, int reset);
  virtual void LoadMovie(void *h, MovieAsset *a);
  virtual int PlayMovie(MovieAsset *a, int seek) ;
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
