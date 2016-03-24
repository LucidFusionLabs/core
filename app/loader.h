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

#ifndef LFL_CORE_APP_LOADER_H__
#define LFL_CORE_APP_LOADER_H__
namespace LFL {
  
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

struct AssetLoader : public Module {
  unique_ptr<AssetLoaderInterface> default_loader;
  AudioAssetLoader *default_audio_loader=0;
  VideoAssetLoader *default_video_loader=0;
  MovieAssetLoader *default_movie_loader=0;
  MovieAsset       *movie_playing=0;
  AssetLoader();
  ~AssetLoader();
  int Init();
};

unique_ptr<AssetLoaderInterface> CreateAssetLoader();
unique_ptr<AssetLoaderInterface> CreateSimpleAssetLoader();

}; // namespace LFL
#endif // LFL_CORE_APP_LOADER_H__
