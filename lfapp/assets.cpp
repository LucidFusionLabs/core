/*
 * $Id: assets.cpp 1334 2014-11-28 09:14:21Z justin $
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

extern "C" {
#ifdef LFL_FFMPEG
#define INT64_C (long long)
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#define AVCODEC_MAX_AUDIO_FRAME_SIZE 192000
#endif
};
#ifdef LFL_PNG
#include <png.h>
#endif

#include "lfapp/lfapp.h"
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/flow.h"
#include "lfapp/gui.h"
#include "lfapp/ipc.h"

extern "C" {
#ifdef LFL_JPEG
#include "jpeglib.h"
#endif
};
#ifdef LFL_GIF
#include "gif_lib.h"
#endif

#ifdef LFL_IPHONE
extern "C" void *iPhoneMusicCreate(const char *filename);
#endif

namespace LFL {
DEFINE_float(shadertoy_blend, 0.5, "Shader blend factor");
DEFINE_int(soundasset_seconds, 10, "Soundasset buffer seconds");

const int SoundAsset::FlagNoRefill = 1;
#ifdef LFL_FFMPEG
const int SoundAsset::FromBufPad = FF_INPUT_BUFFER_PADDING_SIZE;
#else
const int SoundAsset::FromBufPad = 0;
#endif

#ifdef LFL_PNG
static void PngRead (png_structp png_ptr, png_bytep data, png_size_t length) { FromVoid<File*>(png_get_io_ptr(png_ptr))->Read (data, length); }
static void PngWrite(png_structp png_ptr, png_bytep data, png_size_t length) { FromVoid<File*>(png_get_io_ptr(png_ptr))->Write(data, length); }
static void PngFlush(png_structp png_ptr) {}

int PngReader::Read(File *lf, Texture *out) {
  char header[8];
  if (lf->Read(header, sizeof(header)) != sizeof(header)) return ERRORv(-1, "read: ", lf->Filename());
  if (png_sig_cmp(MakeUnsigned(header), 0, 8)) return ERRORv(-1, "png_sig_cmp: ", lf->Filename());

  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
  if (!png_ptr) return ERRORv(-1, "png_create_read_struct: ", lf->Filename());

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) { png_destroy_read_struct(&png_ptr, 0, 0); return ERRORv(-1, "png_create_info_struct: ", lf->Filename()); }

  if (setjmp(png_jmpbuf(png_ptr)))
  { png_destroy_read_struct(&png_ptr, &info_ptr, 0); return ERRORv(-1, "png error: ", lf->Filename()); }

  png_set_read_fn(png_ptr, lf, PngRead);
  png_set_sig_bytes(png_ptr, sizeof(header));
  png_read_info(png_ptr, info_ptr);
  png_byte color_type = png_get_color_type(png_ptr, info_ptr), bit_depth = png_get_bit_depth(png_ptr, info_ptr);
  int number_of_passes = png_set_interlace_handling(png_ptr), opf = Texture::preferred_pf;
  switch (color_type) {
    case PNG_COLOR_TYPE_GRAY:       opf = Pixel::GRAY8;              break;
    case PNG_COLOR_TYPE_GRAY_ALPHA: opf = Pixel::GRAYA8;             break;
    case PNG_COLOR_TYPE_PALETTE:    png_set_palette_to_rgb(png_ptr); // fall thru
    case PNG_COLOR_TYPE_RGB:
                                    if (opf == Pixel::RGBA || opf == Pixel::BGRA) png_set_filler(png_ptr, 0xff, PNG_FILLER_AFTER);
    case PNG_COLOR_TYPE_RGBA: 
                                    // if (opf == Pixel::BGRA || opf == Pixel::BGR24) png_set_bgr(png_ptr);
                                    break;
    default: FATAL("unknown png_get_color_type ", color_type);
  }
  png_read_update_info(png_ptr, info_ptr);

  out->Resize(png_get_image_width(png_ptr, info_ptr), png_get_image_height(png_ptr, info_ptr), opf, Texture::Flag::CreateBuf);
  int linesize = out->LineSize();
  vector<png_bytep> row_pointers;
  CHECK_LE(png_get_rowbytes(png_ptr, info_ptr), linesize);
  for (int y=0; y<out->height; y++) row_pointers.push_back(png_bytep(out->buf + linesize * y));
  png_read_image(png_ptr, &row_pointers[0]);
  png_destroy_read_struct(&png_ptr, &info_ptr, 0);
  return 0;
}

int PngWriter::Write(File *lf, const Texture &tex) {
  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
  if (!png_ptr) return ERRORv(-1, "png_create_read_struct: ", lf->Filename());

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) { png_destroy_write_struct(&png_ptr, 0); return ERRORv(-1, "png_create_info_struct: ", lf->Filename()); }

  if (setjmp(png_jmpbuf(png_ptr)))
  { png_destroy_write_struct(&png_ptr, &info_ptr); return ERRORv(-1, "setjmp: ", lf->Filename()); }
  png_set_write_fn(png_ptr, lf, PngWrite, PngFlush);

  int color_type = 0;
  switch (tex.pf) {
    case Pixel::BGRA:   png_set_bgr(png_ptr);
    case Pixel::RGBA:   color_type = PNG_COLOR_TYPE_RGBA;       break;
    case Pixel::BGR24:  png_set_bgr(png_ptr);
    case Pixel::RGB24:  color_type = PNG_COLOR_TYPE_RGB;        break;
    case Pixel::GRAY8:  color_type = PNG_COLOR_TYPE_GRAY;       break;
    case Pixel::GRAYA8: color_type = PNG_COLOR_TYPE_GRAY_ALPHA; break;
    default:            FATAL("unknown color_type: ", tex.pf);
  }

  png_set_IHDR(png_ptr, info_ptr, tex.width, tex.height, 8, color_type, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
  png_write_info(png_ptr, info_ptr);

  vector<png_bytep> row_pointers;
  for (int ls=tex.LineSize(), y=0; y<tex.height; y++) row_pointers.push_back(png_bytep(tex.buf + y*ls));
  png_write_image(png_ptr, &row_pointers[0]);
  png_write_end(png_ptr, 0);
  png_destroy_write_struct(&png_ptr, &info_ptr);
  return 0;
}
#else /* LFL_PNG */
int PngReader::Read(File *lf, Texture *out) { FATAL("not implemented"); }
int PngWriter::Write(File *lf, const Texture &tex) { FATAL("not implemented"); }
#endif /* LFL_PNG */

int PngWriter::Write(const string &fn, const Texture &tex) {
  LocalFile lf(fn, "w");
  if (!lf.Opened()) return ERRORv(-1, "open: ", fn);
  return PngWriter::Write(&lf, tex);
}

#ifdef LFL_JPEG
const JOCTET EOI_BUFFER[1] = { JPEG_EOI };
struct MyJpegErrorMgr  { jpeg_error_mgr  pub; jmp_buf setjmp_buffer; };
struct MyJpegSourceMgr { jpeg_source_mgr pub; const JOCTET *data; size_t len; } ;

static void JpegErrorExit(j_common_ptr jcs) { longjmp(FromVoid<MyJpegErrorMgr*>(jcs->err)->setjmp_buffer, 1); }
static void JpegInitSource(j_decompress_ptr jds) {}
static void JpegTermSource(j_decompress_ptr jds) {}
static boolean JpegFillInputBuffer(j_decompress_ptr jds) {
  MyJpegSourceMgr *src = FromVoid<MyJpegSourceMgr*>(jds->src);
  src->pub.next_input_byte = EOI_BUFFER;
  src->pub.bytes_in_buffer = 1;
  return true;
}

static void JpegSkipInputData(j_decompress_ptr jds, long len) {
  MyJpegSourceMgr *src = FromVoid<MyJpegSourceMgr*>(jds->src);
  if (src->pub.bytes_in_buffer < len) {
    src->pub.next_input_byte = EOI_BUFFER;
    src->pub.bytes_in_buffer = 1;
  } else {
    src->pub.next_input_byte += len;
    src->pub.bytes_in_buffer -= len;
  }
}

static void JpegMemSrc(j_decompress_ptr jds, const char *buf, size_t len) {
  if (!jds->src) jds->src = FromVoid<jpeg_source_mgr*>((*jds->mem->alloc_small)(j_common_ptr(jds), JPOOL_PERMANENT, sizeof(MyJpegSourceMgr)));
  MyJpegSourceMgr *src = FromVoid<MyJpegSourceMgr*>(jds->src);
  src->pub.init_source       = JpegInitSource;
  src->pub.fill_input_buffer = JpegFillInputBuffer;
  src->pub.skip_input_data   = JpegSkipInputData;
  src->pub.resync_to_restart = jpeg_resync_to_restart;
  src->pub.term_source       = JpegTermSource;
  src->data = MakeUnsigned(buf);
  src->len = len;
  src->pub.bytes_in_buffer = len;
  src->pub.next_input_byte = src->data;
}

static string JpegErrorMessage(j_decompress_ptr jds) {
  char buf[JMSG_LENGTH_MAX];
  jds->err->format_message(j_common_ptr(jds), buf);
  return buf;
}

int JpegReader::Read(File *lf, Texture *out) { return Read(lf->Contents(), out); }
int JpegReader::Read(const string &data, Texture *out) {
  MyJpegErrorMgr jerr;
  jpeg_decompress_struct jds;
  jds.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = JpegErrorExit;
  if (setjmp(jerr.setjmp_buffer)) {
    string error = JpegErrorMessage(&jds);
    jpeg_destroy_decompress(&jds);
    return ERRORv(-1, "jpeg decompress failed ", error);
  }

  jpeg_create_decompress(&jds);
  JpegMemSrc(&jds, data.data(), data.size());
  if (jpeg_read_header(&jds, 1) != 1) { jpeg_destroy_decompress(&jds); return ERRORv(-1, "jpeg decompress failed "); }
  jpeg_start_decompress(&jds);

  if      (jds.output_components == 1) out->pf = Pixel::GRAY8;
#ifndef WIN32
  else if (jds.output_components == 3) out->pf = Pixel::RGBA;
#else
  else if (jds.output_components == 3) out->pf = Pixel::RGB24;
#endif
  else if (jds.output_components == 4) out->pf = Pixel::RGBA;
  else { ERROR("unsupported jpeg components ", jds.output_components); jpeg_destroy_decompress(&jds); return -1; }

#ifndef WIN32
  if      (out->pf == Pixel::RGBA)  jds.out_color_space = JCS_EXT_RGBA;
  else if (out->pf == Pixel::BGRA)  jds.out_color_space = JCS_EXT_BGRA;
  else if (out->pf == Pixel::RGB24) jds.out_color_space = JCS_EXT_RGB;
  else if (out->pf == Pixel::BGR24) jds.out_color_space = JCS_EXT_BGR;
#endif

  out->Resize(jds.output_width, jds.output_height, out->pf, Texture::Flag::CreateBuf);
  for (int linesize = out->LineSize(); jds.output_scanline < jds.output_height;) {
    unsigned char *offset = out->buf + jds.output_scanline * linesize;
    jpeg_read_scanlines(&jds, &offset, 1);
  }
  jpeg_finish_decompress(&jds);
  jpeg_destroy_decompress(&jds);
  return 0;
}
#else /* LFL_JPEG */
int JpegReader::Read(File *lf,           Texture *out) { FATAL("not implemented"); }
int JpegReader::Read(const string &data, Texture *out) { FATAL("not implemented"); }
#endif /* LFL_JPEG */

#ifdef LFL_GIF
static int GIFInput(GifFileType *gif, GifByteType *out, int size) {
  return FromVoid<BufferFile*>(gif->UserData)->Read(out, size);
}

int GIFReader::Read(File *lf,           Texture *out) { return Read(lf->Contents(), out); }
int GIFReader::Read(const string &data, Texture *out) {
  int error_code = 0;
  BufferFile bf(data);
  GifFileType *gif = DGifOpen(&bf, &GIFInput, &error_code);
  if (!gif) { INFO("gif open failed: ", error_code); return -1; }
  if (DGifSlurp(gif) != GIF_OK) { INFO("gif slurp failed"); DGifCloseFile(gif, &error_code); return -1; }
  out->Resize(gif->SWidth, gif->SHeight, Pixel::RGBA, Texture::Flag::CreateBuf);

  SavedImage *image = &gif->SavedImages[0];
  int gif_linesize = image->ImageDesc.Width, ls = out->LineSize(), ps = out->PixelSize(), transparent = -1;
  ColorMapObject *color_map = X_or_Y(gif->SColorMap, image->ImageDesc.ColorMap);
  for (int i=0; i<image->ExtensionBlockCount; i++) {
    ExtensionBlock *block = &image->ExtensionBlocks[i];
    if (block->Function == GRAPHICS_EXT_FUNC_CODE && block->ByteCount == 4 && (block->Bytes[0] & 1))
    { transparent = block->Bytes[3]; break; }
  }
  CHECK_LE(image->ImageDesc.Left + image->ImageDesc.Width,  out->width);
  CHECK_LE(image->ImageDesc.Top  + image->ImageDesc.Height, out->height);
  for (int y=image->ImageDesc.Top, yl=y+image->ImageDesc.Height; y<yl; y++) {
    unsigned char *gif_row = &image->RasterBits[y * gif_linesize], *pix_row = &(out->buf)[y * ls];
    for (int x=image->ImageDesc.Left, xl=x+image->ImageDesc.Width; x<xl; x++) {
      unsigned char gif_in = gif_row[x], *pix_out = &pix_row[x * ps];
      if (gif_in == transparent || gif_in >= color_map->ColorCount) { pix_out[3] = 0; continue; }
      *pix_out++ = color_map->Colors[gif_in].Red;
      *pix_out++ = color_map->Colors[gif_in].Green;
      *pix_out++ = color_map->Colors[gif_in].Blue;
      *pix_out++ = 255;
    }
  }
  DGifCloseFile(gif, &error_code);
  return 0;
}
#else /* LFL_GIF */
int GIFReader::Read(File *lf,           Texture *out) { FATAL("not implemented"); }
int GIFReader::Read(const string &data, Texture *out) { FATAL("not implemented"); }
#endif /* LFL_GIF */

bool WavReader::Open(File *F, WavHeader *out) {
  f = F;
  last = WavHeader::Size;
  return (f && out) ? File::SeekReadSuccess(f, 0, out, WavHeader::Size) : f != nullptr;
}

int WavReader::Read(RingBuf::Handle *B, int off, int num) {
  if (!f->Opened()) return -1;
  basic_string<short> buf(num, 0);
  int offset = WavHeader::Size + off * sizeof(short);
  if (!File::SeekReadSuccess(f, offset, &buf[0], buf.size()*2)) return -1;
  for (int i=0; i<num; i++) B->Write(buf[i] / 32768.0);
  last = offset + buf.size()*2;
  return 0;
}

void WavWriter::Open(File *F) {
  f = F;
  wrote = WavHeader::Size;
  if (f && f->Opened()) Flush();
}

int WavWriter::Write(const RingBuf::Handle *B, bool flush) {
  if (!f->Opened()) return -1;
  basic_string<short> buf(B->Len(), 0);
  for (int i=0, l=B?B->Len():0; i<l; ++i) buf[i] = short(B->Read(i) * 32768.0);
  if (!File::WriteSuccess(f, &buf[0], buf.size()*2)) return -1;
  wrote += buf.size()*2;
  return flush ? Flush() : 0;
}

int WavWriter::Flush() {
  if (!f->Opened()) return -1;
  WavHeader hdr;
  hdr.chunk_id = *reinterpret_cast<const unsigned*>("RIFF");
  hdr.format = *reinterpret_cast<const unsigned*>("WAVE");
  hdr.subchunk_id = *reinterpret_cast<const unsigned*>("fmt ");
  hdr.audio_format = 1;
  hdr.num_channels = 1;
  hdr.sample_rate = FLAGS_sample_rate;
  hdr.byte_rate = hdr.sample_rate * hdr.num_channels * sizeof(short);
  hdr.block_align = hdr.num_channels * sizeof(short);
  hdr.bits_per_sample = 16;
  hdr.subchunk_id2 = *reinterpret_cast<const unsigned*>("data");
  hdr.subchunk_size = 16;
  hdr.subchunk_size2 = wrote - WavHeader::Size;
  hdr.chunk_size = 4 + (8 + hdr.subchunk_size) + (8 + hdr.subchunk_size2);

  if (f->Seek(0, File::Whence::SET) != 0) return -1;
  if (f->Write(&hdr, WavHeader::Size) != WavHeader::Size) return -1;
  if (f->Seek(wrote, File::Whence::SET) != wrote) return -1;
  return 0;
}

struct SimpleAssetLoader : public AudioAssetLoader, public VideoAssetLoader, public MovieAssetLoader {
  SimpleAssetLoader() { INFO("SimpleAssetLoader"); }

  virtual void *LoadFile(const string &filename) {
    unique_ptr<LocalFile> lf = make_unique<LocalFile>(filename, "r");
    if (!lf->Opened()) return ERRORv(nullptr, "open: ", filename);
    return lf.release();
  }
  virtual void UnloadFile(void *h) { delete FromVoid<File*>(h); }
  virtual void *LoadBuf(const char *buf, int len, const char *mimetype) { return new BufferFile(string(buf, len), mimetype); }
  virtual void UnloadBuf(void *h) { delete FromVoid<File*>(h); }

  virtual void *LoadAudioFile(const string &filename) { return LoadFile(filename); }
  virtual void UnloadAudioFile(void *h) { return UnloadFile(h); }
  virtual void *LoadAudioBuf(const char *buf, int len, const char *mimetype) { return LoadBuf(buf, len, mimetype); }
  virtual void UnloadAudioBuf(void *h) { return UnloadBuf(h); }

  virtual void *LoadVideoFile(const string &filename) { return LoadFile(filename); }
  virtual void UnloadVideoFile(void *h) { return UnloadFile(h); }
  virtual void *LoadVideoBuf(const char *buf, int len, const char *mimetype) { return LoadBuf(buf, len, mimetype); }
  virtual void UnloadVideoBuf(void *h) { return UnloadBuf(h); }

  virtual void *LoadMovieFile(const string &filename) { return LoadFile(filename); }
  virtual void UnloadMovieFile(void *h) { return UnloadFile(h); }
  virtual void *LoadMovieBuf(const char *buf, int len, const char *mimetype) { return LoadBuf(buf, len, mimetype); }
  virtual void UnloadMovieBuf(void *h) { return UnloadBuf(h); }

  virtual void LoadVideo(void *h, Texture *out, int load_flag=VideoAssetLoader::Flag::Default) {
    static char jpghdr[2] = { '\xff', '\xd8' }; // 0xff, 0xd8 };
    static char gifhdr[4] = { 'G', 'I', 'F', '8' };
    static char pnghdr[8] = { '\211', 'P', 'N', 'G', '\r', '\n', '\032', '\n' };

    File *f = FromVoid<File*>(h);
    string fn = f->Filename();
    unsigned char hdr[8];
    if (f->Read(hdr, 8) != 8) return ERROR("load ", fn, " : failed");
    f->Reset();

    if (0) {}
#ifdef LFL_PNG
    else if (!memcmp(hdr, pnghdr, sizeof(pnghdr))) {
      if (PngReader::Read(f, out)) return;
    }
#endif
#ifdef LFL_JPEG
    else if (!memcmp(hdr, jpghdr, sizeof(jpghdr))) {
      if (JpegReader::Read(f, out)) return;
    }
#endif
#ifdef LFL_GIF
    else if (!memcmp(hdr, gifhdr, sizeof(gifhdr))) {
      if (GIFReader::Read(f, out)) return;
    }
#endif
    else return ERROR("load ", fn, " : failed");

    if (load_flag & Flag::LoadGL) out->LoadGL();
    if (load_flag & Flag::Clear)  out->ClearBuffer();
  }

  virtual void LoadAudio(void *h, SoundAsset *a, int seconds, int flag) {
    File *f = FromVoid<File*>(h);
    string fn = f->Filename();
    if (SuffixMatch(fn, ".wav", false)) {
      WavHeader wav_header;
      WavReader wav_reader;
      if (!wav_reader.Open(f, &wav_header))
        return ERROR("LoadAudio(", a->name, ", ", fn, ") open failed: ", strerror(errno));

      int wav_samples = (f->Size() - WavHeader::Size) / 2;
      if (wav_header.audio_format != 1 || wav_header.num_channels != 1 ||
          wav_header.bits_per_sample != 16 || wav_header.sample_rate != FLAGS_sample_rate ||
          wav_samples > FLAGS_sample_rate*FLAGS_soundasset_seconds)
        return ERROR("LoadAudio(", a->name, ", ", fn, ") not supported");

      a->channels = 1;
      a->sample_rate = FLAGS_sample_rate;
      a->wav = make_unique<RingBuf>(a->sample_rate, wav_samples);
      RingBuf::Handle H(a->wav.get());
      if (wav_reader.Read(&H, 0, wav_samples))
        return ERROR("LoadAudio(", a->name, ", ", fn, ") read failed: ", strerror(errno));
    }
  }
  virtual int RefillAudio(SoundAsset *a, int reset) { return 0; }

  virtual void LoadMovie(void *h, MovieAsset *a) {}
  virtual int PlayMovie(MovieAsset *a, int seek) { return 0; }
};

#ifdef LFL_ANDROID
struct AndroidAudioAssetLoader : public AudioAssetLoader {
  virtual void *LoadAudioFile(const string &filename) { return AndroidLoadMusicResource(filename.c_str()); }
  virtual void UnloadAudioFile(void *h) {}
  virtual void *LoadAudioBuf(const char *buf, int len, const char *mimetype) { return 0; }
  virtual void UnloadAudioBuf(void *h) {}
  virtual void LoadAudio(void *handle, SoundAsset *a, int seconds, int flag) { a->handle = handle; }
  virtual int RefillAudio(SoundAsset *a, int reset) { return 0; }
};
#else
struct AndroidAudioAssetLoader {};
#endif

#ifdef LFL_IPHONE
struct iPhoneAudioAssetLoader : public AudioAssetLoader {
  virtual void *LoadAudioFile(const string &filename) { return iPhoneMusicCreate(filename.c_str()); }
  virtual void UnloadAudioFile(void *h) {}
  virtual void *LoadAudioBuf(const char *buf, int len, const char *mimetype) { return 0; }
  virtual void UnloadAudioBuf(void *h) {}
  virtual void LoadAudio(void *handle, SoundAsset *a, int seconds, int flag) { a->handle = handle; }
  virtual int RefillAudio(SoundAsset *a, int reset) { return 0; }
};
#else
struct iPhoneAudioAssetLoader {};
#endif

#ifdef LFL_FFMPEG
struct FFBIOFile {
  static void *Alloc(File *f) {
    int bufsize = 16384;
    return avio_alloc_context(FromVoid<unsigned char*>(malloc(bufsize)), bufsize, 0, f, Read, Write, Seek);
  }
  static void Free(void *in) {
    AVIOContext *s = FromVoid<AVIOContext*>(in);
    delete FromVoid<File*>(s->opaque);
    free(s->buffer);
    av_free(s);
  }
  static int     Read(void *f, uint8_t *buf, int buf_size) { return FromVoid<File*>(f)->Read (buf, buf_size); }
  static int    Write(void *f, uint8_t *buf, int buf_size) { return FromVoid<File*>(f)->Write(buf, buf_size); }
  static int64_t Seek(void *f, int64_t offset, int whence) { return FromVoid<File*>(f)->Seek (offset, whence); }
};

struct FFBIOC {
  void const *buf;
  int len, offset;
  FFBIOC(void const *b, int l) : buf(b), len(l), offset(0) {}

  static void *Alloc(void const *buf, int len) {
    static const int bufsize = 32768;
    return avio_alloc_context(FromVoid<unsigned char*>(malloc(bufsize)), bufsize, 0, new FFBIOC(buf, len), Read, Write, Seek);
  }
  static void Free(void *in) {
    AVIOContext *s = FromVoid<AVIOContext*>(in);
    delete FromVoid<FFBIOC*>(s->opaque);
    free(s->buffer);
    av_free(s);
  }
  static int Read(void *opaque, uint8_t *buf, int buf_size) {
    FFBIOC *s = FromVoid<FFBIOC*>(opaque);
    int len = min(buf_size, s->len - s->offset);
    if (len <= 0) return len;

    memcpy(buf, FromVoid<const char*>(s->buf) + s->offset, len);
    s->offset += len;
    return len;
  }
  static int Write(void *opaque, uint8_t *buf, int buf_size) { return -1; }
  static int64_t Seek(void *opaque, int64_t offset, int whence) {
    FFBIOC *s = FromVoid<FFBIOC*>(opaque);
    if      (whence == SEEK_SET) s->offset = offset;
    else if (whence == SEEK_CUR) s->offset += offset;
    else if (whence == SEEK_END) s->offset = s->len + offset;
    else return -1;
    return 0;
  }
};

struct FFMpegAssetLoader : public AudioAssetLoader, public VideoAssetLoader, public MovieAssetLoader {
  FFMpegAssetLoader() { INFO("FFMpegAssetLoader"); }

  virtual void *LoadFile(const string &filename) { return LoadFile(filename, 0); }
  AVFormatContext *LoadFile(const string &filename, AVIOContext **pbOut) {
#if !defined(LFL_ANDROID) && !defined(LFL_IPHONE)
    AVFormatContext *fctx = 0;
    if (avformat_open_input(&fctx, filename.c_str(), 0, 0)) return ERRORv(nullptr, "av_open_input_file: ", filename);
    return fctx;
#else
    unique_ptr<LocalFile> lf = make_unique<LocalFile>(filename, "r");
    if (!lf->opened()) return ERRORv(nullptr, "FFLoadFile: open ", filename);
    void *pb = FFBIOFile::Alloc(lf.release());
    AVFormatContext *ret = Load((AVIOContext*)pb, filename, 0, 0, pbOut);
    if (!ret) FFBIOFile::Free(pb);
    return ret;
#endif
  }

  virtual void UnloadFile(void *h) {
    AVFormatContext *handle = FromVoid<AVFormatContext*>(h);
    for (int i = handle->nb_streams - 1; handle->streams && i >= 0; --i) {
      AVStream* stream = handle->streams[i];
      if (!stream || !stream->codec || !stream->codec->codec) continue;
      stream->discard = AVDISCARD_ALL;
      avcodec_close(stream->codec);
    }
#if defined(LFL_ANDROID) || defined(LFL_IPHONE)
    FFBIOFile::Free(handle->pb);
#endif
    avformat_close_input(&handle);
  }

  virtual void *LoadBuf(const char *buf, int len, const char *filename) { return LoadBuf(buf, len, filename, 0); }
  AVFormatContext *LoadBuf(void const *buf, int len, const char *filename, AVIOContext **pbOut) {
    const char *suffix = strchr(filename, '.');
    void *pb = FFBIOC::Alloc(buf, len);
    AVFormatContext *ret = Load(FromVoid<AVIOContext*>(pb), suffix ? suffix+1 : filename,
                                const_cast<char*>(FromVoid<const char*>(buf)), min(4096, len), pbOut);
    if (!ret) FFBIOC::Free(pb);
    return ret;
  }

  virtual void UnloadBuf(void *h) {
    AVFormatContext *handle = FromVoid<AVFormatContext*>(h);
    FFBIOC::Free(handle->pb);
    avformat_close_input(&handle);
  }

  virtual void *LoadAudioFile(const string &filename) { return LoadFile(filename); }
  virtual void UnloadAudioFile(void *h) { return UnloadFile(h); }
  virtual void *LoadAudioBuf(const char *buf, int len, const char *mimetype) { return LoadBuf(buf, len, mimetype); }
  virtual void UnloadAudioBuf(void *h) { return UnloadBuf(h); }

  virtual void *LoadVideoFile(const string &filename) { return LoadFile(filename); }
  virtual void UnloadVideoFile(void *h) { return UnloadFile(h); }
  virtual void *LoadVideoBuf(const char *buf, int len, const char *mimetype) { return LoadBuf(buf, len, mimetype); }
  virtual void UnloadVideoBuf(void *h) { return UnloadBuf(h); }

  virtual void *LoadMovieFile(const string &filename) { return LoadFile(filename); }
  virtual void UnloadMovieFile(void *h) { return UnloadFile(h); }
  virtual void *LoadMovieBuf(const char *buf, int len, const char *mimetype) { return LoadBuf(buf, len, mimetype); }
  virtual void UnloadMovieBuf(void *h) { return UnloadBuf(h); }

  void LoadVideo(void *handle, Texture *out, int load_flag=VideoAssetLoader::Flag::Default) {
    AVFormatContext *fctx = FromVoid<AVFormatContext*>(handle);
    int video_index = -1, got=0;
    for (int i=0; i<fctx->nb_streams; i++) {
      AVStream *st = fctx->streams[i];
      AVCodecContext *avctx = st->codec;
      if (avctx->codec_type == AVMEDIA_TYPE_VIDEO) video_index = i;
    }
    if (video_index < 0) return ERROR("no stream: ", fctx->nb_streams);

    AVCodecContext *avctx = fctx->streams[video_index]->codec;
    AVCodec *codec = avcodec_find_decoder(avctx->codec_id);
    if (!codec || avcodec_open2(avctx, codec, 0) < 0) return ERROR("avcodec_open2: ", handle);

    AVPacket packet;
    if (av_read_frame(fctx, &packet) < 0) { avcodec_close(avctx); return ERROR("av_read_frame: ", handle); }

    AVFrame *frame = av_frame_alloc(); int ret;
    if ((ret = avcodec_decode_video2(avctx, frame, &got, &packet)) <= 0 || !got) {
      char errstr[128]; av_strerror(ret, errstr, sizeof(errstr));
      ERROR("avcodec_decode_video2: ", codec->name, " ", ret, ": ", errstr);
    } else {
      int pf = Pixel::FromFFMpegId(avctx->pix_fmt);
      out->width  = avctx->width;
      out->height = avctx->height;
      out->LoadGL(*frame->data, point(out->width, out->height), pf, frame->linesize[0]);
      if (!(load_flag & Flag::Clear)) out->LoadBuffer(frame->data[0], point(out->width, out->height), pf, frame->linesize[0]);
      // av_frame_unref(frame);
    }

    av_free_packet(&packet);
    // av_frame_free(&frame);
  }

  void LoadAudio(void *handle, SoundAsset *a, int seconds, int flag) {
    AVFormatContext *fctx = FromVoid<AVFormatContext*>(handle);
    LoadMovie(a, 0, fctx);

    int samples = RefillAudio(a, 1);
    if (samples == SoundAsset::Size(a) && !(flag & SoundAsset::FlagNoRefill)) a->refill = RefillAudioCB;

    if (!a->refill) {
      avcodec_close(fctx->streams[a->handle_arg1]->codec);
      a->resampler.Close();
    }
  }

  void LoadMovie(void *handle, MovieAsset *ma) {
    LoadMovie(&ma->audio, &ma->video, FromVoid<AVFormatContext*>(handle));
    PlayMovie(&ma->audio, &ma->video, FromVoid<AVFormatContext*>(handle), 0);
  }

  int PlayMovie(MovieAsset *ma, int seek) { return PlayMovie(&ma->audio, &ma->video, FromVoid<AVFormatContext*>(ma->handle), seek); }
  int RefillAudio(SoundAsset *a, int reset) { return RefillAudioCB(a, reset); }

  static AVFormatContext *Load(AVIOContext *pb, const string &filename, char *probe_buf, int probe_buflen, AVIOContext **pbOut=0) {
    AVProbeData probe_data;
    memzero(probe_data);
    probe_data.filename = BaseName(filename);

    bool probe_buf_data = probe_buf && probe_buflen;
    if (probe_buf_data) {
      probe_data.buf = MakeUnsigned(probe_buf);
      probe_data.buf_size = probe_buflen;
    }

    AVInputFormat *fmt = av_probe_input_format(&probe_data, probe_buf_data);
    if (!fmt) return ERRORv(nullptr, "no AVInputFormat for ", probe_data.filename);

    AVFormatContext *fctx = avformat_alloc_context(); int ret;
    fctx->flags |= AVFMT_FLAG_CUSTOM_IO;

    fctx->pb = pb;
    if ((ret = avformat_open_input(&fctx, probe_data.filename, fmt, 0))) {
      char errstr[128]; av_strerror(ret, errstr, sizeof(errstr));
      return ERRORv(nullptr, "av_open_input ", probe_data.filename, ": ", ret, " ", errstr);
    }

    if (pbOut) *pbOut = fctx->pb;
    return fctx;
  }

  static int RefillAudioCB(SoundAsset *a, int reset) {
    AVFormatContext *fctx = FromVoid<AVFormatContext*>(a->handle);
    AVCodecContext *avctx = fctx->streams[a->handle_arg1]->codec;
    bool open_resampler = false;
    if (reset) {
      av_seek_frame(fctx, a->handle_arg1, 0, AVSEEK_FLAG_BYTE);
      a->wav->ring.size = SoundAsset::Size(a);
      a->wav->bytes = a->wav->ring.size * a->wav->width;
      open_resampler = true;
    }
    if (!a->wav) {
      a->wav = make_unique<RingBuf>(a->sample_rate, SoundAsset::Size(a));
      open_resampler = true;
    }
    if (open_resampler)
      a->resampler.Open(a->wav.get(), avctx->channels, avctx->sample_rate, Sample::FromFFMpegId(avctx->sample_fmt),
                        a->channels, a->sample_rate, Sample::S16);
    a->wav->ring.back = 0;
    int wrote = PlayMovie(a, 0, fctx, 0);
    if (wrote < SoundAsset::Size(a)) {
      a->wav->ring.size = wrote;
      a->wav->bytes = a->wav->ring.size * a->wav->width;
    }
    return wrote;
  }

  static void LoadMovie(SoundAsset *sa, Asset *va, AVFormatContext *fctx) {
    if (avformat_find_stream_info(fctx, 0) < 0) return ERROR("av_find_stream_info");

    int audio_index = -1, video_index = -1, got=0;
    for (int i=0; i<fctx->nb_streams; i++) {
      AVStream *st = fctx->streams[i];
      AVCodecContext *avctx = st->codec;
      if (avctx->codec_type == AVMEDIA_TYPE_AUDIO) audio_index = i;
      if (avctx->codec_type == AVMEDIA_TYPE_VIDEO) video_index = i;
    }
    if (va) {
      if (video_index < 0) return ERROR("no v-stream: ", fctx->nb_streams);
      AVCodecContext *avctx = fctx->streams[video_index]->codec;
      AVCodec *codec = avcodec_find_decoder(avctx->codec_id);
      if (!codec || avcodec_open2(avctx, codec, 0) < 0) return ERROR("avcodec_open2: ", codec);
      va->tex.CreateBacked(avctx->width, avctx->height, Pixel::BGR32);
    }
    if (sa) {
      if (audio_index < 0) return ERROR("no a-stream: ", fctx->nb_streams);
      AVCodecContext *avctx = fctx->streams[audio_index]->codec;
      AVCodec *codec = avcodec_find_decoder(avctx->codec_id);
      if (!codec || avcodec_open2(avctx, codec, 0) < 0) return ERROR("avcodec_open2: ", codec);

      sa->handle = fctx;
      sa->handle_arg1 = audio_index;
      sa->channels = avctx->channels == 1 ? 1 : FLAGS_chans_out;
      sa->sample_rate = FLAGS_sample_rate;
      sa->seconds = FLAGS_soundasset_seconds;
      sa->wav = make_unique<RingBuf>(sa->sample_rate, SoundAsset::Size(sa));
      sa->resampler.Open(sa->wav.get(), avctx->channels, avctx->sample_rate, Sample::FromFFMpegId(avctx->sample_fmt),
                         sa->channels, sa->sample_rate, Sample::S16);
    }
  }

  static int PlayMovie(SoundAsset *sa, Asset *va, AVFormatContext *fctx, int seek_unused) {
    int begin_resamples_available = sa->resampler.output_available, wrote=0, done=0;
    Allocator *tlsalloc = ThreadLocalStorage::GetAllocator();

    while (!done && wrote != SoundAsset::Size(sa)) {
      AVPacket packet; int ret, got=0;
      if ((ret = av_read_frame(fctx, &packet)) < 0) {
        char errstr[128]; av_strerror(ret, errstr, sizeof(errstr));
        ERROR("av_read_frame: ", errstr); break;
      }

      AVFrame *frame = av_frame_alloc();
      AVCodecContext *avctx = fctx->streams[packet.stream_index]->codec;
      if (sa && packet.stream_index == sa->handle_arg1) {
        if ((ret = avcodec_decode_audio4(avctx, frame, &got, &packet)) <= 0 || !got) {
          char errstr[128]; av_strerror(ret, errstr, sizeof(errstr));
          ERROR("avcodec_decode_audio4: ", errstr);
        } else {
          tlsalloc->Reset();
          short *rsout = FromVoid<short*>(tlsalloc->Malloc(AVCODEC_MAX_AUDIO_FRAME_SIZE + FF_INPUT_BUFFER_PADDING_SIZE));

          int sampsIn  = frame->nb_samples;
          int sampsOut = max(0, SoundAsset::Size(sa) - wrote) / sa->channels;

          sa->resampler.Update(sampsIn, FromVoid<const short**>(frame->extended_data), rsout, Time(-1), sampsOut);
          wrote = sa->resampler.output_available - begin_resamples_available;
          av_frame_unref(frame);
        }
      } else if (va) {
        if ((ret = avcodec_decode_video2(avctx, frame, &got, &packet)) <= 0 || !got) {
          char errstr[128]; av_strerror(ret, errstr, sizeof(errstr));
          ERROR("avcodec_decode_video2 ", ret, ": ", errstr);
        } else {
          screen->gd->BindTexture(GraphicsDevice::Texture2D, va->tex.ID);
          va->tex.UpdateBuffer(*frame->data, point(avctx->width, avctx->height), Pixel::FromFFMpegId(avctx->pix_fmt), 
                               frame->linesize[0], Texture::Flag::Resample);
          va->tex.UpdateGL();
          av_frame_unref(frame);
          done = 1;
        }
      }
      av_free_packet(&packet);
      av_frame_free(&frame);
    }
    return wrote;
  }
};
#else
struct FFMpegAssetLoader {};
#endif

struct AliasWavefrontObjLoader {
  typedef map<string, Material> MtlMap;
  MtlMap MaterialMap;
  string dir;

  unique_ptr<Geometry> LoadOBJ(const string &filename, const float *map_tex_coord=0) {
    dir.assign(filename, 0, DirNameLen(filename, true));

    LocalFile file(filename, "r");
    if (!file.Opened()) return ERRORv(nullptr, "obj.open: ", file.Filename());

    NextRecordReader nr(&file);
    vector<v3> vert, vert_out, norm, norm_out;
    vector<v2> tex, tex_out;
    Material mat; v3 xyz; v2 xy;
    int format=0, material=0, ind[3];

    for (const char *line = nr.NextLine(); line; line = nr.NextLine()) {
      if (!line[0] || line[0] == '#') continue;
      StringWordIter word(line);
      string cmd = word.NextString();
      if (word.Done()) continue;

      if      (cmd == "mtllib") LoadMaterial(word.NextString());
      else if (cmd == "usemtl") {
        MtlMap::iterator it = MaterialMap.find(word.NextString());
        if (it != MaterialMap.end()) { mat = (*it).second; material = 1; }
      }
      else if (cmd == "v" ) { word.ScanN(&xyz[0], 3);             vert.push_back(xyz); }
      else if (cmd == "vn") { word.ScanN(&xyz[0], 3); xyz.Norm(); norm.push_back(xyz); }
      else if (cmd == "vt") {
        word.ScanN(&xy[0], 2);
        if (map_tex_coord) {
          xy.x = map_tex_coord[Texture::minx_coord_ind] + xy.x * (map_tex_coord[Texture::maxx_coord_ind] - map_tex_coord[Texture::minx_coord_ind]);
          xy.y = map_tex_coord[Texture::miny_coord_ind] + xy.y * (map_tex_coord[Texture::maxy_coord_ind] - map_tex_coord[Texture::miny_coord_ind]);
        }
        tex.push_back(xy);
      }
      else if (cmd == "f") {
        vector<v3> face; 
        for (const char *fi = word.Next(); fi; fi = word.Next()) {
          StringWordIter indexword(fi, word.cur_len, isint<'/'>);
          indexword.ScanN(ind, 3);

          int count = (ind[0]!=0) + (ind[1]!=0) + (ind[2]!=0);
          if (!count) continue;
          if (!format) format = count;
          if (format != count) return ERRORv(nullptr, "face format ", format, " != ", count);

          face.push_back(v3(ind[0], ind[1], ind[2]));
        }
        if (face.size() && face.size() < 2) return ERRORv(nullptr, "face size ", face.size());

        int next = 1;
        for (int i=0,j=0; i<face.size(); i++,j++) {
          unsigned ind[3] = { unsigned(face[i].x), unsigned(face[i].y), unsigned(face[i].z) };

          if (ind[0] > int(vert.size())) return ERRORv(nullptr, "index error 1 ", ind[0], ", ", vert.size());
          if (ind[1] > int(tex .size())) return ERRORv(nullptr, "index error 2 ", ind[1], ", ", tex.size());
          if (ind[2] > int(norm.size())) return ERRORv(nullptr, "index error 3 ", ind[2], ", ", norm.size());

          if (ind[0]) vert_out.push_back(vert[ind[0]-1]);
          if (ind[1])  tex_out.push_back(tex [ind[1]-1]);
          if (ind[2]) norm_out.push_back(norm[ind[2]-1]);

          if (i == face.size()-1) break;
          if (!i) i = next-1;
          else if (j == 2) { next=i; i=j=(-1); }
        }
      }
    }

    unique_ptr<Geometry> ret =
      make_unique<Geometry>(GraphicsDevice::Triangles, vert_out.size(), &vert_out[0],
                            norm_out.size() ? &norm_out[0] : 0, tex_out.size() ? &tex_out[0] : 0);
    ret->material = material;
    ret->mat = mat;

    INFO("asset(", filename, ") ", ret->count, " verts ");
    return ret;
  }

  int LoadMaterial(const string &filename) {
    LocalFile file(StrCat(dir, filename), "r");
    if (!file.Opened()) return ERRORv(-1, "LocalFile::open(", file.Filename(), ")");

    NextRecordReader nr(&file);
    Material m;
    string name;
    for (const char *line = nr.NextLine(); line; line = nr.NextLine()) {
      if (!line[0] || line[0] == '#') continue;
      StringWordIter word(line);
      string cmd = word.NextString();
      if (word.Done()) continue;

      if      (cmd == "Ka") word.ScanN(m.ambient .x, 4);
      else if (cmd == "Kd") word.ScanN(m.diffuse .x, 4);
      else if (cmd == "Ks") word.ScanN(m.specular.x, 4);
      else if (cmd == "Ke") word.ScanN(m.emissive.x, 4);
      else if (cmd == "newmtl") {
        if (name.size()) MaterialMap[name] = m;
        name = word.Next();
      }
    }
    if (name.size()) MaterialMap[name] = m;
    return 0;
  }
};

unique_ptr<Geometry> Geometry::LoadOBJ(const string &filename, const float *map_tex_coord) {
  return AliasWavefrontObjLoader().LoadOBJ(filename, map_tex_coord);
}

string Geometry::ExportOBJ(const Geometry *geom, const set<int> *prim_filter, bool prim_filter_invert) {
  int vpp = GraphicsDevice::VertsPerPrimitive(geom->primtype), prims = geom->count / vpp;
  int vert_attr = 1 + (geom->norm_offset >= 0) + (geom->tex_offset >= 0);
  CHECK_EQ(geom->count, prims * vpp);
  string vert_prefix, verts, faces;

  for (int offset = 0, d = 0, f = 0; f < 3; f++, offset += d) {
    if      (f == 0) { vert_prefix = "v";  d = geom->vd; }
    else if (f == 1) { vert_prefix = "vn"; d = geom->vd; if (geom->norm_offset < 0) { d=0; continue; } }
    else if (f == 2) { vert_prefix = "vt"; d = geom->td; if (geom-> tex_offset < 0) { d=0; continue; } }

    for (int prim = 0, prim_filtered = 0; prim < prims; prim++) {
      if (prim_filter) {
        bool in_filter = Contains(*prim_filter, prim);
        if (in_filter == prim_filter_invert) { prim_filtered++; continue; }
      }

      StrAppend(&faces, "f ");
      for (int i = 0; i < vpp; i++) {
        StrAppend(&verts, vert_prefix, " ");
        int vert_ind = prim * vpp + i, float_ind = vert_ind * geom->width + offset;
        for (int j = 0; j < d; j++) StringAppendf(&verts, "%s%f", j?" ":"", geom->vert[float_ind + j]);
        StrAppend(&verts, "\n");

        int out_vert_ind = (prim - prim_filtered) * vpp + i + 1;
        for (int j = 0; j < vert_attr; j++) StrAppend(&faces, j?"/":"", out_vert_ind);
        StrAppend(&faces, i == vpp-1 ? "\n" : " ");
      }
    }
  }
  return StrCat(verts, "\n", faces);
}

void Geometry::SetPosition(const float *v) {
  vector<float> delta(vd);
  if (Vec<float>::Equals(v, &last_position[0], vd)) return;
  for (int j = 0; j<vd; j++) delta[j] = v[j] - last_position[j];
  for (int j = 0; j<vd; j++) last_position[j] = v[j];
  for (int i = 0; i<count; i++) for (int j = 0; j<vd; j++) vert[i*width+j] += delta[j];
}

void Geometry::ScrollTexCoord(float dx, float dx_extra, int *subtract_max_int) {
  int vpp = GraphicsDevice::VertsPerPrimitive(primtype), prims = count / vpp, max_int = 0;
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
  screen->gd->VertexPointer(vd, GraphicsDevice::Float, width_bytes, 0, &vert[0], vert_size, &vert_ind, true, primtype);
}

AssetLoader::AssetLoader() {}
AssetLoader::~AssetLoader() {}

int AssetLoader::Init() {
  simple_loader = make_unique<SimpleAssetLoader>();
#ifdef LFL_FFMPEG
  ffmpeg_loader = make_unique<FFMpegAssetLoader>();
  default_audio_loader = ffmpeg_loader.get();
  default_video_loader = ffmpeg_loader.get();
  default_movie_loader = ffmpeg_loader.get();
#endif
  if (!default_audio_loader || !default_video_loader || !default_movie_loader) {
    if (!default_audio_loader) default_audio_loader = simple_loader.get();
    if (!default_video_loader) default_video_loader = simple_loader.get();
    if (!default_movie_loader) default_movie_loader = simple_loader.get();
  }
#ifdef LFL_ANDROID
  default_audio_loader = (android_audio_loader = make_unique<AndroidAudioAssetLoader>()).get();
#endif
#ifdef LFL_IPHONE
  default_audio_loader = (iphone_audio_loader = make_unique<iPhoneAudioAssetLoader>()).get();
#endif
  return 0;
}

string Asset::FileName(const string &asset_fn) { return StrCat(app->assetdir, asset_fn); }

string Asset::FileContents(const string &asset_fn) {
  auto i = app->asset_cache.find(asset_fn);
  if (i != app->asset_cache.end()) return string(i->second.data(), i->second.size());
  else                             return LocalFile::FileContents(Asset::FileName(asset_fn));
}

File *Asset::OpenFile(const string &asset_fn) {
  auto i = app->asset_cache.find(asset_fn);
  if (i != app->asset_cache.end()) return new BufferFile(i->second);
  else                             return new LocalFile(Asset::FileName(asset_fn), "r");
}

void Asset::Unload() {
  if (parent) parent->Unloaded(this);
  if (tex.ID && screen) tex.ClearGL();
  if (geometry) { delete geometry; geometry = 0; }
  if (hull)     { delete hull;     hull     = 0; }
}

void Asset::Load(void *h, VideoAssetLoader *l) {
  static int next_asset_type_id = 1, next_list_id = 1;
  if (!name.empty()) typeID = next_asset_type_id++;
  if (!geom_fn.empty()) geometry = Geometry::LoadOBJ(StrCat(app->assetdir, geom_fn)).release();
  if (!texture.empty() || h) LoadTexture(h, texture, &tex, l);
}

void Asset::LoadTexture(void *h, const string &asset_fn, Texture *out, VideoAssetLoader *l) {
  if (!FLAGS_lfapp_video) return;
  auto i = app->asset_cache.find(asset_fn);
  if (i != app->asset_cache.end()) return LoadTexture(i->second.data(), asset_fn.c_str(), i->second.size(), out);
  if (!l) l = app->asset_loader->default_video_loader;
  void *handle = h ? h : l->LoadVideoFile(asset_fn[0] == '/' ? asset_fn : StrCat(app->assetdir, asset_fn).c_str());
  if (!handle) return ERROR("load: ", asset_fn);
  l->LoadVideo(handle, out);
  if (!h) l->UnloadVideoFile(handle);
}

void Asset::LoadTexture(const void *FromBuf, const char *filename, int size, Texture *out, int flag) {
  VideoAssetLoader *l = app->asset_loader->default_video_loader;
  // work around ffmpeg image2 format being AVFMT_NO_FILE; ie doesnt work with custom AVIOContext
  if (FileSuffix::Image(filename)) l = app->asset_loader->simple_loader.get();
  void *handle = l->LoadVideoBuf(FromVoid<const char*>(FromBuf), size, filename);
  if (!handle) return;
  l->LoadVideo(handle, out, flag);
  l->UnloadVideoBuf(handle);
}

Texture *Asset::LoadTexture(const MultiProcessFileResource &file, int max_image_size) {
  unique_ptr<Texture> tex = make_unique<Texture>();
  Asset::LoadTexture(file.buf.data(), file.name.data(), file.buf.size(), tex.get(), 0);

  if (tex->BufferSize() >= max_image_size) {
    unique_ptr<Texture> orig_tex = move(tex);
    tex = make_unique<Texture>();
    float scale_factor = sqrt(float(max_image_size)/orig_tex->BufferSize());
    tex->Resize(orig_tex->width*scale_factor, orig_tex->height*scale_factor, Pixel::RGB24, Texture::Flag::CreateBuf);

    VideoResampler resampler;
    resampler.Open(orig_tex->width, orig_tex->height, orig_tex->pf, tex->width, tex->height, tex->pf);
    resampler.Resample(orig_tex->buf, orig_tex->LineSize(), tex->buf, tex->LineSize());
  }

  if (!tex->buf || !tex->width || !tex->height) return 0;
  return tex.release();
}

void SoundAsset::Unload() {
  if (parent) parent->Unloaded(this);
  wav.reset();
  if (handle) ERROR("leak: ", handle);
}

void SoundAsset::Load(void *handle, const char *FN, int Secs, int flag) {
  if (!FLAGS_lfapp_audio) return ERROR("load: ", FN, ": lfapp_audio = ", FLAGS_lfapp_audio);
  if (handle) app->asset_loader->default_audio_loader->LoadAudio(handle, this, Secs, flag);
}

void SoundAsset::Load(void const *buf, int len, char const *FN, int Secs) {
  void *handle = app->asset_loader->default_audio_loader->LoadAudioBuf(FromVoid<const char*>(buf), len, FN);
  if (!handle) return;

  Load(handle, FN, Secs, FlagNoRefill);
  if (!refill) {
    app->asset_loader->default_audio_loader->UnloadAudioBuf(handle);
    handle = 0;
  }
}

void SoundAsset::Load(int Secs, bool unload) {
  string fn;
  void *handle = 0;

  if (!filename.empty()) {
    fn = filename;
    if (fn.length() && isalpha(fn[0])) fn = app->assetdir + fn;

    handle = app->asset_loader->default_audio_loader->LoadAudioFile(fn);
    if (!handle) ERROR("SoundAsset::Load ", fn);
  }

  Load(handle, fn.c_str(), Secs);

#if !defined(LFL_IPHONE) && !defined(LFL_ANDROID) /* XXX */
  if (!refill && handle && unload) {
    app->asset_loader->default_audio_loader->UnloadAudioFile(handle);
    handle = 0;
  }
#endif
}

int SoundAsset::Refill(int reset) { return app->asset_loader->default_audio_loader->RefillAudio(this, reset); }

void MovieAsset::Load(const char *fn) {
  if (!fn || !(handle = app->asset_loader->default_movie_loader->LoadMovieFile(StrCat(app->assetdir, fn)))) return;
  app->asset_loader->default_movie_loader->LoadMovie(handle, this);
  audio.Load();
  video.Load();
}

int MovieAsset::Play(int seek) {
  app->asset_loader->movie_playing = this; int ret;
  if ((ret = app->asset_loader->default_movie_loader->PlayMovie(this, 0) <= 0)) app->asset_loader->movie_playing = 0;
  return ret;
}

/* asset impls */

void glLine(const point &p1, const point &p2, const Color *color) {
  static int verts_ind=-1;
  screen->gd->DisableTexture();

  float verts[] = { /*1*/ float(p1.x), float(p1.y), /*2*/ float(p2.x), float(p2.y) };
  screen->gd->VertexPointer(2, GraphicsDevice::Float, 0, 0, verts, sizeof(verts), &verts_ind, true);

  if (color) screen->gd->Color4f(color->r(), color->g(), color->b(), color->a());
  screen->gd->DrawArrays(GraphicsDevice::LineLoop, 0, 2);
}

void glAxis(Asset*, Entity*) {
  static int vert_id=-1;
  const float scaleFactor = 1;
  const float range = powf(10, scaleFactor);
  const float step = range/10;

  screen->gd->DisableNormals();
  screen->gd->DisableTexture();

  float vert[] = { /*1*/ 0, 0, 0, /*2*/ step, 0, 0, /*3*/ 0, 0, 0, /*4*/ 0, step, 0, /*5*/ 0, 0, 0, /*6*/ 0, 0, step };
  screen->gd->VertexPointer(3, GraphicsDevice::Float, 0, 0, vert, sizeof(vert), &vert_id, false);

  /* origin */
  screen->gd->PointSize(7);
  screen->gd->Color4f(1, 1, 0, 1);
  screen->gd->DrawArrays(GraphicsDevice::Points, 0, 1);

  /* R=X */
  screen->gd->PointSize(5);
  screen->gd->Color4f(1, 0, 0, 1);
  screen->gd->DrawArrays(GraphicsDevice::Points, 1, 1);

  /* G=Y */
  screen->gd->Color4f(0, 1, 0, 1);
  screen->gd->DrawArrays(GraphicsDevice::Points, 3, 1);

  /* B=Z */
  screen->gd->Color4f(0, 0, 1, 1);
  screen->gd->DrawArrays(GraphicsDevice::Points, 5, 1);

  /* axis */
  screen->gd->PointSize(1);
  screen->gd->Color4f(.5, .5, .5, 1);
  screen->gd->DrawArrays(GraphicsDevice::Lines, 0, 5);
}

void glRoom(Asset*, Entity*) {
  screen->gd->DisableNormals();
  screen->gd->DisableTexture();

  static int verts_id=-1;
  float verts[] = {
    /*1*/ 0,0,0, /*2*/ 0,2,0, /*3*/ 2,0,0,
    /*4*/ 0,0,2, /*5*/ 0,2,0, /*6*/ 0,0,0,
    /*7*/ 0,0,0, /*8*/ 2,0,0, /*9*/ 0,0,2 
  };
  screen->gd->VertexPointer(3, GraphicsDevice::Float, 0, 0, verts, sizeof(verts), &verts_id, false);

  screen->gd->Color4f(.8,.8,.8,1);
  screen->gd->DrawArrays(GraphicsDevice::Triangles, 0, 3);

  screen->gd->Color4f(.7,.7,.7,1);
  screen->gd->DrawArrays(GraphicsDevice::Triangles, 3, 3);

  screen->gd->Color4f(.6,.6,.6,1);
  screen->gd->DrawArrays(GraphicsDevice::Triangles, 6, 3);;
}

void glIntersect(int x, int y, Color *c) {
  unique_ptr<Geometry> geom = make_unique<Geometry>(GraphicsDevice::Lines, 4, NullPointer<v2>(), NullPointer<v3>(), NullPointer<v2>(), *c);
  v2 *vert = reinterpret_cast<v2*>(&geom->vert[0]);

  vert[0] = v2(0, y);
  vert[1] = v2(screen->width, y);
  vert[2] = v2(x, 0);
  vert[3] = v2(x, screen->height);

  screen->gd->DisableTexture();
  Scene::Select(geom.get());
  Scene::Draw(geom.get(), 0);
}

void glShadertoyShader(Shader *shader, const Texture *tex) {
  float scale = shader->scale;
  screen->gd->UseShader(shader);
  shader->SetUniform1f("iGlobalTime", ToFSeconds(Now() - app->time_started).count());
  shader->SetUniform1f("iBlend", FLAGS_shadertoy_blend);
  shader->SetUniform4f("iMouse", screen->mouse.x, screen->mouse.y, app->input->MouseButton1Down(), 0);
  shader->SetUniform3f("iResolution", XY_or_Y(scale, screen->pow2_width), XY_or_Y(scale, screen->pow2_height), 0);
  if (tex) shader->SetUniform3f("iChannelResolution", XY_or_Y(scale, tex->width), XY_or_Y(scale, tex->height), 1);
}

void glShadertoyShaderWindows(Shader *shader, const Color &backup_color, const Box &w,                   const Texture *tex) { glShadertoyShaderWindows(shader, backup_color, vector<const Box*>(1, &w), tex); }
void glShadertoyShaderWindows(Shader *shader, const Color &backup_color, const vector<const Box*> &wins, const Texture *tex) {
  if (shader) glShadertoyShader(shader, tex);
  else screen->gd->SetColor(backup_color);
  if (tex) { screen->gd->EnableLayering(); tex->Bind(); }
  else screen->gd->DisableTexture();
  for (auto w : wins) w->Draw(tex ? tex->coord : 0);
  if (shader) screen->gd->UseShader(0);
}

void BoxFilled::Draw(const LFL::Box &b, const Drawable::Attr*) const { b.Draw(); }
void BoxOutline::Draw(const LFL::Box &b, const Drawable::Attr*) const {
  screen->gd->DisableTexture();
  if (line_width <= 1) {
    static int verts_ind = -1;
    float verts[] = { /*1*/ float(b.x),     float(b.y),     /*2*/ float(b.x),     float(b.y+b.h),
                      /*3*/ float(b.x+b.w), float(b.y+b.h), /*4*/ float(b.x+b.w), float(b.y) };
    screen->gd->VertexPointer(2, GraphicsDevice::Float, 0, 0, verts, sizeof(verts), &verts_ind, true);
    screen->gd->DrawArrays(GraphicsDevice::LineLoop, 0, 4);
  } else {
    int lw = line_width-1;
    static int verts_ind = -1;
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
    screen->gd->VertexPointer(2, GraphicsDevice::Float, 0, 0, verts, sizeof(verts), &verts_ind, true);
    screen->gd->DrawArrays(GraphicsDevice::TriangleStrip, 0, 16);
  }
}

void BoxTopLeftOutline::Draw(const LFL::Box &b, const Drawable::Attr*) const {
  screen->gd->DisableTexture();
  if (line_width <= 1) {
    static int verts_ind = -1;
    float verts[] = { /*1*/ float(b.x), float(b.y),     /*2*/ float(b.x),     float(b.y+b.h),
                      /*2*/ float(b.x), float(b.y+b.h), /*4*/ float(b.x+b.w), float(b.y) };
    screen->gd->VertexPointer(2, GraphicsDevice::Float, 0, 0, verts, sizeof(verts), &verts_ind, true);
    screen->gd->DrawArrays(GraphicsDevice::Lines, 0, 4);
  } else {
    int lw = line_width-1;
    static int verts_ind = -1;
    float verts[] = {
      /*1.1*/ float(b.x-lw),     float(b.y-lw),     /*1.2*/ float(b.x-lw),     float(b.y+b.h+lw),
      /*1.4*/ float(b.x),        float(b.y),        /*1.3*/ float(b.x),        float(b.y+b.h),
      /*2.1*/ float(b.x),        float(b.y+b.h),    /*2.2*/ float(b.x-lw),     float(b.y+b.h+lw),
      /*2.4*/ float(b.x+b.w),    float(b.y+b.h),    /*2.3*/ float(b.x+b.w+lw), float(b.y+b.h+lw),
    };
    screen->gd->VertexPointer(2, GraphicsDevice::Float, 0, 0, verts, sizeof(verts), &verts_ind, true);
    screen->gd->DrawArrays(GraphicsDevice::TriangleStrip, 0, 8);
  }
}

void BoxBottomRightOutline::Draw(const LFL::Box &b, const Drawable::Attr*) const {
  screen->gd->DisableTexture();
  if (line_width <= 1) {
    static int verts_ind = -1;
    float verts[] = { /*1*/ float(b.x),     float(b.y), /*4*/ float(b.x+b.w), float(b.y),
                      /*4*/ float(b.x+b.w), float(b.y), /*3*/ float(b.x+b.w), float(b.y+b.h) };
    screen->gd->VertexPointer(2, GraphicsDevice::Float, 0, 0, verts, sizeof(verts), &verts_ind, true);
    screen->gd->DrawArrays(GraphicsDevice::Lines, 0, 4);
  } else {
    int lw = line_width-1;
    static int verts_ind = -1;
    float verts[] = {
      /*3.3*/ float(b.x+b.w+lw), float(b.y+b.h+lw), /*3.4*/ float(b.x+b.w+lw), float(b.y-lw),
      /*3.2*/ float(b.x+b.w),    float(b.y+b.h),    /*3.1*/ float(b.x+b.w),    float(b.y),
      /*4.3*/ float(b.x+b.w),    float(b.y),        /*4.4*/ float(b.x+b.w+lw), float(b.y-lw),
      /*4.2*/ float(b.x),        float(b.y),        /*4.1*/ float(b.x-lw),     float(b.y-lw)
    };
    screen->gd->VertexPointer(2, GraphicsDevice::Float, 0, 0, verts, sizeof(verts), &verts_ind, true);
    screen->gd->DrawArrays(GraphicsDevice::TriangleStrip, 0, 16);
  }
}

Waveform::Waveform(point dim, const Color *c, const Vec<float> *sbh) : width(dim.x), height(dim.y) {
  float xmax=sbh->Len(), ymin=INFINITY, ymax=-INFINITY;
  for (int i=0; i<xmax; i++) {
    ymin = min(ymin,sbh->Read(i));
    ymax = max(ymax,sbh->Read(i));
  }

  geom = make_unique<Geometry>(GraphicsDevice::Lines, int(xmax-1)*2, NullPointer<v2>(), NullPointer<v3>(), NullPointer<v2>(), *c);
  v2 *vert = reinterpret_cast<v2*>(&geom->vert[0]);

  for (int i=0; i<geom->count/2; i++) {
    float x1=(i)  *1.0/xmax, y1=(sbh->Read(i)   - ymin)/(ymax - ymin); 
    float x2=(i+1)*1.0/xmax, y2=(sbh->Read(i+1) - ymin)/(ymax - ymin);

    vert[i*2+0] = v2(x1 * dim.x, -dim.y + y1 * dim.y);
    vert[i*2+1] = v2(x2 * dim.x, -dim.y + y2 * dim.y);
  }
}

void Waveform::Draw(const LFL::Box &w, const Drawable::Attr *a) const {
  if (!geom) return;
  geom->SetPosition(w.Position());
  screen->gd->DisableTexture();
  Scene::Select(geom.get());
  Scene::Draw(geom.get(), 0);
}

Waveform Waveform::Decimated(point dim, const Color *c, const RingBuf::Handle *sbh, int decimateBy) {
  unique_ptr<RingBuf> dsb;
  RingBuf::Handle dsbh;
  if (decimateBy) {
    dsb = unique_ptr<RingBuf>(Decimate(sbh, decimateBy));
    dsbh = RingBuf::Handle(dsb.get());
    sbh = &dsbh;
  }
  Waveform WF(dim, c, sbh);
  return WF;
}

void glSpectogram(Matrix *m, unsigned char *data, int pf, int width, int height, int hjump, float vmax, float clip, bool interpolate, int pd) {
  int ps = Pixel::size(pf);
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

void glSpectogram(Matrix *m, Texture *t, float *max, float clip, int pd) {
  if (!t->ID) t->CreateBacked(m->N, m->M);
  else {
    if (t->width < m->N || t->height < m->M) t->Resize(m->N, m->M);
    else Texture::Coordinates(t->coord, m->N, m->M, NextPowerOfTwo(t->width), NextPowerOfTwo(t->height));
  }

  if (clip == -INFINITY) clip = -65;
  if (!t->buf) return;

  float Max = Matrix::Max(m);
  if (max) *max = Max;

  glSpectogram(m, t->buf, t->pf, m->M, m->N, t->width, Max, clip, 0, pd);

  screen->gd->BindTexture(GraphicsDevice::Texture2D, t->ID);
  t->UpdateGL();
}

void glSpectogram(const RingBuf::Handle *in, Texture *t, Matrix *transform, float *max, float clip) {
  /* 20*log10(abs(specgram(y,2048,sr,hamming(512),256))) */
  Matrix *m = Spectogram(in, 0, 512, 256, 512, vector<double>(), PowerDomain::abs);
  if (transform) m = Matrix::Mult(m, transform, mDelA);
  glSpectogram(m, t, max, clip, PowerDomain::abs);
  delete m;
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

  return make_unique<Geometry>(GraphicsDevice::Triangles, verts.size(), &verts[0], normals ? &norms[0] : 0, nullptr, nullptr);
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
  return make_unique<Geometry>(GraphicsDevice::Triangles, verts.size(), &verts[0], nullptr, &tex[0]);
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
  return make_unique<Geometry>(GraphicsDevice::Triangles, verts.size(), &verts[0], nullptr, &tex[0]);
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
  return make_unique<Geometry>(GraphicsDevice::Triangles, verts.size(), &verts[0], nullptr, &tex[0]);
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
  return make_unique<Geometry>(GraphicsDevice::Triangles, verts.size(), &verts[0], nullptr, &tex[0]);
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
  return make_unique<Geometry>(GraphicsDevice::Triangles, verts.size(), &verts[0], nullptr, &tex[0]);
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
  return make_unique<Geometry>(GraphicsDevice::Triangles, verts.size(), &verts[0], nullptr, &tex[0]);
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
  return make_unique<Geometry>(GraphicsDevice::Lines, verts.size(), &verts[0], nullptr, nullptr, Color(.5,.5,.5));
}

unique_ptr<Geometry> Grid::Grid2D(float x, float y, float range, float step) {
  vector<v2> verts;
  for (float d = 0; d <= range; d += step) {
    verts.push_back(v2(x+range, y+d));
    verts.push_back(v2(x,       y+d));
    verts.push_back(v2(x+d, y+range));
    verts.push_back(v2(x+d, y));
  }
  return make_unique<Geometry>(GraphicsDevice::Lines, verts.size(), &verts[0], nullptr, nullptr, Color(0,0,0));
}

void TextureArray::Load(const string &fmt, const string &prefix, const string &suffix, int N) {
  a.resize(N);
  for (int i=0, l=a.size(); i<l; i++)
    Asset::LoadTexture(StringPrintf(fmt.c_str(), prefix.c_str(), i, suffix.c_str()), &a[i]);
}

void TextureArray::DrawSequence(Asset *out, Entity *e, int *ind) {
  *ind = (*ind + 1) % a.size();
  const Texture *in = &a[*ind];
  out->tex.ID = in->ID;
  if (out->geometry) Scene::Draw(out->geometry, e);
}

Skybox::Skybox() :
  a_left  ("", "", 1, 0, 0, Cube::Create(500, 500, 500).release(), 0, CubeMap::PX, TexGen::LINEAR),
  a_right ("", "", 1, 0, 0, 0,                                     0, CubeMap::NX, 0),
  a_top   ("", "", 1, 0, 0, 0,                                     0, CubeMap::PY, 0),
  a_bottom("", "", 1, 0, 0, 0,                                     0, CubeMap::NY, 0),
  a_front ("", "", 1, 0, 0, 0,                                     0, CubeMap::PZ, 0),
  a_back  ("", "", 1, 0, 0, 0,                                     0, CubeMap::NZ, 0),
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

void Skybox::Draw() {
  screen->gd->DisableNormals();
  screen->gd->DisableVertexColor();
  screen->gd->DisableDepthTest();
  Scene::Draw(&a_left,   0, v_left);
  Scene::Draw(&a_right,  0, v_right);
  Scene::Draw(&a_top,    0, v_top);
  Scene::Draw(&a_bottom, 0, v_bottom);
  Scene::Draw(&a_front,  0, v_front);
  Scene::Draw(&a_back,   0, v_back);
  screen->gd->EnableDepthTest();
}

void LayersInterface::Update() {
  for (auto &i : this->layer) i->Run(TilesInterface::RunFlag::ClearEmpty);
  if (app->main_process) app->main_process->SwapTree(0, this);
}

void LayersInterface::Draw(const Box &b, const point &p) {
  if (layer.size() > 0) layer[0]->Draw(b, p + node[0].scrolled);
  for (int c=node[0].child_offset; c; c=child[c-1].next_child_offset) {
    const Node *n = &node[child[c-1].node_id-1];
    Scissor s(screen->gd, n->box + b.TopLeft());
    if (layer.size() > n->layer_id) layer[n->layer_id]->Draw(b, p + n->scrolled);
  }
}

void TilesInterface::AddDrawableBoxArray(const DrawableBoxArray &box, point p) {
  for (DrawableBoxIterator iter(box.data); !iter.Done(); iter.Increment()) {
    const Drawable::Attr *attr = box.attr.GetAttr(iter.cur_attr1);
    bool attr_set = 0;
    if (attr->bg) {
      ContextOpen();
      if (!attr_set && (attr_set=1)) SetAttr(attr);
      InitDrawBackground(p);
      DrawableBoxRun(iter.Data(), iter.Length(), attr, VectorGet(box.line, iter.cur_attr2))
        .DrawBackground(p, bind(&TilesInterface::DrawBackground, this, _1));
      ContextClose();
    }
    if (1) {
      ContextOpen();
      if (!attr_set && (attr_set=1)) SetAttr(attr);
      InitDrawBox(p);
      DrawableBoxRun(iter.Data(), iter.Length(), attr).Draw(p, bind(&TilesInterface::DrawBox, this, _1, _2, _3));
      ContextClose();
    }
  }
}

void Tiles::InitDrawBox(const point &p) {
  PreAdd(bind(&DrawableBoxRun::draw, DrawableBoxRun(0,0,attr), p));
  if (attr->scissor) AddScissor(*attr->scissor + p);
}
void Tiles::InitDrawBackground(const point &p) {
  PreAdd(bind(&DrawableBoxRun::DrawBackground,
              DrawableBoxRun(0,0,attr), p, &DrawableBoxRun::DefaultDrawBackgroundCB));
}
void Tiles::DrawBox(const Drawable *d, const Box &b, const Drawable::Attr*) {
  AddCallback(&b, bind(&Drawable::Draw, d, b, attr));
}
void Tiles::DrawBackground(const Box &b) {
  AddCallback(&b, bind(&Box::Draw, b, NullPointer<float>()));
}
void Tiles::AddScissor(const Box &b) {
  PreAdd(bind(&Tiles::PushScissor, this, b));
  PostAdd(bind(&GraphicsDevice::PopScissor, screen->gd));
}

#ifdef  LFL_TILES_IPC_DEBUG
#define TilesIPCDebug(...) printf(__VA_ARGS__)
#else
#define TilesIPCDebug(...)
#endif

void TilesIPC::SetAttr(const Drawable::Attr *a) {
  attr = a;
  prepend[context_depth]->cb.Add(MultiProcessPaintResource::SetAttr(*attr));
  TilesIPCDebug("TilesIPC SetAttr %s\n", a->DebugString().c_str());
}
void TilesIPC::InitDrawBox(const point &p) {
  prepend[context_depth]->cb.Add(MultiProcessPaintResource::InitDrawBox(p));
  if (attr->scissor) AddScissor(*attr->scissor + p);
  TilesIPCDebug("TilesIPC InitDrawBox %s\n", p.DebugString().c_str());
}
void TilesIPC::InitDrawBackground(const point &p) {
  prepend[context_depth]->cb.Add(MultiProcessPaintResource::InitDrawBackground(p));
  TilesIPCDebug("TilesIPC InitDrawBackground %s\n", p.DebugString().c_str());
}
void TilesIPC::DrawBox(const Drawable *d, const Box &b, const Drawable::Attr*) {
  if (d) AddCallback(&b, MultiProcessPaintResource::DrawBox(b, d->TexId()));
  TilesIPCDebug("TilesIPC DrawBox %s\n", b.DebugString().c_str());
}
void TilesIPC::DrawBackground(const Box &b) {
  AddCallback(&b, MultiProcessPaintResource::DrawBackground(b));
  TilesIPCDebug("TilesIPC DrawBackground %s\n", b.DebugString().c_str());
}
void TilesIPC::AddScissor(const Box &b) {
  prepend[context_depth]->cb.Add(MultiProcessPaintResource::PushScissor(b));
  append [context_depth]->cb.Add(MultiProcessPaintResource::PopScissor());
  TilesIPCDebug("TilesIPC AddScissor %s\n", b.DebugString().c_str());
}

void TilesIPCClient::Run(int flag) {
  bool clear_empty = (flag & RunFlag::ClearEmpty);
  TilesMatrixIter(&mat) {
    if (!tile->cb.Count() && !clear_empty) continue;
    app->main_process->Paint(layer, point(j, i), flag, tile->cb);
  }
}

int MultiProcessPaintResource::Run(const Box &t) const {
  ProcessAPIClient *s = CheckPointer(app->render_process.get());
  Iterator i(data.buf);
  int si=0, sd=0, count=0; 
  TilesIPCDebug("MPPR Begin\n");
  for (; i.offset + sizeof(int) <= data.size(); count++) {
    int type = *i.Get<int>();
    switch (type) {
      default:                       FATAL("unknown type ", type);
      case SetAttr           ::Type: { auto c=i.Get<SetAttr>           (); c->Update(&attr, app->render_process.get());      i.offset += SetAttr           ::Size; TilesIPCDebug("MPPR SetAttr %s\n",     attr.DebugString().c_str()); } break;
      case InitDrawBox       ::Type: { auto c=i.Get<InitDrawBox>       (); DrawableBoxRun(0,0,&attr).draw(c->p);             i.offset += InitDrawBox       ::Size; TilesIPCDebug("MPPR InitDrawBox %s\n", c->p.DebugString().c_str()); } break;
      case InitDrawBackground::Type: { auto c=i.Get<InitDrawBackground>(); DrawableBoxRun(0,0,&attr).DrawBackground(c->p);   i.offset += InitDrawBackground::Size; TilesIPCDebug("MPPR InitDrawBG %s\n",  c->p.DebugString().c_str()); } break;
      case DrawBackground    ::Type: { auto c=i.Get<DrawBackground>    (); c->b.Draw();                                      i.offset += DrawBackground    ::Size; TilesIPCDebug("MPPR DrawBG %s\n",      c->b.DebugString().c_str()); } break;
      case PushScissor       ::Type: { auto c=i.Get<PushScissor>       (); screen->gd->PushScissorOffset(t, c->b);     si++; i.offset += PushScissor       ::Size; TilesIPCDebug("MPPR PushScissor %s\n", c->b.DebugString().c_str()); } break;
      case PopScissor        ::Type: { auto c=i.Get<PopScissor>        (); screen->gd->PopScissor(); CHECK_LT(sd, si); sd++; i.offset += PopScissor        ::Size; TilesIPCDebug("MPPR PopScissor\n");                                 } break;
      case DrawBox           ::Type: { auto c=i.Get<DrawBox>           ();
                                       auto d=(c->id > 0 && c->id <= s->drawable.size()) ? s->drawable[c->id-1] : Singleton<BoxFilled>::Get();
                                       d->Draw(c->b, &attr); i.offset += DrawBox::Size; TilesIPCDebug("DrawBox MPPR %s\n", c->b.DebugString().c_str());
                                     } break;
    }
  }
  if (si != sd) { ERROR("mismatching scissor ", si, " != ", sd); for (int i=sd; i<si; ++i) screen->gd->PopScissor(); }
  TilesIPCDebug("MPPR End\n");
  return count;
}

void MultiProcessPaintResource::SetAttr::Update(Drawable::Attr *o, ProcessAPIClient *s) const {
  *o = Drawable::Attr((font_id > 0 && font_id <= s->font_table.size()) ? s->font_table[font_id-1] : NULL,
                      hfg ? &fg : NULL, hbg ? &bg : NULL, underline, blend);
  if (hs) o->scissor = &scissor;
  if (tex_id > 0 && tex_id <= s->drawable.size()) o->tex = dynamic_cast<Texture*>(s->drawable[tex_id-1]);
}

}; // namespace LFL
