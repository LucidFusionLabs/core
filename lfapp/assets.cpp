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
#include "lfapp/gui.h"

extern "C" {
#ifdef LFL_JPEG
#include "jpeglib.h"
#endif
};
#ifdef LFL_GIF
#include "gif_lib.h"
#endif

namespace LFL {
DEFINE_int(soundasset_seconds, 10, "Soundasset buffer seconds");

const int SoundAsset::FlagNoRefill = 1;
#ifdef LFL_FFMPEG
const int SoundAsset::FromBufPad = FF_INPUT_BUFFER_PADDING_SIZE;
#else
const int SoundAsset::FromBufPad = 0;
#endif

#ifdef LFL_PNG
int Pixel::FromPngId(int fmt) {
    switch (fmt) {
        case PNG_COLOR_TYPE_RGB:        return Pixel::RGB24;
        case PNG_COLOR_TYPE_RGBA:       return Pixel::RGBA;
        case PNG_COLOR_TYPE_GRAY:       return Pixel::GRAY8;
        case PNG_COLOR_TYPE_GRAY_ALPHA: return Pixel::GRAYA8;
        case PNG_COLOR_TYPE_PALETTE:    ERROR("not supported: ",     fmt); return 0;
        default:                        ERROR("unknown pixel fmt: ", fmt); return 0;
    }
}

int Pixel::ToPngId(int fmt) {
    switch (fmt) {
        case Pixel::RGB24:  return PNG_COLOR_TYPE_RGB; 
        case Pixel::RGBA:   return PNG_COLOR_TYPE_RGBA;
        case Pixel::GRAY8:  return PNG_COLOR_TYPE_GRAY; 
        case Pixel::GRAYA8: return PNG_COLOR_TYPE_GRAY_ALPHA;
        default:            ERROR("unknown pixel fmt: ", fmt); return 0;
    }
}

static void PngRead (png_structp png_ptr, png_bytep data, png_size_t length) { ((File*)png_get_io_ptr(png_ptr))->Read (data, length); }
static void PngWrite(png_structp png_ptr, png_bytep data, png_size_t length) { ((File*)png_get_io_ptr(png_ptr))->Write(data, length); }
static void PngFlush(png_structp png_ptr) {}

int PngReader::Read(File *lf, Texture *out) {
    char header[8];
    if (lf->Read(header, sizeof(header)) != sizeof(header)) { ERROR("read: ", lf->Filename()); return -1; }
    if (png_sig_cmp((png_byte*)header, 0, 8)) { ERROR("png_sig_cmp: ", lf->Filename()); return -1; }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
    if (!png_ptr) { ERROR("png_create_read_struct: ", lf->Filename()); return -1; }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) { ERROR("png_create_info_struct: ", lf->Filename()); png_destroy_read_struct(&png_ptr, 0, 0); return -1; }

    if (setjmp(png_jmpbuf(png_ptr)))
    { ERROR("png error: ", lf->Filename()); png_destroy_read_struct(&png_ptr, &info_ptr, 0); return -1; }
    png_set_read_fn(png_ptr, lf, PngRead);

    png_set_sig_bytes(png_ptr, sizeof(header));
    png_read_info(png_ptr, info_ptr);

    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    int number_of_passes = png_set_interlace_handling(png_ptr), pf;
    png_read_update_info(png_ptr, info_ptr);

    if (!(pf = Pixel::FromPngId(color_type))) { png_destroy_read_struct(&png_ptr, &info_ptr, 0); return -1; }
    out->Resize(png_get_image_width(png_ptr, info_ptr), png_get_image_height(png_ptr, info_ptr), pf, Texture::Flag::CreateBuf);

    int linesize = out->LineSize();
    CHECK_EQ(linesize, png_get_rowbytes(png_ptr, info_ptr));

    vector<png_bytep> row_pointers;
    for (int y=0; y<out->height; y++) row_pointers.push_back((png_bytep)(out->buf + linesize * y));

    png_read_image(png_ptr, &row_pointers[0]);
    png_destroy_read_struct(&png_ptr, &info_ptr, 0);
    return 0;
}

int PngWriter::Write(File *lf, const Texture &tex) {
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
    if (!png_ptr) { ERROR("png_create_read_struct: ", lf->Filename()); return -1; }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) { ERROR("png_create_info_struct: ", lf->Filename()); png_destroy_write_struct(&png_ptr, 0); return -1; }

    if (setjmp(png_jmpbuf(png_ptr)))
    { ERROR("setjmp: ", lf->Filename()); png_destroy_write_struct(&png_ptr, &info_ptr); return -1; }

    png_set_write_fn(png_ptr, lf, PngWrite, PngFlush);

    png_set_IHDR(png_ptr, info_ptr, tex.width, tex.height,
                 8, Pixel::ToPngId(tex.pf), PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    vector<png_bytep> row_pointers;
    for (int linesize=tex.LineSize(), y=0; y<tex.height; y++)
        row_pointers.push_back((png_bytep)(tex.buf + y*linesize));

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
    if (!lf.Opened()) { ERROR("open: ", fn); return -1; }
    return PngWriter::Write(&lf, tex);
}

#ifdef LFL_JPEG
const JOCTET EOI_BUFFER[1] = { JPEG_EOI };
struct MyJpegErrorMgr  { jpeg_error_mgr  pub; jmp_buf setjmp_buffer; };
struct MyJpegSourceMgr { jpeg_source_mgr pub; const JOCTET *data; size_t len; } ;
static void JpegErrorExit(j_common_ptr jcs) { longjmp(((MyJpegErrorMgr*)jcs->err)->setjmp_buffer, 1); }
static void JpegInitSource(j_decompress_ptr jds) {}
static void JpegTermSource(j_decompress_ptr jds) {}
static boolean JpegFillInputBuffer(j_decompress_ptr jds) {
    MyJpegSourceMgr *src = (MyJpegSourceMgr*)jds->src;
    src->pub.next_input_byte = EOI_BUFFER;
    src->pub.bytes_in_buffer = 1;
    return true;
}
static void JpegSkipInputData(j_decompress_ptr jds, long len) {
    MyJpegSourceMgr *src = (MyJpegSourceMgr*)jds->src;
    if (src->pub.bytes_in_buffer < len) {
        src->pub.next_input_byte = EOI_BUFFER;
        src->pub.bytes_in_buffer = 1;
    } else {
        src->pub.next_input_byte += len;
        src->pub.bytes_in_buffer -= len;
    }
}
static void JpegMemSrc(j_decompress_ptr jds, const char *buf, size_t len) {
    if (!jds->src) jds->src = (jpeg_source_mgr*)(*jds->mem->alloc_small)((j_common_ptr)jds, JPOOL_PERMANENT, sizeof(MyJpegSourceMgr));
    MyJpegSourceMgr *src = (MyJpegSourceMgr*)jds->src;
    src->pub.init_source       = JpegInitSource;
    src->pub.fill_input_buffer = JpegFillInputBuffer;
    src->pub.skip_input_data   = JpegSkipInputData;
    src->pub.resync_to_restart = jpeg_resync_to_restart;
    src->pub.term_source       = JpegTermSource;
    src->data = (const JOCTET *)buf;
    src->len = len;
    src->pub.bytes_in_buffer = len;
    src->pub.next_input_byte = src->data;
}
int JpegReader::Read(File *lf,           Texture *out) { return Read(lf->Contents(), out); }
int JpegReader::Read(const string &data, Texture *out) {
    MyJpegErrorMgr jerr;
    jpeg_decompress_struct jds;
    jds.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = JpegErrorExit;
    if (setjmp(jerr.setjmp_buffer)) {
        char buf[JMSG_LENGTH_MAX]; jds.err->format_message((j_common_ptr)&jds, buf);
        INFO("jpeg decompress failed ", buf); jpeg_destroy_decompress(&jds); return -1;
    }

    jpeg_create_decompress(&jds);
    JpegMemSrc(&jds, data.data(), data.size());
    if (jpeg_read_header(&jds, 1) != 1) 
    { INFO("jpeg decompress failed "); jpeg_destroy_decompress(&jds); return -1; }

    jpeg_start_decompress(&jds);
    if      (jds.output_components == 1) out->pf = Pixel::GRAY8;
    else if (jds.output_components == 3) out->pf = Pixel::RGB24;
    else if (jds.output_components == 4) out->pf = Pixel::RGBA;
    else { ERROR("unsupported jpeg components ", jds.output_components); jpeg_destroy_decompress(&jds); return -1; }
    out->Resize(jds.output_width, jds.output_height, out->pf, Texture::Flag::CreateBuf);

    int linesize = out->LineSize();
    while (jds.output_scanline < jds.output_height) {
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
    return ((BufferFile*)gif->UserData)->Read(out, size);
}
int GIFReader::Read(File *lf,           Texture *out) { return Read(lf->Contents(), out); }
int GIFReader::Read(const string &data, Texture *out) {
    int error_code = 0;
    BufferFile bf(data.data(), data.size());
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

struct WavHeader {
    unsigned chunk_id, chunk_size, format, subchunk_id, subchunk_size;
    unsigned short audio_format, num_channels;
    unsigned sampleRate, byte_rate;
    unsigned short block_align, bits_per_sample;
    unsigned subchunk_id2, subchunk_size2;
    static const int Size=44;
};

void WavWriter::Open(File *F) { delete f; f=F; wrote=WavHeader::Size; if (f && f->Opened()) Flush(); }
void WavReader::Open(File *F) { delete f; f=F; last =WavHeader::Size; }

int WavReader::Read(RingBuf::Handle *B, int off, int num) {
    if (!f->Opened()) return -1;
    int offset = WavHeader::Size + off * sizeof(short), bytes = num * sizeof(short);
    if (f->Seek(offset, File::Whence::SET) != offset) return -1;

    short *data = (short *)alloca(bytes);
    if (f->Read(data, bytes) != bytes) return -1;
    for (int i=0; i<num; i++) B->Write(data[i] / 32768.0);
    last = offset + bytes;
    return 0;
}

int WavWriter::Write(const RingBuf::Handle *B, bool flush) {
    if (!f->Opened()) return -1;
    int bytes = B->Len() * sizeof(short);
    short *data = (short *)alloca(bytes);
    for (int i=0; B && i<B->Len(); i++) data[i] = (short)(B->Read(i) * 32768.0);

    if (f->Write(data, bytes) != bytes) return -1;
    wrote += bytes;
    return flush ? Flush() : 0;
}

int WavWriter::Flush() {
    if (!f->Opened()) return -1;
    WavHeader hdr;
    hdr.chunk_id = *(unsigned*)"RIFF";    
    hdr.format = *(unsigned*)"WAVE";
    hdr.subchunk_id = *(unsigned*)"fmt ";
    hdr.audio_format = 1;
    hdr.num_channels = 1;
    hdr.sampleRate = FLAGS_sample_rate;
    hdr.byte_rate = hdr.sampleRate * hdr.num_channels * sizeof(short);
    hdr.block_align = hdr.num_channels * sizeof(short);
    hdr.bits_per_sample = 16;
    hdr.subchunk_id2 = *(unsigned*)"data";
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
        LocalFile *lf = new LocalFile(filename, "r");
        if (!lf->Opened()) { ERROR("open: ", filename); delete lf; return 0; }
        return lf;    
    }
    virtual void UnloadFile(void *h) { delete (File*)h; }
    virtual void *LoadBuf(const char *buf, int len, const char *mimetype) { return new BufferFile(buf, len, mimetype); }
    virtual void UnloadBuf(void *h) { delete (File*)h; }

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

    virtual void LoadVideo(void *handle, Texture *out, bool clear=true) {
        static char jpghdr[2] = { '\xff', '\xd8' }; // 0xff, 0xd8 };
        static char gifhdr[4] = { 'G', 'I', 'F', '8' };
        static char pnghdr[8] = { '\211', 'P', 'N', 'G', '\r', '\n', '\032', '\n' };

        File *f = (File*)handle; string fn = f->Filename(); unsigned char hdr[8];
        if (f->Read(hdr, 8) != 8) { ERROR("load ", fn, " : failed"); return; }
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
        else { ERROR("load ", fn, " : failed"); return; }

        out->LoadGL();
        if (clear) out->ClearBuffer();
    }

    virtual void LoadAudio(void *h, SoundAsset *a, int seconds, int flag) {}
    virtual int RefillAudio(SoundAsset *a, int reset) { return 0; }

    virtual void LoadMovie(void *h, MovieAsset *a) {}
    virtual int PlayMovie(MovieAsset *a, int seek) { return 0; }
};

#ifdef LFL_ANDROID
struct AndroidAudioAssetLoader : public AudioAssetLoader {
    virtual void *LoadAudioFile(const string &filename) { return android_load_music_asset(filename.c_str()); }
    virtual void UnloadAudioFile(void *h) {}
    virtual void *LoadAudioBuf(const char *buf, int len, const char *mimetype) { return 0; }
    virtual void UnloadAudioBuf(void *h) {}
    virtual void LoadAudio(void *handle, SoundAsset *a, int seconds, int flag) { a->handle = handle; }
    virtual int RefillAudio(SoundAsset *a, int reset) { return 0; }
};
#endif

#ifdef LFL_IPHONE
extern void *iphone_load_music_asset(const char *filename);
struct IPhoneAudioAssetLoader : public AudioAssetLoader {
    virtual void *LoadAudioFile(const string &filename) { return iphone_load_music_asset(filename.c_str()); }
    virtual void UnloadAudioFile(void *h) {}
    virtual void *LoadAudioBuf(const char *buf, int len, const char *mimetype) { return 0; }
    virtual void UnloadAudioBuf(void *h) {}
    virtual void LoadAudio(void *handle, SoundAsset *a, int seconds, int flag) { a->handle = handle; }
    virtual int RefillAudio(SoundAsset *a, int reset) { return 0; }
};
#endif

#ifdef LFL_FFMPEG
struct FFBIOFile {
    static void *Alloc(File *f) {
        int bufsize = 16384;
        return avio_alloc_context((unsigned char *)malloc(bufsize), bufsize, 0, f, Read, Write, Seek);
    }
    static void Free(void *in) {
        AVIOContext *s = (AVIOContext*)in;
        delete (File*)s->opaque;
        ::free(s->buffer);
        av_free(s);
    }
    static int     Read(void *f, uint8_t *buf, int buf_size) { return ((File*)f)->Read (buf, buf_size); }
    static int    Write(void *f, uint8_t *buf, int buf_size) { return ((File*)f)->Write(buf, buf_size); }
    static int64_t Seek(void *f, int64_t offset, int whence) { return ((File*)f)->Seek (offset, whence); }
};

struct FFBIOC {
    void const *buf; int len, offset;
    FFBIOC(void const *b, int l) : buf(b), len(l), offset(0) {}

    static void *Alloc(void const *buf, int len) {
        static const int bufsize = 32768;
        return avio_alloc_context((unsigned char *)malloc(bufsize), bufsize, 0, new FFBIOC(buf, len), Read, Write, Seek);
    }
    static void Free(void *in) {
        AVIOContext *s = (AVIOContext*)in;
        delete (FFBIOC*)s->opaque;
        ::free(s->buffer);
        av_free(s);
    }
    static int Read(void *opaque, uint8_t *buf, int buf_size) {
        FFBIOC *s = (FFBIOC*)opaque;
        int len = min(buf_size, s->len - s->offset);
        if (len <= 0) return len;

        memcpy(buf, (char*)s->buf + s->offset, len);
        s->offset += len;
        return len;
    }
    static int Write(void *opaque, uint8_t *buf, int buf_size) { return -1; }
    static int64_t Seek(void *opaque, int64_t offset, int whence) {
        FFBIOC *s = (FFBIOC*)opaque;
        if      (whence == SEEK_SET) s->offset = offset;
        else if (whence == SEEK_CUR) s->offset += offset;
        else if (whence == SEEK_END) s->offset = s->len + offset;
        else return -1;
        return 0;
    }
};

struct FFMpegAssetLoader : public AudioAssetLoader, public VideoAssetLoader, public MovieAssetLoader {
    FFMpegAssetLoader() { INFO("FFMpegAssetLoader"); }
    static AVFormatContext *Load(AVIOContext *pb, const string &filename, char *probe_buf, int probe_buflen, AVIOContext **pbOut=0) {
        AVProbeData probe_data;
        memzero(probe_data);
        probe_data.filename = basename(filename.c_str(),0,0);

        bool probe_buf_data = probe_buf && probe_buflen;
        if (probe_buf_data) {
            probe_data.buf = (unsigned char *)probe_buf;
            probe_data.buf_size = probe_buflen;
        }

        AVInputFormat *fmt = av_probe_input_format(&probe_data, probe_buf_data);
        if (!fmt) { ERROR("no AVInputFormat for ", probe_data.filename); return 0; }

        AVFormatContext *fctx = avformat_alloc_context(); int ret;
        fctx->flags |= AVFMT_FLAG_CUSTOM_IO;

        fctx->pb = pb;
        if ((ret = avformat_open_input(&fctx, probe_data.filename, fmt, 0))) {
            char errstr[128]; av_strerror(ret, errstr, sizeof(errstr));
            ERROR("av_open_input ", probe_data.filename, ": ", ret, " ", errstr);
            return 0;
        }

        if (pbOut) *pbOut = fctx->pb;
        return fctx;
    }

    virtual void *LoadFile(const string &filename) { return LoadFile(filename, 0); }
    AVFormatContext *LoadFile(const string &filename, AVIOContext **pbOut) {
#if !defined(LFL_ANDROID) && !defined(LFL_IPHONE)
        AVFormatContext *fctx = 0;
        if (avformat_open_input(&fctx, filename.c_str(), 0, 0)) { ERROR("av_open_input_file: ", filename); return 0; }
        return fctx;
#else
        LocalFile *lf = new LocalFile(filename, "r");
        if (!lf->opened()) { ERROR("FFLoadFile: open ", filename); delete lf; return 0; }
        void *pb = FFBIOFile::Alloc(lf);
        AVFormatContext *ret = Load((AVIOContext*)pb, filename, 0, 0, pbOut);
        if (!ret) FFBIOFile::Free(pb);
        return ret;
#endif
    }
    virtual void UnloadFile(void *h) {
        AVFormatContext *handle = (AVFormatContext*)h;
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
        AVFormatContext *ret = Load((AVIOContext*)pb, suffix ? suffix+1 : filename, (char*)buf, min(4096, len), pbOut);
        if (!ret) FFBIOC::Free(pb);
        return ret;
    }
    virtual void UnloadBuf(void *h) {
        AVFormatContext *handle = (AVFormatContext*)h;
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

    void LoadVideo(void *handle, Texture *out, bool clear=true) {
        AVFormatContext *fctx = (AVFormatContext*)handle;
        int video_index = -1, got=0;
        for (int i=0; i<fctx->nb_streams; i++) {
            AVStream *st = fctx->streams[i];
            AVCodecContext *avctx = st->codec;
            if (avctx->codec_type == AVMEDIA_TYPE_VIDEO) video_index = i;
        }
        if (video_index < 0) { ERROR("no stream: ", fctx->nb_streams); return; }

        AVCodecContext *avctx = fctx->streams[video_index]->codec;
        AVCodec *codec = avcodec_find_decoder(avctx->codec_id);
        if (!codec || avcodec_open2(avctx, codec, 0) < 0) { ERROR("avcodec_open2: ", handle); return; }

        AVPacket packet;
        if (av_read_frame(fctx, &packet) < 0) { ERROR("av_read_frame: ", handle); avcodec_close(avctx); return; }

        AVFrame *frame = av_frame_alloc(); int ret;
        if ((ret = avcodec_decode_video2(avctx, frame, &got, &packet)) <= 0 || !got) {
            char errstr[128]; av_strerror(ret, errstr, sizeof(errstr));
            ERROR("avcodec_decode_video2: ", codec->name, " ", ret, ": ", errstr);
        } else {
            int pf = Pixel::FromFFMpegId(avctx->pix_fmt);
            out->width  = avctx->width;
            out->height = avctx->height;
            out->LoadGL(*frame->data, out->width, out->height, pf, frame->linesize[0]);
            if (!clear) out->LoadBuffer(frame->data[0], out->width, out->height, pf, frame->linesize[0]);
            // av_frame_unref(frame);
        }

        av_free_packet(&packet);
        // av_frame_free(&frame);
    }

    void LoadAudio(void *handle, SoundAsset *a, int seconds, int flag) {
        AVFormatContext *fctx = (AVFormatContext*)handle;
        LoadMovie(a, 0, fctx);

        int samples = RefillAudio(a, 1);
        if (samples == SoundAssetSize(a) && !(flag & SoundAsset::FlagNoRefill)) a->refill = RefillAudioCB;

        if (!a->refill) {
            avcodec_close(fctx->streams[a->handle_arg1]->codec);
            a->resampler.Close();
        }
    }
    int RefillAudio(SoundAsset *a, int reset) { return RefillAudioCB(a, reset); }
    static int RefillAudioCB(SoundAsset *a, int reset) {
        AVFormatContext *fctx = (AVFormatContext*)a->handle;
        AVCodecContext *avctx = fctx->streams[a->handle_arg1]->codec;
        bool open_resampler = false;
        if (reset) {
            av_seek_frame(fctx, a->handle_arg1, 0, AVSEEK_FLAG_BYTE);
            a->wav->ring.size = SoundAssetSize(a);
            a->wav->bytes = a->wav->ring.size * a->wav->width;
            open_resampler = true;
        }
        if (!a->wav) {
            a->wav = new RingBuf(a->sample_rate, SoundAssetSize(a));
            open_resampler = true;
        }
        if (open_resampler)
            a->resampler.Open(a->wav, avctx->channels, avctx->sample_rate, Sample::FromFFMpegId(avctx->sample_fmt),
                                      a->channels,     a->sample_rate,     Sample::S16);
        a->wav->ring.back = 0;
        int wrote = PlayMovie(a, 0, fctx, 0);
        if (wrote < SoundAssetSize(a)) {
            a->wav->ring.size = wrote;
            a->wav->bytes = a->wav->ring.size * a->wav->width;
        }
        return wrote;
    }

    void LoadMovie(void *handle, MovieAsset *ma) {
        LoadMovie(&ma->audio, &ma->video, (AVFormatContext*)handle);
        PlayMovie(&ma->audio, &ma->video, (AVFormatContext*)handle, 0);
    }
    static void LoadMovie(SoundAsset *sa, Asset *va, AVFormatContext *fctx) {
        if (avformat_find_stream_info(fctx, 0) < 0) { ERROR("av_find_stream_info"); return; }

        int audio_index = -1, video_index = -1, got=0;
        for (int i=0; i<fctx->nb_streams; i++) {
            AVStream *st = fctx->streams[i];
            AVCodecContext *avctx = st->codec;
            if (avctx->codec_type == AVMEDIA_TYPE_AUDIO) audio_index = i;
            if (avctx->codec_type == AVMEDIA_TYPE_VIDEO) video_index = i;
        }
        if (va) {
            if (video_index < 0) { ERROR("no v-stream: ", fctx->nb_streams); return; }
            AVCodecContext *avctx = fctx->streams[video_index]->codec;
            AVCodec *codec = avcodec_find_decoder(avctx->codec_id);
            if (!codec || avcodec_open2(avctx, codec, 0) < 0) { ERROR("avcodec_open2: ", codec); return; }
            va->tex.CreateBacked(avctx->width, avctx->height, Pixel::BGR32);
        }
        if (sa) {
            if (audio_index < 0) { ERROR("no a-stream: ", fctx->nb_streams); return; }
            AVCodecContext *avctx = fctx->streams[audio_index]->codec;
            AVCodec *codec = avcodec_find_decoder(avctx->codec_id);
            if (!codec || avcodec_open2(avctx, codec, 0) < 0) { ERROR("avcodec_open2: ", codec); return; }

            sa->handle = fctx;
            sa->handle_arg1 = audio_index;
            sa->channels = avctx->channels == 1 ? 1 : FLAGS_chans_out;
            sa->sample_rate = FLAGS_sample_rate;
            sa->seconds = FLAGS_soundasset_seconds;
            sa->wav = new RingBuf(sa->sample_rate, SoundAssetSize(sa));
            sa->resampler.Open(sa->wav, avctx->channels, avctx->sample_rate, Sample::FromFFMpegId(avctx->sample_fmt),
                                           sa->channels,    sa->sample_rate, Sample::S16);
        }
    }

    int PlayMovie(MovieAsset *ma, int seek) { return PlayMovie(&ma->audio, &ma->video, (AVFormatContext*)ma->handle, seek); }
    static int PlayMovie(SoundAsset *sa, Asset *va, AVFormatContext *fctx, int seek_unused) {
        int begin_resamples_available = sa->resampler.output_available, wrote=0, done=0;
        Allocator *tlsalloc = ThreadLocalStorage::GetAllocator();

        while (!done && wrote != SoundAssetSize(sa)) {
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
                    short *rsout = (short*)tlsalloc->Malloc(AVCODEC_MAX_AUDIO_FRAME_SIZE + FF_INPUT_BUFFER_PADDING_SIZE);

                    int sampsIn  = frame->nb_samples;
                    int sampsOut = max(0, SoundAssetSize(sa) - wrote) / sa->channels;

                    sa->resampler.Update(sampsIn, (const short**)frame->extended_data, rsout, -1, sampsOut);
                    wrote = sa->resampler.output_available - begin_resamples_available;
                    av_frame_unref(frame);
                }
            } else if (va) {
                if ((ret = avcodec_decode_video2(avctx, frame, &got, &packet)) <= 0 || !got) {
                    char errstr[128]; av_strerror(ret, errstr, sizeof(errstr));
                    ERROR("avcodec_decode_video2 ", ret, ": ", errstr);
                } else {
                    screen->gd->BindTexture(GraphicsDevice::Texture2D, va->tex.ID);
                    va->tex.UpdateBuffer(*frame->data, avctx->width, avctx->height, Pixel::FromFFMpegId(avctx->pix_fmt), 
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
#endif

struct AliasWavefrontObjLoader {
    typedef map<string, Material> MtlMap;
    MtlMap MaterialMap;
    string dir;

    Geometry *LoadOBJ(const string &filename, const float *map_tex_coord=0) {
        dir.assign(filename, 0, dirnamelen(filename.c_str(), filename.size(), true));

        LocalFile file(filename, "r");
        if (!file.Opened()) { ERROR("LocalFile::open(", file.Filename(), ")"); return 0; }

        vector<v3> vert, vertOut, norm, normOut;
        vector<v2> tex, texOut;
        Material mat; v3 xyz; v2 xy;
        int format=0, material=0, ind[3];

        for (const char *cmd = 0, *line = file.NextLine(); line; line = file.NextLine()) {
            if (!line[0] || line[0] == '#') continue;

            StringWordIter word(line);
            if (!(cmd = word.Next())) continue;

            if      (!strcmp(cmd, "mtllib")) LoadMaterial(word.Next());
            else if (!strcmp(cmd, "usemtl")) {
                MtlMap::iterator it = MaterialMap.find(word.Next());
                if (it != MaterialMap.end()) { mat = (*it).second; material = 1; }
            }
            else if (!strcmp(cmd, "v" )) { word.ScanN(&xyz[0], 3);             vert.push_back(xyz); }
            else if (!strcmp(cmd, "vn")) { word.ScanN(&xyz[0], 3); xyz.Norm(); norm.push_back(xyz); }
            else if (!strcmp(cmd, "vt")) {
                word.ScanN(&xy[0], 2);
                if (map_tex_coord) {
                    xy.x = map_tex_coord[Texture::CoordMinX] + xy.x * (map_tex_coord[Texture::CoordMaxX] - map_tex_coord[Texture::CoordMinX]);
                    xy.y = map_tex_coord[Texture::CoordMinY] + xy.y * (map_tex_coord[Texture::CoordMaxY] - map_tex_coord[Texture::CoordMinY]);
                }
                tex.push_back(xy);
            }
            else if (!strcmp(cmd, "f")) {
                vector<v3> face; 
                for (const char *fi = word.Next(); fi; fi = word.Next()) {
                    StringWordIter indexword(fi, 0, isint<'/'>);
                    indexword.ScanN(ind, 3);

                    int count = (ind[0]!=0) + (ind[1]!=0) + (ind[2]!=0);
                    if (!count) continue;
                    if (!format) format = count;
                    if (format != count) { ERROR("face format ", format, " != ", count); return 0; }

                    face.push_back(v3(ind[0], ind[1], ind[2]));
                }
                if (face.size() && face.size() < 2) { ERROR("face size ", face.size()); return 0; }

                int next = 1;
                for (int i=0,j=0; i<face.size(); i++,j++) {
                    unsigned ind[3] = { (unsigned)face[i].x, (unsigned)face[i].y, (unsigned)face[i].z };

                    if (ind[0] > (int) vert.size()) { ERROR("index error 1 ", ind[0], ", ", vert.size()); return 0; }
                    if (ind[1] > (int)  tex.size()) { ERROR("index error 2 ", ind[1], ", ", tex.size());  return 0; }
                    if (ind[2] > (int) norm.size()) { ERROR("index error 3 ", ind[2], ", ", norm.size()); return 0; }

                    if (ind[0]) vertOut.push_back(vert[ind[0]-1]);
                    if (ind[1])  texOut.push_back(tex [ind[1]-1]);
                    if (ind[2]) normOut.push_back(norm[ind[2]-1]);

                    if (i == face.size()-1) break;
                    if (!i) i = next-1;
                    else if (j == 2) { next=i; i=j=(-1); }
                }
            }
        }

        Geometry *ret = new Geometry(GraphicsDevice::Triangles, vertOut.size(), &vertOut[0], normOut.size() ? &normOut[0] : 0, texOut.size() ? &texOut[0] : 0);
        ret->material = material;
        ret->mat = mat;

        INFO("asset(", filename, ") ", ret->count, " verts ");
        return ret;
    }

    int LoadMaterial(const string &filename) {
        LocalFile file(StrCat(dir, filename), "r");
        if (!file.Opened()) { ERROR("LocalFile::open(", file.Filename(), ")"); return -1; }

        Material m; string name;
        for (const char *cmd = 0, *line = file.NextLine(); line; line = file.NextLine()) {
            if (!line[0] || line[0] == '#') continue;

            StringWordIter word(line);
            if (!(cmd = word.Next())) continue;

            if      (!strcmp(cmd, "Ka")) word.ScanN(m.ambient .x, 4);
            else if (!strcmp(cmd, "Kd")) word.ScanN(m.diffuse .x, 4);
            else if (!strcmp(cmd, "Ks")) word.ScanN(m.specular.x, 4);
            else if (!strcmp(cmd, "Ke")) word.ScanN(m.emissive.x, 4);
            else if (!strcmp(cmd, "newmtl")) {
                if (name.size()) MaterialMap[name] = m;
                name = word.Next();
            }
        }
        if (name.size()) MaterialMap[name] = m;
        return 0;
    }
};

Geometry *Geometry::LoadOBJ(const string &filename, const float *map_tex_coord) {
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
    if (ArrayEquals(v, &last_position[0], vd)) return;
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
    screen->gd->VertexPointer(vd, GraphicsDevice::Float, width_bytes, 0, &vert[0], vert_size, &vert_ind, true);
}

int Assets::Init() {
#ifdef LFL_FFMPEG
    static FFMpegAssetLoader *ffmpeg_loader = new FFMpegAssetLoader();
    default_audio_loader = ffmpeg_loader;
    default_video_loader = ffmpeg_loader;
    default_movie_loader = ffmpeg_loader;
#endif
    if (!default_audio_loader || !default_video_loader || !default_movie_loader) {
        if (!default_audio_loader) default_audio_loader = Singleton<SimpleAssetLoader>::Get();
        if (!default_video_loader) default_video_loader = Singleton<SimpleAssetLoader>::Get();
        if (!default_movie_loader) default_movie_loader = Singleton<SimpleAssetLoader>::Get();
    }
#ifdef LFL_ANDROID
    default_audio_loader = new AndroidAudioAssetLoader();
#endif
#ifdef LFL_IPHONE
    default_audio_loader = new IPhoneAudioAssetLoader();
#endif
    return 0;
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
    if (!geom_fn.empty()) geometry = Geometry::LoadOBJ(StrCat(ASSETS_DIR, geom_fn));
    if (!texture.empty() || h) LoadTexture(h, StrCat(ASSETS_DIR, texture), &tex, l);
}

void Asset::Load(const void *FromBuf, const char *filename, int size) {
    VideoAssetLoader *l = app->assets.default_video_loader;
    // work around ffmpeg image2 format being AVFMT_NO_FILE; ie doesnt work with custom AVIOContext
    if (FileSuffix::Image(filename)) l = Singleton<SimpleAssetLoader>::Get();
    void *handle = l->LoadVideoBuf((const char *)FromBuf, size, filename);
    if (!handle) return;
    Load(handle, l);
    l->UnloadVideoBuf(handle);
}

void Asset::LoadTexture(void *h, const string &fn, Texture *out, VideoAssetLoader *l) {
    if (!FLAGS_lfapp_video) return;
    if (!l) l = app->assets.default_video_loader;
    void *handle = h ? h : l->LoadVideoFile(fn);
    if (!handle) { ERROR("load: ", fn); return; }
    l->LoadVideo(handle, out);
    if (!h) l->UnloadVideoFile(handle);
}

void SoundAsset::Unload() {
    if (parent) parent->Unloaded(this);
    delete wav; wav=0;
    if (handle) ERROR("leak: ", handle);
}

void SoundAsset::Load(void *handle, const char *FN, int Secs, int flag) {
    if (!FLAGS_lfapp_audio) { ERROR("load: ", FN, ": lfapp_audio = ", FLAGS_lfapp_audio); return; }
    if (handle) app->assets.default_audio_loader->LoadAudio(handle, this, Secs, flag);
}

void SoundAsset::Load(void const *buf, int len, char const *FN, int Secs) {
    void *handle = app->assets.default_audio_loader->LoadAudioBuf((const char *)buf, len, FN);
    if (!handle) return;

    Load(handle, FN, Secs, FlagNoRefill);
    if (!refill) {
        app->assets.default_audio_loader->UnloadAudioBuf(handle);
        handle = 0;
    }
}

void SoundAsset::Load(int Secs, bool unload) {
    string fn; void *handle = 0;

    if (!filename.empty()) {
        fn = filename;
        if (fn.length() && isalpha(fn[0])) fn = ASSETS_DIR + fn;

        handle = app->assets.default_audio_loader->LoadAudioFile(fn);
        if (!handle) ERROR("SoundAsset::Load ", fn);
    }

    Load(handle, fn.c_str(), Secs);

#if !defined(LFL_IPHONE) && !defined(LFL_ANDROID) /* XXX */
    if (!refill && handle && unload) {
        app->assets.default_audio_loader->UnloadAudioFile(handle);
        handle = 0;
    }
#endif
}

int SoundAsset::Refill(int reset) { return app->assets.default_audio_loader->RefillAudio(this, reset); }

void MovieAsset::Load(const char *fn) {
    if (!fn || !(handle = app->assets.default_movie_loader->LoadMovieFile(StrCat(ASSETS_DIR, fn)))) return;
    app->assets.default_movie_loader->LoadMovie(handle, this);
    audio.Load();
    video.Load();
}

int MovieAsset::Play(int seek) {
    app->assets.movie_playing = this; int ret;
    if ((ret = app->assets.default_movie_loader->PlayMovie(this, 0) <= 0)) app->assets.movie_playing = 0;
    return ret;
}

/* asset impls */

void glLine(float x1, float y1, float x2, float y2, const Color *color) {
    static int verts_ind=-1;
    screen->gd->DisableTexture();

    float verts[] = { /*1*/ x1, y1, /*2*/ x2, y2 };
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
    unique_ptr<Geometry> geom(new Geometry(GraphicsDevice::Lines, 4, (v2*)0, 0, 0, *c));
    v2 *vert = (v2*)&geom->vert[0];

    vert[0] = v2(0, y);
    vert[1] = v2(screen->width, y);
    vert[2] = v2(x, 0);
    vert[3] = v2(x, screen->height);

    screen->gd->DisableTexture();
    Scene::Select(geom.get());
    Scene::Draw(geom.get(), 0);
}

void glTimeResolutionShaderWindows(Shader *shader, const Color &backup_color, const Box &w,             const Texture *tex) { Box wc=w; vector<Box*> wv; wv.push_back(&wc); glTimeResolutionShaderWindows(shader, backup_color, wv, tex); }
void glTimeResolutionShaderWindows(Shader *shader, const Color &backup_color, const vector<Box*> &wins, const Texture *tex) {
    if (!shader) screen->gd->SetColor(backup_color);
    else {
        screen->gd->UseShader(shader);
        shader->SetUniform1f("time", ToSeconds(app->app_time.GetTime()));
        shader->SetUniform2f("resolution", screen->width, screen->height);
    }
    if (tex) { screen->gd->EnableLayering(); tex->Bind(); }
    else screen->gd->DisableTexture();
    for (vector<Box*>::const_iterator i = wins.begin(); i != wins.end(); ++i) (*i)->Draw(tex ? tex->coord : 0);
    if (shader) screen->gd->UseShader(0);
}

void BoxOutline::Draw(const LFL::Box &w) const {
    screen->gd->DisableTexture();
    if (line_width <= 1) {
        static int verts_ind = -1;
        float verts[] = { /*1*/ (float)w.x,     (float)w.y,     /*2*/ (float)w.x,     (float)w.y+w.h,
                          /*3*/ (float)w.x+w.w, (float)w.y+w.h, /*4*/ (float)w.x+w.w, (float)w.y };
        screen->gd->VertexPointer(2, GraphicsDevice::Float, 0, 0, verts, sizeof(verts), &verts_ind, true);
        screen->gd->DrawArrays(GraphicsDevice::LineLoop, 0, 4);
    } else {

    }
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

Waveform::Waveform(point dim, const Color *c, const Vec<float> *sbh) : width(dim.x), height(dim.y), geom(0) {
    float xmax=sbh->Len(), ymin=INFINITY, ymax=-INFINITY;
    for (int i=0; i<xmax; i++) {
        ymin = min(ymin,sbh->Read(i));
        ymax = max(ymax,sbh->Read(i));
    }

    geom = new Geometry(GraphicsDevice::Lines, (int)(xmax-1)*2, (v2*)0, 0, 0, *c);
    v2 *vert = (v2*)&geom->vert[0];

    for (int i=0; i<geom->count/2; i++) {
        float x1=(i)  *1.0/xmax, y1=(sbh->Read(i)   - ymin)/(ymax - ymin); 
        float x2=(i+1)*1.0/xmax, y2=(sbh->Read(i+1) - ymin)/(ymax - ymin);

        vert[i*2+0] = v2(x1 * dim.x, -dim.y + y1 * dim.y);
        vert[i*2+1] = v2(x2 * dim.x, -dim.y + y2 * dim.y);
    }
}

void glSpectogram(Matrix *m, unsigned char *data, int width, int height, int hjump, float vmax, float clip, bool interpolate, int pd) {
    for (int j=0; j<width; j++) {
        for(int i=0; i<height; i++) {
            double v;
            if (!interpolate) v = m->row(j)[i];
            else v = MatrixAsFunc(m, i?(float)i/(height-1):0, j?(float)j/(width-1):0);

            if      (pd == PowerDomain::dB)   { /**/ }
            else if (pd == PowerDomain::abs)  { v = AmplitudeRatioDecibels(v, 1); }
            else if (pd == PowerDomain::abs2) { v = AmplitudeRatioDecibels(sqrt(v), 1); }

            if (v < clip) v = clip;
            v = (v - clip) / (vmax - clip);

            int pixel = 4;
            int ind = j*hjump*pixel + i*pixel;
            data[ind + 0] = (unsigned)(v > 1.0/3.0 ? 255 : v*3*255); v=max(0.0,v-1.0/3.0);
            data[ind + 1] = (unsigned)(v > 1.0/3.0 ? 255 : v*3*255); v=max(0.0,v-1.0/3.0);
            data[ind + 2] = (unsigned)(v > 1.0/3.0 ? 255 : v*3*255); v=max(0.0,v-1.0/3.0);
            data[ind + 3] = (unsigned)255;
        }
    }
}

void glSpectogram(Matrix *m, Asset *a, float *max, float clip, int pd) {
    if (!a->tex.ID) a->tex.CreateBacked(m->N, m->M);
    else {
        if (a->tex.width < m->N || a->tex.height < m->M) a->tex.Resize(m->N, m->M);
        else Texture::Coordinates(a->tex.coord, m->N, m->M, NextPowerOfTwo(a->tex.width), NextPowerOfTwo(a->tex.height));
    }

    if (clip == -INFINITY) clip = -65;
    if (!a->tex.buf) return;

    float Max = Matrix::Max(m);
    if (max) *max = Max;

    glSpectogram(m, a->tex.buf, m->M, m->N, a->tex.width, Max, clip, 0, pd);

    screen->gd->BindTexture(GraphicsDevice::Texture2D, a->tex.ID);
    a->tex.UpdateGL();
}

void glSpectogram(SoundAsset *sa, Asset *a, Matrix *transform, float *max, float clip) {
    const RingBuf::Handle B(sa->wav);

    /* 20*log10(abs(specgram(y,2048,sr,hamming(512),256))) */
    Matrix *m = Spectogram(&B, 0, FLAGS_feat_window, FLAGS_feat_hop, FLAGS_feat_window, 0, PowerDomain::abs);
    if (transform) m = Matrix::Mult(m, transform, mDelA);

    glSpectogram(m, a, max, clip, PowerDomain::abs);
    delete m;
}

/* Cube */

Geometry *Cube::Create(v3 v) { return Create(v.x, v.y, v.z); } 
Geometry *Cube::Create(float rx, float ry, float rz, bool normals) {
    vector<v3> verts, norms;

    verts.push_back(v3(-rx,  ry, -rz)); norms.push_back(v3(0, 0, 1));
    verts.push_back(v3(-rx, -ry, -rz)); norms.push_back(v3(0, 0, 1));
    verts.push_back(v3( rx, -ry, -rz)); norms.push_back(v3(0, 0, 1));
    verts.push_back(v3( rx,  ry, -rz)); norms.push_back(v3(0, 0, 1));
    verts.push_back(v3(-rx,  ry, -rz)); norms.push_back(v3(0, 0, 1));
    verts.push_back(v3( rx, -ry, -rz)); norms.push_back(v3(0, 0, 1));

    verts.push_back(v3( rx,  ry,  rz)); norms.push_back(v3(0, 0, -1));
    verts.push_back(v3( rx, -ry,  rz)); norms.push_back(v3(0, 0, -1));
    verts.push_back(v3(-rx, -ry,  rz)); norms.push_back(v3(0, 0, -1));
    verts.push_back(v3(-rx,  ry,  rz)); norms.push_back(v3(0, 0, -1));
    verts.push_back(v3( rx,  ry,  rz)); norms.push_back(v3(0, 0, -1));
    verts.push_back(v3(-rx, -ry,  rz)); norms.push_back(v3(0, 0, -1));

    verts.push_back(v3(rx,  ry, -rz));  norms.push_back(v3(-1, 0, 0));
    verts.push_back(v3(rx, -ry, -rz));  norms.push_back(v3(-1, 0, 0));
    verts.push_back(v3(rx, -ry,  rz));  norms.push_back(v3(-1, 0, 0));
    verts.push_back(v3(rx,  ry,  rz));  norms.push_back(v3(-1, 0, 0));
    verts.push_back(v3(rx,  ry, -rz));  norms.push_back(v3(-1, 0, 0));
    verts.push_back(v3(rx, -ry,  rz));  norms.push_back(v3(-1, 0, 0));

    verts.push_back(v3(-rx,  ry,  rz)); norms.push_back(v3(1, 0, 0));
    verts.push_back(v3(-rx, -ry,  rz)); norms.push_back(v3(1, 0, 0));
    verts.push_back(v3(-rx, -ry, -rz)); norms.push_back(v3(1, 0, 0));
    verts.push_back(v3(-rx,  ry, -rz)); norms.push_back(v3(1, 0, 0));
    verts.push_back(v3(-rx,  ry,  rz)); norms.push_back(v3(1, 0, 0));
    verts.push_back(v3(-rx, -ry, -rz)); norms.push_back(v3(1, 0, 0));

    verts.push_back(v3( rx,  ry, -rz)); norms.push_back(v3(0, -1, 0));
    verts.push_back(v3( rx,  ry,  rz)); norms.push_back(v3(0, -1, 0));
    verts.push_back(v3(-rx,  ry,  rz)); norms.push_back(v3(0, -1, 0));
    verts.push_back(v3(-rx,  ry, -rz)); norms.push_back(v3(0, -1, 0));
    verts.push_back(v3( rx,  ry, -rz)); norms.push_back(v3(0, -1, 0));
    verts.push_back(v3(-rx,  ry,  rz)); norms.push_back(v3(0, -1, 0));

    verts.push_back(v3( rx, -ry,  rz)); norms.push_back(v3(0, 1, 0));
    verts.push_back(v3( rx, -ry, -rz)); norms.push_back(v3(0, 1, 0));
    verts.push_back(v3(-rx, -ry, -rz)); norms.push_back(v3(0, 1, 0));
    verts.push_back(v3(-rx, -ry,  rz)); norms.push_back(v3(0, 1, 0));
    verts.push_back(v3( rx, -ry,  rz)); norms.push_back(v3(0, 1, 0));
    verts.push_back(v3(-rx, -ry, -rz)); norms.push_back(v3(0, 1, 0));

    return new Geometry(GraphicsDevice::Triangles, verts.size(), &verts[0], normals ? &norms[0] : 0, 0, 0);
}

Geometry *Cube::CreateFrontFace(float r) {
    vector<v3> verts; vector<v2> tex;
    verts.push_back(v3(-r,  r, -r)); tex.push_back(v2(1, 1)); 
    verts.push_back(v3(-r, -r, -r)); tex.push_back(v2(1, 0)); 
    verts.push_back(v3( r, -r, -r)); tex.push_back(v2(0, 0)); 
    verts.push_back(v3( r,  r, -r)); tex.push_back(v2(0, 1)); 
    verts.push_back(v3(-r,  r, -r)); tex.push_back(v2(1, 1)); 
    verts.push_back(v3( r, -r, -r)); tex.push_back(v2(0, 0)); 
    return new Geometry(GraphicsDevice::Triangles, verts.size(), &verts[0], 0, &tex[0]);
}

Geometry *Cube::CreateBackFace(float r) {
    vector<v3> verts; vector<v2> tex;
    verts.push_back(v3( r,  r,  r)); tex.push_back(v2(1, 1)); 
    verts.push_back(v3( r, -r,  r)); tex.push_back(v2(1, 0)); 
    verts.push_back(v3(-r, -r,  r)); tex.push_back(v2(0, 0)); 
    verts.push_back(v3(-r,  r,  r)); tex.push_back(v2(0, 1)); 
    verts.push_back(v3( r,  r,  r)); tex.push_back(v2(1, 1)); 
    verts.push_back(v3(-r, -r,  r)); tex.push_back(v2(0, 0)); 
    return new Geometry(GraphicsDevice::Triangles, verts.size(), &verts[0], 0, &tex[0]);
}

Geometry *Cube::CreateLeftFace(float r) {
    vector<v3> verts; vector<v2> tex;
    verts.push_back(v3(r,  r, -r)); tex.push_back(v2(1, 1));
    verts.push_back(v3(r, -r, -r)); tex.push_back(v2(1, 0));
    verts.push_back(v3(r, -r,  r)); tex.push_back(v2(0, 0));
    verts.push_back(v3(r,  r,  r)); tex.push_back(v2(0, 1));
    verts.push_back(v3(r,  r, -r)); tex.push_back(v2(1, 1));
    verts.push_back(v3(r, -r,  r)); tex.push_back(v2(0, 0));
    return new Geometry(GraphicsDevice::Triangles, verts.size(), &verts[0], 0, &tex[0]);
}

Geometry *Cube::CreateRightFace(float r) {
    vector<v3> verts; vector<v2> tex;
    verts.push_back(v3(-r,  r,  r)); tex.push_back(v2(1, 1));
    verts.push_back(v3(-r, -r,  r)); tex.push_back(v2(1, 0));
    verts.push_back(v3(-r, -r, -r)); tex.push_back(v2(0, 0));
    verts.push_back(v3(-r,  r, -r)); tex.push_back(v2(0, 1));
    verts.push_back(v3(-r,  r,  r)); tex.push_back(v2(1, 1));
    verts.push_back(v3(-r, -r, -r)); tex.push_back(v2(0, 0));
    return new Geometry(GraphicsDevice::Triangles, verts.size(), &verts[0], 0, &tex[0]);
}

Geometry *Cube::CreateTopFace(float r) {
    vector<v3> verts; vector<v2> tex;
    verts.push_back(v3( r,  r, -r)); tex.push_back(v2(1, 1));
    verts.push_back(v3( r,  r,  r)); tex.push_back(v2(1, 0));
    verts.push_back(v3(-r,  r,  r)); tex.push_back(v2(0, 0));
    verts.push_back(v3(-r,  r, -r)); tex.push_back(v2(0, 1));
    verts.push_back(v3( r,  r, -r)); tex.push_back(v2(1, 1));
    verts.push_back(v3(-r,  r,  r)); tex.push_back(v2(0, 0));
    return new Geometry(GraphicsDevice::Triangles, verts.size(), &verts[0], 0, &tex[0]);
}

Geometry *Cube::CreateBottomFace(float r) {
    vector<v3> verts; vector<v2> tex;
    verts.push_back(v3( r, -r,  r)); tex.push_back(v2(1, 1));
    verts.push_back(v3( r, -r, -r)); tex.push_back(v2(1, 0));
    verts.push_back(v3(-r, -r, -r)); tex.push_back(v2(0, 0));
    verts.push_back(v3(-r, -r,  r)); tex.push_back(v2(0, 1));
    verts.push_back(v3( r, -r,  r)); tex.push_back(v2(1, 1));
    verts.push_back(v3(-r, -r, -r)); tex.push_back(v2(0, 0));
    return new Geometry(GraphicsDevice::Triangles, verts.size(), &verts[0], 0, &tex[0]);
}

Geometry *Grid::Grid3D() {
    const float scaleFactor = 1;
    const float range = powf(10, scaleFactor);
    const float step = range/10;

    vector<v3> verts;
    for (float d = -range ; d <= range; d += step) {
        verts.push_back(v3(range,0,d));
        verts.push_back(v3(-range,0,d));
        verts.push_back(v3(d,0,range));
        verts.push_back(v3(d,0,-range));
    }

    return new Geometry(GraphicsDevice::Lines, verts.size(), &verts[0], 0, 0, Color(.5,.5,.5));
}

Geometry *Grid::Grid2D(float x, float y, float range, float step) {
    vector<v2> verts;
    for (float d = 0; d <= range; d += step) {
        verts.push_back(v2(x+range, y+d));
        verts.push_back(v2(x,       y+d));
        verts.push_back(v2(x+d, y+range));
        verts.push_back(v2(x+d, y));
    }

    return new Geometry(GraphicsDevice::Lines, verts.size(), &verts[0], 0, 0, Color(0,0,0));
}

void TextureArray::Load(const string &fmt, const string &prefix, const string &suffix, int N) {
    ind = 0;
    a.resize(N);
    for (int i=0, l=a.size(); i<l; i++)
        Asset::LoadTexture(StringPrintf(fmt.c_str(), prefix.c_str(), i, suffix.c_str()), &a[i]);
}

void TextureArray::DrawSequence(Asset *out, Entity *e) {
    ind = (ind + 1) % a.size();
    const Texture *in = &a[ind];
    out->tex.ID = in->ID;
    if (out->geometry) Scene::Draw(out->geometry, e);
}

void Tiles::AddBoxArray(const BoxArray &box, point p) {
    for (Drawable::Box::Iterator iter(box.data); !iter.Done(); iter.Increment()) {
        Drawable::Attr attr = box.attr.GetAttr(iter.cur_attr1);
        if (attr.bg) {
            ContextOpen();
            TilesPreAdd(this, &BoxRun::DrawBackground, BoxRun(0,0,attr), p, &BoxRun::DefaultDrawBackgroundCB);
            BoxRun(iter.Data(), iter.Length(), attr, VectorGet(box.line, iter.cur_attr2))
                .DrawBackground(p, [&] (const Box &w) { TilesAdd(this, &w, &Box::Draw, w, (float*)0); });
            ContextClose();
        }
        if (1) {
            ContextOpen();
            TilesPreAdd(this, &BoxRun::draw, BoxRun(0,0,attr), p);
            if (attr.scissor) {
                TilesPreAdd (this, &Tiles::PushScissor, this, *attr.scissor + p);
                TilesPostAdd(this, &GraphicsDevice::PopScissor, screen->gd);
            }
            BoxRun(iter.Data(), iter.Length(), attr)
                .Draw(p, [&] (const Drawable *d, const Box &w) { TilesAdd(this, &w, &Drawable::Draw, d, w); });
            ContextClose();
        }
    }
}

void Tiles::Run() {
    bool init = !fb.ID;
    if (init) fb.Create(W, H);
    current_tile = Box(0, 0, W, H);
    screen->gd->DrawMode(DrawMode::_2D);
    screen->gd->ViewPort(current_tile);
    screen->gd->EnableLayering();
    TilesMatrixIter(&mat) {
        if (!tile->cb.Size() && !clear_empty) continue;
        GetScreenCoords(i, j, &current_tile.x, &current_tile.y);
        if (!tile->id) fb.AllocTexture(&tile->id);
        fb.Attach(tile->id);
        screen->gd->MatrixProjection();
        if (clear) screen->gd->Clear();
        screen->gd->LoadIdentity();
        screen->gd->Ortho(current_tile.x, current_tile.x + W, current_tile.y, current_tile.y + H, 0, 100);
        screen->gd->MatrixModelview();
        tile->cb.Run();
    }
    fb.Release();
    screen->gd->RestoreViewport(DrawMode::_2D);
}

void Tiles::Draw(const Box &viewport, int scrolled_x, int scrolled_y) {
    screen->gd->SetColor(Color::white);
    int x1 = viewport.x + scrolled_x, y1 = viewport.y + scrolled_y, x2 = x1 + viewport.w, y2 = y1 + viewport.h;
    for (int ys, y = y1; y < y2; y += ys) {
        int ymh = y % H, yo = y >= 0 ? ymh : (ymh ? (H + ymh) : 0);
        ys = min(y2 - y, H - yo);

        for (int xs, x = x1; x < x2; x += xs) {
            int xo = (x % W), tx, ty;
            xs = min(x2 - x, W - xo);

            GetTileCoords(x, y, &tx, &ty);
            Tile *tile = GetTile(tx, ty);
            if (!tile || !tile->id) continue;

            Box w(x - scrolled_x, y - scrolled_y, xs, ys);
            Texture tex(0, 0, Pixel::RGBA, tile->id);
            tex.coord[0] = (float)xo/W; tex.coord[2] = (float)(xo+xs)/W;
            tex.coord[1] = (float)yo/H; tex.coord[3] = (float)(yo+ys)/H;
            tex.Draw(w);
            // glBox(w, &Color::white);
        }
    }
}

}; // namespace LFL
