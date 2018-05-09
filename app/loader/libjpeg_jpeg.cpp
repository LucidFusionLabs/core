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

#include <setjmp.h>
extern "C" {
#include "jpeglib.h"
};

namespace LFL {
const JOCTET EOI_BUFFER[1] = { JPEG_EOI };
struct MyJpegErrorMgr  { jpeg_error_mgr  pub; jmp_buf setjmp_buffer; };
struct MyJpegSourceMgr { jpeg_source_mgr pub; const JOCTET *data; size_t len; } ;

static void JpegErrorExit(j_common_ptr jcs) { longjmp(reinterpret_cast<MyJpegErrorMgr*>(jcs->err)->setjmp_buffer, 1); }
static void JpegInitSource(j_decompress_ptr jds) {}
static void JpegTermSource(j_decompress_ptr jds) {}
static boolean JpegFillInputBuffer(j_decompress_ptr jds) {
  MyJpegSourceMgr *src = reinterpret_cast<MyJpegSourceMgr*>(jds->src);
  src->pub.next_input_byte = EOI_BUFFER;
  src->pub.bytes_in_buffer = 1;
  return boolean(true);
}

static void JpegSkipInputData(j_decompress_ptr jds, long len) {
  MyJpegSourceMgr *src = reinterpret_cast<MyJpegSourceMgr*>(jds->src);
  if (src->pub.bytes_in_buffer < len) {
    src->pub.next_input_byte = EOI_BUFFER;
    src->pub.bytes_in_buffer = 1;
  } else {
    src->pub.next_input_byte += len;
    src->pub.bytes_in_buffer -= len;
  }
}

static void JpegMemSrc(j_decompress_ptr jds, const char *buf, size_t len) {
  if (!jds->src) jds->src = static_cast<jpeg_source_mgr*>((*jds->mem->alloc_small)(j_common_ptr(jds), JPOOL_PERMANENT, sizeof(MyJpegSourceMgr)));
  MyJpegSourceMgr *src = reinterpret_cast<MyJpegSourceMgr*>(jds->src);
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
  if (jpeg_read_header(&jds, boolean(true)) != 1) { jpeg_destroy_decompress(&jds); return ERRORv(-1, "jpeg decompress failed "); }
  jpeg_start_decompress(&jds);

  if      (jds.output_components == 1) out->pf = Pixel::GRAY8;
#ifndef WIN32
  else if (jds.output_components == 3) out->pf = Pixel::RGBA;
#else
  else if (jds.output_components == 3) out->pf = Pixel::RGB24;
#endif
  else if (jds.output_components == 4) out->pf = Pixel::RGBA;
  else { ERROR("unsupported jpeg components ", jds.output_components); jpeg_destroy_decompress(&jds); return -1; }

#ifdef JCS_EXTENSIONS
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

}; // namespace LFL
