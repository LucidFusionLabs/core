/*
 * $Id: camera.cpp 1330 2014-11-06 03:04:15Z justin $
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

#include <png.h>

namespace LFL {
static void PngRead (png_structp png_ptr, png_bytep data, png_size_t length) { static_cast<File*>(png_get_io_ptr(png_ptr))->Read (data, length); }
static void PngWrite(png_structp png_ptr, png_bytep data, png_size_t length) { static_cast<File*>(png_get_io_ptr(png_ptr))->Write(data, length); }
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
                                    if (opf == Pixel::BGRA || opf == Pixel::BGR24) png_set_bgr(png_ptr);
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

}; // namespace LFL
