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

#include "core/app/gl/view.h"
#include "core/app/ipc.h"
#include <zlib.h>

namespace LFL {
ZLibStream::ZLibStream() : UniqueVoidPtr(new z_stream) { memzerop(FromVoid<z_stream*>(*this)); }
void ZLibStream::reset(void *x) { if (v) delete FromVoid<z_stream*>(*this); v=x; }

ZLibReader::~ZLibReader() { if (stream) inflateEnd(FromVoid<z_stream*>(stream)); }
ZLibReader::ZLibReader(int bs) : blocksize(bs) {
  if (inflateInit(FromVoid<z_stream*>(stream)) != Z_OK) ERROR("inflateInit");
}

bool ZLibReader::Add(const StringPiece &in, bool partial_flush, bool final, int max_out) {
  if (!stream) return false;
  z_stream *zs = FromVoid<z_stream*>(stream);
  zs->next_in = reinterpret_cast<Bytef*>(const_cast<char*>(&in[0]));
  zs->avail_in = in.size();
  bool success;
  int status, len;
  do {
    out.resize(out.size() + blocksize);
    zs->next_out = reinterpret_cast<Bytef*>(&out[0] + out.size() - blocksize);
    zs->avail_out = blocksize;
    status = inflate(zs, partial_flush ? Z_PARTIAL_FLUSH : 0);
    len = blocksize - zs->avail_out;
    if (len < blocksize) out.resize(out.size() - blocksize + len);
    if (max_out && out.size() > max_out) return false;
    success = status == Z_OK || (status == Z_BUF_ERROR && len);
  } while(success && len == blocksize);
  bool ret = final ? (status == Z_STREAM_END) : success;
  if (!ret) ERROR("inflate ", status);
  return ret;
}

string ZLibReader::Decompress(const StringPiece &in) {
  ZLibReader zreader;
  return zreader.Add(in, false, true) ? zreader.out : "";
}

ZLibWriter::~ZLibWriter() { if (stream) deflateEnd(FromVoid<z_stream*>(stream)); }
ZLibWriter::ZLibWriter(int bs) : blocksize(bs) {
  if (deflateInit(FromVoid<z_stream*>(stream), Z_BEST_COMPRESSION) != Z_OK) ERROR("deflateInit");
}

bool ZLibWriter::Add(const StringPiece &in, bool partial_flush, bool final, int max_out) {
  if (!stream) return false;
  z_stream *zs = FromVoid<z_stream*>(stream);
  zs->next_in = reinterpret_cast<Bytef*>(const_cast<char*>(&in[0]));
  zs->avail_in = in.size();
  bool success;
  int status, len;
  do {
    out.resize(out.size() + blocksize);
    zs->next_out = reinterpret_cast<Bytef*>(&out[0] + out.size() - blocksize);
    zs->avail_out = blocksize;
    status = deflate(zs, partial_flush ? Z_PARTIAL_FLUSH : Z_FINISH);
    len = blocksize - zs->avail_out;
    if (len < blocksize) out.resize(out.size() - blocksize + len);
    if (max_out && out.size() > max_out) return false;
    success = status == Z_OK || (status == Z_BUF_ERROR && len);
  } while(success && len == blocksize);
  bool ret = final ? (status == Z_STREAM_END) : success;
  if (!ret) ERROR("deflate ", status);
  return ret;
}

string ZLibWriter::Compress(const StringPiece &in) {
  ZLibWriter zwriter;
  return zwriter.Add(in, false, true) ? zwriter.out : "";
}

int PngWriter::Write(const string &fn, const Texture &tex) {
  LocalFile lf(fn, "w");
  if (!lf.Opened()) return ERRORv(-1, "open: ", fn);
  return PngWriter::Write(&lf, tex);
}

bool WavReader::Open(File *F, WavHeader *out) {
  f = F;
  last = WavHeader::Size;
  return (f && out) ? File::SeekReadSuccess(f, 0, out, WavHeader::Size) : f != nullptr;
}

int WavReader::Read(RingSampler::Handle *B, int off, int num) {
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

int WavWriter::Write(const RingSampler::Handle *B, bool flush) {
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

struct AliasWavefrontObjLoader {
  typedef map<string, Material> MtlMap;
  MtlMap MaterialMap;
  string dir;

  unique_ptr<Geometry> LoadOBJ(File *file, const float *map_tex_coord=0) {
    if (!file->Opened()) return ERRORv(nullptr, "obj.open: ", file->Filename());
    const char *filename = file->Filename();
    dir.assign(filename, 0, DirNameLen(filename, true));

    NextRecordReader nr(file);
    vector<v3> vert, vert_out, norm, norm_out;
    vector<v2> tex, tex_out;
    Material mat; v3 xyz; v2 xy;
    int format=0, material=0, ind[3];

    for (const char *line = nr.NextLine(); line; line = nr.NextLine()) {
      if (!line[0] || line[0] == '#') continue;
      StringWordIter word(line);
      string cmd = word.NextString();
      if (word.Done()) continue;

      if (cmd == "mtllib") {
        LocalFile f(StrCat(dir, word.NextString()), "r");
        LoadMaterial(&f);
      } else if (cmd == "usemtl") {
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

  int LoadMaterial(File *file) {
    if (!file->Opened()) return ERRORv(-1, "LocalFile::open(", file->Filename(), ")");
    NextRecordReader nr(file);
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

unique_ptr<Geometry> Geometry::LoadOBJ(File *f, const float *map_tex_coord) {
  return AliasWavefrontObjLoader().LoadOBJ(f, map_tex_coord);
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

SimpleAssetLoader::SimpleAssetLoader(AssetLoading *p) : parent(p) { INFO("SimpleAssetLoader"); }
void SimpleAssetLoader::UnloadFile(void *h) { delete static_cast<File*>(h); }
void *SimpleAssetLoader::LoadFile(File *f) { return f; }
void *SimpleAssetLoader::LoadFileNamed(const string &filename) {
  unique_ptr<File> f(parent->OpenFile(filename));
  if (!f || !f->Opened()) return ERRORv(nullptr, "open: ", filename);
  return f.release();
}

void SimpleAssetLoader::UnloadAudioFile(AudioAssetLoader::Handle &h) { return UnloadFile(h.get()); }
AudioAssetLoader::Handle SimpleAssetLoader::LoadAudioFile(unique_ptr<File> f) { return AudioAssetLoader::Handle(this, f.release()); }
AudioAssetLoader::Handle SimpleAssetLoader::LoadAudioFileNamed(const string &filename) { return AudioAssetLoader::Handle(this, LoadFileNamed(filename)); }

void SimpleAssetLoader::UnloadVideoFile(VideoAssetLoader::Handle &h) { return UnloadFile(h.get()); }
VideoAssetLoader::Handle SimpleAssetLoader::LoadVideoFile(unique_ptr<File> f) { return VideoAssetLoader::Handle(this, f.release()); }
VideoAssetLoader::Handle SimpleAssetLoader::LoadVideoFileNamed(const string &filename) { return VideoAssetLoader::Handle(this, LoadFileNamed(filename)); }

void SimpleAssetLoader::UnloadMovieFile(MovieAssetLoader::Handle &h) { return UnloadFile(h.get()); }
MovieAssetLoader::Handle SimpleAssetLoader::LoadMovieFile(unique_ptr<File> f) { return MovieAssetLoader::Handle(this, f.release()); }
MovieAssetLoader::Handle SimpleAssetLoader::LoadMovieFileNamed(const string &filename) { return MovieAssetLoader::Handle(this, LoadFileNamed(filename)); }

void SimpleAssetLoader::LoadVideo(VideoAssetLoader::Handle &h, Texture *out, int load_flag) {
  static char jpghdr[2] = { '\xff', '\xd8' }; // 0xff, 0xd8
  static char gifhdr[4] = { 'G', 'I', 'F', '8' };
  static char pnghdr[8] = { '\211', 'P', 'N', 'G', '\r', '\n', '\032', '\n' };

  File *f = FromVoid<File*>(h);
  string fn = f->Filename();
  unsigned char hdr[8];
  if (f->Read(hdr, 8) != 8) return ERROR("load ", fn, " : failed");
  f->Reset();

  if (!memcmp(hdr, pnghdr, sizeof(pnghdr))) {
    if (PngReader::Read(f, out)) return;
  }
  else if (!memcmp(hdr, jpghdr, sizeof(jpghdr))) {
    if (JpegReader::Read(f, out)) return;
  }
  else if (!memcmp(hdr, gifhdr, sizeof(gifhdr))) {
    if (GIFReader::Read(f, out)) return;
  }
  else return ERROR("load ", fn, " : failed");

  if (load_flag & Flag::LoadGL) out->LoadGL((load_flag & Flag::RepeatGL) ? Texture::Flag::RepeatGL : 0);
  if (load_flag & Flag::Clear)  out->ClearBuffer();
}

void SimpleAssetLoader::LoadAudio(AudioAssetLoader::Handle &h, SoundAsset *a, int seconds, int flag) {
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
      return ERROR("LoadAudio(", a->name, ", ", fn, ") not supported or ", wav_samples, " > ",
                   FLAGS_sample_rate*FLAGS_soundasset_seconds);

    a->channels = 1;
    a->sample_rate = FLAGS_sample_rate;
    a->wav = make_unique<RingSampler>(a->sample_rate, wav_samples);
    RingSampler::Handle H(a->wav.get());
    if (wav_reader.Read(&H, 0, wav_samples))
      return ERROR("LoadAudio(", a->name, ", ", fn, ") read failed: ", strerror(errno));
  } else if (SuffixMatch(fn, ".ogg", false)) {
    if (!(a->handle = OGGReader::OpenFile(this, fn, &a->sample_rate, &a->channels, 0)))
      return ERROR("LoadOGG(", a->name, ", ", fn, ") failed");
    a->seconds = seconds;
    a->wav = make_unique<RingSampler>(a->sample_rate, SoundAsset::Size(a));
    RingSampler::Handle H(a->wav.get());
    int samples = OGGReader::Read(a->handle, a->channels, a->sample_rate*a->seconds, &H, 0);
    if (samples == SoundAsset::Size(a) && !(flag & SoundAsset::FlagNoRefill))
      a->refill = bind(&SimpleAssetLoader::RefillAudio, this, _1, _2);
    if (!a->refill) { OGGReader::Close(a->handle); a->handle=0; }
  }
}

int SimpleAssetLoader::RefillAudio(SoundAsset *a, int reset) {
  if (!a->handle) return 0;
  a->wav->ring.back = 0;
  RingSampler::Handle H(a->wav.get());
  int wrote = OGGReader::Read(a->handle, a->channels, a->sample_rate*a->seconds, &H, reset);
  if (wrote < SoundAsset::Size(a)) {
    a->wav->ring.size = wrote;
    a->wav->bytes = a->wav->ring.size * a->wav->width;
  }
  return wrote;
}

void SimpleAssetLoader::LoadMovie(MovieAssetLoader::Handle &h, MovieAsset *a) {}
int SimpleAssetLoader::PlayMovie(MovieAsset *a, int seek) { return 0; }

unique_ptr<AssetLoaderInterface> CreateSimpleAssetLoader(AssetLoading *p) { return make_unique<SimpleAssetLoader>(p); }

void SimpleVideoResampler::RGB2BGRCopyPixels(unsigned char *dst, const unsigned char *src, int l, int bpp) {
  for (int k = 0; k < l; k++) for (int i = 0; i < bpp; i++) dst[k*bpp+(!i?2:(i==2?0:i))] = src[k*bpp+i];
}

bool SimpleVideoResampler::Supports(int f) { return f == Pixel::RGB24 || f == Pixel::BGR24 || f == Pixel::RGB32 || f == Pixel::BGR32 || f == Pixel::RGBA || f == Pixel::ARGB; }

bool SimpleVideoResampler::Opened() const { return s_fmt && d_fmt && s_width && d_width && s_height && d_height; }

void SimpleVideoResampler::Open(int sw, int sh, int sf, int dw, int dh, int df) {
  s_fmt = sf; s_width = sw; s_height = sh;
  d_fmt = df; d_width = dw; d_height = dh;
  // INFO("resample ", BlankNull(Pixel::Name(s_fmt)), " -> ", BlankNull(Pixel::Name(d_fmt)), " : (", sw, ",", sh, ") -> (", dw, ",", dh, ")");
}

void SimpleVideoResampler::Resample(const unsigned char *sb, int sls, unsigned char *db, int dls, bool flip_x, bool flip_y) {
  if (!Opened()) return ERROR("resample not opened()");

  int spw = Pixel::Size(s_fmt), dpw = Pixel::Size(d_fmt);
  if (spw * s_width > sls) return ERROR(spw * s_width, " > ", sls);
  if (dpw * d_width > dls) return ERROR(dpw * d_width, " > ", dls);

  if (s_width == d_width && s_height == d_height) {
    for (int y=0; y<d_height; y++) {
      for (int x=0; x<d_width; x++) {
        const unsigned char *sp = (sb + sls * y                           + x                          * spw);
        /**/  unsigned char *dp = (db + dls * (flip_y ? d_height-1-y : y) + (flip_x ? d_width-1-x : x) * dpw);
        CopyPixel(s_fmt, d_fmt, sp, dp, x == 0, x == d_width-1);
      }
    }
  } else {
    for (int poi=0; poi<4 && poi<4; poi++) {
      int pi = Pixel::GetRGBAIndex(s_fmt, poi), po = Pixel::GetRGBAIndex(d_fmt, poi);
      if (pi < 0 || po < 0) continue;
      Matrix M(s_height, s_width);
      CopyColorChannelsToMatrix(sb, s_width, s_height, spw, sls, 0, 0, &M, pi);
      for (int y=0; y<d_height; y++) {
        for (int x=0; x<d_width; x++) {
          unsigned char *dp = (db + dls * (flip_y ? d_height-1-y : y) + (flip_x ? d_width-1-x : x) * dpw);
          *(dp + po) = MatrixAsFunc(&M, x?float(x)/(d_width-1):0, y?float(y)/(d_height-1):0) * 255;
        }
      }
    }
  }
}

void SimpleVideoResampler::CopyPixel(int s_fmt, int d_fmt, const unsigned char *sp, unsigned char *dp, bool sxb, bool sxe, int f) {
  unsigned char r, g, b, a;
  switch (s_fmt) {
    case Pixel::RGB24: r = *sp++; g = *sp++; b = *sp++; a = ((f & Flag::TransparentBlack) && !r && !g && !b) ? 0 : 255; break;
    case Pixel::BGR24: b = *sp++; g = *sp++; r = *sp++; a = ((f & Flag::TransparentBlack) && !r && !g && !b) ? 0 : 255; break;
    case Pixel::RGB32: r = *sp++; g = *sp++; b = *sp++; a=255; sp++; break;
    case Pixel::BGR32: r = *sp++; g = *sp++; b = *sp++; a=255; sp++; break;
    case Pixel::RGBA:  r = *sp++; g = *sp++; b = *sp++; a=*sp++; break;
    case Pixel::BGRA:  b = *sp++; g = *sp++; r = *sp++; a=*sp++; break;
    case Pixel::ARGB:  a = *sp++; r = *sp++; g = *sp++; b=*sp++; break;
    case Pixel::GRAY8: r = 255;   g = 255;   b = 255;   a=*sp++; break;
    // case Pixel::GRAY8: r = g = b = a = *sp++; break;
    case Pixel::LCD: 
      r = (sxb ? 0 : *(sp-1)) / 3.0 + *sp / 3.0 + (          *(sp+1)) / 3.0; sp++; 
      g = (          *(sp-1)) / 3.0 + *sp / 3.0 + (          *(sp+1)) / 3.0; sp++; 
      b = (          *(sp-1)) / 3.0 + *sp / 3.0 + (sxe ? 0 : *(sp+1)) / 3.0; sp++;
      a = ((f & Flag::TransparentBlack) && !r && !g && !b) ? 0 : 255;
      break;
    default: return ERROR("s_fmt ", s_fmt, " not supported");
  }
  switch (d_fmt) {
    case Pixel::RGB24: *dp++ = r; *dp++ = g; *dp++ = b; break;
    case Pixel::BGR24: *dp++ = b; *dp++ = g; *dp++ = r; break;
    case Pixel::RGB32: *dp++ = r; *dp++ = g; *dp++ = b; *dp++ = a; break;
    case Pixel::BGR32: *dp++ = b; *dp++ = g; *dp++ = r; *dp++ = a; break;
    case Pixel::RGBA:  *dp++ = r; *dp++ = g; *dp++ = b; *dp++ = a; break;
    case Pixel::BGRA:  *dp++ = b; *dp++ = g; *dp++ = r; *dp++ = a; break;
    case Pixel::ARGB:  *dp++ = a; *dp++ = r; *dp++ = g; *dp++ = b; break;
    default: return ERROR("d_fmt ", d_fmt, " not supported");
  }
}

void SimpleVideoResampler::Blit(const unsigned char *src, unsigned char *dst, int w, int h,
                                int sf, int sls, int sx, int sy,
                                int df, int dls, int dx, int dy, int flag) {
  bool flip_y = flag & Flag::FlipY;
  int sw = Pixel::Size(sf), dw = Pixel::Size(df); 
  for (int yi = 0; yi < h; ++yi) {
    for (int xi = 0; xi < w; ++xi) {
      int sind = flip_y ? sy + h - yi - 1 : sy + yi;
      const unsigned char *sp = src + (sls*(sind)    + (sx + xi)*sw);
      unsigned       char *dp = dst + (dls*(dy + yi) + (dx + xi)*dw);
      CopyPixel(sf, df, sp, dp, xi == 0, xi == w-1, flag);
    }
  }
}

void SimpleVideoResampler::Filter(unsigned char *buf, int w, int h,
                                  int pf, int ls, int x, int y,
                                  Matrix *kernel, int channel, int flag) {
  Matrix M(h, w), out(h, w);
  int pw = Pixel::Size(pf);
  CopyColorChannelsToMatrix(buf, w, h, pw, ls, x, y, &M, ColorChannel::PixelOffset(channel));
  Matrix::Convolve(&M, kernel, &out, (flag & Flag::ZeroOnly) ? mZeroOnly : 0);
  CopyMatrixToColorChannels(&out, w, h, pw, ls, x, y, buf, ColorChannel::PixelOffset(channel));
}

void SimpleVideoResampler::Fill(unsigned char *dst, int len, int pf, const Color &c) {
  int pw = Pixel::Size(pf); 
  unsigned char pixel[4] = { uint8_t(c.R()), uint8_t(c.G()), uint8_t(c.B()), uint8_t(c.A()) };
  for (int i = 0; i != len; ++i) CopyPixel(Pixel::RGBA, pf, pixel, dst + i*pw);
}

void SimpleVideoResampler::Fill(unsigned char *dst, int w, int h,
                                int pf, int ls, int x, int y, const Color &c) {
  int pw = Pixel::Size(pf); 
  unsigned char pixel[4] = { uint8_t(c.R()), uint8_t(c.G()), uint8_t(c.B()), uint8_t(c.A()) };
  for (int yi = 0; yi < h; ++yi) {
    for (int xi = 0; xi < w; ++xi) {
      unsigned char *dp = dst + (ls*(y + yi) + (x + xi)*pw);
      CopyPixel(Pixel::RGBA, pf, pixel, dp);
    }
  }
}

void SimpleVideoResampler::CopyColorChannelsToMatrix(const unsigned char *buf, int w, int h,
                                                     int pw, int ls, int x, int y,
                                                     Matrix *out, int po) {
  MatrixIter(out) { 
    const unsigned char *p = buf + (ls*(y + i) + (x + j)*pw);
    out->row(i)[j] = *(p + po) / 255.0;
  }
}

void SimpleVideoResampler::CopyMatrixToColorChannels(const Matrix *M, int w, int h,
                                                     int pw, int ls, int x, int y,
                                                     unsigned char *out, int po) {
  MatrixIter(M) { 
    unsigned char *p = out + (ls*(y + i) + (x + j)*pw);
    *(p + po) = M->row(i)[j] * 255.0;
  }
}

AssetLoader::AssetLoader(AssetLoading *l) : loading(l) {}
AssetLoader::~AssetLoader() {}

int AssetLoader::Init() {
  default_loader = CreateAssetLoader(loading);
  default_audio_loader = default_loader.get();
  default_video_loader = default_loader.get();
  default_movie_loader = default_loader.get();
  return 0;
}

}; // namespace LFL
