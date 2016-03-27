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

#include "core/app/gui.h"
#include "core/app/ipc.h"

namespace LFL {
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

struct SimpleAssetLoader : public AssetLoaderInterface {
  SimpleAssetLoader() { INFO("SimpleAssetLoader"); }

  virtual void *LoadFile(const string &filename) {
    unique_ptr<LocalFile> lf = make_unique<LocalFile>(filename, "r");
    if (!lf->Opened()) return ERRORv(nullptr, "open: ", filename);
    return lf.release();
  }
  virtual void UnloadFile(void *h) { delete static_cast<File*>(h); }
  virtual void *LoadBuf(const char *buf, int len, const char *mimetype) { return new BufferFile(string(buf, len), mimetype); }
  virtual void UnloadBuf(void *h) { delete static_cast<File*>(h); }

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

    File *f = static_cast<File*>(h);
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

    if (load_flag & Flag::LoadGL) out->LoadGL();
    if (load_flag & Flag::Clear)  out->ClearBuffer();
  }

  virtual void LoadAudio(void *h, SoundAsset *a, int seconds, int flag) {
    File *f = static_cast<File*>(h);
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
    }
  }
  virtual int RefillAudio(SoundAsset *a, int reset) { return 0; }

  virtual void LoadMovie(void *h, MovieAsset *a) {}
  virtual int PlayMovie(MovieAsset *a, int seek) { return 0; }
};

unique_ptr<AssetLoaderInterface> CreateSimpleAssetLoader() { return make_unique<SimpleAssetLoader>(); }

AssetLoader::AssetLoader() {}
AssetLoader::~AssetLoader() {}

int AssetLoader::Init() {
  default_loader = CreateAssetLoader();
  default_audio_loader = default_loader.get();
  default_video_loader = default_loader.get();
  default_movie_loader = default_loader.get();
  return 0;
}

}; // namespace LFL
