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

#ifndef LFL_CORE_APP_FILE_H__
#define LFL_CORE_APP_FILE_H__
namespace LFL {
  
struct IOVec { long long offset; ptrdiff_t len; };
struct IOVector : public vector<IOVec> { int Append(const IOVec&); };

struct MIMEType {
  static bool Jpg(const string &mt) { return mt == "image/jpg" || mt == "image/jpeg"; }
  static bool Png(const string &mt) { return mt == "image/png"; }
};

struct FileName {
  static bool Match(const string &url, const string &fn, bool match_case) { 
    return url == fn ||
      (url.size() > fn.size() && *(url.end() - fn.size() - 1) == '/' && SuffixMatch(url, fn, match_case));
  }
};

struct FileType {
  enum { HTML=1, Image=2, Jpg=3, Png=4, Gif=5, Bmp=6, C=7, CPP=8, CMake=9 };
};

struct FileSuffix {
  static bool HTML (const string &url) { return SuffixMatch(url, ".html", 0) || SuffixMatch(url, ".txt", 0); }
  static bool Image(const string &url) { return Jpg(url) || Png(url) || Gif(url) || Bmp(url); }
  static bool Jpg  (const string &url) { return SuffixMatch(url, ".jpg", 0) || SuffixMatch(url, ".jpeg", 0); }
  static bool Png  (const string &url) { return SuffixMatch(url, ".png", 0); }
  static bool Gif  (const string &url) { return SuffixMatch(url, ".gif", 0); }
  static bool Bmp  (const string &url) { return SuffixMatch(url, ".bmp", 0); }
  static bool C    (const string &url) { return SuffixMatch(url, ".c", 0) || SuffixMatch(url, ".h"); }
  static bool CPP  (const string &url) { return C(url) || SuffixMatch(url, ".cpp", 0) || SuffixMatch(url, ".cc", 0); }
  static bool CMake(const string &url) { return SuffixMatch(url, ".cmake", 0) || FileName::Match(url, "CMakeLists.txt", 0); }
};

struct File {
  virtual ~File() {}
  virtual bool Opened() const = 0;
  virtual void Close() = 0;
  virtual bool Open(const string &path, const string &mode, bool pre_create=0) = 0;
  virtual const char *Filename() const = 0;
  virtual int Size() = 0;
  virtual void Reset() = 0;

  struct Whence { enum { SET=SEEK_SET, CUR=SEEK_CUR, END=SEEK_END }; int x; };
  virtual long long Seek(long long offset, int whence) = 0;
  virtual int Read(void *buf, size_t size) = 0;
  virtual int Write(const void *buf, size_t size=-1) = 0;
  virtual bool Flush() { return false; }

  virtual unique_ptr<File> Create() { return NULL; }
  virtual bool ReplaceWith(unique_ptr<File>) { return false; }

  string Contents();
  string ReadString(int size);
  string ReadString(long long pos, int size);
  int ReadIOV(void *buf, const IOVec*, int iovlen);
  int WriteString(const string &b) { return Write(b.c_str(), b.size()); }
  int Rewrite(const ArrayPiece<IOVec> &v, const function<string(int)> &e);
  int Rewrite(const ArrayPiece<IOVec> &v, const function<string(int)> &e, File *out_file);

  static bool ReadSuccess(File *f, void *out, int len) { return f->Read(out, len) == len; }
  static bool SeekSuccess(File *f, long long pos) { return f->Seek(pos, Whence::SET) == pos; }
  static bool SeekReadSuccess(File *f, long long pos, void *out, int len) { return SeekSuccess(f, pos) ? ReadSuccess(f, out, len) : false; }
  static bool WriteSuccess(File *f, void *out, int len) { return f->Write(out, len) == len; }
};

struct BufferFile : public File {
  StringPiece ptr;
  string buf, fn;
  int rdo=0, wro=0;
  bool owner;
  BufferFile(string           &&s, const char *FN=0) : buf(s), fn(FN?FN:""), owner(1) {}
  BufferFile(const string      &s, const char *FN=0) : buf(s), fn(FN?FN:""), owner(1) {}
  BufferFile(const StringPiece &s, const char *FN=0) : ptr(s), fn(FN?FN:""), owner(0) {}
  ~BufferFile() { Close(); }

  bool Opened() const override { return true; }
  bool Open(const string &path, const string &mode, bool pre_create=0) override { return false; }
  const char *Filename() const override { return fn.c_str(); }
  int Size() override { return owner ? buf.size() : ptr.len; }
  void Reset() override { rdo=wro=0; }
  void Close() override { ptr.assign(0,0); buf.clear(); Reset(); }

  long long Seek(long long pos, int whence) override;
  int Read(void *out, size_t size) override;
  int Write(const void *in, size_t size=-1) override;

  unique_ptr<File> Create() override;
  bool ReplaceWith(unique_ptr<File>) override;
};

struct LocalFile : public File {
  static const char Slash, ExecutableSuffix[];

  FILE *impl=0;
  string fn;
  bool writable=0;
  virtual ~LocalFile() { Close(); }
  LocalFile() {}
  LocalFile(const string &path, const string &mode, bool pre_create=0) : impl(0) { Open(path, mode, pre_create); }

  bool Opened() const override { return impl; }
  bool Open(const string &path, const string &mode, bool pre_create=0) override;
  const char *Filename() const override { return fn.c_str(); }
  int Size() override;
  void Reset() override;
  void Close() override;

  long long Seek(long long pos, int whence) override;
  int Read(void *buf, size_t size) override;
  int Write(const void *buf, size_t size=-1) override;
  bool Flush() override;

  unique_ptr<File> Create() override;
  bool ReplaceWith(unique_ptr<File>) override;

  static int WhenceMap(int n);
  static bool chdir(const string &dir);
  static bool mkdir(const string &dir, int mode);
  static bool unlink(const string &fn);
  static int IsFile(const string &localfilename);
  static int IsDirectory(const string &localfilename);
  static int CreateTemporary(ApplicationInfo*, const string &prefix, string *localfilename_out);
  static string CreateTemporaryName(ApplicationInfo*, const string &prefix);
  static string CreateTemporaryNameTemplate(ApplicationInfo*, const string &prefix);
  static string CurrentDirectory(int max_size=1024);
  static string JoinPath(const string &x, const string &y);
  static string FileContents(const string &localfilename) { return LocalFile(localfilename, "r").Contents(); }
  static int WriteFile(const string &path, const StringPiece &sp) {
    LocalFile file(path, "w");
    return file.Opened() ? file.Write(sp.data(), sp.size()) : -1;
  }

  static FILE *FOpen(const char *fn, const char *mode);
  static FILE *FdOpen(int, const char *mode);
  static FILE *FReopen(const char *fn, const char *mode, FILE *stream);
};

struct SearchPaths {
  vector<string> path;
  SearchPaths(const char *paths);
  string Find(const string &fn);
};

struct FileLineIter : public StringIter {
  File *f;
  NextRecordReader nr;
  FileLineIter(File *F, int fo=0) : f(F), nr(f, fo) {}
  const char *Next() override { return nr.NextLine(); }
  void Reset() override { f->Reset(); nr.Reset(); }
  bool Done() const override { return nr.buf_offset < 0; }
  const char *Begin() const override { return 0; }
  const char *Current() const override { return nr.buf.c_str() + nr.record_offset; }
  int CurrentOffset() const override { return nr.file_offset; }
  int CurrentLength() const override { return nr.record_len; }
  int TotalLength() const override { return 0; }
};

struct LocalFileLineIter : public StringIter {
  LocalFile f;
  NextRecordReader nr;
  LocalFileLineIter(const string &path) : f(path, "r"), nr(&f) {};
  const char *Next() override { return nr.NextLine(); }
  void Reset() override { f.Reset(); nr.Reset(); }
  bool Done() const override { return nr.buf_offset < 0; }
  const char *Begin() const override { return 0; }
  const char *Current() const override { return nr.buf.c_str() + nr.record_offset; }
  int CurrentOffset() const override { return nr.file_offset; }
  int CurrentLength() const override { return nr.record_len; }
  int TotalLength() const override { return 0; }
};

struct BufferFileLineIter : public StringIter {
  BufferFile f;
  NextRecordReader nr;
  BufferFileLineIter(const string &s) : f(s), nr(&f) {};
  const char *Next() override { return nr.NextLine(); }
  void Reset() override { f.Reset(); nr.Reset(); }
  bool Done() const override { return nr.buf_offset < 0; }
  const char *Begin() const override { return 0; }
  const char *Current() const override { return nr.buf.c_str() + nr.record_offset; }
  int CurrentOffset() const override { return nr.file_offset; }
  int CurrentLength() const override { return nr.record_len; }
  int TotalLength() const override { return 0; }
};

struct DirectoryIter {
  typedef map<string, int> Map;
  string pathname;
  Map filemap;
  Map::iterator iter;
  const char *P=0, *S=0;
  bool init=0;
  DirectoryIter() {}
  DirectoryIter(const string &path, int dirs=0, const char *FilePrefix=0, const char *FileSuffix=0);
  const char *Next();
  static void Add(void *self, const char *k, int v) { reinterpret_cast<DirectoryIter*>(self)->filemap[k] = v; }
};

struct ArchiveIter {
  string buf;
  void *impl=0, *entry=0;
  ArchiveIter(const char *path);
  ArchiveIter() {}
  ~ArchiveIter();
  void Skip();
  const char *Next();
  long long Size();
  bool LoadData();
};

struct FileOffset {
  unsigned offset, y, x;
  FileOffset(unsigned O=0, unsigned Y=0, unsigned X=0) : offset(O), y(Y), x(X) {}
};

struct FileNameAndOffset : public FileOffset {
  string fn;
  FileNameAndOffset(const string &F=string(), unsigned O=0, unsigned Y=0, unsigned X=0) : FileOffset(O, Y, X), fn(F) {}
};

struct VersionedFileName {
  const char *dir, *_class, *var;
  VersionedFileName(const char *D=0, const char *C=0, const char *V=0) : dir(D), _class(C), var(V) {}
  string DebugString() const { return StrCat(BlankNull(dir), BlankNull(_class), BlankNull(var)); }
};

struct ContainerFileHeader {
  static const int size = sizeof(int)*2, magic = 0xfefe;
  int flag, len; 
  ContainerFileHeader() : len(0) { SetFlag(0); }
  ContainerFileHeader(int f) : len(0) { SetFlag(f); }
  ContainerFileHeader(const char *text);

  void Validate() const { if (((flag>>16)&0xffff) != magic) FATAL("magic check"); }
  void SetLength(int v) { Validate(); len = v; }
  void SetFlag(unsigned short v) { flag = (magic<<16) | v; }
  unsigned short GetFlag() const { return flag & 0xffff; }
};

struct ContainerFile {
  unique_ptr<File> file;
  bool done=0;
  int read_offset=0, write_offset=-1;
  NextRecordReader nr;
  ContainerFile(const string &fn=string()) : nr(file.get()) { Open(fn); }

  bool Opened() const { return file && file->Opened(); }
  void Open(const string &fn);
  int Add(const StringPiece &msg, int status=0);
  bool Update(int offset, const ContainerFileHeader *ph, const StringPiece &msg);
  bool UpdateFlag(int offset, int status);
  bool Get(StringPiece *out, int offset, int status=-1);
  bool Next(StringPiece *out, int *offsetOut=0, int status=-1);
  bool Next(ContainerFileHeader *hdr, StringPiece *out, int *offsetOut=0, int status=-1);

  static int WriteEntry(File*, ContainerFileHeader *hdr, const StringPiece &msg, bool flush=0);
  static int WriteEntry(File*, const ContainerFileHeader *hdr, const StringPiece &msg, bool flush=0);
  static int WriteEntryFlag(File*, const ContainerFileHeader *hdr, bool flush=0);
};

struct ProtoFile : public ContainerFile {
  using ContainerFile::ContainerFile;
  int Add(const Proto *msg, int status=0);
  bool Update(int offset, const ContainerFileHeader *ph, const Proto *msg);
  bool Get(Proto *out, int offset, int status=-1);
  bool Next(Proto *out, int *offsetOut=0, int status=-1);
  bool Next(ContainerFileHeader *hdr, Proto *out, int *offsetOut=0, int status=-1);
};

struct FlatFile : public ContainerFile {
  using ContainerFile::ContainerFile;
  int Add(const FlatBufferPiece &msg, int status=0);
  bool Update(int offset, const ContainerFileHeader *ph, const FlatBufferPiece &msg);
  template <class X> const X* Get(                         int o,     int s=-1) { StringPiece b; return ContainerFile::Get (   &b, o,  s) ? flatbuffers::GetRoot<X>(b.buf) : nullptr; }
  template <class X> const X* Next(                        int *oo=0, int s=-1) { StringPiece b; return ContainerFile::Next(   &b, oo, s) ? flatbuffers::GetRoot<X>(b.buf) : nullptr; }
  template <class X> const X* Next(ContainerFileHeader *h, int *oo=0, int s=-1) { StringPiece b; return ContainerFile::Next(h, &b, oo, s) ? flatbuffers::GetRoot<X>(b.buf) : nullptr; } 
};

struct StringFile {
  optional_ptr<vector<string>> F;
  string H;
  StringFile() {}
  StringFile(optional_ptr<vector<string>> f, const string &h=string()) : F(move(f)), H(h) {}

  void Clear() { F.reset(); H.clear(); }
  void Print(const string &name, bool nl=1);
  int Lines() const { return F ? F->size() : 0; }
  string Line(int i) const { return (F && i < F->size()) ? (*F)[i] : ""; }
  void MoveTo(unique_ptr<vector<string>> *Fo, string *Ho) { if (Fo) *Fo=F.release(); if (Ho) *Ho=H; Clear(); }

  int ReadVersioned (const VersionedFileName &fn, ApplicationLifetime *life=0, int iter=-1);
  int WriteVersioned(const VersionedFileName &fn, int iter, const string &hdr=string());
  int WriteVersioned(const char *D, const char *C, const char *V, int iter, const string &hdr=string())
  { return WriteVersioned(VersionedFileName(D, C, V), iter, hdr); }

  int Read(const string &path, int header=1);
  int Read(IterWordIter *word, int header);

  int Write(File         *file, const string &name);
  int Write(const string &path, const string &name);
  static int WriteRow(File *file, const string &rowval);

  static int Read(const string &fn, unique_ptr<vector<string>> *F, string *H)
  { StringFile f; int ret=f.Read(fn); f.MoveTo(F, H); return ret; }
  static int ReadVersioned(const VersionedFileName &fn, unique_ptr<vector<string>> *F, string *H, int iter=-1)
  { StringFile f; int ret=f.ReadVersioned(fn); f.MoveTo(F, H); return ret; }
  static int ReadVersioned(const char *D, const char *C, const char *V, unique_ptr<vector<string>> *F, string *H, int iter=-1)
  { return ReadVersioned(VersionedFileName(D, C, V), F, H, iter); }
};

struct SettingsFile {
  static const char *VarType() { return "string"; }
  static const char *VarName() { return "settings"; }
  static const char *Separator() { return " = "; }

  static int Load(ApplicationInfo*);
  static int Read(const string &dir, const string &name);
  static int Write(const vector<string> &fields, const string &dir, const string &name);
  static int Save(ApplicationInfo*, const vector<string> &fields);
};

struct MatrixFile {
  struct Header { enum { NONE=0, DIM_PLUS=1, DIM=2 }; };
  struct BinaryHeader{ int magic, M, N, name, transcript, data, unused1, unused2; };

  optional_ptr<Matrix> F;
  string H;
  MatrixFile() { Clear(); }
  MatrixFile(optional_ptr<Matrix> f, const string &h=string()) : F(move(f)), H(h) {}

  void Clear() { F.reset(); H.clear(); }
  const char *Text() { return H.c_str(); }
  void MoveTo(unique_ptr<Matrix> *Fo, string *Ho) { if (Fo) *Fo=F.release(); if (Ho) *Ho=H; Clear(); }

  int ReadVersioned       (const VersionedFileName &fn, ApplicationLifetime *life=0, int iteration=-1);
  int WriteVersioned      (const VersionedFileName &fn, int iteration);
  int WriteVersionedBinary(const VersionedFileName &fn, int iteration);
  int WriteVersioned(const char *D, const char *C, const char *V, int iter)
  { return WriteVersioned(VersionedFileName(D, C, V), iter); }
  int WriteVersionedBinary(const char *D, const char *C, const char *V, int iter)
  { return WriteVersionedBinary(VersionedFileName(D, C, V), iter); }

  int Read(IterWordIter *word, int header=1);
  int Read(File         *file, int header=1, int(*IsSpace)(int)=0);
  int Read(const string &path, int header=1, int(*IsSpace)(int)=0);
  int ReadBinary(const string &path);

  int Write      (File         *file, const string &name);
  int Write      (const string &path, const string &name);
  int WriteBinary(File         *file, const string &name);
  int WriteBinary(const string &path, const string &name);

  static string Filename(const VersionedFileName &fn, const string &suf, int iter) { return Filename(fn._class, fn.var, suf, iter); }
  static string Filename(const string &_class, const string &var, const string &suffix, int iteration);
  static int FindHighestIteration(const VersionedFileName &fn, const string &suffix, ApplicationLifetime *life=0);
  static int FindHighestIteration(const VersionedFileName &fn, const string &suffix1, const string &suffix2, ApplicationLifetime *life=0);
  static int ReadHeader    (IterWordIter *word, string *hdrout);
  static int ReadDimensions(IterWordIter *word, int *M, int *N);
  static int WriteHeader      (File *file, const string &name, const string &hdr, int M, int N);
  static int WriteBinaryHeader(File *file, const string &name, const string &hdr, int M, int N);
  static int WriteRow         (File *file, const double *row, int N, bool lastrow=0);

  static int Read(IterWordIter *fd, unique_ptr<Matrix> *F, string *H)
  { MatrixFile f; int ret=f.Read(fd); f.MoveTo(F, H); return ret; }
  static int Read(const string &fn, unique_ptr<Matrix> *F, string *H)
  { MatrixFile f; int ret=f.Read(fn); f.MoveTo(F, H); return ret; }
  static int ReadVersioned(const VersionedFileName &fn, unique_ptr<Matrix> *F, string *H, int iter=-1)
  { MatrixFile f; int ret=f.ReadVersioned(fn); f.MoveTo(F, H); return ret; }
  static int ReadVersioned(const char *D, const char *C, const char *V, unique_ptr<Matrix> *F, string *H=0, int iter=-1)
  { return ReadVersioned(VersionedFileName(D, C, V), F, H, iter); }
};

struct MatrixArchiveInputFile {
  unique_ptr<IterWordIter> file;
  int index=0;
  MatrixArchiveInputFile(const string &name=string());
  ~MatrixArchiveInputFile();

  void Close();
  int Open(const string &name);
  int Read(unique_ptr<Matrix> *out, string *hdrout);
  int Read(MatrixFile *f) { unique_ptr<Matrix> x; int v=Read(&x, &f->H); f->F=move(x); return v; }
  int Skip();
  string Filename();
  static int Count(const string &name);
};

struct MatrixArchiveOutputFile {
  unique_ptr<File> file;
  MatrixArchiveOutputFile(const string &name=string());
  ~MatrixArchiveOutputFile();

  void Close();
  int Open(const string &name);
  int Write(Matrix*, const string &hdr, const string &name);
  int Write(const MatrixFile *f, const string &name) { return Write(f->F.get(), f->H, name); }
};

template <class X, void (*Assign)(double *, X), bool (*Equals)(const double*, X)>
struct HashMatrixT {
  optional_ptr<Matrix> map;
  int VPE;
  HashMatrixT(optional_ptr<Matrix> M=optional_ptr<Matrix>(), int vpe=0) : map(move(M)), VPE(vpe) {}

  const double *Get(X hash) const {
    const double *hashrow = map->row(hash % map->M);
    for (int k=0, l=map->N/VPE; k<l; k++) if (Equals(&hashrow[k*VPE], hash)) return &hashrow[k*VPE];
    return 0;
  }

  double *Get(X hash) {
    double *hashrow = map->row(hash % map->M);
    for (int k=0, l=map->N/VPE; k<l; k++) if (Equals(&hashrow[k*VPE], hash)) return &hashrow[k*VPE];
    return 0;
  }

  double *Set(X hash) {
    long long ind = hash % map->M;
    double *hashrow = map->row(ind);
    for (int k=0; k<map->N/VPE; k++) {
      int hri = k*VPE;
      if (hashrow[hri]) {
        if (Equals(&hashrow[hri], hash)) { ERROR("hash collision or duplicate insert ", hash); break; }
        continue;
      }
      Assign(&hashrow[hri], hash);
      return &hashrow[hri];
    }
    return 0;
  }
};

template <class X, void (*Assign)(double *, X), bool (*Equals)(const double*, X)>
struct HashMatrixFileT {
  File *lf;
  int M, N, hdr_size, VPE;
  HashMatrixFileT(File *F=0, int m=0, int n=0, int hs=0, int vpe=0) : lf(F), M(m), N(n), hdr_size(hs), VPE(vpe) {}

  double *SetBinary(X hash, double *hashrow) {
    long long ind = hash % M, row_size = N * sizeof(double), offset = hdr_size + ind * row_size, ret;
    if ((ret = lf->Seek(offset, File::Whence::SET)) != offset) { ERROR("seek: ", offset,   " != ", ret); return 0; } 
    if ((ret = lf->Read(hashrow, row_size))       != row_size) { ERROR("read: ", row_size, " != ", ret); return 0; }

    for (int k=0; k<N/VPE; k++) {
      int hri = k*VPE;
      if (hashrow[hri]) {
        if (Equals(&hashrow[hri], hash)) { ERROR("hash collision or duplicate insert ", hash); break; }
        continue;
      }
      int hri_offset = offset + hri * sizeof(double);
      if ((ret = lf->Seek(hri_offset, File::Whence::SET)) != hri_offset) { ERROR("seek: ", hri_offset, " != ", ret); return 0; } 
      Assign(&hashrow[hri], hash);
      return &hashrow[hri];
    }
    return 0;
  }
  void SetBinaryFlush(const double *hashrow) {
    int write_size = VPE * sizeof(double);
    if (lf->Write(hashrow, write_size) != write_size) ERROR("read: ", write_size);
  }
};

struct HashMatrixF {
  static void Assign(/**/  double *hashrow, unsigned hash) { if (1) hashrow[0] =  hash; }
  static bool Equals(const double *hashrow, unsigned hash) { return hashrow[0] == hash; }
};
struct HashMatrix     : public HashMatrixT    <unsigned, &HashMatrixF::Assign, &HashMatrixF::Equals> { using HashMatrixT::HashMatrixT; };
struct HashMatrixFile : public HashMatrixFileT<unsigned, &HashMatrixF::Assign, &HashMatrixF::Equals> { using HashMatrixFileT::HashMatrixFileT; };

struct HashMatrix64F {
  static void Assign(double *hashrow, unsigned long long hash) {
    hashrow[0] = unsigned(hash>>32);
    hashrow[1] = unsigned(hash&0xffffffff);
  }
  static bool Equals(const double *hashrow, unsigned long long hash) {
    return hashrow[0] == unsigned(hash>>32) && hashrow[1] == unsigned(hash&0xffffffff);
  }
};
struct HashMatrix64     : public HashMatrixT    <unsigned long long, &HashMatrix64F::Assign, &HashMatrix64F::Equals> {};
struct HashMatrix64File : public HashMatrixFileT<unsigned long long, &HashMatrix64F::Assign, &HashMatrix64F::Equals> {};

}; // namespace LFL
#endif // LFL_CORE_APP_FILE_H__
