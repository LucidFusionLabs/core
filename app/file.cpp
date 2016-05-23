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

#include <sys/stat.h>
#ifndef LFL_WINDOWS
#include <dirent.h>
#ifndef LFL_ANDROID
#include <sys/fcntl.h>
#endif
#endif

namespace LFL {
int IOVector::Append(const IOVec &v) {
  if (v.len > 0 && this->size() && this->back().len > 0 &&
      (this->back().offset + this->back().len == v.offset)) this->back().len += v.len;
  else this->push_back(v);
  return v.len;
}
  
string File::Contents() {
  if (!Opened()) return "";
  int l = Size();
  if (!l) return "";
  Reset();

  string ret;
  ret.resize(l);
  Read(&ret[0], l);
  return ret;
}

string File::ReadString(long long pos, int size) { return SeekSuccess(this, pos) ? ReadString(size) : string(); }
string File::ReadString(int size) {
  string ret(size, 0);
  ret.resize(max(0, Read(&ret[0], ret.size())));
  return ret;
}

int File::ReadIOV(void *buf, const IOVec *v, int iovlen) {
  int ret = 0;
  char *b = static_cast<char*>(buf);
  for (const IOVec *i = v, *e = i + iovlen; i != e; ++i) {
    Seek(i->offset, File::Whence::SET);
    Read(b + ret, i->len);
    ret += i->len;
  }
  return ret;
}

int File::Rewrite(const ArrayPiece<IOVec> &v, const function<string(int)> &encode_f) {
  File *new_file = Create();
  int ret = Rewrite(v, encode_f, new_file);
  return ReplaceWith(new_file) ? ret : -1;
}

int File::Rewrite(const ArrayPiece<IOVec> &v, const function<string(int)> &encode_f, File *new_file) {
  string buf(4096, 0);
  int ret = 0;
  for (const IOVec *i = v.begin(), *e = v.end(); i != e; ++i) {
    if (i->len < 0) {
      string encoded = encode_f(-i->len-1);
      CHECK_EQ(encoded.size(), new_file->Write(encoded.data(), encoded.size()));
      ret += encoded.size();
    } else {
      CHECK_EQ(i->offset, Seek(i->offset, Whence::SET));
      for (int j=0, l; j<i->len; j+=l) {
        l = min(ptrdiff_t(buf.size()), i->len-j); 
        CHECK_EQ(l, Read(&buf[0], l));
        CHECK_EQ(l, new_file->Write(buf.data(), l));
        ret += l;
      }
    }
  }
  return ret;
}

long long BufferFile::Seek(long long offset, int whence) {
  int s = owner ? buf.size() : ptr.len;
  if      (whence == SEEK_CUR) offset = rdo + offset;
  else if (whence == SEEK_END) offset = s + offset;
  if (offset < 0 || offset >= s) return -1;
  return rdo = wro = offset;
}

int BufferFile::Read(void *out, size_t size) {
  size_t l = min(size, (owner ? buf.size() : ptr.len) - rdo);
  memcpy(out, (owner ? buf.data() : ptr.buf) + rdo, l);
  rdo += l;
  return l;
}

int BufferFile::Write(const void *In, size_t size) {
  CHECK(owner);
  const char *in = static_cast<const char*>(In);
  if (size == -1) size = strlen(in);
  size_t l = min(size, buf.size() - wro);
  buf.replace(wro, l, in, l);
  if (size > l) buf.append(in + l, size - l);
  wro += size;
  return size;
}

File *BufferFile::Create() {
  BufferFile *ret = new BufferFile(string());
  if (fn.size()) ret->fn = StrCat(fn, ".new");
  return ret;
}

bool BufferFile::ReplaceWith(File *nf) {
  BufferFile *new_file = dynamic_cast<BufferFile*>(nf);
  if (!new_file) return false;
  swap(*this, *new_file);
  swap(this->fn, new_file->fn);
  delete new_file;
  return true;
}

#ifdef LFL_WINDOWS
const char LocalFile::Slash = '\\';
const char LocalFile::ExecutableSuffix[] = ".exe";

int LocalFile::IsFile(const string &filename) {
  if (filename.empty()) return true;
  DWORD attr = ::GetFileAttributes(filename.c_str());
  if (attr == INVALID_FILE_ATTRIBUTES) return ERRORv(0, "GetFileAttributes(", filename, ") failed: ", strerror(errno));
  return attr & FILE_ATTRIBUTE_NORMAL;
}

int LocalFile::IsDirectory(const string &filename) {
  if (filename.empty()) return true;
  DWORD attr = ::GetFileAttributes(filename.c_str());
  if (attr == INVALID_FILE_ATTRIBUTES) return ERRORv(0, "GetFileAttributes(", filename, ") failed: ", strerror(errno));
  return attr & FILE_ATTRIBUTE_DIRECTORY;
}

#else // LFL_WINDOWS
const char LocalFile::Slash = '/';
const char LocalFile::ExecutableSuffix[] = "";

int LocalFile::IsFile(const string &filename) {
  if (filename.empty()) return false;
  struct stat buf;
  if (stat(filename.c_str(), &buf)) return false;
  return !(buf.st_mode & S_IFDIR);
}

int LocalFile::IsDirectory(const string &filename) {
  if (filename.empty()) return true;
  struct stat buf;
  if (stat(filename.c_str(), &buf)) return false;
  return buf.st_mode & S_IFDIR;
}

int LocalFile::CreateTemporary(const string &prefix, string *name) {
  string v;
  if (!name) name = &v;
  *name = CreateTemporaryNameTemplate(prefix);

  int fd = -1;
  if ((fd = mkstemp(&(*name)[0])) < 0) return ERRORv(-1, "mkstemp ", *name, ": ", strerror(errno));
  return fd;
}

string LocalFile::CreateTemporaryName(const string &prefix) {
  string ret = CreateTemporaryNameTemplate(prefix);
  CHECK(mktemp(&ret[0]));
  return ret;
}

string LocalFile::CreateTemporaryNameTemplate(const string &prefix) {
#ifdef LFL_APPLE
  string dir = "/var/tmp/";
#else
  string dir = app->dldir;
#endif
  return StrCat(dir, app->name, "_", prefix, ".XXXXXXXX");
}
#endif // LFL_WINDOWS

bool LocalFile::mkdir(const string &dir, int mode) {
#ifdef LFL_WINDOWS
  return _mkdir(dir.c_str()) == 0;
#else
  return ::mkdir(dir.c_str(), mode) == 0;
#endif
}

bool LocalFile::unlink(const string &fn) {
  return ::unlink(fn.c_str()) == 0;
}

int LocalFile::WhenceMap(int n) {
  if      (n == Whence::SET) return SEEK_SET;
  else if (n == Whence::CUR) return SEEK_CUR;
  else if (n == Whence::END) return SEEK_END;
  else return -1;
}

bool LocalFile::Open(const string &path, const string &mode, bool pre_create) {
  fn = path;
  if (pre_create) {
    FILE *created = fopen(fn.c_str(), "a");
    if (created) fclose(created);
  }
#ifdef LFL_WINDOWS
  if (!(impl = fopen(fn.c_str(), StrCat(mode, "b").c_str()))) return 0;
#else
  if (!(impl = fopen(fn.c_str(), mode.c_str()))) return 0;
#endif
#if 0
  char filepath[MAXPATHLEN];
  if (fcntl(fileno(static_cast<FILE*>(impl)), F_GETPATH, filepath) != -1) fn = filepath;
#endif
  if (!Opened()) return false;
  writable = strchr(mode.c_str(), 'w');
  return true;
}

void LocalFile::Reset() {
  fseek(static_cast<FILE*>(impl), 0, SEEK_SET);
}

int LocalFile::Size() {
  if (!impl) return -1;

  int place = ftell(static_cast<FILE*>(impl));
  fseek(static_cast<FILE*>(impl), 0, SEEK_END);

  int ret = ftell(static_cast<FILE*>(impl));
  fseek(static_cast<FILE*>(impl), place, SEEK_SET);
  return ret;
}

void LocalFile::Close() {
  if (impl) fclose(static_cast<FILE*>(impl));
  impl = 0;
}

long long LocalFile::Seek(long long offset, int whence) {
  long long ret = fseek(static_cast<FILE*>(impl), offset, WhenceMap(whence));
  if (ret < 0) return ret;
  if (whence == Whence::SET) ret = offset;
  else ret = ftell(static_cast<FILE*>(impl));
  return ret;
}

int LocalFile::Read(void *buf, size_t size) {
  int ret = fread(buf, 1, size, static_cast<FILE*>(impl));
  if (ret < 0) return ret;
  return ret;
}

int LocalFile::Write(const void *buf, size_t size) {
  int ret = fwrite(buf, 1, size!=-1?size:strlen(static_cast<const char*>(buf)), static_cast<FILE*>(impl));
  if (ret < 0) return ret;
  return ret;
}

bool LocalFile::Flush() { fflush(static_cast<FILE*>(impl)); return true; }
File *LocalFile::Create() { return new LocalFile(StrCat(fn, ".new"), "w+"); }
bool LocalFile::ReplaceWith(File *nf) {
  LocalFile *new_file = dynamic_cast<LocalFile*>(nf);
  if (!new_file) return false;
#ifdef LFL_WINDOWS
  _unlink(fn.c_str());
#endif
  int ret = rename(new_file->fn.c_str(), fn.c_str());
  swap(*this, *new_file);
  swap(this->fn, new_file->fn);
  delete new_file;
  return !ret;
}

string LocalFile::CurrentDirectory(int max_size) {
  string ret(max_size, 0); 
  getcwd(&ret[0], ret.size());
  ret.resize(strlen(ret.data()));
  return ret;
}

string LocalFile::JoinPath(const string &x, const string &y) {
  string p = (y.size() && y[0] == '/') ? "" : x;
  return StrCat(p, p.empty() ? "" : (p.back() == LocalFile::Slash ? "" : StrCat(LocalFile::Slash)),
                p.size() && PrefixMatch(y, "./") ? y.substr(2) : y);
}

SearchPaths::SearchPaths(const char *paths) {
  StringWordIter words(StringPiece::Unbounded(BlankNull(paths)), isint<':'>);
  for (string word = words.NextString(); !words.Done(); word = words.NextString()) path.push_back(word);
}

string SearchPaths::Find(const string &fn) {
  for (auto &p : path) {
    string f = StrCat(p, LocalFile::Slash, fn);
    if (LocalFile::IsFile(f)) return f;
  }
  return "";
}

DirectoryIter::DirectoryIter(const string &path, int dirs, const char *Pref, const char *Suf) : P(Pref), S(Suf) {
  if (LocalFile::IsDirectory(path)) pathname = path;
  else {
    INFO("DirectoryIter: \"", path, "\" not a directory");
    if (LocalFile(path, "r").Opened()) {
      pathname = string(path, DirNameLen(path)) + LocalFile::Slash;
      filemap[BaseName(path)] = 1;
    }
    return;
  }
#ifdef LFL_WINDOWS
  _finddatai64_t f; int h;
  string match = StrCat(path, "*.*");
  if ((h = _findfirsti64(match.c_str(), &f)) < 0) return;

  do {
    if (!strcmp(f.name, ".")) continue;

    bool isdir = f.attrib & _A_SUBDIR;
    if (dirs >= 0 && isdir != dirs) continue;

    filemap[StrCat(f.name, isdir?"/":"")] = 1;
  }
  while (!_findnexti64(h, &f));

  _findclose(h);
#else /* LFL_WINDOWS */
  DIR *dir;
  dirent *dent;
  string dirname = path;
  if (dirname.empty()) dirname = ".";
  if (dirname.size() > 1 && dirname[dirname.size()-1] == '/') dirname.erase(dirname.size()-1);
  if (!(dir = opendir(dirname.c_str()))) return;

  while ((dent = readdir(dir))) {
    if (!strcmp(dent->d_name, ".")) continue;
    if (dent->d_type != DT_DIR && dent->d_type != DT_REG && dent->d_type != DT_LNK) continue;

    bool isdir = dent->d_type == DT_DIR;
    if (dirs >= 0 && isdir != dirs) continue;

    filemap[StrCat(dent->d_name, isdir?"/":"")] = 1;
  }

  closedir(dir);
#endif /* LFL_WINDOWS */
}

const char *DirectoryIter::Next() {
  const char *fn=0;
  for (;;) {
    if (!init) { init=1; iter = filemap.begin(); }
    else iter++;
    if (iter == filemap.end()) return 0;
    const char *k = (*iter).first.c_str();

    if (!strcmp(k, "../")) continue;
    if (P && !PrefixMatch(k, P)) continue;
    if (S && !SuffixMatch(k, S)) continue;
    return k;
  }
}

/* ContainerFile */

ContainerFileHeader::ContainerFileHeader(const char *text) {
  memcpy(&flag, text, sizeof(int));
  memcpy(&len, text+sizeof(int), sizeof(int));
  Validate();
}

void ContainerFile::Open(const string &fn) {
  if (file) delete file;
  file = fn.size() ? new LocalFile(fn, "r+", true) : 0;
  read_offset = 0;
  write_offset = -1;
  done = (file ? file->Size() : 0) <= 0;
  nr.Init(file);
}

int ContainerFile::Add(const StringPiece &msg, int status) {
  done = 0;
  write_offset = file->Seek(0, File::Whence::END);

  ContainerFileHeader ph(status);
  int wrote = WriteEntry(file, &ph, move(msg), true);
  nr.SetFileOffset(wrote > 0 ? write_offset + wrote : write_offset);
  return wrote > 0;
} 

bool ContainerFile::Update(int offset, const ContainerFileHeader *ph, const StringPiece &msg) {
  if (offset < 0 || (write_offset = file->Seek(offset, File::Whence::SET)) != offset) return false;
  int wrote = WriteEntry(file, ph, move(msg), true);
  nr.SetFileOffset(wrote > 0 ? offset + wrote : offset);
  return wrote > 0;
}

bool ContainerFile::UpdateFlag(int offset, int status) {
  if (offset < 0 || (write_offset = file->Seek(offset, File::Whence::SET)) != offset) return false;
  ContainerFileHeader ph(status);
  int wrote = WriteEntryFlag(file, &ph, true);
  nr.SetFileOffset(wrote > 0 ? offset + wrote : offset);
  return wrote > 0;
}

bool ContainerFile::Get(StringPiece *out, int offset, int status) {
  int record_offset;
  write_offset = 0;
  file->Seek(offset, File::Whence::SET);
  bool ret = Next(out, &record_offset, status);
  if (!ret) return 0;
  return offset == record_offset;
} 

bool ContainerFile::Next(StringPiece *out, int *offsetOut, int status) {
  ContainerFileHeader hdr;
  return Next(&hdr, out, offsetOut, status);
}

bool ContainerFile::Next(ContainerFileHeader *hdr, StringPiece *out, int *offsetOut, int status) {
  if (done) return false;

  if (write_offset >= 0) {
    write_offset = -1;
    file->Seek(read_offset, File::Whence::SET);
  }

  for (;;) {
    const char *text; int offset;
    if (!(text = nr.NextContainerFileEntry(&offset, &read_offset, hdr))) { done=true; return false; }
    *out = StringPiece(text, hdr->len);
    if (status >= 0 && status != hdr->GetFlag()) continue;
    if (offsetOut) *offsetOut = offset;
    return true;
  }
}

int ContainerFile::WriteEntry(File *f, const ContainerFileHeader *hdr, const StringPiece &v, bool doflush) {
  CHECK_EQ(hdr->len, v.size());
  if (f->Write(hdr, ContainerFileHeader::size) != ContainerFileHeader::size) return -1;
  if (f->Write(v.data(), v.size()) != v.size()) return -1;
  if (doflush) f->Flush();
  return ContainerFileHeader::size + v.size();
}

int ContainerFile::WriteEntry(File *f, ContainerFileHeader *hdr, const StringPiece &v, bool doflush) {
  hdr->SetLength(v.size());
  if (f->Write(hdr, ContainerFileHeader::size) != ContainerFileHeader::size) return -1;
  if (f->Write(v.data(), v.size()) != v.size()) return -1;
  if (doflush) f->Flush();
  return ContainerFileHeader::size + v.size();
}

int ContainerFile::WriteEntryFlag(File *f, const ContainerFileHeader *hdr, bool doflush) {
  int ret = f->Write(&hdr->flag, sizeof(int)) == sizeof(int) ? sizeof(int) : -1;
  if (doflush) f->Flush();
  return ret;
}

/* FlatFile */

int FlatFile::Add(const FlatBufferPiece &msg, int status) {
  return ContainerFile::Add(MakeStringPiece(msg), status);
}

bool FlatFile::Update(int offset, const ContainerFileHeader *ph, const FlatBufferPiece &msg) {
  return ContainerFile::Update(offset, ph, MakeStringPiece(msg));
}

/* StringFile */

void StringFile::Print(const string &name, bool nl) {
  string s; INFO(name, " Strings(", Lines(), ") = ");
  if (F) for (int i=0, l=Lines(); i<l; i++) { if (nl) INFO((*F)[i]); else s += string((*F)[i]) + ", "; }
  if (!nl) INFO(s);
}

int StringFile::ReadVersioned(const VersionedFileName &fn, int iteration) {
  if (iteration == -1) if ((iteration = MatrixFile::FindHighestIteration(fn, "string")) == -1) return -1;
  if (Read(StrCat(fn.dir, MatrixFile::Filename(fn, "string", iteration)))) return -1;
  return iteration;
}

int StringFile::WriteVersioned(const VersionedFileName &fn, int iteration, const string &hdr) { 
  string name = MatrixFile::Filename(fn, "string", iteration);
  return Write(StrCat(fn.dir, name), name);
}

int StringFile::Read(const string &path, int header) {
  LocalFileLineIter lfi(path);
  if (!lfi.f.Opened()) return -1;
  IterWordIter word(&lfi);
  return Read(&word, header);
}

int StringFile::Read(IterWordIter *word, int header) {
  int M, N;
  if (header == MatrixFile::Header::DIM_PLUS) {
    const char *name=word->iter->Next();
    if (!name) return -1;

    const char *transcript=word->iter->Next();
    if (!transcript) return -1;
    H.assign(transcript, word->iter->CurrentLength());

    const char *format=word->iter->Next();
    if (!format) return -1;
  }
  if (header != MatrixFile::Header::NONE) {
    M = int(atof(word->NextString()));
    N = int(atof(word->NextString()));
  } else {
    if (MatrixFile::ReadDimensions(word, &M, &N)) return -1;
  }

  if (!F) F = new vector<string>;
  else F->clear(); 
  for (string line = word->iter->NextString(); !word->iter->Done(); line = word->iter->NextString()) F->push_back(line);
  return 0;
}

int StringFile::Write(File *file, const string &name) {    
  if (!file) return -1;
  if (MatrixFile::WriteHeader(file, name, H.c_str(), Lines(), 1) < 0) return -1;
  if (F) for (auto i = F->begin(); i != F->end(); i++) WriteRow(file, (*i));
  return 0;
}

int StringFile::Write(const string &path, const string &name) {
  LocalFile file(path, "w");
  if (!file.Opened()) return -1;
  return Write(&file, name);
}

int StringFile::WriteRow(File *file, const string &rowval) {
  string row = StrCat(rowval, "\r\n");
  if (file->Write(row.c_str(), row.size()) != row.size()) return -1;
  return 0;
}

/* MatrixFile */

int MatrixFile::ReadVersioned(const VersionedFileName &fn, int iteration) {
  static const char *fileext[] = { "matbin", "matrix" };
  if (iteration == -1) if ((iteration = FindHighestIteration(fn, fileext[0], fileext[1])) == -1) return -1;

  bool found = 0;
  if (!found) if (!ReadBinary(string(fn.dir) + Filename(fn, fileext[0], iteration))) found=1;
  if (!found) if (!Read      (string(fn.dir) + Filename(fn, fileext[1], iteration))) found=1;
  return found ? iteration : -1;
}

int MatrixFile::WriteVersioned(const VersionedFileName &fn, int iter) {
  string name=Filename(fn, "matrix", iter), pn=string(fn.dir) + name;
  return Write(pn.c_str(), name.c_str());
}

int MatrixFile::WriteVersionedBinary(const VersionedFileName &fn, int iter) {
  string name=Filename(fn, "matbin", iter), pn=string(fn.dir) + name;
  return WriteBinary(pn.c_str(), name.c_str());
}

int MatrixFile::Read(const string &path, int header, int(*IsSpace)(int)) {
  LocalFile lf(path, "r");
  if (!lf.Opened()) return -1;
  return Read(&lf, header, IsSpace);
}

int MatrixFile::Read(File *f, int header, int(*IsSpace)(int)) {
  FileLineIter fli(f);
  IterWordIter word(&fli);
  if (IsSpace) word.word.IsSpace = IsSpace;
  return Read(&word, header);
}

int MatrixFile::Read(IterWordIter *word, int header) {
  int M, N;
  if (header == Header::DIM_PLUS) { if (ReadHeader(word, &H) < 0) return -1; }
  if (header != Header::NONE) {
    M = int(atof(word->NextString()));
    N = int(atof(word->NextString()));
  } else {
    if (ReadDimensions(word, &M, &N)) return -1;
  }

  if (!F) F = new Matrix(M,N);
  else if (F->M != M || F->N != N) (F->Open(M, N));

  MatrixIter(F) {
    double *ov = &F->row(i)[j];
    string w = word->NextString();
    if (word->Done()) FATAL("%s", "MatrixFile: unexpected EOF");
    if (w == "-1.#INF00" || w == "-inf") { *ov = -INFINITY; continue; }
    *ov = atof(w);
  }
  return 0;
}

int MatrixFile::ReadBinary(const string &path) {
  unique_ptr<MMapAllocator> mmap = MMapAllocator::Open(path.c_str(), false, false);
  if (!mmap) return -1;

  char *buf = static_cast<char*>(mmap->addr);
  BinaryHeader *hdr = static_cast<BinaryHeader*>(mmap->addr);
  H = buf + hdr->transcript;
  long long databytes = mmap->size - hdr->data;
  long long matrixbytes = hdr->M * hdr->N * sizeof(double);
  if (databytes < matrixbytes) {
    ERRORf("%lld (%lld %d) < %lld (%d, %d)", databytes, mmap->size, hdr->data, matrixbytes, hdr->M, hdr->N);
    return -1;
  }

  if (F) FATAL("unexpected arg %p", this);
  F = new Matrix();
  F->AssignDataPtr(hdr->M, hdr->N, reinterpret_cast<double*>(buf + hdr->data), mmap.release());
  return 0;
}

int MatrixFile::Write(File *file, const string &name) {    
  if (!file) return -1;
  if (WriteHeader(file, name, Text(), F->M, F->N) < 0) return -1;
  MatrixRowIter(F) {
    bool lastrow = (i == F->M-1);
    if (WriteRow(file, F->row(i), F->N, lastrow)) return -1;
  }
  return 0;
}

int MatrixFile::Write(const string &path, const string &name) {
  LocalFile file(path, "w");
  if (!file.Opened()) return -1;
  int ret = Write(&file, name);
  file.Close();
  return ret;
}

int MatrixFile::WriteBinary(File *file, const string &name) {    
  if (!file) return -1;
  if (WriteBinaryHeader(file, name, Text(), F->M, F->N) < 0) return -1;
  if (file->Write(F->m, F->bytes) != F->bytes) return -1;
  return 0;
}

int MatrixFile::WriteBinary(const string &path, const string &name) {
  LocalFile file(path, "w");
  if (!file.Opened()) return -1;
  int ret = WriteBinary(&file, name);
  file.Close();
  return ret;
}

string MatrixFile::Filename(const string &Class, const string &Var, const string &Suffix, int iteration) {
  return StringPrintf("%s.%04d.%s.%s", Class.c_str(), iteration, Var.c_str(), Suffix.c_str());
}

int MatrixFile::FindHighestIteration(const VersionedFileName &fn, const string &Suffix) {
  int iteration = -1, iter;
  string pref = StrCat(fn._class, ".");
  string suf  = StrCat(".", fn.var, ".", Suffix);

  DirectoryIter d(fn.dir, 0, pref.c_str(), suf.c_str());
  for (const char *f = d.Next(); app->run && f; f = d.Next()) {
    if ((iter = atoi(f + pref.length())) > iteration) iteration = iter;
  }
  return iteration;
}

int MatrixFile::FindHighestIteration(const VersionedFileName &fn, const string &Suffix1, const string &Suffix2) {
  int iter1 = FindHighestIteration(fn, Suffix1);
  int iter2 = FindHighestIteration(fn, Suffix2);
  return iter1 >= iter2 ? iter1 : iter2;
}

int MatrixFile::ReadHeader(IterWordIter *word, string *hdrout) {
  const char *name=word->iter->Next();
  if (!name) return -1;

  const char *transcript=word->iter->Next();
  if (!transcript) return -1;
  hdrout->assign(transcript, word->iter->CurrentLength());

  const char *format=word->iter->Next();
  if (!format) return -1;

  return 0;
}

int MatrixFile::ReadDimensions(IterWordIter *word, int *M, int *N) {
  int last_line_count = 1, rows = 0, cols = 0;
  for (const char *w = word->Next(); /**/; w = word->Next()) {
    if (last_line_count != word->first_count) {
      if (word->first_count == 2) *N = cols;
      else if (*N != cols) FATAL(*N, " != ", cols);

      last_line_count = word->first_count;
      cols = 0;
      rows++;
    }
    cols++;
    if (!w) break;
  }
  word->Reset();
  *M = rows;
  return !(*M > 0 && *N > 0);
}

int MatrixFile::WriteHeader(File *file, const string &name, const string &hdr, int M, int N) {
  string buf = StringPrintf("%s\r\n%s\r\nMatrix\r\n%d %d\r\n", name.c_str(), hdr.c_str(), M, N);
  if (file->Write(buf.c_str(), buf.size()) != buf.size()) return -1;
  return buf.size();
}

int MatrixFile::WriteBinaryHeader(File *file, const string &name, const string &hdr, int M, int N) {
  int nl=name.size()+1, hl=hdr.size()+1;
  int pnl=NextMultipleOfN(nl, 32), phl=NextMultipleOfN(hl, 32);
  BinaryHeader hdrbuf = { int(0xdeadbeef), M, N, int(sizeof(BinaryHeader)), int(sizeof(BinaryHeader)+pnl), int(sizeof(BinaryHeader)+pnl+phl), 0, 0 };
  if (file->Write(&hdrbuf, sizeof(BinaryHeader)) != sizeof(BinaryHeader)) return -1;
  string buf(pnl+phl, 0);
  strncpy(&buf[0], name.c_str(), nl);
  strncpy(&buf[0]+pnl, hdr.c_str(), hl);
  if (file->Write(buf.data(), pnl+phl) != pnl+phl) return -1;
  return sizeof(BinaryHeader)+pnl+phl;
}

int MatrixFile::WriteRow(File *file, const double *row, int N, bool lastrow) {
  char buf[16384]; int l=0;
  for (int j=0; j<N; j++) {
    bool lastcol = (j == N-1);
    const char *delim = (lastcol /*&& (lastrow || N != 1)*/) ? "\r\n" : " ";
    double val = row[j];
    unsigned ival = unsigned(val);
    if (val == ival) l += sprint(buf+l,sizeof(buf)-l, "%u%s", ival, delim);
    else             l += sprint(buf+l,sizeof(buf)-l, "%f%s", val, delim);
  }
  if (file->Write(buf, l) != l) return -1;
  return 0;
}

/* SettingsFile */

int SettingsFile::Read(const string &dir, const string &name) {
  StringFile settings; int lastiter=0;
  VersionedFileName vfn(dir.c_str(), name.c_str(), VarName());
  if (settings.ReadVersioned(vfn, lastiter) < 0) return ERRORv(-1, name, ".", lastiter, ".name");
  for (int i=0, l=settings.Lines(); i<l; i++) {
    const char *line = (*settings.F)[i].c_str(), *sep = strstr(line, Separator());
    if (sep) Singleton<FlagMap>::Get()->Set(string(line, sep-line), sep + strlen(Separator()));
  }
  return 0;
}

int SettingsFile::Write(const vector<string> &fields, const string &dir, const string &name) {
  LocalFile settings(string(dir) + MatrixFile::Filename(name, VarName(), VarType(), 0), "w");
  MatrixFile::WriteHeader(&settings, BaseName(settings.fn), Join(fields, ",").c_str(), fields.size(), 1);
  for (vector<string>::const_iterator i = fields.begin(); i != fields.end(); i++) {
    StringFile::WriteRow(&settings, StrCat(*i, Separator(), Singleton<FlagMap>::Get()->Get(*i)).c_str());
  }
  return 0;
}

/* Matrix Archive */ 

MatrixArchiveOutputFile::~MatrixArchiveOutputFile() { Close(); }
MatrixArchiveOutputFile::MatrixArchiveOutputFile(const string &name) : file(0) { if (name.size()) Open(name); }
void MatrixArchiveOutputFile::Close() { if (file) { delete file; file=0; } }
int MatrixArchiveOutputFile::Open(const string &name) { file = new LocalFile(name, "w"); if (file->Opened()) return 0; Close(); return -1; }
int MatrixArchiveOutputFile::Write(Matrix *m, const string &hdr, const string &name) { return MatrixFile(m, hdr).Write(file, name); } 

MatrixArchiveInputFile::~MatrixArchiveInputFile() { Close(); }
MatrixArchiveInputFile::MatrixArchiveInputFile(const string &name) : file(0), index(0) { if (name.size()) Open(name); }
void MatrixArchiveInputFile::Close() {if (file) { delete file; file=0; } index=0; }
int MatrixArchiveInputFile::Open(const string &name) { LocalFileLineIter *lfi=new LocalFileLineIter(name); file=new IterWordIter(lfi, true); return !lfi->f.Opened(); }
int MatrixArchiveInputFile::Read(Matrix **out, string *hdrout) { index++; return MatrixFile::Read(file, out, hdrout); }
int MatrixArchiveInputFile::Skip() { index++; return MatrixFile().Read(file, 1); }
string MatrixArchiveInputFile::Filename() { if (!file) return ""; return ""; } // file->file->f.filename(); }
int MatrixArchiveInputFile::Count(const string &name) { MatrixArchiveInputFile a(name); int ret=0; while (a.Skip() != -1) ret++; return ret; }

/* GraphVizFileFile */

string GraphVizFile::Footer() { return "}\r\n"; }
string GraphVizFile::DigraphHeader(const string &name) {
  return StrCat("digraph ", name, " {\r\n"
                "rankdir=LR;\r\n"
                "size=\"8,5\"\r\n"
                "node [style = solid];\r\n"
                "node [shape = circle];\r\n");
}

string GraphVizFile::NodeColor(const string &s) { return StrCat("node [color = ", s, "];\r\n"); }
string GraphVizFile::NodeShape(const string &s) { return StrCat("node [shape = ", s, "];\r\n"); }
string GraphVizFile::NodeStyle(const string &s) { return StrCat("node [style = ", s, "];\r\n"); }

void GraphVizFile::AppendNode(string *out, const string &n1, const string &label) {
  StrAppend(out, "\"", n1, "\"",
            (label.size() ? StrCat(" [ label = \"", label, "\" ] ") : ""),
            ";\r\n");
}

void GraphVizFile::AppendEdge(string *out, const string &n1, const string &n2, const string &label) {
  StrAppend(out, "\"", n1, "\" -> \"", n2, "\"",
            (label.size() ? StrCat(" [ label = \"", label, "\" ] ") : ""),
            ";\r\n");
}

}; // namespace LFL
