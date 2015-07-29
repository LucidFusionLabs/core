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

#ifdef LFL_PROTOBUF
#include <google/protobuf/message.h>
#endif

#include "lfapp/lfapp.h"

#include <sys/stat.h>
#ifndef WIN32
#include <dirent.h>
#endif

#ifdef LFL_LIBARCHIVE
#include "libarchive/archive.h"
#include "libarchive/archive_entry.h"
#endif

namespace LFL {
string File::Contents() {
    if (!Opened()) return "";
    int l = Size();
    if (!l) return "";
    Reset();

    string ret;
    ret.resize(l);
    Read((char*)ret.data(), l);
    return ret;
}

const char *File::NextLine(int *offset, int *nextoffset) {
    const char *nl;
    if (!(nl = nr.GetNextRecord(this, offset, nextoffset, LFL::NextLine))) return 0;
    if (nl) nr.buf[nr.record_offset + nr.record_len] = 0;
    return nl;
}

const char *File::NextLineRaw(int *offset, int *nextoffset) {
    const char *nl;
    if (!(nl = nr.GetNextRecord(this, offset, nextoffset, LFL::NextLineRaw))) return 0;
    if (nl) nr.buf[nr.record_offset + nr.record_len] = 0;
    return nl;
}

const char *File::NextChunk(int *offset, int *nextoffset) {
    const char *nc;
    if (!(nc = nr.GetNextRecord(this, offset, nextoffset, LFL::NextChunk<4096>))) return 0;
    if (nc) nr.buf[nr.record_offset + nr.record_len] = 0;
    return nc;
}

const char *File::NextProto(int *offset, int *nextoffset, ProtoHeader *bhout) {
    const char *np;
    if (!(np = nr.GetNextRecord(this, offset, nextoffset, LFL::NextProto))) return 0;
    if (bhout) *bhout = ProtoHeader(np);
    return np + ProtoHeader::size;
}

const char *File::NextRecord::GetNextRecord(File *f, int *offsetOut, int *nextoffsetOut, NextRecordCB nextcb) {
    const char *next, *text; int left; bool read_short = false;
    if (buf_dirty) buf_offset = buf.size();
    for (;;) {
        left = buf.size() - buf_offset;
        text = buf.data() + buf_offset;
        if (!buf_dirty && left>0 && (next = nextcb(StringPiece(text, left), read_short, &record_len))) {

            if (offsetOut) *offsetOut = file_offset - buf.size() + buf_offset;
            if (nextoffsetOut) *nextoffsetOut = file_offset - buf.size() + (next - buf.data());

            record_offset = buf_offset;
            buf_offset = next-buf.data();
            return text;
        }
        if (read_short) {
            buf_offset = -1;
            return 0;
        }

        buf.erase(0, buf_offset);
        int buf_filled = buf.size();
        buf.resize(buf.size() < 4096 ? 4096 : buf.size()*2);
        int len = f->Read((char*)buf.data()+buf_filled, buf.size()-buf_filled);
        read_short = len < buf.size()-buf_filled;
        buf.resize(max(len,0) + buf_filled);
        buf_dirty = false;
        buf_offset = 0;
    }
}

int File::WriteProto(const ProtoHeader *hdr, const Proto *msg, bool doflush) {
#ifdef LFL_PROTOBUF
    std::string v = msg->SerializeAsString();
    CHECK_EQ(hdr->len, v.size());
    v.insert(0, (const char *)hdr, ProtoHeader::size);
    int ret = (Write(v.c_str(), v.size()) == v.size()) ? v.size() : -1;
    if (doflush) Flush();
    return ret;
#else
    return -1;
#endif
}

int File::WriteProto(ProtoHeader *hdr, const Proto *msg, bool doflush) {
#ifdef LFL_PROTOBUF
    std::string v = msg->SerializeAsString();
    hdr->SetLength(v.size());
    v.insert(0, (const char *)hdr, ProtoHeader::size);
    int ret = (Write(v.c_str(), v.size()) == v.size()) ? v.size() : -1;
    if (doflush) Flush();
    return ret;
#else
    return -1;
#endif
}

int File::WriteProtoFlag(const ProtoHeader *hdr, bool doflush) {
    int ret = Write(&hdr->flag, sizeof(int)) == sizeof(int) ? sizeof(int) : -1;
    if (doflush) Flush();
    return ret;
}

long long BufferFile::Seek(long long offset, int whence) {
    if (offset < 0 || offset >= (owner ? buf.size() : ptr.len)) return -1;
    nr.buf_dirty = true;
    return rdo = wro = nr.file_offset = offset;
}

int BufferFile::Read(void *out, size_t size) {
    size_t l = min(size, (owner ? buf.size() : ptr.len) - rdo);
    memcpy(out, (owner ? buf.data() : ptr.buf) + rdo, l);
    rdo += l;
    nr.file_offset += l;
    nr.buf_dirty = true;
    return l;
}

int BufferFile::Write(const void *In, size_t size) {
    CHECK(owner);
    const char *in = (const char *)In;
    if (size == -1) size = strlen(in);
    size_t l = min(size, buf.size() - wro);
    buf.replace(wro, l, in, l);
    if (size > l) buf.append(in + l, size - l);
    wro += size;
    nr.file_offset += size;
    nr.buf_dirty = true;
    return size;
}

#ifdef WIN32
const char LocalFile::Slash = '\\';
const char LocalFile::ExecutableSuffix[] = ".exe";
int LocalFile::IsDirectory(const string &filename) {
    if (filename.empty()) return true;
    DWORD attr = ::GetFileAttributes(filename.c_str());
    if (attr == INVALID_FILE_ATTRIBUTES) { ERROR("GetFileAttributes(", filename, ") failed: ", strerror(errno)); return 0; }
    return attr & FILE_ATTRIBUTE_DIRECTORY;
}
#else // WIN32
const char LocalFile::Slash = '/';
const char LocalFile::ExecutableSuffix[] = "";
int LocalFile::IsDirectory(const string &filename) {
    if (filename.empty()) return true;
#ifdef LFL_ANDROID
    ERROR("XXX Android IsDirectory");
    return 0;
#else
    struct stat buf;
    if (stat(filename.c_str(), &buf)) { ERROR("stat(", filename, ") failed: ", strerror(errno)); return 0; }
    return buf.st_mode & S_IFDIR;
#endif
}
#endif // WIN32

#ifdef LFL_ANDROID
#if 0
bool LocalFile::Open(const char *path, const char *mode, bool pre_create) {
    char *b=0; int l=0, ret;
    FILE *f = fopen(path, mode);
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    l = ftell(f);
    fseek(f, 0, SEEK_SET);
    b = (char*)malloc(l);
    fread(b, l, 1, f);
    fclose(f);
    impl = new BufferFile(string(b, l));
    ((BufferFile*)impl)->free = true;
    return true;
}
#endif
bool LocalFile::Open(const string &path, const string &mode, bool pre_create) {
    if ((writable = strchr(mode.c_str(), 'w'))) {
        impl = AndroidFileOpenWriter(path.c_str());
        return impl;
    }

    char *b=0;
    int l=0, ret=0;
    bool internal_path = 0; // !strchr(path.c_str(), '/');
    if (internal_path) { if ((ret = AndroidFileRead (path.c_str(), &b, &l))) { ERROR("AndroidFileRead ",  path); return false; } }
    else               { if ((ret = AndroidAssetRead(path.c_str(), &b, &l))) { ERROR("AndroidAssetRead ", path); return false; } }

    impl = new BufferFile(string(b, l));
    free(b);
    return true;
}

void LocalFile::Reset() { if (impl && !writable) ((BufferFile*)impl)->Reset(); }
int LocalFile::Size() { return (impl && !writable) ? ((BufferFile*)impl)->Size() : -1; }
void LocalFile::Close() { if (impl) { if (writable) AndroidFileCloseWriter(impl); else delete ((BufferFile*)impl); impl=0; } }
long long LocalFile::Seek(long long offset, int whence) { return (impl && !writable) ? ((BufferFile*)impl)->Seek(offset, whence) : -1; }
int LocalFile::Read(void *buf, size_t size) { return (impl && !writable) ? ((BufferFile*)impl)->Read(buf, size) : -1; }
int LocalFile::Write(const void *buf, size_t size) { return impl ? (writable ? AndroidFileWrite(impl, (const char*)buf, size) : ((BufferFile*)impl)->Write(buf, size)) : -1; }
bool LocalFile::Flush() { return false; }

#else /* LFL_ANDROID */
bool LocalFile::mkdir(const string &dir, int mode) {
#ifdef _WIN32
    return _mkdir(dir.c_str()) == 0;
#else
    return ::mkdir(dir.c_str(), mode) == 0;
#endif
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
#ifdef _WIN32
    if (!(impl = fopen(fn.c_str(), StrCat(mode, "b").c_str()))) return 0;
#else
    if (!(impl = fopen(fn.c_str(), mode.c_str()))) return 0;
#endif
    nr.Reset();

    if (!Opened()) return false;
    writable = strchr(mode.c_str(), 'w');
    return true;
}

void LocalFile::Reset() {
    fseek((FILE*)impl, 0, SEEK_SET);
    nr.Reset();
}

int LocalFile::Size() {
    if (!impl) return -1;

    int place = ftell((FILE*)impl);
    fseek((FILE*)impl, 0, SEEK_END);

    int ret = ftell((FILE*)impl);
    fseek((FILE*)impl, place, SEEK_SET);
    return ret;
}

void LocalFile::Close() {
    if (impl) fclose((FILE*)impl);
    impl = 0;
    nr.Reset();
}

long long LocalFile::Seek(long long offset, int whence) {
    long long ret = fseek((FILE*)impl, offset, WhenceMap(whence));
    if (ret < 0) return ret;
    if (whence == Whence::SET) ret = offset;
    else ret = ftell((FILE*)impl);
    nr.file_offset = ret;
    nr.buf_dirty = true;
    return ret;
}

int LocalFile::Read(void *buf, size_t size) {
    int ret = fread(buf, 1, size, (FILE*)impl);
    if (ret < 0) return ret;
    nr.file_offset += ret;
    nr.buf_dirty = true;
    return ret;
}

int LocalFile::Write(const void *buf, size_t size) {
    int ret = fwrite(buf, 1, size!=-1?size:strlen((char*)buf), (FILE*)impl);
    if (ret < 0) return ret;
    nr.file_offset += ret;
    nr.buf_dirty = true;
    return ret;
}

bool LocalFile::Flush() { fflush((FILE*)impl); return true; }
#endif /* LFL_ANDROID */

string LocalFile::CurrentDirectory(int max_size) {
    string ret(max_size, 0); 
    getcwd((char*)ret.data(), ret.size());
    ret.resize(strlen(ret.data()));
    return ret;
}

string LocalFile::JoinPath(const string &x, const string &y) {
    string p = (y.size() && y[0] == '/') ? "" : x;
    return StrCat(p, p.empty() ? "" : (p.back() == LocalFile::Slash ? "" : StrCat(LocalFile::Slash)),
                  p.size() && PrefixMatch(y, "./") ? y.substr(2) : y);
}

DirectoryIter::DirectoryIter(const string &path, int dirs, const char *Pref, const char *Suf) : P(Pref), S(Suf), init(0) {
    if (LocalFile::IsDirectory(path)) pathname = path;
    else {
        INFO("DirectoryIter: \"", path, "\" not a directory");
        if (LocalFile(path, "r").Opened()) {
            pathname = string(path, DirNameLen(path)) + LocalFile::Slash;
            filemap[BaseName(path)] = 1;
        }
        return;
    }
#ifdef _WIN32
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
#else /* _WIN32 */
    DIR *dir; dirent *dent; string dirname=path;
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
#endif /* _WIN32 */
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

#ifdef LFL_LIBARCHIVE
ArchiveIter::~ArchiveIter() { if (impl) archive_read_finish((archive*)impl); }
ArchiveIter::ArchiveIter(const char *path) {
    if (!(impl = archive_read_new())) return;
    if (archive_read_support_format_zip          ((archive*)impl) != 0) INFO("no zip support");
    if (archive_read_support_format_tar          ((archive*)impl) != 0) INFO("no tar support");
    if (archive_read_support_compression_gzip    ((archive*)impl) != 0) INFO("no gzip support");
    if (archive_read_support_compression_none    ((archive*)impl) != 0) INFO("no none support");
    if (archive_read_support_compression_compress((archive*)impl) != 0) INFO("no compress support");
    if (archive_read_open_filename((archive*)impl, path, 65536) != 0) {
        archive_read_finish((archive*)impl);
        impl = nullptr;
    }
}
const char *ArchiveIter::Next() {
    if (!impl) return 0;
    int ret = archive_read_next_header((archive*)impl, (archive_entry**)&entry);
    if (ret) {
        if (const char *errstr = archive_error_string((archive*)impl)) ERROR("read_next: ", ret, " ", errstr);
        return 0;
    }
    return archive_entry_pathname((archive_entry*)entry);
}
const void *ArchiveIter::Data() {
    buf.resize(Size());
    CHECK_EQ(buf.size(), archive_read_data((archive*)impl, &buf[0], buf.size()));
    return buf.c_str();
}
void ArchiveIter::Skip() { archive_read_data_skip((archive*)impl); }
long long ArchiveIter::Size() { return archive_entry_size((archive_entry*)entry); }

#else /* LFL_LIBARCHIVE */
ArchiveIter::~ArchiveIter() {}
ArchiveIter::ArchiveIter(const char *path) {}
const char *ArchiveIter::Next() { return 0; }
long long ArchiveIter::Size() { return 0; }
const void *ArchiveIter::Data() { return 0; }
void ArchiveIter::Skip() {}
#endif /* LFL_LIBARCHIVE */

/* ProtoFile */

void ProtoFile::Open(const char *fn) {
    if (file) delete file;
    file = fn ? new LocalFile(fn, "r+", true) : 0;
    read_offset = 0;
    write_offset = -1;
    done = (file ? file->Size() : 0) <= 0;
}

int ProtoFile::Add(const Proto *msg, int status) {
    done = 0;
    write_offset = file->Seek(0, File::Whence::END);

    ProtoHeader ph(status);
    return file->WriteProto(&ph, msg, true) > 0;
}

bool ProtoFile::Update(int offset, const ProtoHeader *ph, const Proto *msg) {
    if (offset < 0 || (write_offset = file->Seek(offset, File::Whence::SET)) != offset) return false;
    return file->WriteProto(ph, msg, true) > 0;
}

bool ProtoFile::Update(int offset, int status) {
    if (offset < 0 || (write_offset = file->Seek(offset, File::Whence::SET)) != offset) return false;
    ProtoHeader ph(status);
    return file->WriteProtoFlag(&ph, true) > 0;
}

bool ProtoFile::Get(Proto *out, int offset, int status) {
    int record_offset;
    write_offset = 0;
    file->Seek(offset, File::Whence::SET);
    bool ret = Next(out, &record_offset, status);
    if (!ret) return 0;
    return offset == record_offset;
}

bool ProtoFile::Next(Proto *out, int *offsetOut, int status) { ProtoHeader hdr; return Next(&hdr, out, offsetOut, status); }
bool ProtoFile::Next(ProtoHeader *hdr, Proto *out, int *offsetOut, int status) {
    if (done) return false;

    if (write_offset >= 0) {
        write_offset = -1;
        file->Seek(read_offset, File::Whence::SET);
    }

    for (;;) {
        const char *text; int offset;
        if (!(text = file->NextProto(&offset, &read_offset, hdr))) { done=true; return false; }
#ifdef LFL_PROTOBUF
        if (!out->ParseFromArray(text, hdr->len)) { ERROR("parse failed, shutting down"); done=1; app->run=0; return false; }
#endif
        if (status >= 0 && status != hdr->GetFlag()) continue;
        if (offsetOut) *offsetOut = offset;
        return true;
    }
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
        M = (int)atof(IterNextString(word));
        N = (int)atof(IterNextString(word));
    } else {
        if (MatrixFile::ReadDimensions(word, &M, &N)) return -1;
    }

    if (!F) F = new vector<string>;
    else F->clear(); 
    for (string line = IterNextString(word->iter); !word->iter->Done(); line = IterNextString(word->iter)) F->push_back(line);
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
        M = (int)atof(IterNextString(word));
        N = (int)atof(IterNextString(word));
    } else {
        if (ReadDimensions(word, &M, &N)) return -1;
    }
    
    if (!F) F = new Matrix(M,N);
    else if (F->M != M || F->N != N) (F->Open(M, N));

    MatrixIter(F) {
        double *ov = &F->row(i)[j];
        string w = IterNextString(word);
        if (word->Done()) FATAL("%s", "MatrixFile: unexpected EOF");
        if (w == "-1.#INF00" || w == "-inf") { *ov = -INFINITY; continue; }
        *ov = atof(w);
    }
    return 0;
}

int MatrixFile::ReadBinary(const string &path) {
    MMapAlloc *mmap = MMapAlloc::Open(path.c_str(), false, false);
    if (!mmap) return -1;

    char *buf = (char *)mmap->addr;
    BinaryHeader *hdr = (BinaryHeader*)buf;
    H = buf + hdr->transcript;
    long long databytes = mmap->size - hdr->data;
    long long matrixbytes = hdr->M * hdr->N * sizeof(double);
    if (databytes < matrixbytes) {
        ERRORf("%lld (%lld %d) < %lld (%d, %d)", databytes, mmap->size, hdr->data, matrixbytes, hdr->M, hdr->N);
        delete mmap;
        return -1;
    }

    if (F) FATAL("unexpected arg %p", this);
    F = new Matrix();
    F->AssignDataPtr(hdr->M, hdr->N, (double*)(buf + hdr->data), mmap);
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
    BinaryHeader hdrbuf = { (int)0xdeadbeef, M, N, (int)sizeof(BinaryHeader), (int)sizeof(BinaryHeader)+pnl, (int)sizeof(BinaryHeader)+pnl+phl, 0, 0 };
    if (file->Write(&hdrbuf, sizeof(BinaryHeader)) != sizeof(BinaryHeader)) return -1;
    char *buf = (char*)alloca(pnl+phl);
    memset(buf, 0, pnl+phl);
    strncpy(buf, name.c_str(), nl);
    strncpy(buf+pnl, hdr.c_str(), hl);
    if (file->Write(buf, pnl+phl) != pnl+phl) return -1;
    return sizeof(BinaryHeader)+pnl+phl;
}

int MatrixFile::WriteRow(File *file, const double *row, int N, bool lastrow) {
    char buf[16384]; int l=0;
    for (int j=0; j<N; j++) {
        bool lastcol = (j == N-1);
        const char *delim = (lastcol /*&& (lastrow || N != 1)*/) ? "\r\n" : " ";
        double val = row[j];
        unsigned ival = (unsigned)val;
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
    if (settings.ReadVersioned(vfn, lastiter) < 0) { ERROR(name, ".", lastiter, ".name"); return -1; }
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

MatrixArchiveOut::~MatrixArchiveOut() { Close(); }
MatrixArchiveOut::MatrixArchiveOut(const string &name) : file(0) { if (name.size()) Open(name); }
void MatrixArchiveOut::Close() { if (file) { delete file; file=0; } }
int MatrixArchiveOut::Open(const string &name) { file = new LocalFile(name, "w"); if (file->Opened()) return 0; Close(); return -1; }
int MatrixArchiveOut::Write(Matrix *m, const string &hdr, const string &name) { return MatrixFile(m, hdr).Write(file, name); } 

MatrixArchiveIn::~MatrixArchiveIn() { Close(); }
MatrixArchiveIn::MatrixArchiveIn(const string &name) : file(0), index(0) { if (name.size()) Open(name); }
void MatrixArchiveIn::Close() {if (file) { delete file; file=0; } index=0; }
int MatrixArchiveIn::Open(const string &name) { LocalFileLineIter *lfi=new LocalFileLineIter(name); file=new IterWordIter(lfi, true); return !lfi->f.Opened(); }
int MatrixArchiveIn::Read(Matrix **out, string *hdrout) { index++; return MatrixFile::Read(file, out, hdrout); }
int MatrixArchiveIn::Skip() { index++; return MatrixFile().Read(file, 1); }
string MatrixArchiveIn::Filename() { if (!file) return ""; return ""; } // file->file->f.filename(); }
int MatrixArchiveIn::Count(const string &name) { MatrixArchiveIn a(name); int ret=0; while (a.Skip() != -1) ret++; return ret; }

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
