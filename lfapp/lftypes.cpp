/*
 * $Id: lftypes.cpp 1334 2014-11-28 09:14:21Z justin $
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
#include "lfapp/network.h"

#ifdef LFL_JUDY
#ifdef _WIN32
#define JU_WIN
#endif
#define LFL_INCLUDE_JUDY_IMPL
#include "judymap.h"
#endif

namespace LFL {
ReallocHeap::ReallocHeap(int startSize) : heap(0), size(startSize), len(0) {}
ReallocHeap::~ReallocHeap() { free(heap); }

int ReallocHeap::Alloc(int bytes) {
    int ret=len, reallocB=!heap, newSize=len+bytes;
    if (!size) {
        if (!heap) size=65536;
        else FATAL("corrupt heap %p %d %d", heap, len, size);
    }
    while (size < newSize) { size *= 2; reallocB=1; }
    if (reallocB) {
        char *h = (char*)realloc(heap, size);
        if (!h) return -1;
        heap = h;
    }
    len += bytes;
    return ret;
}

void RingBuf::Resize(int SPS, int SPB, int Width) {
    if (SPS != samplesPerSec || SPB != ring.size || Width != width) { 
        ring.size = SPB;
        samplesPerSec = SPS;
        width = Width ? Width : sizeof(float);
        bytes = ring.size * width;
        if (buf) alloc->Free(buf);
        buf = (char*)alloc->Malloc(bytes);
        memset(buf, 0, bytes);
        if (stamp) alloc->Free(stamp);
        stamp = (Time*)alloc->Malloc(ring.size * sizeof(Time));
        memset(stamp, 0, ring.size * sizeof(Time));
    }
    ring.back = 0;
}

void *RingBuf::Write(int writeFlag, Time timestamp) {
    void *ret = (void*)(buf + ring.back*width);
    if (writeFlag & Stamp) stamp[ring.back] = timestamp != -1 ? timestamp : Now() * 1000;
    if (!(writeFlag & Peek)) ring.back = ring.Index(1);
    return ret;
}

int RingBuf::Dist(int indexB, int indexE) const { return Since(Bucket(indexB), Bucket(indexE)); }

int RingBuf::Since(int index, int Next) const {
    Next = Next>=0 ? Next : ring.back;
    return (Next < index ? ring.size : 0) + Next - index;
}

void *RingBuf::Read(int index, int Next) const { 
    Next = Next>=0 ? Next : ring.back;
    int ind = Bucket(Next+index);
    return (void *)(buf + ind * width);
}

Time RingBuf::ReadTimestamp(int index, int Next) const { 
    Next = Next>=0 ? Next : ring.back;
    int ind = Bucket(Next+index);
    return stamp[ind];
}

void RingBuf::Handle::CopyFrom(const RingBuf::Handle *src) {
    next=0; int N=Len(), B=0;
    if (N > src->Len()) { B=N-src->Len(); N=src->Len(); }
    for (int i=0; i<N; i++) Write(src->Read(-N+i));
    for (int i=0; i<B; i++) Write(0.0);
}

/* Serializable */

void Serializable::Header::Out(Stream *o) const { o->Htons( id); o->Htons( seq); }
void Serializable::Header::In(const Stream *i)  { i->Ntohs(&id); i->Ntohs(&seq); }

string Serializable::ToString(unsigned short seq) { string ret; ToString(&ret, seq); return ret; }

void Serializable::ToString(string *out, unsigned short seq) {
    out->resize(Header::size + Size());
    return ToString((char*)out->data(), out->size(), seq);
}

void Serializable::ToString(char *buf, int len, unsigned short seq) {
    MutableStream os(buf, len);
    Header hdr = { (unsigned short)Type(), seq };
    hdr.Out(&os);
    Out(&os);
}

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
        H = transcript;

        const char *format=word->iter->Next();
        if (!format) return -1;
    }
    if (header != MatrixFile::Header::NONE) {
        M = (int)atof(word->Next());
        N = (int)atof(word->Next());
    } else {
        if (MatrixFile::ReadDimensions(word, &M, &N)) return -1;
    }

    if (!F) F = new vector<string>;
    else F->clear(); 
    for (const char *line=word->iter->Next(); line; line=word->iter->Next()) F->push_back(line);
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
    if (!found) {
        string pn = string(fn.dir) + Filename(fn, fileext[0], iteration);
        if (!ReadBinary(pn.c_str())) found=1;
    }
    if (!found) {
        string pn = string(fn.dir) + Filename(fn, fileext[1], iteration);
        if (!Read(pn.c_str())) found=1;
    }
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

int MatrixFile::Read(const string &path, int header, int (*IsSpace)(int)) {      
    LocalFileLineIter lfi(path);
    if (!lfi.f.Opened()) return -1;
    IterWordIter word(&lfi);
    if (IsSpace) word.word.IsSpace = IsSpace;
    return Read(&word, header);
}

int MatrixFile::Read(IterWordIter *word, int header) {
    int M, N;
    if (header == Header::DIM_PLUS) { if (ReadHeader(word, &H) < 0) return -1; }
    if (header != Header::NONE) {
        M = (int)atof(word->Next());
        N = (int)atof(word->Next());
    } else {
        if (ReadDimensions(word, &M, &N)) return -1;
    }
    
    if (!F) F = new Matrix(M,N);
    else if (F->M != M || F->N != N) (F->Open(M, N));

    MatrixIter(F) {
        double *ov = &F->row(i)[j];
        const char *w = word->Next();
        if (!w) FATAL("%s", "MatrixFile: unexpected EOF");
        if (!strcmp(w, "-1.#INF00") || !strcmp(w, "-inf")) { *ov = -INFINITY; continue; }
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
    *hdrout = transcript;

    const char *format=word->iter->Next();
    if (!format) return -1;

    return 0;
}

int MatrixFile::ReadDimensions(IterWordIter *word, int *M, int *N) {
    int last_line_count = 1, rows = 0, cols = 0;
    for (const char *w = word->Next(); /**/; w = word->Next()) {
        if (last_line_count != word->line_count) {
            if (word->line_count == 2) *N = cols;
            else if (*N != cols) FATAL(*N, " != ", cols);

            last_line_count = word->line_count;
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
    MatrixFile::WriteHeader(&settings, basename(settings.fn.c_str(),0,0), Join(fields, ",").c_str(), fields.size(), 1);
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

}; // namespace LFL
