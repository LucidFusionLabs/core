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

int ReallocHeap::alloc(int bytes) {
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

void RingBuf::resize(int SPS, int SPB, int Width) {
    if (SPS != samplesPerSec || SPB != ring.size || Width != width) { 
        ring.size = SPB;
        samplesPerSec = SPS;
        width = Width ? Width : sizeof(float);
        bytes = ring.size * width;
        if (buf) alloc->free(buf);
        buf = (char*)alloc->malloc(bytes);
        memset(buf, 0, bytes);
        if (stamp) alloc->free(stamp);
        stamp = (Time*)alloc->malloc(ring.size * sizeof(Time));
        memset(stamp, 0, ring.size * sizeof(Time));
    }
    ring.back = 0;
}

void *RingBuf::write(int writeFlag, Time timestamp) {
    void *ret = (void*)(buf + ring.back*width);
    if (writeFlag & Stamp) stamp[ring.back] = timestamp != -1 ? timestamp : Now() * 1000;
    if (!(writeFlag & Peek)) ring.back = ring.Index(1);
    return ret;
}

int RingBuf::dist(int indexB, int indexE) const { return since(bucket(indexB), bucket(indexE)); }

int RingBuf::since(int index, int Next) const {
    Next = Next>=0 ? Next : ring.back;
    if (index == Next) return ring.size;
    else if (index < Next) return Next - index;
    else return ring.size - index + Next;
}

void *RingBuf::read(int index, int Next) const { 
    Next = Next>=0 ? Next : ring.back;
    int ind = bucket(Next+index);
    return (void *)(buf + ind * width);
}

Time RingBuf::readtimestamp(int index, int Next) const { 
    Next = Next>=0 ? Next : ring.back;
    int ind = bucket(Next+index);
    return stamp[ind];
}

void RingBuf::Handle::CopyFrom(const RingBuf::Handle *src) {
    next=0; int N=len(), B=0;
    if (N > src->len()) { B=N-src->len(); N=src->len(); }
    for (int i=0; i<N; i++) write(src->read(-N+i));
    for (int i=0; i<B; i++) write(0.0);
}

/* Serializable */

void Serializable::Header::out(Stream *o) const { o->Htons( id); o->Htons( seq); }
void Serializable::Header::in(const Stream *i)  { i->Ntohs(&id); i->Ntohs(&seq); }

string Serializable::ToString(unsigned short seq) { string ret; ToString(&ret, seq); return ret; }

void Serializable::ToString(string *out, unsigned short seq) {
    out->resize(Header::size + size());
    return ToString((char*)out->data(), out->size(), seq);
}

void Serializable::ToString(char *buf, int len, unsigned short seq) {
    MutableStream os(buf, len);
    Header hdr = { (unsigned short)type(), seq };
    hdr.out(&os);
    out(&os);
}

/* ProtoFile */

void ProtoFile::open(const char *fn) {
    if (file) delete file;
    file = fn ? new LocalFile(fn, "r+", true) : 0;
    read_offset = 0;
    write_offset = -1;
    done = (file ? file->size() : 0) <= 0;
}

int ProtoFile::add(const Proto *msg, int status) {
    done = 0;
    write_offset = file->seek(0, File::Whence::END);

    ProtoHeader ph(status);
    return file->writeproto(&ph, msg, true) > 0;
}

bool ProtoFile::update(int offset, const ProtoHeader *ph, const Proto *msg) {
    if (offset < 0 || (write_offset = file->seek(offset, File::Whence::SET)) != offset) return false;
    return file->writeproto(ph, msg, true) > 0;
}

bool ProtoFile::update(int offset, int status) {
    if (offset < 0 || (write_offset = file->seek(offset, File::Whence::SET)) != offset) return false;
    ProtoHeader ph(status);
    return file->writeprotoflag(&ph, true) > 0;
}

bool ProtoFile::get(Proto *out, int offset, int status) {
    int record_offset;
    write_offset = 0;
    file->seek(offset, File::Whence::SET);
    bool ret = next(out, &record_offset, status);
    if (!ret) return 0;
    return offset == record_offset;
}

bool ProtoFile::next(Proto *out, int *offsetOut, int status) { ProtoHeader hdr; return next(&hdr, out, offsetOut, status); }
bool ProtoFile::next(ProtoHeader *hdr, Proto *out, int *offsetOut, int status) {
    if (done) return false;

    if (write_offset >= 0) {
        write_offset = -1;
        file->seek(read_offset, File::Whence::SET);
    }

    for (;;) {
        const char *text; int offset;
        if (!(text = file->nextproto(&offset, &read_offset, hdr))) { done=true; return false; }
#ifdef LFL_PROTOBUF
        if (!out->ParseFromArray(text, hdr->len)) { ERROR("parse failed, shutting down"); done=1; app->run=0; return false; }
#endif
        if (status >= 0 && status != hdr->get_flag()) continue;
        if (offsetOut) *offsetOut = offset;
        return true;
    }
}

/* StringFile */

int StringFile::read(string &hdrout, StringFile *out, const char *path, int header) {      
    LocalFileLineIter lfi(path);
    if (!lfi.f.opened()) return -1;
    IterWordIter word(&lfi);
    return read(hdrout, out, &word, header);
}

int StringFile::read(string &hdrout, StringFile *out, IterWordIter *word, int header) {
    int M, N;
    if (header == MatrixFile::Header::DIM_PLUS) {
        const char *name=word->iter->next();
        if (!name) return -1;
        
        const char *transcript=word->iter->next();
        if (!transcript) return -1;
        hdrout = transcript;

        const char *format=word->iter->next();
        if (!format) return -1;
    }
    if (header != MatrixFile::Header::NONE) {
        M = (int)atof(word->next());
        N = (int)atof(word->next());
    } else {
        if (MatrixFile::readDimensions(word, &M, &N)) return -1;
    }

    out->clear();
    vector<int> offsets;
    for (const char *line=word->iter->next(); line; line=word->iter->next()) {
        int offset = out->b.alloc(strlen(line)+1);
        strcpy(out->b.heap + offset, line);
        offsets.push_back(offset);
    }

    out->lines = offsets.size();
    int lines_index_offset = out->b.alloc(sizeof(char*) * out->lines);
    out->line = (char**)(out->b.heap + lines_index_offset);
    for (int i=0; i<out->lines; i++) out->line[i] = out->b.heap + offsets[i];
    return 0;
}

int StringFile::writeRow(File *file, const char *rowval) {
    string row = StrCat(rowval, "\r\n");
    if (file->write(row.c_str(), row.size()) != row.size()) return -1;
    return 0;
}

int StringFile::write(const string &hdrout, const vector<string> &out, File *file, const char *name) {    
    if (!file) return -1;
    if (MatrixFile::writeHeader(file, name, hdrout.c_str(), out.size(), 1) < 0) return -1;

    for (vector<string>::const_iterator i = out.begin(); i != out.end(); i++) writeRow(file, (*i).c_str());
    return 0;
}

int StringFile::write(const string &hdrout, const vector<string> &out, const char *path, const char *name) {
    LocalFile file(path, "w");
    if (!file.opened()) return -1;
    return write(hdrout, out, &file, name);
}

int StringFile::ReadFile(const char *Dir, const char *Class, const char *Var, StringFile *modelOut, string *flagOut, int iteration) {
    if (iteration == -1) if ((iteration = MatrixFile::findHighestIteration(Dir, Class, Var, "string")) == -1) return -1;

    string hdr, name=MatrixFile::filename(Class, Var, "string", iteration), pn=string(Dir) + name;
    if (read(hdr, modelOut, pn.c_str())) return -1;
    if (flagOut) *flagOut = hdr;
    return iteration;
}

int StringFile::WriteFile(const char *Dir, const char *Class, const char *Var, const vector<string> &model, int iteration, const char *transcript) { 
    string hdr=transcript?transcript:"", name=MatrixFile::filename(Class, Var, "string", iteration), pn=string(Dir) + name;
    return write(hdr, model, pn.c_str(), name.c_str());
}

void StringFile::print(const char *name, bool nl) {
    INFO(name, " Strings(", lines, ") = ");

    string s;
    for (int i=0; i<lines; i++) {
        if (nl) INFO(line[i]);
        else s += string(line[i]) + ", ";
    }

    if (!nl) INFO(s);
}

/* MatrixFile */

string MatrixFile::filename(const char *Class, const char *Var, const char *Suffix, int iteration) {
    return StringPrintf("%s.%04d.%s.%s", Class, iteration, Var, Suffix);
}

int MatrixFile::findHighestIteration(const char *Dir, const char *Class, const char *Var, const char *Suffix) {
    int iteration = -1, iter;
    string pref = StrCat(Class, ".");
    string suf  = StrCat(".", Var, ".", Suffix);

    DirectoryIter d(Dir, 0, pref.c_str(), suf.c_str());
    for (const char *fn = d.next(); app->run && fn; fn = d.next()) {
        if ((iter = atoi(fn+pref.length())) > iteration) iteration = iter;
    }
    return iteration;
}

int MatrixFile::findHighestIteration(const char *Dir, const char *Class, const char *Var, const char *Suffix1, const char *Suffix2) {
    int iter1 = findHighestIteration(Dir, Class, Var, Suffix1);
    int iter2 = findHighestIteration(Dir, Class, Var, Suffix2);
    return iter1 >= iter2 ? iter1 : iter2;
}

int MatrixFile::read(string &hdrout, Matrix **out, const char *path, int header, int (*IsSpace)(int)) {      
    LocalFileLineIter lfi(path);
    if (!lfi.f.opened()) return -1;
    IterWordIter word(&lfi);
    if (IsSpace) word.word.IsSpace = IsSpace;
    return read(hdrout, out, &word, header);
}

int MatrixFile::readHeader(string &hdrout, IterWordIter *word) {
    const char *name=word->iter->next();
    if (!name) return -1;

    const char *transcript=word->iter->next();
    if (!transcript) return -1;
    hdrout = transcript;

    const char *format=word->iter->next();
    if (!format) return -1;

    return 0;
}

int MatrixFile::readDimensions(IterWordIter *word, int *M, int *N) {
    int last_line_count = 1, rows = 0, cols = 0;
    for (const char *w = word->next(); /**/; w = word->next()) {
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
    word->reset();
    *M = rows;
    return !(*M > 0 && *N > 0);
}

int MatrixFile::read(string &hdrout, Matrix **out, IterWordIter *word, int header) {
    int M, N;
    if (header == Header::DIM_PLUS) { if (readHeader(hdrout, word) < 0) return -1; }
    if (header != Header::NONE) {
        M = (int)atof(word->next());
        N = (int)atof(word->next());
    } else {
        if (readDimensions(word, &M, &N)) return -1;
    }

    if (*out) { if ((*out)->M != M || (*out)->N != N) ((*out)->open(M, N)); }
    else { *out = new Matrix(M,N); }

    MatrixIter(*out) {
        double *ov = &(*out)->row(i)[j];
        const char *w = word->next();
        if (!w) FATAL("%s", "MatrixFile: unexpected EOF");
        if (!strcmp(w, "-1.#INF00") || !strcmp(w, "-inf")) { *ov = -INFINITY; continue; }
        *ov = atof(w);
    }
    return 0;
}

int MatrixFile::read_binary(string &hdrout, Matrix **out, const char *path) {
    MMapAlloc *mmap = MMapAlloc::open(path, false, false);
    if (!mmap) return -1;

    char *buf = (char *)mmap->addr;
    BinaryHeader *hdr = (BinaryHeader*)buf;
    hdrout = buf + hdr->transcript;
    long long databytes = mmap->size - hdr->data;
    long long matrixbytes = hdr->M * hdr->N * sizeof(double);
    if (databytes < matrixbytes) {
        ERRORf("%lld (%lld %d) < %lld (%d, %d)", databytes, mmap->size, hdr->data, matrixbytes, hdr->M, hdr->N);
        delete mmap;
        return -1;
    }

    if (*out) FATAL("unexpected arg %p", out);
    *out = new Matrix();
    (*out)->assignDataPtr(hdr->M, hdr->N, (double*)(buf + hdr->data), mmap);
    return 0;
}

int MatrixFile::writeHeader(File *file, const char *name, const char *hdr, int M, int N) {
    string buf = StringPrintf("%s\r\n%s\r\nMatrix\r\n%d %d\r\n", name, hdr, M, N);
    if (file->write(buf.c_str(), buf.size()) != buf.size()) return -1;
    return buf.size();
}

int MatrixFile::writeBinaryHeader(File *file, const char *name, const char *hdr, int M, int N) {
    int nl=strlen(name)+1, hl=strlen(hdr)+1;
    int pnl=next_multiple_of_n(nl, 32), phl=next_multiple_of_n(hl, 32);
    BinaryHeader hdrbuf = { (int)0xdeadbeef, M, N, (int)sizeof(BinaryHeader), (int)sizeof(BinaryHeader)+pnl, (int)sizeof(BinaryHeader)+pnl+phl, 0, 0 };
    if (file->write(&hdrbuf, sizeof(BinaryHeader)) != sizeof(BinaryHeader)) return -1;
    char *buf = (char*)alloca(pnl+phl);
    memset(buf, 0, pnl+phl);
    strncpy(buf, name, nl);
    strncpy(buf+pnl, hdr, hl);
    if (file->write(buf, pnl+phl) != pnl+phl) return -1;
    return sizeof(BinaryHeader)+pnl+phl;
}

int MatrixFile::writeRow(File *file, const double *row, int N, bool lastrow) {
    char buf[16384]; int l=0;
    for (int j=0; j<N; j++) {
        bool lastcol = (j == N-1);
        const char *delim = (lastcol /*&& (lastrow || N != 1)*/) ? "\r\n" : " ";
        double val = row[j];
        unsigned ival = (unsigned)val;
        if (val == ival) l += sprint(buf+l,sizeof(buf)-l, "%u%s", ival, delim);
        else             l += sprint(buf+l,sizeof(buf)-l, "%f%s", val, delim);
    }
    if (file->write(buf, l) != l) return -1;
    return 0;
}

int MatrixFile::write(const string &hdrout, Matrix *F, File *file, const char *name) {    
    if (!file) return -1;
    if (writeHeader(file, name, hdrout.c_str(), F->M, F->N) < 0) return -1;
    MatrixRowIter(F) {
        bool lastrow = (i == F->M-1);
        if (writeRow(file, F->row(i), F->N, lastrow)) return -1;
    }
    return 0;
}

int MatrixFile::write(const string &hdrout, Matrix *out, const char *path, const char *name) {
    LocalFile file(path, "w");
    if (!file.opened()) return -1;

    int ret = write(hdrout, out, &file, name);
    file.close();
    return ret;
}

int MatrixFile::write_binary(const string &hdrout, Matrix *F, File *file, const char *name) {    
    if (!file) return -1;
    if (writeBinaryHeader(file, name, hdrout.c_str(), F->M, F->N) < 0) return -1;
    if (file->write(F->m, F->bytes) != F->bytes) return -1;
    return 0;
}

int MatrixFile::write_binary(const string &hdrout, Matrix *out, const char *path, const char *name) {
    LocalFile file(path, "w");
    if (!file.opened()) return -1;

    int ret = write_binary(hdrout, out, &file, name);
    file.close();
    return ret;
}

int MatrixFile::ReadFile(const char *Dir, const char *Class, const char *Var, Matrix **modelOut, string *flagOut, int iteration) {
    static const char *fileext[] = { "matbin", "matrix" };
    if (iteration == -1) if ((iteration = findHighestIteration(Dir, Class, Var, fileext[0], fileext[1])) == -1) return -1;

    string hdr; bool found=0;
    if (!found) {
        string pn = string(Dir) + filename(Class, Var, fileext[0], iteration);
        if (!read_binary(hdr, modelOut, pn.c_str())) found=1;
    }
    if (!found) {
        string pn = string(Dir) + filename(Class, Var, fileext[1], iteration);
        if (!read(hdr, modelOut, pn.c_str())) found=1;
    }
    if (!found) return -1;

    if (flagOut) *flagOut = hdr;
    return iteration;
}

int MatrixFile::WriteFile(const char *Dir, const char *Class, const char *Var, Matrix *model, int iteration, const char *transcript) {        
    string hdr=transcript ? transcript : "", name=filename(Class, Var, "matrix", iteration), pn=string(Dir) + name;
    return write(hdr, model, pn.c_str(), name.c_str());
}

int MatrixFile::WriteFileBinary(const char *Dir, const char *Class, const char *Var, Matrix *model, int iteration, const char *transcript) {        
    string hdr=transcript ? transcript : "", name=filename(Class, Var, "matbin", iteration), pn=string(Dir) + name;
    return write_binary(hdr, model, pn.c_str(), name.c_str());
}

/* SettingsFile */

int SettingsFile::read(const char *dir, const char *name) {
    StringFile settings; string fields; int lastiter=0;
    if (StringFile::ReadFile(dir, name, VarName(), &settings, &fields, lastiter) < 0) { ERROR(name, ".", lastiter, ".name"); return -1; }
    for (int i=0; i<settings.lines; i++) {
        const char *line = settings.line[i];
        const char *sep = strstr(line, Separator());
        if (!sep) continue;
        Singleton<FlagMap>::Get()->Set(string(line, sep-line), sep + strlen(Separator()));
    }
    return 0;
}

int SettingsFile::write(const vector<string> &fields, const char *dir, const char *name) {
    LocalFile settings(string(dir) + MatrixFile::filename(name, VarName(), VarType(), 0), "w");
    MatrixFile::writeHeader(&settings, basename(settings.fn.c_str(),0,0), Join(fields, ",").c_str(), fields.size(), 1);
    for (vector<string>::const_iterator i = fields.begin(); i != fields.end(); i++) {
        StringFile::writeRow(&settings, StrCat(*i, Separator(), Singleton<FlagMap>::Get()->Get(*i)).c_str());
    }
    return 0;
}

/* Matrix Archive */ 

MatrixArchiveOut::~MatrixArchiveOut() { close(); }
MatrixArchiveOut::MatrixArchiveOut(const char *name) : file(0) { if (name) open(name); }
void MatrixArchiveOut::close() { if (file) { delete file; file=0; } }
int MatrixArchiveOut::open(const char *name) { file = new LocalFile(name, "w"); if (file->opened()) return 0; close(); return -1; }
int MatrixArchiveOut::write(const string &hdr, Matrix *m, const char *name) { return MatrixFile::write(hdr, m, file, name); } 

MatrixArchiveIn::~MatrixArchiveIn() { close(); }
MatrixArchiveIn::MatrixArchiveIn(const char *name) : file(0), index(0) { if (name) open(name); }
void MatrixArchiveIn::close() {if (file) { delete file; file=0; } index=0; }
int MatrixArchiveIn::open(const char *name) { LocalFileLineIter *lfi=new LocalFileLineIter(name); file=new IterWordIter(lfi, true); return !lfi->f.opened(); }
int MatrixArchiveIn::read(string &hdrout, Matrix **out) { index++; return MatrixFile::read(hdrout, out, file, 1); }
int MatrixArchiveIn::skip() { string hdr; Matrix *skip=0; index++; int ret = MatrixFile::read(hdr, &skip, file, 1); delete skip; return ret; }
const char *MatrixArchiveIn::filename() { if (!file) return ""; return ""; } // file->file->f.filename(); }
int MatrixArchiveIn::count(const char *name) { MatrixArchiveIn a(name); int ret=0; while(a.skip() != -1) ret++; return ret; }

}; // namespace LFL
