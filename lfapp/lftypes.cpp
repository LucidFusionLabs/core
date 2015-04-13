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

}; // namespace LFL
