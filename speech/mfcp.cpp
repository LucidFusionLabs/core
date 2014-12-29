/*
 * $Id: mfcp.cpp 1336 2014-12-08 09:29:59Z justin $
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

#include "lfapp/lfapp.h"
#include "ml/hmm.h"
#include "speech.h"
#include "ml/corpus.h"
#include "corpus.h"

namespace LFL {
DEFINE_bool(verify,       false,       "Verify copied matrix file");

}; // namespace LFL
using namespace LFL;

extern "C" {
int main(int argc, const char *argv[]) {
    /* lfapp init */
#ifdef _WIN32
    open_console = 1;
#endif
    int ac=1; const char *av[] = { "" };
    if (app->Create(ac, av, __FILE__)) { app->Free(); return -1; }
    FLAGS_lfapp_audio = FLAGS_lfapp_video = FLAGS_lfapp_input = FLAGS_lfapp_camera = FLAGS_lfapp_network = 0;
    if (app->Init()) { app->Free(); return -1; }

    /* app init */
    if (argc<2) { INFO(argv[0], " <file.matrix> [file.matbin]"); app->Free(); return -1; }
    const char *in=argv[1], *out=argc>2 ? argv[2] : 0;

    if (!out) {
        static string prefix=".matrix", filename=argv[1];
        if (!SuffixMatch(filename, prefix, false)) { ERROR("unrecognized input filename: ", filename); return -1; }
        filename = filename.substr(0, filename.size() - prefix.size()) + ".matbin";
        out = filename.c_str();
    }

    if (0) {
        /* read input */
        INFO("input = ", in);
        MatrixFile mf;
        if (mf.Read(in)) { ERROR("read: ", in); return -1; }

        /* write output */
        INFO("output = ", out);
        if (mf.WriteBinary(out, basename(out,0,0))) { ERROR("write ", out); return -1; }
    }
    else {
        /* open input */
        INFO("input = ", in);
        LocalFileLineIter lfi(in);
        if (!lfi.f.Opened()) { ERROR("FileWordIter: ", in); return -1; }
        IterWordIter word(&lfi);

        string hdr;
        if (MatrixFile::ReadHeader(&word, &hdr) < 0) { ERROR("readHeader: ", -1); return -1; }
        int M=atof(word.Next()), N=atof(word.Next()), ret;

        /* open output */
        INFO("output = ", out);
        LocalFile file(out, "w");
        if (!file.Opened()) { ERROR("LocalFile: ", strerror(errno)); return -1;  }
        if (MatrixFile::WriteBinaryHeader(&file, basename(out,0,0), hdr.c_str(), M, N) < 0) { ERROR("writeBinaryHeader: ", -1); return -1; }

        /* read & write */
        double *row = (double *)alloca(N*sizeof(double));
        for (int i=0; i<M; i++) {
            for (int j=0; j<N; j++) row[j] = atof(word.Next()); 
            if ((ret = file.Write(row, N*sizeof(double))) != N*sizeof(double)) FATAL("file write ret: ", ret);
        }
    }

    if (FLAGS_verify) {
        MatrixFile mf;
        if (mf.Read(in)) { ERROR("read: ", in); return -1; }

        MatrixFile nf;
        if (nf.ReadBinary(out)) { ERROR("read_binary: ", out); return -1; }

        if (mf.H != nf.H) ERROR("mismatching text '", mf.H, "' != '", nf.H, "'");

        Matrix *A=mf.F, *B=nf.F;
        if (A->M != B->M || A->N != B->N) { ERROR("dim mismatch ", A->M, " != ", B->M, " || ", A->N, " != ", B->N); return -1; }

        MatrixIter(A) {
            if (A->row(i)[j] - B->row(i)[j] > 1e-6) ERROR("val mismatch (", i, ", ", j, ") ", A->row(i)[j], " != ", B->row(i)[j]);
        }

        INFO("input and output verified ", 1);
    }

    return 0;
}
}

