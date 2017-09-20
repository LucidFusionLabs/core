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

#include "core/ml/hmm.h"
#include "speech.h"
#include "core/ml/corpus.h"
#include "corpus.h"

namespace LFL {
Application *app;
DEFINE_bool(verify,       false,       "Verify copied matrix file");
}; // namespace LFL
using namespace LFL;

extern "C" LFApp *MyAppCreate(int argc, const char* const* argv) {
#ifdef _WIN32
  FLAGS_open_console = 1;
#endif
  app = CreateApplication(argc, argv).release();
  app->focused = CreateWindow(app).release();
  return app;
}

extern "C" int MyAppMain() {
  /* lfapp init */
  if (app->Create(__FILE__)) return -1;
  if (app->Init()) return -1;

  /* app init */
  if (app->argc<2) { INFO(app->argv[0], " <file.matrix> [file.matbin]"); return -1; }
  const char *in=app->argv[1], *out=app->argc>2 ? app->argv[2] : 0;

  if (!out) {
    static string prefix=".matrix", filename=app->argv[1];
    if (!SuffixMatch(filename, prefix, false)) return ERRORv(-1, "unrecognized input filename: ", filename);
    filename = filename.substr(0, filename.size() - prefix.size()) + ".matbin";
    out = filename.c_str();
  }

  if (0) {
    /* read input */
    INFO("input = ", in);
    MatrixFile mf;
    if (mf.Read(in)) return ERRORv(-1, "read: ", in);

    /* write output */
    INFO("output = ", out);
    if (mf.WriteBinary(out, BaseName(out))) return ERRORv(-1, "write ", out);
  }
  else {
    /* open input */
    INFO("input = ", in);
    LocalFileLineIter lfi(in);
    if (!lfi.f.Opened()) return ERRORv(-1, "FileWordIter: ", in);
    IterWordIter word(&lfi);

    string hdr;
    if (MatrixFile::ReadHeader(&word, &hdr) < 0) return ERRORv(-1, "readHeader: ", -1);
    int M=atof(word.NextString()), N=atof(word.NextString()), ret;

    /* open output */
    INFO("output = ", out);
    LocalFile file(out, "w");
    if (!file.Opened()) return ERRORv(-1, "LocalFile: ", strerror(errno));
    if (MatrixFile::WriteBinaryHeader(&file, BaseName(out), hdr.c_str(), M, N) < 0) return ERRORv(-1, "writeBinaryHeader: ", -1);

    /* read & write */
    vector<double> row(N, 0);
    for (int i=0; i<M; i++) {
      for (int j=0; j<N; j++) row[j] = atof(word.NextString()); 
      if ((ret = file.Write(row.data(), N*sizeof(double))) != N*sizeof(double)) FATAL("file write ret: ", ret);
    }
  }

  if (FLAGS_verify) {
    MatrixFile mf;
    if (mf.Read(in)) return ERRORv(-1, "read: ", in);

    MatrixFile nf;
    if (nf.ReadBinary(out)) return ERRORv(-1, "read_binary: ", out);

    if (mf.H != nf.H) ERROR("mismatching text '", mf.H, "' != '", nf.H, "'");

    Matrix *A=mf.F.get(), *B=nf.F.get();
    if (A->M != B->M || A->N != B->N) return ERRORv(-1, "dim mismatch ", A->M, " != ", B->M, " || ", A->N, " != ", B->N);

    MatrixIter(A) {
      if (A->row(i)[j] - B->row(i)[j] > 1e-6) ERROR("val mismatch (", i, ", ", j, ") ", A->row(i)[j], " != ", B->row(i)[j]);
    }

    INFO("input and output verified ", 1);
  }

  return 0;
}
