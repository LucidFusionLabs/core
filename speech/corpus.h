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

#ifndef LFL_SPEECH_CORPUS_H__
#define LFL_SPEECH_CORPUS_H__
namespace LFL {

struct WavCorpus : public Corpus {
  const static int MaxSecs = 60, MinWindows = 16;
  typedef function<void(const string&, SoundAsset*, const char*)> WavCB;

  WavCB wav_cb;
  AssetLoading *loader;
  WavCorpus(AssetLoading *l, const WavCB &cb) : wav_cb(cb), loader(l) {}

  void RunFile(const string &fn);
  void RunBuf(const string &srcdir, const string &afn, const char *adata, int adatalen, const char *transcript) {
    SoundAsset wav(loader);
    wav.filename = afn;
    wav.Load(adata, adatalen, afn.c_str(), MaxSecs);
    RunWAV(srcdir, &wav, transcript);
  }
  void RunFile(const string &srcdir, const string &afn, const char *transcript) {
    SoundAsset wav(loader);
    wav.filename = afn;
    wav.Load(MaxSecs);
    RunWAV(srcdir, &wav, transcript);
  }
  void RunWAV(const string &srcdir, SoundAsset *wav, const char *transcript) {
    if (!wav->wav) ERROR("no audio data for ", wav->filename); 
    else if (wav->wav->ring.size < MinWindows * FLAGS_feat_window) ERROR(wav->wav->ring.size, " audio samples for ", wav->filename, " < MinSamples ", MinWindows * FLAGS_feat_window);
    else wav_cb(srcdir, wav, transcript); /* thunk */
    wav->filename.clear();
    wav->Unload();
  }
};

struct VoxForgeTgzFile { 
  WavCorpus *wav_corpus;
  map<string, string> wav2transcript;
  VoxForgeTgzFile(WavCorpus *wc) : wav_corpus(wc) {}

  void Run(const string &file, ApplicationLifetime *lifetime=0) {
    ArchiveIter a(file.c_str());
    if (!a.impl) return;
    for (const char *afn = a.Next(); (!lifetime || lifetime->run) && afn; afn = a.Next()) {
      if (SuffixMatch(afn, "etc/prompts", false)) HandlePrompts(       a.buf.data(), a.buf.size());
      else if (BaseDir(afn, "wav"))               HandleWAV(file, afn, a.buf.data(), a.buf.size());
    }
  }
  void HandlePrompts(const char *promptsdata, int promptsdatalen) {
    string promptsfile(promptsdata, promptsdatalen);
    LocalFileLineIter prompts(promptsfile.c_str());

    for (const char *line = prompts.Next(); line; line = prompts.Next()) {
      StringWordIter words(line); 
      string key = words.NextString();
      int val = words.next_offset;
      if (key.empty() || val<0) continue;
      wav2transcript[BaseName(key)] = &line[val];
    }
  }
  void HandleWAV(const string &srcdir, const string &afn, const char *ad, int as) {
    int bnl;
    const char *bn = BaseName(afn, &bnl);
    string name = string(bn, bnl);

    auto ti = wav2transcript.find(name);
    if (ti == wav2transcript.end()) { ERROR("wav2transcript missing ", name); return; }

    wav_corpus->RunBuf(srcdir, afn, ad, as, ti->second.c_str());
  }
};

struct FeatCorpus {
  typedef function<void(const char*, Matrix*, Matrix*, const char*)> FeatCB;

  static int FeatIter(const char *featdir, const FeatCB &cb, ApplicationLifetime *lifetime=0) {
    DirectoryIter d(featdir); int count=0;

    for (const char *fn = d.Next(); (!lifetime || lifetime->run) && fn; fn = d.Next()) {
      bool matrixF=SuffixMatch(fn, ".feat", false), txtF=SuffixMatch(fn, ".txt", false), listF=SuffixMatch(fn, ".featlist", false);
      string pn = string(featdir) + fn;

      if      (matrixF || txtF) count += FeatIterMatrix(pn.c_str(), matrixF, cb);
      else if (listF)           count += FeatIterMatList(pn.c_str(), cb);
    }
    return count;
  }

  static int FeatIterMatrix(const char *fn, bool hdr, const FeatCB &cb) {
    MatrixFile feat;
    if (feat.Read(fn, hdr)) return 0;
    AddFeature(&feat, fn, cb);
    return 1;
  }

  static int FeatIterMatList(const char *fn, const FeatCB &cb, ApplicationLifetime *lifetime=0) {
    MatrixArchiveInputFile featlist;
    featlist.Open(fn);
    int count=0;

    while (!lifetime || lifetime->run) {
      MatrixFile feat;
      featlist.Read(&feat);
      if (!feat.F) break;
      if (feat.F->M < 12) { ERROR("feature '", feat.H, "' too short ", feat.F->M, ", skipping"); count++; continue; }
      string afn = ArchiveFilename(fn, count);
      AddFeature(&feat, afn.c_str(), cb);
      count++;
    }
    return count;
  }

  static void AddFeature(MatrixFile *feat, const char *pn, const FeatCB &cb) {
    unique_ptr<Matrix> orig(feat->F->Clone());
    feat->F = Features::FromFeat(feat->F.release(), Features::Flag::Full);
    DEBUG("processing %s : %s", pn,  feat->Text());
    cb(pn, orig.get(), feat->F.get(), feat->Text());
  }

  static string ArchiveFilename(const char *fn, int index) { return StrCat(fn, ",", index); }

  static int ArchiveFilenameIndex(const char *name, string *fn) {
    string n1, n2; if (Split(name, isint<','>, &n1, &n2)) return -1;
    if (fn) *fn = n1;
    return atoi(n2.c_str());
  } 
};

struct PathCorpus {
  typedef function<void(AcousticModel::Compiled*, Matrix*, double, Time, Matrix*, Matrix*, const char*)> PathCB;

  static void AddPath(MatrixArchiveOutputFile *out, Matrix *viterbi, const char *uttfilename) {
    MatrixFile f(viterbi, BaseName(uttfilename));
    out->Write(&f, "viterbi");
    f.Clear();
  }

  static int PathIter(const char *featdir, MatrixArchiveInputFile *paths, PathCB cb, ApplicationLifetime *lifetime=0) {
    MatrixArchiveInputFile utts; string lastarchive; int listfile, count=0;

    while (!lifetime || lifetime->run) {
      Timer vtime;
      MatrixFile path, utt;

      /* read utterance viterbi path */
      if (paths->Read(&path)) { ERROR("read utterance path: ", paths->index); break; }

      string uttfilename = string(featdir) + path.Text();
      int uttindex = FeatCorpus::ArchiveFilenameIndex(uttfilename.c_str(), &uttfilename);
      if (uttindex < 0) {
        /* not archive, read from file */
        utt.Read(uttfilename.c_str());
      }
      else {
        /* archive url */
        if (StringChanged(lastarchive, uttfilename.c_str())) {
          utts.Close();
          utts.Open(uttfilename);
        }

        /* seek to archive index */
        while (utts.index < uttindex) utts.Skip();
        if (utts.index != uttindex) { ERROR("skipping ", uttfilename, " ", utts.index, " != ", uttindex); continue; }
        utts.Read(&utt);
      }
      if (!utt.F) { ERROR("skipping ", uttfilename); continue; }
      if (path.F->M != utt.F->M) { ERROR("path/utt mismatch offset ", uttindex, " len ", path.F->M, " != ", utt.F->M, " transcript='", utt.Text(), "' file=", uttfilename); continue; }

      DEBUG("processing %s", uttfilename.c_str());
      unique_ptr<Matrix> orig(utt.F->Clone());
      utt.F = Features::FromFeat(utt.F.release(), Features::Flag::Full);
      cb(0, path.F.get(), 0, vtime.GetTime(), orig.get(), utt.F.get(), utt.Text());
      count++;
    }
    return count;
  }

  static bool StringChanged(string &last, const char *next) { if (!strcmp(last.c_str(), next)) return 0; last=next; return 1; }
};

void WavCorpus::RunFile(const string &fn) { 
  string dn = string(fn, DirNameLen(fn));

  /* /corpus/wav/voxforge/k-20090202-afe.tgz */ 
  if (SuffixMatch(fn, ".tgz", false) || SuffixMatch(fn, ".tar.gz", false) || SuffixMatch(fn, ".tar", false)) {
    VoxForgeTgzFile vftgz(this);
    vftgz.Run(fn);
  } else if (SuffixMatch(fn, ".wav", false)) { /* TIMIT style: .wav with matching .txt */
    string tfn = string(fn, fn.size()-3) + "txt";
    LocalFile tf(tfn.c_str(), "r");
    if (!tf.Opened()) return;

    NextRecordReader nr(&tf);
    StringWordIter words(nr.NextLine());
    if (bool skip_two_words=true) { words.Next(); words.Next(); }
    string transcript = toupper(words.in.data() + words.next_offset);
    transcript = togrep(transcript.c_str(), isalnum, isspace);

    WavCorpus::RunFile(dn, fn, transcript.c_str());
  }
}

}; // namespace LFL
#endif // LFL_SPEECH_CORPUS_H__
