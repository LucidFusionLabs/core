/*
 * $Id: corpus.h 1336 2014-12-08 09:29:59Z justin $
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

#ifndef __LFL_CODER_CORPUS_H__
#define __LFL_CODER_CORPUS_H__
namespace LFL {

struct WavCorpus : public Corpus {

    typedef void (*WavCB)(const char *sd, SoundAsset *wav, const char *transcript, void *arg);

    void Run(const char *fn, void *Cb, void *arg);

    const static int MaxSecs = 60, MinWindows = 16;

    static void Thunk(const char *srcdir, const char *afn, const char *adata, int adatalen, const char *transcript, WavCB cb, void *cbarg) {
        SoundAsset wav; wav.filename = afn;
        wav.Load(adata, adatalen, afn, MaxSecs);
        Thunk(srcdir, &wav, transcript, cb, cbarg);
    }

    static void Thunk(const char *srcdir, const char *afn, const char *transcript, WavCB cb, void *cbarg) {
        SoundAsset wav; wav.filename = afn;
        wav.Load(MaxSecs);
        Thunk(srcdir, &wav, transcript, cb, cbarg);
    }

    static void Thunk(const char *srcdir, SoundAsset *wav, const char *transcript, WavCB cb, void *cbarg) {
        if (!wav->wav) ERROR("no audio data for ", wav->filename); 
        else if (wav->wav->ring.size < MinWindows * FLAGS_feat_window) ERROR(wav->wav->ring.size, " audio samples for ", wav->filename, " < MinSamples ", MinWindows * FLAGS_feat_window);
        else cb(srcdir, wav, transcript, cbarg); /* thunk */

        wav->filename.clear();
        wav->Unload();
    }
};

struct VoxForgeTgzFile { 
    VoxForgeTgzFile() {}

    ArchiveIter a;
    int open(const char *file) { ArchiveIter A(file); a=A; A.impl=0; return !a.impl; }
    
    typedef map<string, string> s2smap;
    s2smap wav2transcript;

    void wav_iter(const char *srcdir, WavCorpus::WavCB cb, void *cbarg) {
        for (const char *afn = a.next(); Running() && afn; afn = a.next()) {
            if (SuffixMatch(afn, "etc/prompts", false)) handle_prompts((const char *)a.data(), a.size());
            else if (basedir(afn, "wav")) handle_wav_filename(srcdir, afn, cb, cbarg);
        }
    }

    void handle_prompts(const char *promptsdata, int promptsdatalen) {
        string promptsfile(promptsdata, promptsdatalen);
        LocalFileLineIter prompts(promptsfile.c_str());

        for (const char *line = prompts.next(); line; line = prompts.next()) {
            StringWordIter words(line); const char *key=words.next(); int val=words.offset;
            if (!key || val<0) continue;
            wav2transcript[basename(key,0,0)] = &line[val];
        }
    }

    void handle_wav_filename(const char *srcdir, const char *afn, WavCorpus::WavCB cb, void *cbarg) {
        int bnl; const char *bn = basename(afn, 0, &bnl);
        string name = string(bn, bnl);

        const char *transcript = s2sval(wav2transcript, name.c_str());
        if (!transcript) { ERROR("wav2transcript missing ", name); return; }

        WavCorpus::Thunk(srcdir, afn, (char*)a.data(), a.size(), transcript, cb, cbarg);
    }
    
    static const char *s2sval(s2smap &map, const char *key) { s2smap::iterator i=map.find(key); return i != map.end() ? (*i).second.c_str() : 0; }
};

struct FeatCorpus {
    typedef void (*FeatCB)(const char *filename, Matrix *origFeatures, Matrix *fullFeatures, const char *transcript, void *arg);

    static int feat_iter(const char *featdir, FeatCB cb, void *arg) {
        DirectoryIter d(featdir); int count=0;

        for (const char *fn = d.next(); Running() && fn; fn = d.next()) {
            bool matrixF=SuffixMatch(fn, ".feat", false), txtF=SuffixMatch(fn, ".txt", false), listF=SuffixMatch(fn, ".featlist", false);
            string pn = string(featdir) + fn;

            if      (matrixF || txtF) count += feat_iter_matrix(pn.c_str(), matrixF, cb, arg);
            else if (listF)           count += feat_iter_matlist(pn.c_str(), cb, arg);
        }
        return count;
    }

    static int feat_iter_matrix(const char *fn, bool hdr, FeatCB cb, void *arg) {
        MatrixFile feat;
        if (feat.read(fn, hdr)) return 0;
        add_feature(&feat, fn, cb, arg);
        return 1;
    }

    static int feat_iter_matlist(const char *fn, FeatCB cb, void *arg) {
        MatrixArchiveIn featlist;
        featlist.open(fn);
        int count=0;

        while (Running()) {
            MatrixFile feat;
            featlist.read(&feat);
            if (!feat.F) break;
            if (feat.F->M < 12) { ERROR("feature '", feat.T, "' too short ", feat.F->M, ", skipping"); count++; continue; }
            string afn = archive_filename(fn, count);
            add_feature(&feat, afn.c_str(), cb, arg);
            count++;
        }
        return count;
    }

    static void add_feature(MatrixFile *feat, const char *pn, FeatCB cb, void *arg) {
        Matrix *orig = feat->F->clone();
        feat->F = Features::fromFeat(feat->F, Features::Flag::Full);
        DEBUG("processing %s : %s", pn,  feat->text());
        cb(pn, orig, feat->F, feat->text(), arg);
        delete orig;
    }
    
    static string archive_filename(const char *fn, int index) { return StrCat(fn, ",", index); }

    static int archive_filename_index(const char *name, string *fn) {
        string n1, n2; if (Split(name, isint<','>, &n1, &n2)) return -1;
        if (fn) *fn = n1;
        return atoi(n2.c_str());
    } 
};

struct PathCorpus {
    typedef void (*PathCB)(AcousticModel::Compiled *, Matrix *viterbi, double vprob, double vtime, Matrix *MFCC, Matrix *features, const char *transcript, void *arg);

    static void add_path(MatrixArchiveOut *out, Matrix *viterbi, const char *uttfilename) {
        MatrixFile f(viterbi, basename(uttfilename,0,0));
        out->write(&f, "viterbi");
        f.clear();
    }

    static int path_iter(const char *featdir, MatrixArchiveIn *paths, PathCB cb, void *arg) {
        MatrixArchiveIn utts; string lastarchive; int listfile, count=0;

        while (Running()) {
            Timer vtime;
            MatrixFile path, utt;

            /* read utterance viterbi path */
            if (paths->read(&path)) { ERROR("read utterance path: ", paths->index); break; }

            string uttfilename = string(featdir) + path.text();
            int uttindex = FeatCorpus::archive_filename_index(uttfilename.c_str(), &uttfilename);
            if (uttindex < 0) {
                /* not archive, read from file */
                utt.read(uttfilename.c_str());
            }
            else {
                /* archive url */
                if (string_changed(lastarchive, uttfilename.c_str())) {
                    utts.close();
                    utts.open(uttfilename.c_str());
                }

                /* seek to archive index */
                while (utts.index < uttindex) utts.skip();
                if (utts.index != uttindex) { ERROR("skipping ", uttfilename, " ", utts.index, " != ", uttindex); continue; }
                utts.read(&utt);
            }
            if (!utt.F) { ERROR("skipping ", uttfilename); continue; }
            if (path.F->M != utt.F->M) { ERROR("path/utt mismatch offset ", uttindex, " len ", path.F->M, " != ", utt.F->M, " transcript='", utt.text(), "' file=", uttfilename); continue; }

            DEBUG("processing %s", uttfilename.c_str());
            Matrix *orig = utt.F->clone();
            utt.F = Features::fromFeat(utt.F, Features::Flag::Full);
            cb(0, path.F, 0, vtime.time(), orig, utt.F, utt.text(), arg);
            delete orig;
            count++;
        }
        return count;
    }

    static bool string_changed(string &last, const char *next) { if (!strcmp(last.c_str(), next)) return 0; last=next; return 1; }
};

void WavCorpus::Run(const char *fn, void *Cb, void *arg) { 
    WavCB cb = (WavCB)Cb;
    string dn = string(fn, dirnamelen(fn));

    /* /corpus/wav/voxforge/k-20090202-afe.tgz */ 
    if (SuffixMatch(fn, ".tgz", false) || SuffixMatch(fn, ".tar.gz", false) || SuffixMatch(fn, ".tar", false)) {
        VoxForgeTgzFile vftgz;
        if (vftgz.open(fn)) return;
        vftgz.wav_iter(dn.c_str(), cb, arg);
    }
    else if (SuffixMatch(fn, ".wav", false)) { /* TIMIT style: .wav with matching .txt */
        string tfn = string(fn, strlen(fn)-3) + "txt";
        LocalFile tf(tfn.c_str(), "r");
        if (!tf.opened()) return;

        StringWordIter words(tf.nextline());
        if (bool skip_two_words=true) { words.next(); words.next(); }
        string transcript = toupper(words.in + words.offset);
        transcript = togrep(transcript.c_str(), isalnum, isspace);

        WavCorpus::Thunk(dn.c_str(), fn, transcript.c_str(), cb, arg);
    }
}

}; // namespace LFL
#endif // __LFL_CODER_CORPUS_H__
