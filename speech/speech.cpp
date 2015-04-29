/*
 * $Id: speech.cpp 1336 2014-12-08 09:29:59Z justin $
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
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/flow.h"
#include "lfapp/gui.h"
#include "speech.h"

#ifdef LFL_CUDA
#include "lfcuda/lfcuda.h"
CudaAcousticModel *CAM = 0;
#endif

namespace LFL {
DEFINE_bool(TriphoneModel, false, "Using triphone model");

/* phonemes */
const char *Phoneme::name(char in) {
#   undef  XX
#   define XX(phone) else if (phone == in) return #phone;
    if (0) {}
#   include "phones.h"
    return 0;
}

char Phoneme::id(const char *in, int len) {
    if (!len) len=strlen(in);
#   undef  XX
#   define XX(phone) else if(strlen(#phone)==len && !strncmp(#phone,in,len)) return phone;
    if (0) {}
#   include "phones.h"
    return -1;
}

PronunciationDict *PronunciationDict::instance() {
    static PronunciationDict *inst = 0;
    if (inst) return inst;
    inst = new PronunciationDict();

    LocalFileLineIter file(StrCat(app->assetdir, "cmudict.txt"));
    if (!file.f.Opened()) { ERROR("no ", app->assetdir, "cmudict.txt"); return 0; }    

    readDictionary(&file, inst);
    return inst;
}

int PronunciationDict::readDictionary(StringIter *in, PronunciationDict *out) {
    for (const char *line = in->Next(); line; line = in->Next()) {
        const char *c=line; if ((*c) == ';') continue; /* skip comments */

        /* Format: word <two spaces> pronunciation */
        c += LengthChar(c, notspace);
        if (!*c || !isspace(*(c+1))) continue;
        *(char *)c = 0;

        const char *word=line, *pronunciation=c+2; /* (k,v) */

        char phone[1024], accent[1024], phones; /* pronunciation phone-ID sequence */
        if ((phones = readPronunciation(pronunciation, -1, phone, accent, sizeof(phone)-1)) <= 0) continue;
        phone[phones]=0;

        /* insert(word, valbufOffset) */
        int offset = out->val.Alloc(phones*2+1);
        out->word_pronunciation[word] = offset;

        memcpy(&out->val.buf[offset], phone, phones+1);
        memcpy(&out->val.buf[offset+phones+1], accent, phones);
    }
    return 0;
}

int PronunciationDict::readPronunciation(const char *in, int len, char *phonesOut, char *accentOut, int outlen) {
    StringWordIter phones(StringPiece(in, len)); int outi=0;
    for (const char *phone_text=phones.Next(); phone_text; phone_text=phones.Next()) {
        if (outi >= outlen) return -1;
        string phone(phone_text, phones.cur_len);

        int accent = LengthChar(phone.c_str(), isalpha);
        char stress = phone[accent];
        ((char *)phone_text)[accent]=0;
        accent = isdigit(stress) ? stress - '0' : 0;

        if (!(phonesOut[outi] = Phoneme::id(phone.c_str(), 0))) return -1;
        accentOut[outi] = accent;
        outi++;
    }
    return outi;
}

const char *PronunciationDict::pronounce(const char *in) {
    Map::iterator i = word_pronunciation.find(toupper(in));
    if (i == word_pronunciation.end()) return 0;
    int pronunciationIndex = (*i).second;
    return &val.buf[pronunciationIndex];
}

int PronunciationDict::pronounce(const char *utterance, const char **w, const char **wa, int *phones, int max) {
    *phones=0;
    int words=0;
    StringWordIter script(utterance);
    for (string word = IterNextString(&script); !script.Done(); word = IterNextString(&script)) {
        const char *pronunciation = pronounce(word.c_str());
        if (!pronunciation || words+1 >= max) { DEBUG("pronunciation %s count=%d", word.c_str(), words); return -1; }

        int len = strlen(pronunciation);
        (*phones) += len;

        w[words] = pronunciation;
        wa[words] = pronunciation + len + 1; /* accent */
        words++;
    }
    return words;
}

/* features */
int Features::FilterZeroth=0;
int Features::Deltas=1;
int Features::DeltaDeltas=1;
int Features::MeanNormalization=0;
int Features::VarianceNormalization=0;

Matrix *Features::filterZeroth(Matrix *features) {
    Matrix *fz = new Matrix(features->M, features->N-1);
    MatrixRowIter(fz) {
        double *in=features->row(i), *out=fz->row(i);
        MatrixColIter(fz) out[j] = in[j+1];
    }
    delete features;
    return fz;
}

void Features::meanAndVarianceNormalization(int D, double *feat, const double *mean, const double *var) {
    Vector::Sub(feat, mean, D);
    if (var) Vector::Mult(feat, var, D);
}

void Features::deltaCoefficients(int D, const double *n2, double *f, const double *p2) {
    for (int j=0; j<D; j++)
        f[D+j] = p2[j] - n2[j];
}

void Features::deltaDeltaCoefficients(int D, const double *n3, const double *n1, double *f,  const double *p1, const double *p3) {
    for (int j=0; j<D; j++) {
        double d1 = p3[j] - n1[j];
        double d2 = p1[j] - n3[j];
        f[D*2+j] = d1 - d2;
    }
}

void Features::patchDeltaCoefficients(int D, const double *in, double *out1, double *out2) {
    Vector::Assign(out1, in, D);
    Vector::Assign(out2, in, D);
}

void Features::patchDeltaDeltaCoefficients(int D, const double *in, double *out1, double *out2, double *out3) {
    Vector::Assign(out1, in, D);
    Vector::Assign(out2, in, D);
    Vector::Assign(out3, in, D);
}

Matrix *Features::deltaCoefficients(Matrix *in, bool dd) {
    int M=in->M, D=in->N;
    Matrix *features = new Matrix(M, D*(2+dd));
    features->AssignR(in);
    delete in;

    /* copy sphinx */
    MatrixRowIter(features) {
        if (i-2 < 0 || i+2 >= M) continue;
        deltaCoefficients(D, features->row(i-2), features->row(i), features->row(i+2));
 
        if (i-3 < 0 || i+3 >= M || !dd) continue;
        deltaDeltaCoefficients(D, features->row(i-3), features->row(i-1), features->row(i), features->row(i+1), features->row(i+3));
    }

    /* delta - patch ends */
    patchDeltaCoefficients(D, &features->row(    2)[D], &features->row(    1)[D], &features->row(    0)[D]);
    patchDeltaCoefficients(D, &features->row(M-1-2)[D], &features->row(M-1-1)[D], &features->row(M-1-0)[D]);

    /* delta delta - patch ends */
    if (dd) {
        patchDeltaDeltaCoefficients(D, &features->row(    3)[D*2], &features->row(    2)[D*2], &features->row(    1)[D*2], &features->row(    0)[D*2]);
        patchDeltaDeltaCoefficients(D, &features->row(M-1-3)[D*2], &features->row(M-1-2)[D*2], &features->row(M-1-1)[D*2], &features->row(M-1-0)[D*2]);
    }

    return features;
}

Matrix *Features::fromFeat(Matrix *features, int flag, bool filterzeroth, bool deltas, bool deltadeltas, bool meannorm, bool varnorm) {
    if (flag == Flag::Full) {
        if (filterzeroth) features = filterZeroth(features);

        if (meannorm) {
            double *mean = (double *)alloca(features->N * sizeof(double));
            double *var = (double *)alloca(features->N * sizeof(double));

            Vector::Assign(mean, 0.0, features->N);
            MatrixRowIter(features) Vector::Add(mean, features->row(i), features->N);
            Vector::Div(mean, features->M, features->N);

            if (varnorm) {                
                Vector::Assign(var, 0.0, features->N);
                MatrixIter(features) {
                    double diff = features->row(i)[j] - mean[j];
                    var[j] += diff*diff;
                }
                MatrixColIter(features) var[j] = sqrt(var[j] / features->M);
            }

            MatrixRowIter(features) {
                meanAndVarianceNormalization(features->N, features->row(i), mean, VarianceNormalization ? var : 0);
            }            
        }

        if (deltas) features = deltaCoefficients(features, deltadeltas);
    }
    return features;
}

Matrix *Features::fromFeat(Matrix *features, int flag) {
    return fromFeat(features, flag, FilterZeroth, Deltas, DeltaDeltas, MeanNormalization, VarianceNormalization);
}

Matrix *Features::fromAsset(SoundAsset *wav, int flag) {
    if (wav->channels > 1) ERROR("Features::fromAsset called on SoundAsset with ", wav->channels, " channels");
    RingBuf::Handle B(wav->wav);
    Matrix *features = fromBuf(&B);
    return fromFeat(features, flag);
}

Matrix *Features::fromBuf(const RingBuf::Handle *in, Matrix *out, vector<StatefulFilter> *filter, Allocator *alloc) {
    Matrix *features = 0;
    if      (FLAGS_feat_type == "MFCC") features = MFCC(in, out, alloc);
    else if (FLAGS_feat_type == "PLP")  features = PLP(in, out, filter, alloc);
    return features;
}

RingBuf *Features::reverse(const Matrix *in, int samplerate, const Matrix *f0, Allocator *alloc) {
    RingBuf *wav = 0;
    if      (FLAGS_feat_type == "MFCC") wav = InvMFCC(in, samplerate, f0);
    else if (FLAGS_feat_type == "PLP")  wav = InvPLP(in, samplerate, alloc);
    return wav;
}

int Features::dimension() {
    if      (FLAGS_feat_type == "MFCC") return FLAGS_feat_cepcoefs;
    else if (FLAGS_feat_type == "PLP")  return feat_lpccoefs+1;
    return 0;
}

/* AcousticModel */

double *AcousticModel::State::transit(Compiled *model, Matrix *transit, State *Rstate, unsigned *toOut) {
    int toind = Rstate->val.emission_index;
    if (toind < 0 || toind >= model->states) FATAL("OOB toind ", toind);
    unsigned to = model->state[toind].id();
    if (toOut) *toOut = to;
    return AcousticModel::State::transit(transit, to);
}

double *AcousticModel::State::transit(Matrix *trans, unsigned K) { double k[3]={0,(double)K,0}; return (double*)bsearch(k, trans->m, trans->M, sizeof(double)*TransitCols, transitionSort); }

int AcousticModel::State::transitionSort(const void *a, const void *b) { return DoubleSortR((void*)((double*)a+1), (void*)((double*)b+1)); }

void AcousticModel::State::sortTransitionMap(double *trans, int M) { qsort(trans, M, sizeof(double)*TransitCols, transitionSort); }

void AcousticModel::State::sortTransitionMap(Matrix *trans) {
    if (!trans || trans->N != TransitCols) FATAL("error ", trans?trans->N:0, " != ", TransitCols);
    return sortTransitionMap(trans->m, trans->M);
}

int AcousticModel::State::localizeTransitionMap(Matrix *transIn, Matrix *transOut, Compiled *scope, int scopeState, int minSamples, Compiled *dubdref, const char *srcname) {
    if (!transIn || !transIn->M) return 0;
    if (transIn->N != TransitCols || transOut->N != TransitCols) { ERROR("wrong dimensions (", transIn->N, "|", transOut->N, ") != ", TransitCols); return -1; }
    if (transIn->M > transOut->M) { ERROR("rows overflow ", transIn->M, " > ", transOut->M); return -1; }

    MatrixRowIter(transIn) {
        int si = scopeState-1, skipped=-1;
        do {
            skipped++;
            for (si++; si<scope->states; si++) {
                unsigned match = scope->state[si].id();
                if (dubdref) {
                    if (srcname) { 
                        int ap, an, a = parseName(srcname, 0, &ap, &an);
                        int bp, bn, b = parseName(scope->state[si].name.c_str(), 0, &bp, &bn);
                        if (!triphoneConnect(ap, a, an, bp, b, bn)) continue;
                    }

                    int matchind = scope->state[si].val.emission_index;
                    if (matchind < 0 || matchind >= dubdref->states) FATAL("OOB matchind ", matchind);
                    match = dubdref->state[matchind].id();
                }
                if (transIn->row(i)[TC_Edge] == match) break;
            }
            if (si >= scope->states) { ERRORf("cant find %d %f dim(%d,%d) scope(%d,%d) skipped=%d", i, transIn->row(i)[TC_Edge], transIn->M, transIn->N, scopeState, minSamples, skipped); return -1; }
        }
        while (minSamples && scope->state[si].val.samples < minSamples);

        transOut->row(i)[TC_Edge] = si;
        transOut->row(i)[TC_Cost] = transIn->row(i)[TC_Cost];
    }
    return 0;
}

unsigned AcousticModel::State::tied(Matrix *tiedstates, int pp, int p, int pn, int k) {
    int tied_ind = pp*LFL_PHONES*LFL_PHONES*StatesPerPhone + p*LFL_PHONES*StatesPerPhone + pn*StatesPerPhone + k;
    double *r = tiedstates->row(tied_ind);
    if (!r) FATAL("tied ret ", r);
    unsigned thash = r[0];
    return thash;
}

void AcousticModel::State::assign(State *out, int *outind, int outsize, StateCollection *model, int pp, int p, int pn, int states, int val) {
    Matrix *tiedstates = model->tiedStates();
    State *s;
    for (int k=0; k<states; k++) {
        if ((*outind) >= outsize) FATAL("overflow ", *outind, " >= ", outsize);

        string n = AcousticModel::name(pp, p, pn, k);
        unsigned ohash = fnv32(n.c_str()), hash = ohash, thash = tiedstates ? AcousticModel::State::tied(tiedstates, pp, p, pn, k) : 0;
        bool tied = tiedstates && thash != hash;
        if (tied) hash = thash;
        if (!(s = model->getState(hash))) FATAL("no state ", n, " (tied=", tied, ")");
        if (!tied) { if (n != s->name) FATAL("mismatching states ", n, " ", s->name); }
        else { if (0) INFO("tied ", n, " to ", s->name); }

        int ind = *outind;
        out[ind].assignPtr(s);
        out[ind].val.samples = val;
        out[ind].transition.Absorb(out[ind].transition.Clone());

        if (tied) {
            static unsigned silhash = fnv32("Model_SIL_State_00");

            out[ind].alloc = Singleton<MallocAlloc>::Get();
            out[ind].name = n;

            MatrixIter(&out[ind].transition) { out[ind].transition.row(i)[TC_Self] = ohash; }
            sortTransitionMap(&out[ind].transition);
        }
        else {
            MatrixIter(&out[ind].transition) if (out[ind].transition.row(i)[TC_Self] != hash) FATAL("mismatching transit ", hash, " ", out[ind].transition.row(i)[TC_Self]); 
        }

        (*outind)++;
    }
}

int AcousticModel::parseName(const string &name, int *phonemeStateOut, int *prevPhoneOut, int *nextPhoneOut) {
    static string prefix="Model_", suffix="_State_";
    if (strncmp(name.c_str(), prefix.c_str(), prefix.size())) return -1;

    const char *vn = name.c_str() + prefix.size(), *sn;
    if (!(sn = strstr(vn, suffix.c_str()))) return -1;
    if (phonemeStateOut) *phonemeStateOut = atoi(sn + suffix.size());
    if (!prevPhoneOut && !nextPhoneOut) return Phoneme::id(vn, sn-vn);

    string phoneme(vn, sn-vn);
    const char *pp= phoneme.c_str(), *px, *pn;
    if (!(px = strchr(phoneme.c_str(), '_')) || !(pn = strchr(px+1, '_'))) {
        if (prevPhoneOut) *prevPhoneOut = 0;
        if (nextPhoneOut) *nextPhoneOut = 0;
        return Phoneme::id(phoneme.c_str());
    }

    if (prevPhoneOut) *prevPhoneOut = Phoneme::id(pp, px-pp);
    if (nextPhoneOut) *nextPhoneOut = Phoneme::id(pn+1);
    return Phoneme::id(px+1, pn-px-1);
}

string AcousticModel::flags() {
    string s = StrCat("sr=", FLAGS_sample_rate, ",type=", FLAGS_feat_type);

    if (FLAGS_feat_minfreq)                        StrAppend(&s, ",minfreq=", FLAGS_feat_minfreq);
    if (FLAGS_feat_maxfreq != FLAGS_sample_rate/2) StrAppend(&s, ",maxfreq=", FLAGS_feat_maxfreq);

    StrAppend(&s, ",fftwin=", FLAGS_feat_window, ",ffthop=", FLAGS_feat_hop);

    if (FLAGS_feat_type == "MFCC") {
        if (FLAGS_feat_preemphasis) StrAppend(&s, ",preemph=", FLAGS_feat_preemphasis_filter);

        StrAppend(&s, ",mels=", FLAGS_feat_melbands, ",ceps=", FLAGS_feat_cepcoefs);
    }

    if (Features::FilterZeroth)          s += ",filterzeroth";
    if (Features::MeanNormalization)     s += ",cmn";
    if (Features::VarianceNormalization) s += ",cvn";
    if (Features::Deltas)                s += ",delta";
    if (Features::DeltaDeltas)           s += ",deltadelta";

    if (FLAGS_TriphoneModel) s += ",triphone";

    return s;
}

void AcousticModel::loadflags(const char *flags) {
    StringWordIter iter(flags, iscomma);
    for (string k = IterNextString(&iter); !iter.Done(); k = IterNextString(&iter)) {
        char *v; double val;
        if ((v = (char*)strchr(k.c_str(), '='))) { *v++=0; val=atof(v); }

        if      (k == "sr")         FLAGS_sample_rate = val;
        else if (k == "type")       FLAGS_feat_type   = v;
        else if (k == "minfreq")    FLAGS_feat_minfreq  = val;
        else if (k == "maxfreq")    FLAGS_feat_maxfreq  = val;
        else if (k == "ffthop")     FLAGS_feat_hop      = val;
        else if (k == "fftwin")     FLAGS_feat_window   = val;
        else if (k == "mels")       FLAGS_feat_melbands = val;
        else if (k == "ceps")       FLAGS_feat_cepcoefs = val;
        else if (k == "filterzeroth") Features::FilterZeroth          = 1;
        else if (k == "cmn")          Features::MeanNormalization     = 1;
        else if (k == "cvn")          Features::VarianceNormalization = 1;
        else if (k == "delta")        Features::Deltas                = 1;
        else if (k == "deltadelta")   Features::DeltaDeltas           = 1;
        else if (k == "preemph") { FLAGS_feat_preemphasis_filter = val; FLAGS_feat_preemphasis = 1; }
        else if (k == "triphone") { FLAGS_TriphoneModel = 1; }
    }
    INFO("loaded flags: ", flags);
}

int AcousticModel::write(StateCollection *model, const char *name, const char *dir, int iteration, int minSamples) {
    int ret=0, states=0, means=0, transits=0, D=0, K=0;
    string flagtext=flags();

    AcousticModel::StateCollection::Iterator iter;
    for (model->beginState(&iter); !iter.done; model->nextState(&iter)) {
        AcousticModel::State *s = iter.v;
        if (minSamples && s->val.samples < minSamples) continue;
        if (!D) D = s->emission.mean.N;
        if (!K) K = s->emission.mean.M;
        if (D != s->emission.mean.N || D != s->emission.diagcov.N) FATAL("D mismatch ", D, " ", s->emission.mean.N, " ", s->emission.diagcov.N);
        if (K != s->emission.mean.M || K != s->emission.diagcov.M) FATAL("K mismatch ", K, " ", s->emission.mean.M, " ", s->emission.diagcov.M);

        /* count */
        states++;
        means += s->emission.mean.M;
        transits += s->transition.M;
    }

    /* open map */
    const int buckets=5, values=4;
    Matrix map_data(NextPrime(states*4), buckets*values);
    HashMatrix map(&map_data, values);

    /* open data */
    LocalFile names  (string(dir) + MatrixFile::Filename(name, "name",       "string", iteration), "w");
    LocalFile initial(string(dir) + MatrixFile::Filename(name, "prior",      "matrix", iteration), "w");
    LocalFile transit(string(dir) + MatrixFile::Filename(name, "transition", "matrix", iteration), "w");
    LocalFile prior  (string(dir) + MatrixFile::Filename(name, "emPrior",    "matrix", iteration), "w");
    LocalFile mean   (string(dir) + MatrixFile::Filename(name, "emMeans",    "matrix", iteration), "w");
    LocalFile covar  (string(dir) + MatrixFile::Filename(name, "emCov",      "matrix", iteration), "w");

    /* write data headers */
    MatrixFile::WriteHeader(&names,   BaseName(names.Filename()),   flagtext, states,   1);
    MatrixFile::WriteHeader(&initial, BaseName(initial.Filename()), flagtext, states,   1);
    MatrixFile::WriteHeader(&transit, BaseName(transit.Filename()), flagtext, transits, TransitCols);
    MatrixFile::WriteHeader(&prior,   BaseName(prior.Filename()),   flagtext, states,   K);
    MatrixFile::WriteHeader(&mean,    BaseName(mean.Filename()),    flagtext, means,    D);
    MatrixFile::WriteHeader(&covar,   BaseName(covar.Filename()),   flagtext, means,    D);

    /* write data */
    states=means=transits=0;
    for (model->beginState(&iter); !iter.done; model->nextState(&iter)) {
        AcousticModel::State *s = (AcousticModel::State*)iter.v;
        if (minSamples && s->val.samples < minSamples) continue;

        StringFile::WriteRow(&names, s->name);
        MatrixFile::WriteRow(&initial, &s->prior, 1);

        MatrixFile::WriteRow(&prior, s->emission.prior.m, s->emission.prior.M);

        MatrixRowIter(&s->emission.mean)    { MatrixFile::WriteRow(&mean,  s->emission.mean.row(i),    s->emission.mean.N);  }
        MatrixRowIter(&s->emission.diagcov) { MatrixFile::WriteRow(&covar, s->emission.diagcov.row(i), s->emission.diagcov.N); }

        Matrix tx(s->transition);
        State::sortTransitionMap(tx.m, tx.M);
        MatrixRowIter(&tx) {
            tx.row(i)[TC_Self] = s->id();
            MatrixFile::WriteRow(&transit, tx.row(i), tx.N);
        }

        /* build map */
        double *he = map.Set(s->id());
        if (!he) FATAL("Matrix hash collision: ", s->name);
        he[1] = states;
        he[2] = means;
        he[3] = transits;

        /* count */
        states++;
        means += s->emission.mean.M;
        transits += s->transition.M;
    }

    /* write map & tied */
    if (1                   && MatrixFile(map.map,             flagtext).WriteVersioned(VersionedFileName(dir, name, "map"),        iteration) < 0) { ERROR(name, " write map");     ret=-1; }
    if (model->tiedStates() && MatrixFile(model->tiedStates(), flagtext).WriteVersioned(VersionedFileName(dir, name, "tiedstates"), iteration) < 0) { ERROR(name, " write tied");    ret=-1; }
    if (model->phoneTx()    && MatrixFile(model->phoneTx(),    flagtext).WriteVersioned(VersionedFileName(dir, name, "phonetx"),    iteration) < 0) { ERROR(name, " write phonetx"); ret=-1; }

    return ret;
}

int AcousticModel::toCUDA(AcousticModel::Compiled *model) {
#ifdef LFL_CUDA
    if (!lfapp_cuda) return 0;
    int K=model->state[0].emission.mean.M, D=model->state[0].emission.mean.N, ret;
    if (!CAM) CAM = new CudaAcousticModel(model->states, K, D);
    if (K != CAM->K || D != CAM->D || model->states != CAM->states) FATALf("toCUDA mismatch %d != %d || %d != %d || %d != %d", K, CAM->K, D, CAM->D, model->states, CAM->states);
    if ((ret = CudaAcousticModel::load(CAM, model))) ERROR("CudaAcousticModel::load ret=", ret);
    return ret;
#else
    return 0;
#endif
}

AcousticModel::Compiled *AcousticModel::fullyConnected(Compiled *model) {
    Compiled *hmm = new Compiled(model->states);
    Matrix tx(model->states, TransitCols);
    double prob = log(1.0 / model->states);
    for (int i=0; i<model->states; i++) {
        hmm->state[i].assignPtr(&model->state[i]);
        hmm->state[i].prior = prob;
        tx.row(i)[TC_Edge] = i;
        tx.row(i)[TC_Cost] = prob;
    }
    AcousticModel::State::sortTransitionMap(&tx);
    for (int i=0; i<model->states; i++) {
        AcousticModel::State *s = &hmm->state[i];
        s->transition = tx;
        MatrixRowIter(&s->transition) s->transition.row(i)[TC_Self] = s->id();
    }
    return hmm;
}

AcousticModel::Compiled *AcousticModel::fromUtterance(Compiled *model, const char *transcript, bool UseTransit) {
    return FLAGS_TriphoneModel ? fromUtterance3(model, transcript, UseTransit) : fromUtterance1(model, transcript, UseTransit);
}

/* context independent utterance model */
AcousticModel::Compiled *AcousticModel::fromUtterance1(AcousticModel::Compiled *model, const char *transcript, bool UseTransit) {
    /* get pronunciation */
    static const int maxwords=1024;
    PronunciationDict *dict = PronunciationDict::instance();
    const char *w[maxwords], *wa[maxwords]; int words, phones, len=0;
    if ((words = dict->pronounce(transcript, w, wa, &phones, maxwords)) <= 0) return 0;
    Compiled *hmm = new Compiled(phones*StatesPerPhone+words+1);

    /* assign states */
    State::assign(hmm->state, &len, hmm->states, model, 0, Phoneme::SIL, 0, 1);
    for (int i=0; i<words; i++) {
        for (int j=0, pl=strlen(w[i]); j<pl; j++) State::assign(hmm->state, &len, hmm->states, model, 0, w[i][j], 0, StatesPerPhone);
        State::assign(hmm->state, &len, hmm->states, model, 0, Phoneme::SIL, 0, 1);
    }
    hmm->states = len;

    if (FLAGS_lfapp_debug) {
        INFO("lattice pre patch");
        AcousticHMM::printLattice(hmm, model);
    }

    /* patch & localize transitions */
    for (int i=0; i<hmm->states; i++) {
        AcousticModel::State *s = &hmm->state[i];
        bool lastState = (i+1 == hmm->states);
        bool nextStateSil = (!lastState && isSilence(hmm->state[i+1].name));
        bool addtx = nextStateSil && i+2<hmm->states;
        Matrix *trans = new Matrix(2-lastState+addtx, TransitCols, s->id());
        
        double mintx = INFINITY;
        for (int ti=0; ti<s->transition.M; ti++) if (s->transition.row(ti)[TC_Cost] < mintx) mintx = s->transition.row(ti)[TC_Cost];

        double *trself = AcousticModel::State::transit(model, &s->transition, s);
        if (!trself && UseTransit) ERROR("no transition to self for ", s->name);
        trans->row(0)[TC_Edge] = i;
        trans->row(0)[TC_Cost] = trself ? trself[TC_Cost] : mintx;

        if (!lastState) {
            State *sj = &hmm->state[i+1];
            double *trnext = AcousticModel::State::transit(model, &s->transition, sj);
            if (!trnext && UseTransit) ERROR("no transition to next for ", s->name);
            trans->row(1)[TC_Edge] = i+1;
            trans->row(1)[TC_Cost] = trnext ? trnext[TC_Cost] : mintx;
        }

        if (addtx) {
            State *sj = &hmm->state[i+2];
            double *trnext = AcousticModel::State::transit(model, &s->transition, sj);
            if (!trnext && UseTransit) ERROR("no transition to next for ", s->name);
            trans->row(2)[TC_Edge] = i+2;
            trans->row(2)[TC_Cost] = trnext ? trnext[TC_Cost] : mintx;
        }

        s->transition.Absorb(trans);
    }
    return hmm;
}

/* triphone utterance model */
AcousticModel::Compiled *AcousticModel::fromUtterance3(AcousticModel::Compiled *model, const char *transcript, bool UseTransit) {
    /* get pronunciation */
    PronunciationDict *dict = PronunciationDict::instance();
    static const int maxwords=1024;
    const char *w[maxwords], *wa[maxwords];
    int words, phones, len=0, depth=0, prevPhonesInWord=0, prevEndWordPhone=0;
    if ((words = dict->pronounce(transcript, w, wa, &phones, maxwords)) < 0) { DEBUG("pronounce '%s' failed", transcript); return 0; }
    Compiled *hmm = new Compiled(phones*StatesPerPhone+words*4*StatesPerPhone+StatesPerPhone+1);

    /* assign states */
    State::assign(hmm->state, &len, hmm->states, model, 0, Phoneme::SIL, 0, 1, depth++);
    for (int i=0; i<words; i++) {
        bool lastword = i == words-1;
        const char *pronunciation = w[i];
        int phonesInWord = strlen(pronunciation);

        if (1) { /* add cross-word begin triphone states */
            int pn = phonesInWord>1 ? pronunciation[1] : (!lastword ? w[i+1][0] : 0);

            /* paralell phoneme paths - no depth increase */
            if (phonesInWord==1 || prevEndWordPhone)       State::assign(hmm->state, &len, hmm->states, model, prevEndWordPhone, pronunciation[0], pn, StatesPerPhone, depth);
            if (phonesInWord==1 && prevEndWordPhone && pn) State::assign(hmm->state, &len, hmm->states, model, 0,                pronunciation[0], pn, StatesPerPhone, depth);
        }

        int lastphone=0; /* add inner word triphone states and begin/end diphone states */
        for (int j=0; j<phonesInWord; j++) {
            int nextphone = j+1 < phonesInWord ? pronunciation[j+1] : 0;
            int phone = pronunciation[j];

            State::assign(hmm->state, &len, hmm->states, model, lastphone, phone, nextphone, StatesPerPhone, depth++);
            lastphone = phone;
        }
        depth--;

        if (!lastword) { /* add cross-word end triphone states */
            int p = pronunciation[phonesInWord-1]; 
            int pp = phonesInWord>1 ? pronunciation[phonesInWord-2] : prevEndWordPhone;
            int pn = w[i+1][0];

            /* paralell phoneme paths - no depth increase */
            if (phonesInWord==1 && pn && pp) State::assign(hmm->state, &len, hmm->states, model, pp, p, 0,  StatesPerPhone, depth);
            if (phonesInWord>1)              State::assign(hmm->state, &len, hmm->states, model, pp, p, pn, StatesPerPhone, depth);
        }
        depth++;

        /* add silence after word */
        State::assign(hmm->state, &len, hmm->states, model, 0, Phoneme::SIL, 0, 1, depth++);
        prevPhonesInWord = phonesInWord;
        prevEndWordPhone = lastphone;
    }
    hmm->states = len;
    
    if (FLAGS_lfapp_debug) {
        INFO("lattice pre patch");
        AcousticHMM::printLattice(hmm, model);
    }

    /* patch transitions */
    for (int i=0; i<hmm->states-1; i++) {
        State *s = &hmm->state[i];
        vector<pair<unsigned, float> > tx;
        int as, ap, an, a = parseName(s->name, &as, &ap, &an);

        double mincost = INFINITY;
        for (int k=0; k<s->transition.M; k++) {
            double v = s->transition.row(k)[TC_Cost];
            if (v < mincost) mincost = v;
        }
        if (!isfinite(mincost)) mincost = -16;

        if (a == Phoneme::SIL || as == StatesPerPhone-1) {
            /* determine matching next triphone states */
            for (int j=i+1; j<hmm->states; j++) {
                State *sj = &hmm->state[j];

                int bs, bp, bn, b = parseName(sj->name, &bs, &bp, &bn);
                if (bs != 0) continue;

                if (State::triphoneConnect(ap, a, an, bp, b, bn)) {
                    unsigned to; double cost=mincost, *tr = AcousticModel::State::transit(model, &s->transition, sj, &to);
                    if (tr) cost = tr[TC_Cost];
                    else if (UseTransit) ERROR("no transition from ", s->name, " ", s->id(), " to ", sj->name, ", patched w min ", cost);
                    tx.push_back(pair<unsigned, float>(to, cost));
                }
                else if (tx.size()) break;
            }
            if (!tx.size()) FATAL("cant find transition for ", i, " ", s->name);
        }
        else {
            State *sj = &hmm->state[i+1];
            unsigned to; double cost=mincost, *tr = AcousticModel::State::transit(model, &s->transition, sj, &to);
            if (tr) cost = tr[TC_Cost];
            else if (UseTransit) ERROR("no transition from ", s->name, " ", s->id(), " to ", sj->name, ", patched w min ", cost);
            tx.push_back(pair<unsigned, float>(to, cost));
        }

        /* add localized transition to self for states patched below */
        double strcost=mincost, *str = AcousticModel::State::transit(model, &s->transition, s);
        if (!str) ERROR("no transition to self ", s->name, " ", s->id(), ", patched w min ", strcost);
        else strcost = str[TC_Cost];

        double prob=log(1.0/tx.size()+1);
        s->transition.Open(tx.size()+1, TransitCols, s->id());
        s->transition.row(tx.size())[TC_Edge] = i;
        s->transition.row(tx.size())[TC_Cost] = strcost;
        s->transition.M--;

        /* assemble transit map */
        for (int j=0; j<tx.size(); j++) {
            s->transition.row(j)[TC_Edge] = tx[j].first;
            s->transition.row(j)[TC_Cost] = tx[j].second;
        }
    }

    /* patch final silence to transit only to self */
    State *final = &hmm->state[len-1];
    double fstrcost=-16, *fstr = AcousticModel::State::transit(model, &final->transition, final);
    if (!fstr) ERROR("no transition from final to self ", final->name, " ", final->id(), ", patched w min ", fstrcost);
    else fstrcost = fstr[TC_Cost];

    final->transition.Open(1, TransitCols, final->id());
    final->transition.row(0)[TC_Edge] = len-1;
    final->transition.row(0)[TC_Cost] = fstrcost;

    if (FLAGS_lfapp_debug) {
        INFO("lattice pre localization");
        AcousticHMM::printLattice(hmm, model);
    }

    /* localize transitions */
    for (int i=0; i<hmm->states-1; i++) {
        State *s = &hmm->state[i];
        int as, ap, an, a = parseName(s->name, &as, &ap, &an);
        bool patch = (a == Phoneme::SIL || as == StatesPerPhone-1);

        /* resolve to forward states using val.samples minSamples filter on paralell phoneme 'depth' */
        if (State::localizeTransitionMap(&s->transition, &s->transition, hmm, i, s->val.samples + patch, model, patch ? s->name.c_str() : 0)) { ERROR("localize transition ", i, " ", s->name.c_str(), " '", transcript, "'"); return 0; }

        /* transition to self for patched state */ 
        s->transition.M++;
    }

    if (FLAGS_lfapp_debug) {
        INFO("lattice post localization");
        AcousticHMM::printLattice(hmm, model);
    }

    return hmm;
}

AcousticModel::Compiled *AcousticModel::fromModel1(StateCollection *model, bool rewriteTransitions) {
    int states = model->getStateCount(), len = 0;
    if (states <= 0) return 0;

    Compiled *hmm = new Compiled(states);
    AcousticModel::StateCollection::Iterator iter;
    State *sil=0;
    for (model->beginState(&iter); !iter.done; model->nextState(&iter)) {
        State *s = iter.v;
        if (isSilence(s->name)) { sil=s; continue; }
        hmm->state[len++].assignPtr(s);
    }
    hmm->state[len++].assignPtr(sil);

    for (int i=0; i<hmm->states; i++) {
        State *s = &hmm->state[i];
        s->transition.Absorb(s->transition.Clone());
        if (State::localizeTransitionMap(&s->transition, &s->transition, hmm, 0)) { ERROR("localize transition ", i, " ", s->name); return 0; }
    }

    if (rewriteTransitions) {
        double tp = log((double)1/LFL_PHONES);
        Matrix *trans = new Matrix(LFL_PHONES, TransitCols);
        MatrixRowIter(trans) {
            trans->row(i)[TC_Edge] = i*StatesPerPhone;
            trans->row(i)[TC_Cost] = tp;
        }
        for (int i=0; i<hmm->states; i++) {
            State *s = &hmm->state[i];
            string n = s->name;
            if (!isSilence(s->name) && n.substr(n.size()-8) != "State_02") continue;

            s->transition.Absorb(trans->Clone()); 
            MatrixRowIter(&s->transition) s->transition.row(i)[TC_Self] = s->id();
        }
        delete trans;
    }

    return hmm;
}

int AcousticModelFile::Open(const char *name, const char *dir, int lastiter, bool rebuild_transit) {
    Reset();
    string flags;
    lastiter = MatrixFile::ReadVersioned(dir, name, "transition", &transit, &flags, lastiter);
    if (!transit) { ERROR("no acoustic model: ", name); return -1; }

    if (flags.size()) AcousticModel::loadflags(flags.c_str());

    if (MatrixFile::ReadVersioned(dir, name, "prior",      &initial, 0, lastiter)<0) { ERROR(name, ".", lastiter, ".prior"  ); return -1; }
    if (MatrixFile::ReadVersioned(dir, name, "emMeans",    &mean,    0, lastiter)<0) { ERROR(name, ".", lastiter, ".emMean" ); return -1; }
    if (MatrixFile::ReadVersioned(dir, name, "emCov",      &covar,   0, lastiter)<0) { ERROR(name, ".", lastiter, ".emCov"  ); return -1; }
    if (MatrixFile::ReadVersioned(dir, name, "emPrior",    &prior,   0, lastiter)<0) { ERROR(name, ".", lastiter, ".emPrior"); return -1; }
    if (MatrixFile::ReadVersioned(dir, name, "map",        &map.map, 0, lastiter)<0) { ERROR(name, ".", lastiter, ".map"    ); return -1; }
    if (StringFile::ReadVersioned(dir, name, "name",       &names,   0, lastiter)<0) { ERROR(name, ".", lastiter, ".name"   ); return -1; }
    if (MatrixFile::ReadVersioned(dir, name, "tiedstates", &tied,    0, lastiter)<0) { ERROR(name, ".", lastiter, ".tied"   );       /**/ }

    if (prior->M != names->size()) { ERROR("mismatch ", prior->M, " != ", names->size()); return -1; }
    int M=prior->M, K=prior->N, N=mean->N, transind=0;

    LFL_STL_NAMESPACE::map<unsigned, pair<int, int> > txmap;
    if (rebuild_transit) {
        unsigned last = -1; int count = 0, beg;
        MatrixRowIter(transit) {
            unsigned self = transit->row(i)[TC_Self];
            if (self != last) {
                if (count) txmap[last] = pair<int, int>(beg, count);
                last = self;
                count = 1;
                beg = i;
            }
            else count++;
        }
        if (count) txmap[last] = pair<int, int>(beg, count);
    }

    AcousticModel::Compiled::open(M);
    for (int i=0; i<states; i++) {
        state[i].name = (*names)[i];
        state[i].prior = initial->row(i)[0];
        state[i].emission.AssignDataPtr(K, N, mean->row(i*K), covar->row(i*K), prior->row(i));
        state[i].val.emission_index = i;

        unsigned hash = state[i].id();
        if (rebuild_transit) {
            LFL_STL_NAMESPACE::map<unsigned, pair<int, int> >::iterator it = txmap.find(hash);
            if (it == txmap.end()) { ERROR("find state ", state[i].name, " in transition failed ", hash); return -1; }
            state[i].transition.AssignDataPtr((*it).second.second, transit->N, transit->row((*it).second.first));
        }
        else {
            double *he = getHashEntry(hash);
            if (!he) { ERROR("find state ", state[i].name, " in map failed"); return -1; }

            unsigned transbegin=he[3], transits=0;
            while (transbegin+transits < transit->M && transit->row(transbegin+transits)[TC_Self] == hash) transits++;
            if (!transits) { ERROR(transits, " transits for ", state[i].name); return -1; }

            state[i].transition.AssignDataPtr(transits, transit->N, transit->row(transbegin));
        }

        state[i].txself = -INFINITY;
        for (int j=0, l=state[i].transition.M; j<l; j++) {
            double *txrow = state[i].transition.row(j);
            if (txrow[TC_Self] == txrow[TC_Edge]) { state[i].txself = txrow[TC_Cost]; break; }
        }
        if (!isfinite(state[i].txself)) ERROR("state ", state[i].name, " no self transition prob ", state[i].txself);

        if (0) {
            INFO((*names)[i], " ", fnv32((*names)[i].c_str()));
            Matrix::Print(&state[i].transition, (*names)[i]);
            Matrix::Print(&state[i].emission.prior, (*names)[i]);
            Matrix::Print(&state[i].emission.mean, (*names)[i]);
            Matrix::Print(&state[i].emission.diagcov, (*names)[i]);
        }
    }
    return lastiter;
}

/* AcousticHMM */

double AcousticHMM::forwardBackward(AcousticModel::Compiled *model, Matrix *observations, int InitMax, double beamWidth, HMM::BaumWelchAccum *BWaccum, int flag) {
    int NBest=1, M=observations->M, N=model->states;
    Matrix alpha(M, N, -INFINITY), beta(M, N, -INFINITY), gamma(M, N, -INFINITY), xi(N, N, -INFINITY);

    HMM::ActiveStateIndex active(N, NBest, beamWidth, InitMax);
    TransitMap transit(model, flag & Flag::UseTransit);
    EmissionMatrix emit(model, observations, flag & Flag::UsePrior);
    HMM::BaumWelchWrapper BWwrapper;

    double bp, fp=HMM::forwardBackward(&active, &transit, &emit, &alpha, &beta, &gamma, &xi, &bp, &BWwrapper);
    if (isnan(fp) || isnan(bp) || isinf(fp) || isinf(bp)) { INFO("NaN or INF forward ", fp, " or backward ", bp); return 0; }

    double diff = fabs(fp - bp);
    if (diff > 1) INFO("forward and backwards probabilities differ by ", diff, " (", fp, " and ", bp, ")");
    else BWwrapper.add(BWaccum);
    return fp;
}

double AcousticHMM::viterbi(AcousticModel::Compiled *model, Matrix *observations, Matrix *path, int InitMax, double beamWidth, int flag) {
    int NBest=1, M=observations->M, N=model->states;
    TransitMap transit(model, flag & Flag::UseTransit);
    EmissionArray emit(model, observations, flag & Flag::UsePrior);

#if 0
    matrix<HMM::Token> backtrace(M, beamWidth, HMM::Token()), viterbi(M, 1, HMM::Token());
    HMM::TokenBacktraceMatrix<HMM::Token> tbm(&backtrace);
    HMM::TokenPasser<HMM::Token> beam(N, NBest, beamWidth, &tbm);

    int endstate = HMM::forward(&beam, &transit, &emit, &beam, &beam) / NBest;
    HMM::Token::tracePath(&viterbi, &backtrace, endstate, M);
    MatrixIter(&viterbi) path->row(i)[j] = viterbi.row(i)[j].ind;
    return backtrace.row(M-1)[endstate].val;
#else
    Matrix lambda(M, N, -INFINITY), backtrace(M, N, -1);
    HMM::ActiveStateIndex active(N, NBest, beamWidth, InitMax);
    HMM::BeamMatrix beam(&lambda, &backtrace, NBest);

    int endstate = HMM::forward(&active, &transit, &emit, &beam, Singleton<HMM::Algorithm::Viterbi>::Get()) / NBest;
    HMM::tracePath(path, &backtrace, endstate, M);
    return lambda.row(M-1)[endstate];
#endif
}

double AcousticHMM::uniformViterbi(AcousticModel::Compiled *model, Matrix *observations, Matrix *viterbi, int Unused1, double Unused2, int uu3) {
    if (!DimCheck("HMM::viterbi", viterbi->M, observations->M)) return -INFINITY;

    int obvs = observations->M, states=0;
    for (int i=0; i<model->states; i++) {
        if (AcousticModel::isSilence(model->state[i].name)) continue;
        states++;
    }

    double spmF = (double)obvs/states;
    int spm = (int)spmF, len=0;
    spmF -= spm;

    for (int i=0; i<model->states; i++) {
        if (AcousticModel::isSilence(model->state[i].name)) continue;
        for (int j=0; j<spm && len<obvs; j++) viterbi->row(len++)[0] = i;
        if (len<obvs && Rand(0.0, 1.0) < spmF) viterbi->row(len++)[0] = i;
    }
    while (len<obvs) viterbi->row(len++)[0] = model->states-1;
    return 0;
}

void AcousticHMM::printLattice(AcousticModel::Compiled *hmm, AcousticModel::Compiled *model) {
    for (int i=0; i<hmm->states; i++) {
        AcousticModel::State *s = &hmm->state[i];
        string v = StringPrintf("%03d ", i);
        StrAppend(&v, s->name, " (", s->id(), " ", model ? model->state[s->val.emission_index].id() : 0, ")");
        StrAppend(&v, " ei=", s->val.emission_index, " sc=", s->val.samples, " tx = ");
        for (int j=0; j<s->transition.M; j++) {
            StringAppendf(&v, "%u = %.02f, ", (unsigned)s->transition.row(j)[TC_Edge], s->transition.row(j)[TC_Cost]);
        }
        INFO(v);
    }
}

void AcousticHMM::EmissionArray::calc(AcousticModel::Compiled *model, HMM::ActiveState *active, int time_index, const double *observation, double *emission, double *posterior, double *cudaposterior) {
    active->time_index = time_index;
    int active_states = active->size();
#ifdef LFL_CUDA
    if (lfapp_cuda) {
        if (posterior && !cudaposterior) FATAL("posterior ", posterior, " requires cudaposterior ", cudaposterior);
        double *cudaemission = (double*)alloca(active_states*sizeof(double));
        int *beam = (int*)alloca(active_states*sizeof(int));
        int K = model->state[0].emission.mean.M;

        HMM::ActiveState::Iterator stateI; int ind=0;
        for (active->begin(time_index, &stateI); !stateI.done && ind<active_states; active->next(&stateI)) {
            AcousticModel::State *si = &model->state[stateI.index];
            beam[ind++] = si->val.emission_index;
        }

        if (CudaAcousticModel::calcEmissions(CAM, observation, beam, active->size(), cudaemission, cudaposterior))
            ERROR("cuda calc emissions failed: ", CAM);

        ind = 0;
        for (active->begin(time_index, &stateI); !stateI.done && ind<active_states; active->next(&stateI)) {
            AcousticModel::State *si = &model->state[stateI.index];
            if (posterior) memcpy(posterior+stateI.index*K, cudaposterior+ind*K, K*sizeof(double));
            emission[stateI.index] = cudaemission[ind++];
        }
    }
    else
#endif
    {
        HMM::ActiveState::Iterator stateI; int ind=0;
        for (active->begin(time_index, &stateI); !stateI.done && ind<active_states; active->next(&stateI), ind++) {
            AcousticModel::State *si = &model->state[stateI.index];
            emission[stateI.index] = si->emission.PDF(observation, posterior ? posterior + stateI.index * si->emission.mean.M : 0);
        }
    }
}

/* decoder */

Matrix *Decoder::decodeFile(AcousticModel::Compiled *model, const char *fn, double beamWidth) {
    SoundAsset input("input", fn, 0, 0, 0, 0);
    input.Load();
    if (!input.wav) { ERROR(fn, " not found"); return 0; }

    Matrix *features = Features::fromAsset(&input, Features::Flag::Full);
    Matrix *ret = decodeFeatures(model, features, beamWidth);
    delete features;
    return ret;
}

Matrix *Decoder::decodeFeatures(AcousticModel::Compiled *model, Matrix *features, double beamWidth, int flag) {
    if (!DimCheck("decodeFeatures", features->N, model->state[0].emission.mean.N)) return 0;
    if (FLAGS_lfapp_debug) AcousticHMM::printLattice(model);

    Matrix *viterbi = new Matrix(features->M, 1); Timer vtime;
    double vprob = AcousticHMM::viterbi(model, features, viterbi, 0, beamWidth, flag);
    if (flag & AcousticHMM::Flag::Visualize) Decoder::visualizeFeatures(model, features, viterbi, vprob, vtime.GetTime(), flag & AcousticHMM::Flag::Interactive);

    return viterbi;
}

string Decoder::transcript(const AcousticModel::Compiled *model, const Matrix *viterbi, Allocator *alloc) {
    string ret; int laststate=-1;
    for (Decoder::PhoneIter iter(model, viterbi); !iter.done(); iter.next()) {
        if (!iter.phone) continue;
        StrAppend(&ret, ret.size()?" ":"", Phoneme::name(iter.phone));
    }
    if (ret.empty()) ret = "<none>";
    return ret;
}

void Decoder::visualizeFeatures(AcousticModel::Compiled *model, Matrix *MFCC, Matrix *viterbi, double vprob, Time vtime, bool interactive) {
    static PhoneticSegmentationGUI *segments = 0;
    static bool interactive_done;

    Replace(&segments, new PhoneticSegmentationGUI(screen, model, viterbi, "visbuf"));
    interactive_done = 0;

    SoundAsset sa;
    sa.name = "visbuf";
    sa.Load();
    sa.channels = 1;
    sa.sample_rate = FLAGS_sample_rate;
    sa.wav = Features::reverse(MFCC, FLAGS_sample_rate);
    RingBuf::Handle B = RingBuf::Handle(sa.wav);

    Matrix *spect = Spectogram(&B, 0, FLAGS_feat_window, FLAGS_feat_hop, FLAGS_feat_window, 0, PowerDomain::dB);
    Asset *snap = app->shell.asset("snap");
    glSpectogram(spect, snap, 0);
    delete spect;

    if (FLAGS_lfapp_audio) app->audio.QueueMixBuf(&B);
    INFO("vprob=", vprob, " vtime=", vtime.count());
    Font *font = Fonts::Default();

    Box wcc = Box(5,345, 400,100);
    while (Running() && (app->audio.Out.size() || (interactive && !interactive_done))) {
        app->PreFrame(app->frame_time.GetTime(true).count());

        screen->gd->DrawMode(DrawMode::_2D);
        app->shell.asset("snap")->tex.Draw(wcc); // 4);

        int levels=10;
        float percent = 1-(float)app->audio.Out.size()/app->audio.outlast;
        font->Draw(StringPrintf("time=%d vprob=%f percent=%f next=%d", vtime.count(), vprob, percent, interactive_done), point(10, 440));

        percent -= feat_progressbar_c*FLAGS_sample_rate*FLAGS_chans_out/app->audio.outlast;
        if (percent >= 0 && percent <= 1) {
            for (int i=1; i<=levels; i++) font->Draw("|", point(wcc.x+percent*wcc.w, wcc.y-30*i));
        }

        int count=0;
        for (Decoder::PhoneIter iter(model, viterbi); !iter.done(); iter.next()) {
            if (!iter.phone) continue;
            int r = count++%levels+1;
            font->Draw(Phoneme::name(iter.phone), point(wcc.x+(float)iter.beg*wcc.w/viterbi->M, wcc.y-r*30));
        }

        if (interactive) {
            static Font *norm = Fonts::Get(FLAGS_default_font, "", 12, Color::grey70);
            segments->Frame(wcc, norm);

            // gui.mouse.Activate();
            // static Widget::Button next(&gui, 0, norm, "next", Callback([&](){interactive_done=1;}));
            // Box::FromScreen(2/3.0, .95, 1/3.0, .05, .0001, .0001);
            // next.Draw();

            screen->DrawDialogs();
        }

        app->PostFrame();
        MSleep(1);
    }

    sa.Unload();
}

int resynthesize(Audio *s, const SoundAsset *sa) {
    if (!sa->wav) return -1;
    RingBuf::Handle B(sa->wav);
    Matrix *m = Features::fromBuf(&B);
    Matrix *f0 = F0Stream(&B, 0, FLAGS_feat_window, FLAGS_feat_hop);

    int ret = -1;
    RingBuf *resynth = Features::reverse(m, B.Rate(), f0);
    if (resynth) {
        B = RingBuf::Handle(resynth);
        s->QueueMixBuf(&B);	
    }

    delete m;
    delete f0;
    delete resynth;
    return 0;
}

}; // namespace LFL
