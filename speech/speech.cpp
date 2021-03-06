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

#include "core/app/app.h"
#include "core/app/shell.h"
#include "core/ml/hmm.h"
#include "core/web/dom.h"
#include "core/web/css.h"
#include "core/app/flow.h"
#include "core/app/gl/view.h"
#include "speech.h"

#ifdef LFL_CUDA
#include "core/app/cuda/speech.h"
unique_ptr<CudaAcousticModel> CAM;
#endif

namespace LFL {
DEFINE_string(feat_type, "MFCC", "Feature type");
DEFINE_bool(feat_preemphasis, true, "Use pre-emphasis filter");
DEFINE_bool(feat_dither, false, "Dither input");
DEFINE_double(feat_minfreq, 0, "Minimum feature frequency");
DEFINE_double(feat_maxfreq, FLAGS_sample_rate/2, "Maximum feature frequency");
DEFINE_double(feat_preemphasis_filter, -0.97, "Preemphasis filter = { 1, X }");
DEFINE_int(feat_window, 512, "Feature window length");
DEFINE_int(feat_hop, 256, "Feature window advance");
DEFINE_int(feat_melbands, 40, "Mel bands used to create MFCC features");
DEFINE_int(feat_cepcoefs, 20, "Number of cepstrum coefficients");
DEFINE_bool(triphone_model, false, "Using triphone model");
DEFINE_bool(speech_recognition_debug, false, "Debug speech recognition");

/* phonemes */
const char *Phoneme::Name(char in) {
# undef  XX
# define XX(phone) else if (phone == in) return #phone;
  if (0) {}
# include "phones.h"
  return 0;
}

char Phoneme::Id(const char *in, int len) {
  if (!len) len=strlen(in);
# undef  XX
# define XX(phone) else if(strlen(#phone)==len && !strncmp(#phone,in,len)) return phone;
  if (0) {}
# include "phones.h"
  return -1;
}

PronunciationDict *PronunciationDict::Instance(AssetLoading *loader) {
  static unique_ptr<PronunciationDict> inst;
  if (inst) return inst.get();
  inst = make_unique<PronunciationDict>();

  string fn = loader->FileName("cmudict.txt");
  LocalFileLineIter file(fn);
  if (!file.f.Opened()) return ERRORv(inst.get(), "no ", fn);

  ReadDictionary(&file, inst.get());
  return inst.get();
}

int PronunciationDict::ReadDictionary(StringIter *in, PronunciationDict *out) {
  for (const char *line = in->Next(); line; line = in->Next()) {
    const char *c=line; if ((*c) == ';') continue; /* skip comments */

    /* Format: word <two spaces> pronunciation */
    c += LengthChar(c, notspace);
    if (!*c || !isspace(*(c+1))) continue;
    *const_cast<char*>(c) = 0;

    const char *word=line, *pronunciation=c+2; /* (k,v) */

    char phone[1024], accent[1024], phones; /* pronunciation phone-ID sequence */
    if ((phones = ReadPronunciation(pronunciation, -1, phone, accent, sizeof(phone)-1)) <= 0) continue;
    phone[phones]=0;

    /* insert(word, valbufOffset) */
    int offset = out->val.Alloc(phones*2+1);
    out->word_pronunciation[word] = offset;

    memcpy(&out->val.buf[offset], phone, phones+1);
    memcpy(&out->val.buf[offset+phones+1], accent, phones);
  }
  return 0;
}

int PronunciationDict::ReadPronunciation(const char *in, int len, char *phonesOut, char *accentOut, int outlen) {
  StringWordIter phones(StringPiece(in, len)); int outi=0;
  for (const char *phone_text=phones.Next(); phone_text; phone_text=phones.Next()) {
    if (outi >= outlen) return -1;
    string phone(phone_text, phones.cur_len);

    int accent = LengthChar(phone.c_str(), isalpha);
    char stress = phone[accent];
    const_cast<char*>(phone_text)[accent]=0;
    accent = isdigit(stress) ? stress - '0' : 0;

    if (!(phonesOut[outi] = Phoneme::Id(phone.c_str(), 0))) return -1;
    accentOut[outi] = accent;
    outi++;
  }
  return outi;
}

const char *PronunciationDict::Pronounce(const char *in) {
  Map::iterator i = word_pronunciation.find(toupper(in));
  if (i == word_pronunciation.end()) return 0;
  int pronunciationIndex = (*i).second;
  return &val.buf[pronunciationIndex];
}

int PronunciationDict::Pronounce(const char *utterance, const char **w, const char **wa, int *phones, int max) {
  *phones=0;
  int words=0;
  StringWordIter script(utterance);
  for (string word = script.NextString(); !script.Done(); word = script.NextString()) {
    const char *pronunciation = Pronounce(word.c_str());
    if (!pronunciation || words+1 >= max) { DEBUG("pronunciation %s count=%d", word.c_str(), words); return -1; }

    int len = strlen(pronunciation);
    (*phones) += len;

    w[words] = pronunciation;
    wa[words] = pronunciation + len + 1; /* accent */
    words++;
  }
  return words;
}

/* Features */

int Features::filter_zeroth=0;
int Features::deltas=1;
int Features::deltadeltas=1;
int Features::mean_normalization=0;
int Features::variance_normalization=0;
int Features::lpccoefs  = 12;
int Features::barkbands = 21;
double Features::rastaB[5] = { .2, .1, 0, -.1, -.2 };
double Features::rastaA[2] = { 1, -.94 };

unique_ptr<Matrix> Features::EqualLoudnessCurve(int outrows, double max) {
  double maxbark = HZToBark(max), stepbark = maxbark/(outrows-1);   
  auto ret = make_unique<Matrix>(outrows, 1);
  MatrixIter(ret) {
    double bandcfhz = BarkToHZ(i*stepbark), fsq = pow(bandcfhz, 2);
    ret->row(i)[j] = pow(fsq/(fsq + 1.6e5), 2) * ((fsq + 1.44e6) / (fsq + 9.61e6));
  }
  return ret;
}

vector<double> Features::LifterMatrixROSA(int n, double L, bool inverse) {
  vector<double> dm(n);
  for (int i = 0; i < n; i++) {
    dm[i] = !i ? 1 : pow(i, L);
    if (inverse) dm[i] = 1 / dm[i];
  }
  return dm;
}

vector<double> Features::LifterMatrixHTK(int n, double L, bool inverse) {
  vector<double> dm(n);
  for (int i = 0; i < n; i++) {
    dm[i] = !i ? 1 : (1+L/2*sin(i*M_PI/L));
    if (inverse) dm[i] = 1 / dm[i];
  }
  return dm;
}

unique_ptr<Matrix> Features::FFT2Bark(int outrows, double minfreq, double maxfreq, int fftlen, int samplerate) {
  auto m = make_unique<Matrix>(outrows, fftlen/2);
  double minbark = HZToBark(minfreq), maxbark = HZToBark(maxfreq);
  double nyqbark = maxbark - minbark, stepbark = nyqbark/(outrows-1);

  vector<double> binbark(fftlen/2);
  for (int i=0; i<fftlen/2; i++)
    binbark[i] = HZToBark(double(i)/fftlen*samplerate);

  for (int i=0; i<outrows; i++) {
    double *row = m->row(i), midbark = minbark + i*stepbark;

    for (int j=0; j<fftlen/2; j++) {
      double lofreq = (binbark[j] - midbark) - 0.5;
      double hifreq = (binbark[j] - midbark) + 0.5;
      row[j] = pow(10, min(0.0, min(hifreq, -2.5*lofreq)));
    }
  }
  return m;
}

unique_ptr<Matrix> Features::FFT2Mel(int outrows, double minfreq, double maxfreq, int fftlen, int samplerate) {
  unique_ptr<Matrix> m = make_unique<Matrix>(outrows, fftlen/2);
  double minmel = HZToMel(minfreq), maxmel = HZToMel(maxfreq);

  vector<double> mel(outrows+2);
  for (int i=0; i<outrows+2; i++)
    mel[i] = MelToHZ(minmel + (double(i)/(outrows+1)) * (maxmel-minmel)); 

  for (int i=0; i<outrows; i++) {
    double *row = m->row(i);

    for (int j=0; j<fftlen/2; j++) {
      double binfreq = double(j)/fftlen*samplerate;
      double loslope = ( binfreq - mel[i+0]) / (mel[i+1] - mel[i+0]);
      double hislope = (-binfreq + mel[i+2]) / (mel[i+2] - mel[i+1]);

      row[j] = max(0.0, min(loslope, hislope));
    }
  }
  return m;
}

unique_ptr<Matrix> Features::Mel2FFT(int outrows, double minfreq, double maxfreq, int fftlen, int samplerate) {
  auto m = FFT2Mel(outrows, minfreq, maxfreq, fftlen, samplerate);
  auto w = Matrix::Mult(m.get(), move(m), mTrnpA);

  float diagmean=0; int diagcount=0;
  for (int i=0; i<w->M && i<w->N; i++) { diagmean += w->row(i)[i]; diagcount++; }
  diagmean /= (diagcount+1);

  m = Matrix::Transpose(move(m));
  for (int i=0; i<m->M; i++) {
    double sum=0;
    for (int j=0; j<m->N; j++) sum += w->row(i)[j];

    if (sum < diagmean) sum = diagmean;
    for (int j=0; j<m->N; j++) m->row(i)[j] /= sum;
  }
  return m;
};

unique_ptr<Matrix> Features::PLP(const RingSampler::Handle *in, Matrix *out,
                                 vector<StatefulFilter> *rastaFilters, Allocator *alloc) {
  /* http://seed.ucsd.edu/mediawiki/images/5/5c/PLP.pdf */
  if (!alloc) alloc = Singleton<MallocAllocator>::Set();

  /* plp = melfcc(x, sr, 'lifterexp', -22, 'nbands', 21, 'maxfreq', sr/2, 'fbtype', 'bark', 'modelorder', 12, 'usecmp',1, 'wintime', 512/sr, 'hoptime', 256/sr, 'preemph', 0, 'dcttype', 1) */
  static unique_ptr<const Matrix> barktrans = Matrix::Transpose(FFT2Bark(barkbands, FLAGS_feat_minfreq, FLAGS_feat_maxfreq, FLAGS_feat_window, FLAGS_sample_rate));

  /* fft */
  int frames = (in->Len() - (FLAGS_feat_window - FLAGS_feat_hop)) / FLAGS_feat_hop;
  Matrix fftbuf(frames, FLAGS_feat_window/2, 0.0, 0, alloc);
  if (!Spectogram(in, &fftbuf, FLAGS_feat_window, FLAGS_feat_hop, FLAGS_feat_window,
                  vector<double>(), PowerDomain::abs2, 32768)) return 0;
  if (FLAGS_feat_dither) MatrixIter(&fftbuf) fftbuf.row(i)[j] += FLAGS_feat_window;

  /* to bark scale critcal bands */
  Matrix barkbuf(fftbuf.M, barktrans->N, 0.0, 0, alloc);
  Matrix::Mult(&fftbuf, barktrans.get(), &barkbuf);

  if (1) { /* rasta */
    MatrixIter(&barkbuf) barkbuf.row(i)[j] = log(barkbuf.row(i)[j] ? barkbuf.row(i)[j] : 1e7);
    Matrix rastabuf(barkbuf.M, barkbuf.N, 0.0, 0, alloc);

    if (rastaFilters) {
      if (!rastaFilters->size()) MatrixColIter(&rastabuf) rastaFilters->push_back(StatefulFilter());
      if (rastaFilters->size() != rastabuf.N) FATAL("mismatch ", rastaFilters, " != ", rastabuf.N);
    } 

    MatrixColIter(&rastabuf) {
      ColMatPtrRingSampler bandTrajectory(&barkbuf, j), rastaOut(&rastabuf, j);
      RingSampler::HandleT<double> bt(&bandTrajectory), ro(&rastaOut);
      StatefulFilter RFbuf, *RF = rastaFilters ? &(*rastaFilters)[j] : &RFbuf;
      int processed = 0;

      if (!RF->samples) RF->Open(sizeof(rastaB)/sizeof(double), rastaB, 0, rastaA);
      if (RF->samples < 4) {
        processed = RF->Filter(&bt, 0, 0, min(4-RF->samples, bt.Len()));
        for (int i=0; i<processed; i++) ro.Write(0.0);
      }
      if (RF->samples == 4) RF->filterLenA = sizeof(rastaA)/sizeof(double);
      RF->Filter(&bt, &ro, processed);
    }
    MatrixIter(&barkbuf) barkbuf.row(i)[j] = exp(rastabuf.row(i)[j]);
  }

  static unique_ptr<const Matrix> equalLoudnessCurve = EqualLoudnessCurve(barkbuf.N, FLAGS_sample_rate/2);
  MatrixIter(&barkbuf) {
    /* preemphasize = weight critical bands by equal loudness curve */
    barkbuf.row(i)[j] *= equalLoudnessCurve->row(j)[0];

    /* cube root compress */
    barkbuf.row(i)[j] = pow(barkbuf.row(i)[j], 0.33);
  }
  MatrixRowIter(&barkbuf) {
    barkbuf.row(i)[0]           = barkbuf.row(i)[1];
    barkbuf.row(i)[barkbuf.N-1] = barkbuf.row(i)[barkbuf.N-2];
  }

  /* auto-correlation */
  static unique_ptr<const Matrix> IDFTtrans = Matrix::Transpose(IDFT((barkbands-1)*2, barkbands));
  Matrix xcorr(barkbuf.M, IDFTtrans->N, 0.0, 0, alloc);
  Matrix::Mult(&barkbuf, IDFTtrans.get(), &xcorr);
  MatrixIter(&xcorr) xcorr.row(i)[j] /= IDFTtrans->N;

  /* levinson durbin recursion to LPC */
  int order = lpccoefs, ceps = order+1;
  Matrix LPC(xcorr.M, ceps, 0.0, 0, alloc);
  vector<double> reflect(order), lpc(order);
  MatrixRowIter(&xcorr) {
    double err = LevinsonDurbin(order, xcorr.row(i), &reflect[0], &lpc[0]);
    MatrixColIter(&LPC) LPC.row(i)[j] = (!j ? 1.0 : lpc[j-1]) / err;
  }

  /* prepare output */
  unique_ptr<Matrix> ret;
  if (out) { if (out->M != LPC.M || out->N != LPC.N) return nullptr; }
  else out = (ret = make_unique<Matrix>(LPC.M, LPC.N)).get();

  /* first cepstral coefficient is log(error) from levinson durbin recursion */
  MatrixRowIter(out) out->row(i)[0] = -log(LPC.row(i)[0]);

  /* normalize LPC */
  MatrixRowIter(&LPC) {
    double denom = LPC.row(i)[0];
    MatrixColIter(&LPC) LPC.row(i)[j] /= denom;
  }

  /* LPC to cepstral coefficients */
  MatrixRowIter(out) for (int j=1; j<out->N; j++) {
    double sum = 0;
    for (int j2=1; j2<j; j2++) sum += (j-j2) * LPC.row(i)[j2] * out->row(i)[j-j2];
    out->row(i)[j] = -LPC.row(i)[j] - sum / j;
  }

  /* lifter */
  static vector<double> lifter = LifterMatrixROSA(ceps, 0.6); // lifterMatrixHTK(ceps, 22);
  out->MultdiagR(lifter.data(), ceps);

  return ret;
}

unique_ptr<RingSampler> Features::InvPLP(const Matrix *in, int samplerate, Allocator *alloc) {
  /* [dr,aspec,spec] = invmelfcc() */
  if (!alloc) alloc = Singleton<MallocAllocator>::Set();
  Matrix plpcc(in->M, in->N, 0.0, 0, alloc);
  plpcc.AssignL(in);

  int ceps = lpccoefs+1;
  static vector<double> unlifter = LifterMatrixHTK(ceps, 22, true); // lifterMatrixROSA(ceps, 0.6, true);
  plpcc.MultdiagR(unlifter.data(), ceps);
  Matrix::Print(&plpcc, "sec plp");

  return nullptr;
}

unique_ptr<Matrix> Features::MFCC(const RingSampler::Handle *in, Matrix *out, Allocator *alloc) {
  if (!alloc) alloc = Singleton<MallocAllocator>::Set();

  /* [mm,aspc,pspc] = melfcc(y/32768, sr, 'maxfreq', sr/2, 'numcep', 20, 'nbands', 40, 'fbtype', 'htkmel', 'dcttype', 3, 'usecmp', 0, 'wintime', 512/44100, 'hoptime', 256/44100, 'dither', 0, 'lifterexp', 0) */
  static unique_ptr<const Matrix> meltrans = Matrix::Transpose(FFT2Mel(FLAGS_feat_melbands, FLAGS_feat_minfreq, FLAGS_feat_maxfreq, FLAGS_feat_window, FLAGS_sample_rate));
  static unique_ptr<const Matrix> ceptrans = Matrix::Transpose(DCT2(FLAGS_feat_cepcoefs, FLAGS_feat_melbands));

  /* fft */
  int frames = (in->Len() - (FLAGS_feat_window - FLAGS_feat_hop)) / FLAGS_feat_hop;
  Matrix fftbuf(frames, FLAGS_feat_window/2, 0.0, 0, alloc);
  if (!Spectogram(in, &fftbuf, FLAGS_feat_window, FLAGS_feat_hop, FLAGS_feat_window,
                  FLAGS_feat_preemphasis ? PreEmphasisFilter(FLAGS_feat_preemphasis_filter) : vector<double>(),
                  PowerDomain::abs2)) return 0;

  /* to mel scale */
  Matrix melbuf(fftbuf.M, meltrans->N, 0.0, 0, alloc);
  Matrix::Mult(&fftbuf, meltrans.get(), &melbuf);

  /* log */
  MatrixIter(&melbuf) melbuf.row(i)[j] = log(melbuf.row(i)[j]);

  /* prepare output */
  unique_ptr<Matrix> ret;
  if (out) { if (out->M != melbuf.M || out->N != ceptrans->N) return nullptr; }
  else out = (ret = make_unique<Matrix>(melbuf.M, ceptrans->N)).get();

  /* to cepstral coefs */
  Matrix::Mult(&melbuf, ceptrans.get(), out);
  return ret;
}

unique_ptr<RingSampler> Features::InvMFCC(const Matrix *in, int samplerate, const Matrix *f0) {
  /* [dr,aspec,spec] = invmelfcc() */
  auto transform = DCT2(in->N, FLAGS_feat_melbands);
  for (int j=0; j<transform->N; j++) transform->row(0)[j] /= 2;

  auto m = Matrix::Mult(in, move(transform));
  MatrixIter(m) m->row(i)[j] = exp(m->row(i)[j]);

  transform = Mel2FFT(FLAGS_feat_melbands, FLAGS_feat_minfreq, FLAGS_feat_maxfreq, FLAGS_feat_window, samplerate);
  m = Matrix::Mult(move(m), move(transform), mTrnpB);

  if (0 && f0) {
    if (f0->M != in->M || m->M != in->M) return 0;
    unique_ptr<RingSampler> outbuf = make_unique<RingSampler>(samplerate, FLAGS_feat_window + FLAGS_feat_hop * (m->M-1));
    RingSampler::Handle out(outbuf.get());

    for (int i=0; i<m->M; i++) {
      float F0 = f0->row(i)[0];
      float wavs[] = { F0, F0*2, F0*4, F0*8, F0*16, F0*32 };

      for (int j=0; j<sizeofarray(wavs); j++) {
        double f = (2 * M_PI * wavs[j]) / out.Rate();
        int ind = int(wavs[j] * m->N / out.Rate());
        double a = m->row(i)[ind];

        for (int k=0; k<FLAGS_feat_window; k++) 
          *out.Index(i*FLAGS_feat_hop + k) += a*sin(f*i);
      }
    }

    return outbuf;
  }
  else {
    unique_ptr<RingSampler> outbuf = make_unique<RingSampler>(samplerate, FLAGS_feat_window + FLAGS_feat_hop * (m->M-1));
    RingSampler::Handle out(outbuf.get());
    { 
      double f = (2 * M_PI * 440) / out.Rate();
      for (int i=0; i<out.Len(); i++) out.Write(sin(f*i)/16 + Rand(-1.0,1.0)/2);
    }

    unique_ptr<Matrix> spec(Spectogram(&out, 0, FLAGS_feat_window, FLAGS_feat_hop, FLAGS_feat_window, vector<double>(), PowerDomain::complex));
    if (m->M != spec->M || m->N != spec->N) return nullptr;
    MatrixIter(spec) {
      Complex v = {sqrt(m->row(i)[j]), 0}; 
      spec->crow(i)[j].Mult(v);
    }
    return ISpectogram(spec.get(), FLAGS_feat_window, FLAGS_feat_hop, FLAGS_feat_window/2, samplerate);
  }
}

unique_ptr<Matrix> Features::FilterZeroth(unique_ptr<Matrix> features) {
  auto fz = make_unique<Matrix>(features->M, features->N - 1);
  MatrixRowIter(fz) {
    double *in = features->row(i), *out = fz->row(i);
    MatrixColIter(fz) out[j] = in[j+1];
  }
  return fz;
}

void Features::MeanAndVarianceNormalization(int D, double *feat, const double *mean, const double *var) {
  Vector::Sub(feat, mean, D);
  if (var) Vector::Mult(feat, var, D);
}

void Features::DeltaCoefficients(int D, const double *n2, double *f, const double *p2) {
  for (int j=0; j<D; j++)
    f[D+j] = p2[j] - n2[j];
}

void Features::DeltaDeltaCoefficients(int D, const double *n3, const double *n1, double *f,  const double *p1, const double *p3) {
  for (int j=0; j<D; j++) {
    double d1 = p3[j] - n1[j];
    double d2 = p1[j] - n3[j];
    f[D*2+j] = d1 - d2;
  }
}

void Features::PatchDeltaCoefficients(int D, const double *in, double *out1, double *out2) {
  Vector::Assign(out1, in, D);
  Vector::Assign(out2, in, D);
}

void Features::PatchDeltaDeltaCoefficients(int D, const double *in, double *out1, double *out2, double *out3) {
  Vector::Assign(out1, in, D);
  Vector::Assign(out2, in, D);
  Vector::Assign(out3, in, D);
}

unique_ptr<Matrix> Features::DeltaCoefficients(unique_ptr<Matrix> in, bool dd) {
  int M = in->M, D = in->N;
  auto features = make_unique<Matrix>(M, D*(2+dd));
  features->AssignR(in.get());

  /* copy sphinx */
  MatrixRowIter(features) {
    if (i-2 < 0 || i+2 >= M) continue;
    DeltaCoefficients(D, features->row(i-2), features->row(i), features->row(i+2));

    if (i-3 < 0 || i+3 >= M || !dd) continue;
    DeltaDeltaCoefficients(D, features->row(i-3), features->row(i-1), features->row(i), features->row(i+1), features->row(i+3));
  }

  /* delta - patch ends */
  PatchDeltaCoefficients(D, &features->row(    2)[D], &features->row(    1)[D], &features->row(    0)[D]);
  PatchDeltaCoefficients(D, &features->row(M-1-2)[D], &features->row(M-1-1)[D], &features->row(M-1-0)[D]);

  /* delta delta - patch ends */
  if (dd) {
    PatchDeltaDeltaCoefficients(D, &features->row(    3)[D*2], &features->row(    2)[D*2], &features->row(    1)[D*2], &features->row(    0)[D*2]);
    PatchDeltaDeltaCoefficients(D, &features->row(M-1-3)[D*2], &features->row(M-1-2)[D*2], &features->row(M-1-1)[D*2], &features->row(M-1-0)[D*2]);
  }

  return features;
}

unique_ptr<Matrix> Features::FromFeat(unique_ptr<Matrix> features, int flag, bool filterzeroth, bool deltas, bool deltadeltas, bool meannorm, bool varnorm) {
  if (flag == Flag::Full) {
    if (filterzeroth) features = FilterZeroth(move(features));

    if (meannorm) {
      vector<double> mean(features->N);
      vector<double> var(features->N);

      Vector::Assign(&mean[0], 0.0, features->N);
      MatrixRowIter(features) Vector::Add(&mean[0], features->row(i), features->N);
      Vector::Div(&mean[0], features->M, features->N);

      if (varnorm) {                
        Vector::Assign(&var[0], 0.0, features->N);
        MatrixIter(features) {
          double diff = features->row(i)[j] - mean[j];
          var[j] += diff*diff;
        }
        MatrixColIter(features) var[j] = sqrt(var[j] / features->M);
      }

      MatrixRowIter(features) {
        MeanAndVarianceNormalization(features->N, features->row(i), &mean[0], variance_normalization ? &var[0] : 0);
      }            
    }

    if (deltas) features = DeltaCoefficients(move(features), deltadeltas);
  }
  return features;
}

unique_ptr<Matrix> Features::FromFeat(unique_ptr<Matrix> features, int flag) {
  return FromFeat(move(features), flag, filter_zeroth, deltas, deltadeltas, mean_normalization, variance_normalization);
}

unique_ptr<Matrix> Features::FromAsset(SoundAsset *wav, int flag) {
  if (wav->channels > 1) ERROR("Features::fromAsset called on SoundAsset with ", wav->channels, " channels");
  RingSampler::Handle B(wav->wav.get());
  auto features = FromBuf(&B);
  return FromFeat(move(features), flag);
}

unique_ptr<Matrix> Features::FromBuf(const RingSampler::Handle *in, Matrix *out, vector<StatefulFilter> *filter, Allocator *alloc) {
  unique_ptr<Matrix> features;
  if      (FLAGS_feat_type == "MFCC") features = MFCC(in, out, alloc);
  else if (FLAGS_feat_type == "PLP")  features = PLP(in, out, filter, alloc);
  return features;
}

unique_ptr<RingSampler> Features::Reverse(const Matrix *in, int samplerate, const Matrix *f0, Allocator *alloc) {
  unique_ptr<RingSampler> wav;
  if      (FLAGS_feat_type == "MFCC") wav = InvMFCC(in, samplerate, f0);
  else if (FLAGS_feat_type == "PLP")  wav = InvPLP(in, samplerate, alloc);
  return wav;
}

int Features::Dimension() {
  if      (FLAGS_feat_type == "MFCC") return FLAGS_feat_cepcoefs;
  else if (FLAGS_feat_type == "PLP")  return lpccoefs+1;
  return 0;
}

/* AcousticModel */

double *AcousticModel::State::Transit(Compiled *model, Matrix *transit, State *Rstate, unsigned *toOut) {
  int toind = Rstate->val.emission_index;
  if (toind < 0 || toind >= model->state.size()) FATAL("OOB toind ", toind);
  unsigned to = model->state[toind].Id();
  if (toOut) *toOut = to;
  return AcousticModel::State::Transit(transit, to);
}

double *AcousticModel::State::Transit(Matrix *trans, unsigned K) { double k[3]={0,double(K),0}; return static_cast<double*>(bsearch(k, trans->m, trans->M, sizeof(double)*TransitCols, TransitionSort)); }

int AcousticModel::State::TransitionSort(const void *a, const void *b) { return DoubleSortR(Void(static_cast<const double*>(a)+1), Void(static_cast<const double*>(b)+1)); }

void AcousticModel::State::SortTransitionMap(double *trans, int M) { qsort(trans, M, sizeof(double)*TransitCols, TransitionSort); }

void AcousticModel::State::SortTransitionMap(Matrix *trans) {
  if (!trans || trans->N != TransitCols) FATAL("error ", trans?trans->N:0, " != ", TransitCols);
  return SortTransitionMap(trans->m, trans->M);
}

int AcousticModel::State::LocalizeTransitionMap(Matrix *transIn, Matrix *transOut, Compiled *scope, int scopeState, int minSamples, Compiled *dubdref, const char *srcname) {
  if (!transIn || !transIn->M) return 0;
  if (transIn->N != TransitCols || transOut->N != TransitCols) { ERROR("wrong dimensions (", transIn->N, "|", transOut->N, ") != ", TransitCols); return -1; }
  if (transIn->M > transOut->M) { ERROR("rows overflow ", transIn->M, " > ", transOut->M); return -1; }

  MatrixRowIter(transIn) {
    int si = scopeState-1, skipped=-1;
    do {
      skipped++;
      for (si++; si<scope->state.size(); si++) {
        unsigned match = scope->state[si].Id();
        if (dubdref) {
          if (srcname) { 
            int ap, an, a = ParseName(srcname, 0, &ap, &an);
            int bp, bn, b = ParseName(scope->state[si].name.c_str(), 0, &bp, &bn);
            if (!TriphoneConnect(ap, a, an, bp, b, bn)) continue;
          }

          int matchind = scope->state[si].val.emission_index;
          if (matchind < 0 || matchind >= dubdref->state.size()) FATAL("OOB matchind ", matchind);
          match = dubdref->state[matchind].Id();
        }
        if (transIn->row(i)[TC_Edge] == match) break;
      }
      if (si >= scope->state.size()) { ERRORf("cant find %d %f dim(%d,%d) scope(%d,%d) skipped=%d", i, transIn->row(i)[TC_Edge], transIn->M, transIn->N, scopeState, minSamples, skipped); return -1; }
    }
    while (minSamples && scope->state[si].val.samples < minSamples);

    transOut->row(i)[TC_Edge] = si;
    transOut->row(i)[TC_Cost] = transIn->row(i)[TC_Cost];
  }
  return 0;
}

unsigned AcousticModel::State::Tied(Matrix *tiedstates, int pp, int p, int pn, int k) {
  int tied_ind = pp*LFL_PHONES*LFL_PHONES*StatesPerPhone + p*LFL_PHONES*StatesPerPhone + pn*StatesPerPhone + k;
  double *r = tiedstates->row(tied_ind);
  if (!r) FATAL("tied ret ", r);
  unsigned thash = r[0];
  return thash;
}

void AcousticModel::State::Assign(State *out, int *outind, int outsize, StateCollection *model, int pp, int p, int pn, int states, int val) {
  Matrix *tiedstates = model->TiedStates();
  State *s;
  for (int k=0; k<states; k++) {
    if ((*outind) >= outsize) FATAL("overflow ", *outind, " >= ", outsize);

    string n = AcousticModel::Name(pp, p, pn, k);
    unsigned ohash = fnv32(n.c_str()), hash = ohash, thash = tiedstates ? AcousticModel::State::Tied(tiedstates, pp, p, pn, k) : 0;
    bool tied = tiedstates && thash != hash;
    if (tied) hash = thash;
    if (!(s = model->GetState(hash))) FATAL("no state ", n, " (tied=", tied, ")");
    if (!tied) { if (n != s->name) FATAL("mismatching states ", n, " ", s->name); }
    else { if (0) INFO("tied ", n, " to ", s->name); }

    int ind = *outind;
    out[ind].AssignPtr(s);
    out[ind].val.samples = val;
    out[ind].transition.Absorb(out[ind].transition.Clone());

    if (tied) {
      static unsigned silhash = fnv32("Model_SIL_State_00");

      out[ind].alloc = Singleton<MallocAllocator>::Set();
      out[ind].name = n;

      MatrixIter(&out[ind].transition) { out[ind].transition.row(i)[TC_Self] = ohash; }
      SortTransitionMap(&out[ind].transition);
    }
    else {
      MatrixIter(&out[ind].transition) if (out[ind].transition.row(i)[TC_Self] != hash) FATAL("mismatching transit ", hash, " ", out[ind].transition.row(i)[TC_Self]); 
    }

    (*outind)++;
  }
}

int AcousticModel::ParseName(const string &name, int *phonemeStateOut, int *prevPhoneOut, int *nextPhoneOut) {
  static string prefix="Model_", suffix="_State_";
  if (strncmp(name.c_str(), prefix.c_str(), prefix.size())) return -1;

  const char *vn = name.c_str() + prefix.size(), *sn;
  if (!(sn = strstr(vn, suffix.c_str()))) return -1;
  if (phonemeStateOut) *phonemeStateOut = atoi(sn + suffix.size());
  if (!prevPhoneOut && !nextPhoneOut) return Phoneme::Id(vn, sn-vn);

  string phoneme(vn, sn-vn);
  const char *pp= phoneme.c_str(), *px, *pn;
  if (!(px = strchr(phoneme.c_str(), '_')) || !(pn = strchr(px+1, '_'))) {
    if (prevPhoneOut) *prevPhoneOut = 0;
    if (nextPhoneOut) *nextPhoneOut = 0;
    return Phoneme::Id(phoneme.c_str());
  }

  if (prevPhoneOut) *prevPhoneOut = Phoneme::Id(pp, px-pp);
  if (nextPhoneOut) *nextPhoneOut = Phoneme::Id(pn+1);
  return Phoneme::Id(px+1, pn-px-1);
}

string AcousticModel::Flags() {
  string s = StrCat("sr=", FLAGS_sample_rate, ",type=", FLAGS_feat_type);

  if (FLAGS_feat_minfreq)                        StrAppend(&s, ",minfreq=", FLAGS_feat_minfreq);
  if (FLAGS_feat_maxfreq != FLAGS_sample_rate/2) StrAppend(&s, ",maxfreq=", FLAGS_feat_maxfreq);

  StrAppend(&s, ",fftwin=", FLAGS_feat_window, ",ffthop=", FLAGS_feat_hop);

  if (FLAGS_feat_type == "MFCC") {
    if (FLAGS_feat_preemphasis) StrAppend(&s, ",preemph=", FLAGS_feat_preemphasis_filter);

    StrAppend(&s, ",mels=", FLAGS_feat_melbands, ",ceps=", FLAGS_feat_cepcoefs);
  }

  if (Features::filter_zeroth)          s += ",filterzeroth";
  if (Features::mean_normalization)     s += ",cmn";
  if (Features::variance_normalization) s += ",cvn";
  if (Features::deltas)                 s += ",delta";
  if (Features::deltadeltas)            s += ",deltadelta";

  if (FLAGS_triphone_model) s += ",triphone";

  return s;
}

void AcousticModel::LoadFlags(const char *flags) {
  StringWordIter iter(flags, iscomma);
  for (string k = iter.NextString(); !iter.Done(); k = iter.NextString()) {
    char *v; double val;
    if ((v = const_cast<char*>(strchr(k.c_str(), '=')))) { *v++=0; val=atof(v); }

    if      (k == "sr")         FLAGS_sample_rate = val;
    else if (k == "type")       FLAGS_feat_type   = v;
    else if (k == "minfreq")    FLAGS_feat_minfreq  = val;
    else if (k == "maxfreq")    FLAGS_feat_maxfreq  = val;
    else if (k == "ffthop")     FLAGS_feat_hop      = val;
    else if (k == "fftwin")     FLAGS_feat_window   = val;
    else if (k == "mels")       FLAGS_feat_melbands = val;
    else if (k == "ceps")       FLAGS_feat_cepcoefs = val;
    else if (k == "filterzeroth") Features::filter_zeroth          = 1;
    else if (k == "cmn")          Features::mean_normalization     = 1;
    else if (k == "cvn")          Features::variance_normalization = 1;
    else if (k == "delta")        Features::deltas                 = 1;
    else if (k == "deltadelta")   Features::deltadeltas            = 1;
    else if (k == "preemph") { FLAGS_feat_preemphasis_filter = val; FLAGS_feat_preemphasis = 1; }
    else if (k == "triphone") { FLAGS_triphone_model = 1; }
  }
  INFO("loaded flags: ", flags);
}

int AcousticModel::Write(StateCollection *model, const char *name, const char *dir, int iteration, int minSamples) {
  int ret=0, states=0, means=0, transits=0, D=0, K=0;
  string flagtext=Flags();

  AcousticModel::StateCollection::Iterator iter;
  for (model->BeginState(&iter); !iter.done; model->NextState(&iter)) {
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
  for (model->BeginState(&iter); !iter.done; model->NextState(&iter)) {
    AcousticModel::State *s = static_cast<AcousticModel::State*>(iter.v);
    if (minSamples && s->val.samples < minSamples) continue;

    StringFile::WriteRow(&names, s->name);
    MatrixFile::WriteRow(&initial, &s->prior, 1);

    MatrixFile::WriteRow(&prior, s->emission.prior.m, s->emission.prior.M);

    MatrixRowIter(&s->emission.mean)    { MatrixFile::WriteRow(&mean,  s->emission.mean.row(i),    s->emission.mean.N);  }
    MatrixRowIter(&s->emission.diagcov) { MatrixFile::WriteRow(&covar, s->emission.diagcov.row(i), s->emission.diagcov.N); }

    Matrix tx(s->transition);
    State::SortTransitionMap(tx.m, tx.M);
    MatrixRowIter(&tx) {
      tx.row(i)[TC_Self] = s->Id();
      MatrixFile::WriteRow(&transit, tx.row(i), tx.N);
    }

    /* build map */
    double *he = map.Set(s->Id());
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
  if (1                   && MatrixFile(map.map.get(),       flagtext).WriteVersioned(VersionedFileName(dir, name, "map"),        iteration) < 0) { ERROR(name, " write map");     ret=-1; }
  if (model->TiedStates() && MatrixFile(model->TiedStates(), flagtext).WriteVersioned(VersionedFileName(dir, name, "tiedstates"), iteration) < 0) { ERROR(name, " write tied");    ret=-1; }
  if (model->PhoneTx()    && MatrixFile(model->PhoneTx(),    flagtext).WriteVersioned(VersionedFileName(dir, name, "phonetx"),    iteration) < 0) { ERROR(name, " write phonetx"); ret=-1; }

  return ret;
}

int AcousticModel::ToCUDA(AcousticModel::Compiled *model) {
#ifdef LFL_CUDA
  if (!app->cuda) return 0;
  int K=model->state[0].emission.mean.M, D=model->state[0].emission.mean.N, ret;
  if (!CAM) CAM = make_unique<CudaAcousticModel>(model->state.size(), K, D);
  if (K != CAM->K || D != CAM->D || model->state.size() != CAM->states) FATALf("toCUDA mismatch %d != %d || %d != %d || %d != %d", K, CAM->K, D, CAM->D, model->state.size(), CAM->states);
  if ((ret = CudaAcousticModel::load(CAM, model))) ERROR("CudaAcousticModel::load ret=", ret);
  return ret;
#else
  return 0;
#endif
}

unique_ptr<AcousticModel::Compiled> AcousticModel::FullyConnected(Compiled *model) {
  auto hmm = make_unique<Compiled>(model->state.size());
  Matrix tx(model->state.size(), TransitCols);
  double prob = log(1.0 / model->state.size());
  for (int i=0; i<model->state.size(); i++) {
    hmm->state[i].AssignPtr(&model->state[i]);
    hmm->state[i].prior = prob;
    tx.row(i)[TC_Edge] = i;
    tx.row(i)[TC_Cost] = prob;
  }
  AcousticModel::State::SortTransitionMap(&tx);
  for (int i=0; i<model->state.size(); i++) {
    AcousticModel::State *s = &hmm->state[i];
    s->transition = tx;
    MatrixRowIter(&s->transition) s->transition.row(i)[TC_Self] = s->Id();
  }
  return hmm;
}

unique_ptr<AcousticModel::Compiled> AcousticModel::FromUtterance(AssetLoading *loader, Compiled *model, const char *transcript, bool UseTransit) {
  return FLAGS_triphone_model ? FromUtterance3(loader, model, transcript, UseTransit) : FromUtterance1(loader, model, transcript, UseTransit);
}

/* context independent utterance model */
unique_ptr<AcousticModel::Compiled> AcousticModel::FromUtterance1(AssetLoading *loader, AcousticModel::Compiled *model, const char *transcript, bool UseTransit) {
  /* get pronunciation */
  static const int maxwords=1024;
  PronunciationDict *dict = PronunciationDict::Instance(loader);
  const char *w[maxwords], *wa[maxwords]; int words, phones, len=0;
  if ((words = dict->Pronounce(transcript, w, wa, &phones, maxwords)) <= 0) return 0;
  auto hmm = make_unique<Compiled>(phones*StatesPerPhone+words+1);

  /* assign states */
  State::Assign(&hmm->state[0], &len, hmm->state.size(), model, 0, Phoneme::SIL, 0, 1);
  for (int i=0; i<words; i++) {
    for (int j=0, pl=strlen(w[i]); j<pl; j++) State::Assign(&hmm->state[0], &len, hmm->state.size(), model, 0, w[i][j], 0, StatesPerPhone);
    State::Assign(&hmm->state[0], &len, hmm->state.size(), model, 0, Phoneme::SIL, 0, 1);
  }
  hmm->state.resize(len);

  if (FLAGS_speech_recognition_debug) {
    INFO("lattice pre patch");
    AcousticHMM::PrintLattice(hmm.get(), model);
  }

  /* patch & localize transitions */
  for (int i=0; i<hmm->state.size(); i++) {
    AcousticModel::State *s = &hmm->state[i];
    bool lastState = (i+1 == hmm->state.size());
    bool nextStateSil = (!lastState && IsSilence(hmm->state[i+1].name));
    bool addtx = nextStateSil && i+2<hmm->state.size();
    auto trans = make_unique<Matrix>(2-lastState+addtx, TransitCols, s->Id());

    double mintx = INFINITY;
    for (int ti=0; ti<s->transition.M; ti++) if (s->transition.row(ti)[TC_Cost] < mintx) mintx = s->transition.row(ti)[TC_Cost];

    double *trself = AcousticModel::State::Transit(model, &s->transition, s);
    if (!trself && UseTransit) ERROR("no transition to self for ", s->name);
    trans->row(0)[TC_Edge] = i;
    trans->row(0)[TC_Cost] = trself ? trself[TC_Cost] : mintx;

    if (!lastState) {
      State *sj = &hmm->state[i+1];
      double *trnext = AcousticModel::State::Transit(model, &s->transition, sj);
      if (!trnext && UseTransit) ERROR("no transition to next for ", s->name);
      trans->row(1)[TC_Edge] = i+1;
      trans->row(1)[TC_Cost] = trnext ? trnext[TC_Cost] : mintx;
    }

    if (addtx) {
      State *sj = &hmm->state[i+2];
      double *trnext = AcousticModel::State::Transit(model, &s->transition, sj);
      if (!trnext && UseTransit) ERROR("no transition to next for ", s->name);
      trans->row(2)[TC_Edge] = i+2;
      trans->row(2)[TC_Cost] = trnext ? trnext[TC_Cost] : mintx;
    }

    s->transition.Absorb(move(trans));
  }
  return hmm;
}

/* triphone utterance model */
unique_ptr<AcousticModel::Compiled> AcousticModel::FromUtterance3(AssetLoading *loader, AcousticModel::Compiled *model, const char *transcript, bool UseTransit) {
  /* get pronunciation */
  PronunciationDict *dict = PronunciationDict::Instance(loader);
  static const int maxwords=1024;
  const char *w[maxwords], *wa[maxwords];
  int words, phones, len=0, depth=0, prevPhonesInWord=0, prevEndWordPhone=0;
  if ((words = dict->Pronounce(transcript, w, wa, &phones, maxwords)) < 0) { DEBUG("pronounce '%s' failed", transcript); return 0; }
  auto hmm = make_unique<Compiled>(phones*StatesPerPhone+words*4*StatesPerPhone+StatesPerPhone+1);

  /* assign states */
  State::Assign(&hmm->state[0], &len, hmm->state.size(), model, 0, Phoneme::SIL, 0, 1, depth++);
  for (int i=0; i<words; i++) {
    bool lastword = i == words-1;
    const char *pronunciation = w[i];
    int phonesInWord = strlen(pronunciation);

    if (1) { /* add cross-word begin triphone states */
      int pn = phonesInWord>1 ? pronunciation[1] : (!lastword ? w[i+1][0] : 0);

      /* paralell phoneme paths - no depth increase */
      if (phonesInWord==1 || prevEndWordPhone)       State::Assign(&hmm->state[0], &len, hmm->state.size(), model, prevEndWordPhone, pronunciation[0], pn, StatesPerPhone, depth);
      if (phonesInWord==1 && prevEndWordPhone && pn) State::Assign(&hmm->state[0], &len, hmm->state.size(), model, 0,                pronunciation[0], pn, StatesPerPhone, depth);
    }

    int lastphone=0; /* add inner word triphone states and begin/end diphone states */
    for (int j=0; j<phonesInWord; j++) {
      int nextphone = j+1 < phonesInWord ? pronunciation[j+1] : 0;
      int phone = pronunciation[j];

      State::Assign(&hmm->state[0], &len, hmm->state.size(), model, lastphone, phone, nextphone, StatesPerPhone, depth++);
      lastphone = phone;
    }
    depth--;

    if (!lastword) { /* add cross-word end triphone states */
      int p = pronunciation[phonesInWord-1]; 
      int pp = phonesInWord>1 ? pronunciation[phonesInWord-2] : prevEndWordPhone;
      int pn = w[i+1][0];

      /* paralell phoneme paths - no depth increase */
      if (phonesInWord==1 && pn && pp) State::Assign(&hmm->state[0], &len, hmm->state.size(), model, pp, p, 0,  StatesPerPhone, depth);
      if (phonesInWord>1)              State::Assign(&hmm->state[0], &len, hmm->state.size(), model, pp, p, pn, StatesPerPhone, depth);
    }
    depth++;

    /* add silence after word */
    State::Assign(&hmm->state[0], &len, hmm->state.size(), model, 0, Phoneme::SIL, 0, 1, depth++);
    prevPhonesInWord = phonesInWord;
    prevEndWordPhone = lastphone;
  }
  hmm->state.resize(len);

  if (FLAGS_speech_recognition_debug) {
    INFO("lattice pre patch");
    AcousticHMM::PrintLattice(hmm.get(), model);
  }

  /* patch transitions */
  for (int i=0; i<hmm->state.size()-1; i++) {
    State *s = &hmm->state[i];
    vector<pair<unsigned, float> > tx;
    int as, ap, an, a = ParseName(s->name, &as, &ap, &an);

    double mincost = INFINITY;
    for (int k=0; k<s->transition.M; k++) {
      double v = s->transition.row(k)[TC_Cost];
      if (v < mincost) mincost = v;
    }
    if (!isfinite(mincost)) mincost = -16;

    if (a == Phoneme::SIL || as == StatesPerPhone-1) {
      /* determine matching next triphone states */
      for (int j=i+1; j<hmm->state.size(); j++) {
        State *sj = &hmm->state[j];

        int bs, bp, bn, b = ParseName(sj->name, &bs, &bp, &bn);
        if (bs != 0) continue;

        if (State::TriphoneConnect(ap, a, an, bp, b, bn)) {
          unsigned to; double cost=mincost, *tr = AcousticModel::State::Transit(model, &s->transition, sj, &to);
          if (tr) cost = tr[TC_Cost];
          else if (UseTransit) ERROR("no transition from ", s->name, " ", s->Id(), " to ", sj->name, ", patched w min ", cost);
          tx.push_back(pair<unsigned, float>(to, cost));
        }
        else if (tx.size()) break;
      }
      if (!tx.size()) FATAL("cant find transition for ", i, " ", s->name);
    }
    else {
      State *sj = &hmm->state[i+1];
      unsigned to; double cost=mincost, *tr = AcousticModel::State::Transit(model, &s->transition, sj, &to);
      if (tr) cost = tr[TC_Cost];
      else if (UseTransit) ERROR("no transition from ", s->name, " ", s->Id(), " to ", sj->name, ", patched w min ", cost);
      tx.push_back(pair<unsigned, float>(to, cost));
    }

    /* add localized transition to self for states patched below */
    double strcost=mincost, *str = AcousticModel::State::Transit(model, &s->transition, s);
    if (!str) ERROR("no transition to self ", s->name, " ", s->Id(), ", patched w min ", strcost);
    else strcost = str[TC_Cost];

    double prob=log(1.0/tx.size()+1);
    s->transition.Open(tx.size()+1, TransitCols, s->Id());
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
  double fstrcost=-16, *fstr = AcousticModel::State::Transit(model, &final->transition, final);
  if (!fstr) ERROR("no transition from final to self ", final->name, " ", final->Id(), ", patched w min ", fstrcost);
  else fstrcost = fstr[TC_Cost];

  final->transition.Open(1, TransitCols, final->Id());
  final->transition.row(0)[TC_Edge] = len-1;
  final->transition.row(0)[TC_Cost] = fstrcost;

  if (FLAGS_speech_recognition_debug) {
    INFO("lattice pre localization");
    AcousticHMM::PrintLattice(hmm.get(), model);
  }

  /* localize transitions */
  for (int i=0; i<hmm->state.size()-1; i++) {
    State *s = &hmm->state[i];
    int as, ap, an, a = ParseName(s->name, &as, &ap, &an);
    bool patch = (a == Phoneme::SIL || as == StatesPerPhone-1);

    /* resolve to forward states using val.samples minSamples filter on paralell phoneme 'depth' */
    if (State::LocalizeTransitionMap(&s->transition, &s->transition, hmm.get(), i, s->val.samples + patch, model, patch ? s->name.c_str() : 0)) { ERROR("localize transition ", i, " ", s->name.c_str(), " '", transcript, "'"); return 0; }

    /* transition to self for patched state */ 
    s->transition.M++;
  }

  if (FLAGS_speech_recognition_debug) {
    INFO("lattice post localization");
    AcousticHMM::PrintLattice(hmm.get(), model);
  }

  return hmm;
}

unique_ptr<AcousticModel::Compiled> AcousticModel::FromModel1(StateCollection *model, bool rewriteTransitions) {
  int states = model->GetStateCount(), len = 0;
  if (states <= 0) return 0;

  auto hmm = make_unique<Compiled>(states);
  AcousticModel::StateCollection::Iterator iter;
  State *sil=0;
  for (model->BeginState(&iter); !iter.done; model->NextState(&iter)) {
    State *s = iter.v;
    if (IsSilence(s->name)) { sil=s; continue; }
    hmm->state[len++].AssignPtr(s);
  }
  hmm->state[len++].AssignPtr(sil);

  for (int i=0; i<hmm->state.size(); i++) {
    State *s = &hmm->state[i];
    s->transition.Absorb(s->transition.Clone());
    if (State::LocalizeTransitionMap(&s->transition, &s->transition, hmm.get(), 0)) { ERROR("localize transition ", i, " ", s->name); return 0; }
  }

  if (rewriteTransitions) {
    double tp = log(1.0/LFL_PHONES);
    auto trans = make_unique<Matrix>(LFL_PHONES, TransitCols);
    MatrixRowIter(trans) {
      trans->row(i)[TC_Edge] = i*StatesPerPhone;
      trans->row(i)[TC_Cost] = tp;
    }
    for (int i=0; i<hmm->state.size(); i++) {
      State *s = &hmm->state[i];
      string n = s->name;
      if (!IsSilence(s->name) && n.substr(n.size()-8) != "State_02") continue;

      s->transition.Absorb(trans->Clone()); 
      MatrixRowIter(&s->transition) s->transition.row(i)[TC_Self] = s->Id();
    }
  }

  return hmm;
}

int AcousticModelFile::Open(FileSystem *fs, const char *name, const char *dir, int lastiter, bool rebuild_transit) {
  Reset();
  string flags;
  lastiter = MatrixFile::ReadVersioned(fs, dir, name, "transition", &transit, &flags, lastiter);
  if (!transit) { ERROR("no acoustic model: ", name); return -1; }

  if (flags.size()) AcousticModel::LoadFlags(flags.c_str());

  unique_ptr<Matrix> mapdata;
  if (MatrixFile::ReadVersioned(fs, dir, name, "prior",      &initial, 0, lastiter)<0) { ERROR(name, ".", lastiter, ".prior"  ); return -1; }
  if (MatrixFile::ReadVersioned(fs, dir, name, "emMeans",    &mean,    0, lastiter)<0) { ERROR(name, ".", lastiter, ".emMean" ); return -1; }
  if (MatrixFile::ReadVersioned(fs, dir, name, "emCov",      &covar,   0, lastiter)<0) { ERROR(name, ".", lastiter, ".emCov"  ); return -1; }
  if (MatrixFile::ReadVersioned(fs, dir, name, "emPrior",    &prior,   0, lastiter)<0) { ERROR(name, ".", lastiter, ".emPrior"); return -1; }
  if (MatrixFile::ReadVersioned(fs, dir, name, "map",        &mapdata, 0, lastiter)<0) { ERROR(name, ".", lastiter, ".map"    ); return -1; }
  if (StringFile::ReadVersioned(fs, dir, name, "name",       &names,   0, lastiter)<0) { ERROR(name, ".", lastiter, ".name"   ); return -1; }
  if (MatrixFile::ReadVersioned(fs, dir, name, "tiedstates", &tied,    0, lastiter)<0) { ERROR(name, ".", lastiter, ".tied"   );       /**/ }
  map.map = move(mapdata);

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

  AcousticModel::Compiled::Open(M);
  for (int i=0; i<state.size(); i++) {
    state[i].name = (*names)[i];
    state[i].prior = initial->row(i)[0];
    state[i].emission.AssignDataPtr(K, N, mean->row(i*K), covar->row(i*K), prior->row(i));
    state[i].val.emission_index = i;

    unsigned hash = state[i].Id();
    if (rebuild_transit) {
      LFL_STL_NAMESPACE::map<unsigned, pair<int, int> >::iterator it = txmap.find(hash);
      if (it == txmap.end()) { ERROR("find state ", state[i].name, " in transition failed ", hash); return -1; }
      state[i].transition.AssignDataPtr((*it).second.second, transit->N, transit->row((*it).second.first));
    }
    else {
      double *he = GetHashEntry(hash);
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

double AcousticHMM::ForwardBackward(AcousticModel::Compiled *model, Matrix *observations, int InitMax, double beamWidth, HMM::BaumWelchAccum *BWaccum, int flag) {
  int NBest=1, M=observations->M, N=model->state.size();
  Matrix alpha(M, N, -INFINITY), beta(M, N, -INFINITY), gamma(M, N, -INFINITY), xi(N, N, -INFINITY);

  HMM::ActiveStateIndex active(N, NBest, beamWidth, InitMax);
  TransitMap transit(model, flag & Flag::UseTransit);
  EmissionMatrix emit(model, observations, flag & Flag::UsePrior);
  HMM::BaumWelchWrapper BWwrapper;

  double bp, fp=HMM::ForwardBackward(&active, &transit, &emit, &alpha, &beta, &gamma, &xi, &bp, &BWwrapper);
  if (isnan(fp) || isnan(bp) || isinf(fp) || isinf(bp)) { INFO("NaN or INF forward ", fp, " or backward ", bp); return 0; }

  double diff = fabs(fp - bp);
  if (diff > 1) INFO("forward and backwards probabilities differ by ", diff, " (", fp, " and ", bp, ")");
  else BWwrapper.Add(BWaccum);
  return fp;
}

double AcousticHMM::Viterbi(AcousticModel::Compiled *model, Matrix *observations, Matrix *path, int InitMax, double beamWidth, int flag) {
  int NBest=1, M=observations->M, N=model->state.size();
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

  int endstate = HMM::Forward(&active, &transit, &emit, &beam, Singleton<HMM::Algorithm::Viterbi>::Set()) / NBest;
  HMM::TracePath(path, &backtrace, endstate, M);
  return lambda.row(M-1)[endstate];
#endif
}

double AcousticHMM::UniformViterbi(AcousticModel::Compiled *model, Matrix *observations, Matrix *viterbi, int Unused1, double Unused2, int uu3) {
  if (!DimCheck("HMM::viterbi", viterbi->M, observations->M)) return -INFINITY;

  int obvs = observations->M, states=0;
  for (int i=0; i<model->state.size(); i++) {
    if (AcousticModel::IsSilence(model->state[i].name)) continue;
    states++;
  }

  double spmF = double(obvs)/states;
  int spm = int(spmF), len=0;
  spmF -= spm;

  for (int i=0; i<model->state.size(); i++) {
    if (AcousticModel::IsSilence(model->state[i].name)) continue;
    for (int j=0; j<spm && len<obvs; j++) viterbi->row(len++)[0] = i;
    if (len<obvs && Rand(0.0, 1.0) < spmF) viterbi->row(len++)[0] = i;
  }
  while (len<obvs) viterbi->row(len++)[0] = model->state.size()-1;
  return 0;
}

void AcousticHMM::PrintLattice(AcousticModel::Compiled *hmm, AcousticModel::Compiled *model) {
  for (int i=0; i<hmm->state.size(); i++) {
    AcousticModel::State *s = &hmm->state[i];
    string v = StringPrintf("%03d ", i);
    StrAppend(&v, s->name, " (", s->Id(), " ", model ? model->state[s->val.emission_index].Id() : 0, ")");
    StrAppend(&v, " ei=", s->val.emission_index, " sc=", s->val.samples, " tx = ");
    for (int j=0; j<s->transition.M; j++) {
      StringAppendf(&v, "%u = %.02f, ", unsigned(s->transition.row(j)[TC_Edge]), s->transition.row(j)[TC_Cost]);
    }
    INFO(v);
  }
}

void AcousticHMM::EmissionArray::Calc(AcousticModel::Compiled *model, HMM::ActiveState *active, int time_index, const double *observation, double *emission, double *posterior, double *cudaposterior) {
  active->time_index = time_index;
  int active_states = active->Size();
#ifdef LFL_CUDA
  if (app->cuda) {
    if (posterior && !cudaposterior) FATAL("posterior ", posterior, " requires cudaposterior ", cudaposterior);
    vector<double> cudaemission(active_states);
    vector<int> beam(active_states);
    int K = model->state[0].emission.mean.M;

    HMM::ActiveState::Iterator stateI; int ind=0;
    for (active->begin(time_index, &stateI); !stateI.done && ind<active_states; active->next(&stateI)) {
      AcousticModel::State *si = &model->state[stateI.index];
      beam[ind++] = si->val.emission_index;
    }

    if (CudaAcousticModel::calcEmissions(CAM, observation, &beam[0], active->size(), &cudaemission[0], cudaposterior))
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
    for (active->Begin(time_index, &stateI); !stateI.done && ind<active_states; active->Next(&stateI), ind++) {
      AcousticModel::State *si = &model->state[stateI.index];
      emission[stateI.index] = si->emission.PDF(observation, posterior ? posterior + stateI.index * si->emission.mean.M : 0);
    }
  }
}

/* decoder */

unique_ptr<Matrix> Decoder::DecodeFile(AssetLoading *loader, AcousticModel::Compiled *model, const char *fn, double beamWidth) {
  SoundAsset input(loader, "input", fn, 0, 0, 0, 0);
  input.Load();
  if (!input.wav) { ERROR(fn, " not found"); return 0; }

  auto features = Features::FromAsset(&input, Features::Flag::Full);
  return DecodeFeatures(model, features.get(), beamWidth);
}

unique_ptr<Matrix> Decoder::DecodeFeatures(AcousticModel::Compiled *model, Matrix *features, double beamWidth, int flag, Window *visualize) {
  if (!DimCheck("DecodeFeatures", features->N, model->state[0].emission.mean.N)) return 0;
  if (FLAGS_speech_recognition_debug) AcousticHMM::PrintLattice(model);

  Timer vtime;
  auto viterbi = make_unique<Matrix>(features->M, 1);
  double vprob = AcousticHMM::Viterbi(model, features, viterbi.get(), 0, beamWidth, flag);
  if (visualize) Decoder::VisualizeFeatures(visualize, model, features, viterbi.get(), vprob, vtime.GetTime(), flag & AcousticHMM::Flag::Interactive);

  return viterbi;
}

string Decoder::Transcript(const AcousticModel::Compiled *model, const Matrix *viterbi, Allocator *alloc) {
  string ret; int laststate=-1;
  for (Decoder::PhoneIter iter(model, viterbi); !iter.Done(); iter.Next()) {
    if (!iter.phone) continue;
    StrAppend(&ret, ret.size()?" ":"", Phoneme::Name(iter.phone));
  }
  if (ret.empty()) ret = "<none>";
  return ret;
}

void Decoder::VisualizeFeatures(Window *w, AcousticModel::Compiled *model, Matrix *MFCC, Matrix *viterbi, double vprob, Time vtime, bool interactive) {
  static unique_ptr<PhoneticSegmentationView> segments; 
  static bool interactive_done;

  auto app = w->parent;
  segments = make_unique<PhoneticSegmentationView>(w, model, viterbi, "visbuf");
  interactive_done = 0;

  GraphicsContext gc(segments->root->gd);
  SoundAsset sa(app);
  sa.name = "visbuf";
  sa.Load();
  sa.channels = 1;
  sa.sample_rate = FLAGS_sample_rate;
  sa.wav = unique_ptr<RingSampler>(Features::Reverse(MFCC, FLAGS_sample_rate));
  RingSampler::Handle B = RingSampler::Handle(sa.wav.get());

  auto spect = Spectogram(&B, 0, FLAGS_feat_window, FLAGS_feat_hop, FLAGS_feat_window, vector<double>(), PowerDomain::dB);
  Asset *snap = app->asset("snap");
  SpectogramAsset().Draw(gc.gd, spect.get(), &snap->tex, 0);

  if (FLAGS_enable_audio) app->audio->QueueMixBuf(&B);
  INFO("vprob=", vprob, " vtime=", vtime.count());
  Font *font = segments->root->default_font;

  Box wcc = Box(5,345, 400,100);
  while (app->run && (app->audio->Out.size() || (interactive && !interactive_done))) {
    app->HandleEvents(app->frame_time.GetTime(true).count());

    gc.gd->DrawMode(DrawMode::_2D);
    app->asset("snap")->tex.Draw(&gc, wcc); // 4);

    int levels=10;
    float percent = 1-float(app->audio->Out.size())/app->audio->outlast;
    font->Draw(gc.gd, StringPrintf("time=%d vprob=%f percent=%f next=%d", vtime.count(), vprob, percent, interactive_done), point(10, 440));

    percent -= Audio::VisualDelay()*FLAGS_sample_rate*FLAGS_chans_out/app->audio->outlast;
    if (percent >= 0 && percent <= 1) {
      for (int i=1; i<=levels; i++) font->Draw(gc.gd, "|", point(wcc.x+percent*wcc.w, wcc.y-30*i));
    }

    int count=0;
    for (Decoder::PhoneIter iter(model, viterbi); !iter.Done(); iter.Next()) {
      if (!iter.phone) continue;
      int r = count++%levels+1;
      font->Draw(gc.gd, Phoneme::Name(iter.phone), point(wcc.x+float(iter.beg)*wcc.w/viterbi->M, wcc.y-r*30));
    }

    if (interactive) {
      static Font *norm = app->fonts->Get(w->gl_h, FLAGS_font, "", 12, Color::grey70);
      segments->Frame(wcc, norm);

      // gui.mouse.Activate();
      // static Widget::Button next(&gui, 0, norm, "next", Callback([&](){interactive_done=1;}));
      // Box::FromScreen(2/3.0, .95, 1/3.0, .05, .0001, .0001);
      // next.Draw();

      segments->root->DrawDialogs();
    }

    MSleep(1);
  }

  sa.Unload();
}

int Resynthesize(Audio *s, const SoundAsset *sa) {
  if (!sa->wav) return -1;
  RingSampler::Handle B(sa->wav.get());
  auto m = Features::FromBuf(&B);
  auto f0 = F0Stream(&B, 0, FLAGS_feat_window, FLAGS_feat_hop);
  auto resynth = Features::Reverse(m.get(), B.Rate(), f0.get());
  if (resynth) {
    B = RingSampler::Handle(resynth.get());
    s->QueueMixBuf(&B);	
  }
  return 0;
}

}; // namespace LFL
