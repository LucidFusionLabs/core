/*
 * $Id: audio.cpp 1330 2014-11-06 03:04:15Z justin $
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

namespace LFL {
DEFINE_int(sample_rate, 16000, "Audio sample rate");
DEFINE_int(sample_secs, 3, "Seconds of RingSampler audio");
DEFINE_int(chans_in, -1, "Audio input channels");
DEFINE_int(chans_out, -1, "Audio output channels");
DEFINE_int(audio_input_device, -1, "Audio input device index");
DEFINE_int(audio_output_device, -1, "Audio output device index");
DEFINE_bool(print_audio_devices, false, "Print audio device list");

int Sample::Size(int fmt) {
  switch (fmt) {
    case Sample::U8:     case Sample::U8P:     return 1;
    case Sample::S16:    case Sample::S16P:    return 2;
    case Sample::S32:    case Sample::S32P:    return 4;
    case Sample::FLOAT:  case Sample::FLOATP:  return 4;
    case Sample::DOUBLE: case Sample::DOUBLEP: return 8;
    default: return ERRORv(0, "unknown sample fmt: ", fmt);
  }
}

int Audio::Init() {
  INFO("Audio::Init()");
  impl = CreateAudioModule(this);
  if (!impl) FLAGS_lfapp_audio = false;
  if (impl && impl->Init()) return -1;

  IL = make_unique<RingSampler>(FLAGS_sample_rate*FLAGS_sample_secs);
  IR = make_unique<RingSampler>(FLAGS_sample_rate*FLAGS_sample_secs);
  RL = RingSampler::Handle(IL.get());
  RR = RingSampler::Handle(IR.get());

  if (impl && impl->Start()) return -1;
  if (FLAGS_chans_out < 0) FLAGS_chans_out = 0;
  return 0;
}

int Audio::Start() {
  return 0;
}

int Audio::Frame(unsigned clicks) {
  {
    ScopedMutex ML(inlock);

    /* frame align stream handles */
    RL.next = IL->ring.back;
    RR.next = IR->ring.back;

    mic_samples = samples_read - samples_read_last;
    samples_read_last = samples_read;
  }

  const int refillWhen = FLAGS_sample_rate*FLAGS_chans_out/2;

  if (app->asset_loader->movie_playing) {
    app->asset_loader->movie_playing->Play(0);
  } else if ((playing || loop) && Out.size() < refillWhen) {
    // QueueMix(playing ? playing : loop, !playing ? MixFlag::Reset : 0, -1, -1);
    app->RunInMainThread(bind(&Audio::QueueMix, this,
                              playing ? playing : loop, !playing ? MixFlag::Reset : 0, -1, -1));
  }
  return 0;
}

int Audio::Free() {
  if (impl) impl->Free();
  return 0;
}

void Audio::QueueMixBuf(const RingSampler::Handle *B, int channels, int flag) {
  ScopedMutex ML(outlock);
  outlast = B->Len() * FLAGS_chans_out;
  bool mix = flag & MixFlag::Mix, dont_queue = flag & MixFlag::DontQueue;

  for (int i=0; i<B->Len(); i++) {
    if (channels == 1) {
      for (int j=0; j<FLAGS_chans_out; j++) {
        int ind = i*FLAGS_chans_out + j;
        bool mixable = mix && ind < Out.size();
        if (dont_queue && !mixable) return;
        if (mixable) Out[ind] = Clamp(Out[ind] + B->Read(i), -32767, 32768);
        else Out.push_back(B->Read(i));
      }
    }
    else if (channels == FLAGS_chans_out) {
      bool mixable = mix && i < Out.size();
      if (dont_queue && !mixable) return;
      if (mixable) Out[i] = Clamp(Out[i] + B->Read(i), -32767, 32768);
      else Out.push_back(B->Read(i));
    }
    else FATAL("QueueMix unsupported channel combo ", channels, ", ", FLAGS_chans_out);
  }
}

void Audio::QueueMix(SoundAsset *sa, int flag, int begin, int len) {
  if (!sa->wav) return ERROR("QueueMix: asset(", sa->name, ") missing wave data", sa->name);
  if (sa->refill) {
    int samples = sa->refill(sa, flag & MixFlag::Reset);
    playing = (samples == SoundAsset::Size(sa)) ? sa : 0;
  }
  RingSampler::Handle B(sa->wav.get(), begin, len);
  QueueMixBuf(&B, sa->channels, flag);
}

int Audio::Snapshot(SoundAsset *out) {
  ScopedMutex ML(inlock);
  RingSampler::Handle(out->wav.get()).CopyFrom(&RL);
  return 0;
}

double Audio::VisualDelay() {
#ifdef LFL_APPLE
  return .1;
#else
  return .4;
#endif
}

int Sinthesize(Audio *s, int hz1, int hz2, int hz3) {
  RingSampler tonebuf(FLAGS_sample_rate, FLAGS_sample_rate);
  RingSampler::Handle tone(&tonebuf);
  double f = 2 * M_PI / tone.Rate();
  for (int i=0; i<tone.Len(); i++) {
    double v=sin(f*hz1*i);
    if (hz2) v+=sin(f*hz2*i);
    if (hz3) v+=sin(f*hz3*i);
    tone.Write(v);
  }
  s->QueueMixBuf(&tone);
  return 0;
}

double LowPassFilter(int n, int i, int maxfreq) {
  int ind = i-(n-1)/2;
  double R=2.0*maxfreq/FLAGS_sample_rate;
  return R*Sinc(ind*R);
}

double HighPassFilter(int n, int i, int minfreq) {
  int ind = i-(n-1)/2;
  double impulse = ind ? 0 : 1;
  return impulse - LowPassFilter(n,i,minfreq);
}

float PseudoEnergy(const RingSampler::Handle *in, int window, int offset) {
  float e=0;
  for (int i = 0; i < window; i++) e += fabs(in->Read(offset + i) * 32768.0);
  return e;
}

int ZeroCrossings(const RingSampler::Handle *in, int window, int offset) {
  int zcr=0;
  float last = in->Read(offset);
  for (int i = 1; i < window; i++) {
    float cur = in->Read(offset + i);
    if ((cur < 0 && last >= 0) || (cur >= 0 && last < 0)) zcr++;
    last = cur;
  }
  return zcr;
}

RingSampler *Decimate(const RingSampler::Handle *inh, int factor) {
  const int SPS=inh->Rate(), SPB=inh->Len();
  unique_ptr<RingSampler> outbuf = make_unique<RingSampler>(SPS/factor, SPB/factor);
  RingSampler::Handle out(outbuf.get());

  double sum = 0;
  for (int i = 0, count = 0; i < SPB; i++) {
    sum += inh->Read(-SPB + i);
    if (++count >= factor) {
      out.Write(sum / count);
      count = 0; 
      sum = 0;
    }
  }
  return outbuf.release();
}

int CrossCorrelateTDOA(const RingSampler::Handle *a, const RingSampler::Handle *b, int window, int offset, int samps) {
  float max=0; int maxind=0;
  float res[100];

  /* a : b */
  for (int i=0; i<window; i++) {
    float sum=0;
    for (int j=0; j<samps; j++) sum += a->Read(offset+j) * b->Read(offset+i+j);
    if (sum > max) { max=sum; maxind=i; }
    res[i] = max;
  }

  /* b : a */
  for (int i=1; i<window; i++) {
    float sum=0;
    for (int j=0; j<samps; j++) sum += b->Read(offset+j) * a->Read(offset+i+j);
    if (sum > max) { max=sum; maxind=-i; }
  }

  return maxind;
}

microseconds AudioResamplerInterface::MonotonouslyIncreasingTimestamp(microseconds laststamp, microseconds stamp, microseconds *step, int steps) {
  if (laststamp > stamp) {
    microseconds end = stamp + (steps-1) * (*step);
    *step = max(microseconds(1), (end - laststamp) / steps);
    stamp = laststamp + *step;
  }
  return stamp;
}

/* StatefulFilter */

void StatefulFilter::Open(int FilterLenB, const double *FilterB, int FilterLenA, const double *FilterA) {
  next = samples = 0;

  filterB    = FilterB;    filterA    = FilterA;
  filterLenB = FilterLenB; filterLenA = FilterLenA;

  size = max(filterLenA, filterLenB);
  memset(state, 0, sizeof(state));
}

double StatefulFilter::Filter(double sample) {
  for (int i=0; i<filterLenB; i++) {
    int si = (next+i) % size;
    state[si] += sample * filterB[i];
  }
  double ret = state[next];

  for (int i=1; i<filterLenA; i++) {
    int si = (next+i) % size;
    state[si] -= ret * filterA[i];
  }
  state[next] = 0;

  samples++;
  next = (next+1) % size;
  return ret;
}

int StatefulFilter::Filter(const RingSampler::Handle *in, RingSampler::Handle *out, int start, int length) {
  int count = 0;
  if (!length) length = in->Len();
  for (int i=start; i<length; i++) {
    double yi = Filter(in->Read(i));
    if (out) out->Write(yi);
    count++;
  }
  return count;
}

/* stateless filters */

double Filter(const RingSampler::Handle *in, int offset1,
              const RingSampler::Handle *out, int offset2,
              int filterlenB, const double *filterB,
              int filterlenA, const double *filterA,
              double initialcondition, bool nohistory) {    
  double yi = initialcondition;
  for (int k=0; k<filterlenB; k++) {
    double xk = nohistory && offset1-k < 0 ? 0 : in->Read(offset1-k);
    yi += xk * filterB[k];
  }
  bool IIR = out && filterA && filterlenA;
  if (!IIR) return yi;

  for (int k=0; k<filterlenA; k++) {
    double yk = nohistory && offset2-k < 0 ? 0 : out->Read(offset2-k);
    yi -= yk * filterA[k];
  }
  return yi;
}

double Filter(const RingSampler::Handle *in, int offset, int filterlen, const double *filter, bool nohistory) { return LFL::Filter(in, offset, 0, 0, filterlen, filter, 0, 0, 0, nohistory); }

int Filter(const RingSampler::Handle *in, RingSampler::Handle *out, int filterlenB, const double *filterB, int filterlenA, const double *filterA, int start, double *ic, double iclen, bool nohistory) {
  for (int i=start; i<in->Len(); i++) {
    double yi = LFL::Filter(in, i, out, i, filterlenB, filterB, filterlenA, filterA, (iclen-- > 0) ? *ic++ : 0, nohistory);
    out->Write(yi);
  }
  return 0;
}

int Filter(const RingSampler::Handle *in, RingSampler::Handle *out, int filterlen, const double *filter, bool nohistory) { return LFL::Filter(in, out, filterlen, filter, 0, 0, 0, 0, 0, nohistory); }

/* streaming overlap-add fft/ifft */

int FFT(const RingSampler::Handle *in, int i, int window, int hop, int fftlen, float *out,
        const vector<double> &pef, bool doHamming, int scale) {
  int padding=fftlen-window, split=window/2, ind;
  if (window % 2 != 0) return -1;

  for (int j=0; j<fftlen; j++) {			
    if      (j < split)           { ind = i*hop + split + j; }
    else if (j < split + padding) { out[j] = 0; continue; }
    else                          { ind = i*hop + j - split - padding; }

    double v = pef.size() ? Filter(in, ind, pef.size(), &pef[0]) : in->Read(ind);

    if (doHamming) v *= Hamming(window, ind - i*hop);
    out[j] = v * scale;
  }

  FFT(out, fftlen);
  return 0;
}

int IFFT(const Complex *in, int i, int window, int hop, int fftlen, RingSampler::Handle *out) {
  Complex zero={0,0};
  vector<float> fftbuf(fftlen*2);
  int split=fftlen/2;

  for (int j=0; j<split; j++) {
    fftbuf[j*2]   = in[j].r;
    fftbuf[j*2+1] = in[j].i;
  }
  for (int j=0; j<split; j++) {
    Complex val = !j ? zero : in[split-j].Conjugate();
    fftbuf[(split+j)*2]   = val.r;
    fftbuf[(split+j)*2+1] = val.i;
  }

  IFFT(&fftbuf[0], fftlen);

  for (int j=0; j<split; j++) *out->Index(i*hop +         j) += fftbuf[split+j];
  for (int j=0; j<split; j++) *out->Index(i*hop + split + j) += fftbuf[j];

  double scalef = (1.0/fftlen) * (double(window)/fftlen);
  for (int j=0; j<hop; j++) *out->Index(i*hop + j) *= scalef;
  return 0;
}

/* fft filter */

void FFTFilterCompile(int n, double *filter) {
  RingSampler filtbuf(FLAGS_sample_rate, n);
  RingSampler::Handle filtcoef(&filtbuf);
  for (int i=0; i<n; i++) filtcoef.Write(filter[i]);

  vector<float> fftbuf(n);
  FFT(&filtcoef, 0, n, 0, n, &fftbuf[0]);
  for (int i=0; i<n; i++) filter[i] = fftbuf[i];
}

int FFTFilter(const RingSampler::Handle *in, RingSampler::Handle *out, int window, int hop, const double *filter) {
  int frames = (in->Len() - (window - hop)) / hop;
  vector<float> fftbuf(window);
  vector<Complex> ifftb(window/2);

  for (int j=0; j<in->Len(); j++) *out->Index(j) = 0;

  for (int i=0; i<frames; i++) {
    if (FFT(in, i, window, hop, window, &fftbuf[0])) return -1;

    for (int j=0; j<window/2; j++) {
      Complex x={fft_r(&fftbuf[0], window, j), fft_i(&fftbuf[0], window, j)};
      Complex h={fft_r(filter,     window, j), fft_i(filter,     window, j)};
      ifftb[j] = Complex::Mult(x, h); /* cyclic convolution */
    }

    if (IFFT(&ifftb[0], i, window, hop, window, out)) return -1;
  }
  return 0;
}

Matrix *Spectogram(const RingSampler::Handle *in, Matrix *out, int window, int hop, int fftlen,
                   const vector<double> &preemph, int pd, int scale) {
  bool complex = (pd==PowerDomain::complex);
  int frames = (in->Len() - (window - hop)) / hop;
  vector<float> fftbuf(fftlen);

  Matrix *m;
  if (out) { m=out; if (m->M != frames || m->N != fftlen/2 || (m->flag&Matrix::Flag::Complex && !complex)) return 0; }
  else m = new Matrix(frames, fftlen/2, 0.0, complex);

  for (int i=0; i<frames; i++) {
    double *row = m->row(i);
    if (FFT(in, i, window, hop, fftlen, &fftbuf[0], preemph, true, scale)) { if (!out) delete m; return 0; }

    for (int j=0; j<fftlen/2; j++) {
      if (pd == PowerDomain::complex) {
        row[j*2 + 0] = fft_r(&fftbuf[0], fftlen, j);
        row[j*2 + 1] = fft_i(&fftbuf[0], fftlen, j);
        continue;
      }

      if      (pd == PowerDomain::abs)  row[j] = sqrt(fft_abs2(&fftbuf[0], fftlen, j));
      else if (pd == PowerDomain::abs2) row[j] = fft_abs2(&fftbuf[0], fftlen, j);      
      else if (pd == PowerDomain::dB)   row[j] = AmplitudeRatioDecibels(sqrt(fft_abs2(&fftbuf[0], fftlen, j)), 1);
    }
  }

  return m;
}

RingSampler *ISpectogram(const Matrix *in, int window, int hop, int fftlen, int samplerate) {
  unique_ptr<RingSampler> outbuf = make_unique<RingSampler>(samplerate, fftlen + hop * (in->M-1));
  RingSampler::Handle out(outbuf.get());

  for (int i=0; i<in->M; i++) {
    if (IFFT(in->crow(i), i, window, hop, fftlen, &out)) return nullptr;
  }
  return outbuf.release();
}

Matrix *F0Stream(const RingSampler::Handle *in, Matrix *out, int window, int hop, int method) {
  int frames = (in->Len() - (window - hop)) / hop;

  Matrix *m;
  if (out) { m=out; if (m->M != frames || m->N != 1) return 0; }
  else m = new Matrix(frames, 1);

  for (int i=0; i<frames; i++) {
    double *row = m->row(i); 
    row[0] = FundamentalFrequency(in, window, i*hop, method);
  }
  return m;
}

float FundamentalFrequency(const RingSampler::Handle *in, int window, int offset, int method) {
  if (method == F0EstmMethod::fftbucket) { /* max(fft.bucket) */
    vector<float> samp(window);
    FFT(in, 1, window, offset, window, &samp[0]);

    int maxind=0; double max=0;
    for (int i=1; i<window/2; i++) {
      float abs = sqrt(fft_abs2(&samp[0], window, i));
      if (abs > max) { max=abs; maxind=i; }
    }
    return in->Rate() * maxind / (window/2.0);
  }
  else if (method == F0EstmMethod::xcorr) { /* auto correlation */

    /* zero pad */
    RingSampler zpi(in->Rate(), window*2);
    RingSampler::Handle zpin(&zpi);
    for (int i=0; i<window; i++) *zpin.Index(i) = in->Read(i+offset);

    /* fft */
    vector<float> buf(window*2);
    FFT(&zpin, 0, window*2, 0, window*2, &buf[0]);

    /* to spectrum */
    fft_r(&buf[0], window*2, 0) = pow(fft_r(&buf[0], window*2, 0), 2);
    fft_i(&buf[0], window*2, 0) = pow(fft_i(&buf[0], window*2, 0), 2);
    for (int i=1; i<window; i++) {
      fft_r(&buf[0], window*2, i) = fft_abs2(&buf[0], window*2, i);
      fft_i(&buf[0], window*2, i) = 0;
    }

    /* ifft */
    IFFT(&buf[0], window);

    /* auto correlation */
    int minlag = min(in->Rate() / 500, window*2); /* 500 hz */
    int maxlag = min(in->Rate() / 50 + 1, window*2); /* 50 hz */
    vector<float> xcorr(maxlag);

    /* divide by window for ifft and zxc for auto-correlation coefficients */
    float zxc = buf[0] / window;
    for (int i=0; i<maxlag; i++)
      xcorr[i] = buf[i] / window / zxc;

    /* max(xcorr[minlag,maxlag]) */
    int maxind=-1; double max=-INFINITY;
    for (int i=minlag; i<maxlag; i++)
      if (xcorr[i] > max) { max=xcorr[i]; maxind=i; }

    /* xcorr index to hz */
    return float(in->Rate()) / (maxind+1);
  }
  else if (method == F0EstmMethod::cepstral) { /* cepstral analysis */
    vector<float> buf(window);
    FFT(in, 1, window, offset, window, &buf[0]);

    /* to log spectrum */
    RingSampler spe(in->Rate(), window);
    RingSampler::Handle spec(&spe);
    *spec.Index(0) = log(fabs(fft_r(&buf[0], window, 0)));
    for (int i=0; i<window/2; i++) *spec.Index(i) = log(sqrt(fft_abs2(&buf[0], window, i)));
    *spec.Index(window/2) = log(fabs(fft_i(&buf[0], window, 0)));
    for (int i=1; i<window/2; i++) *spec.Index(window/2+i) = *spec.Index(window/2-i);

    /* fft */
    FFT(&spec, 0, window, 0, window, &buf[0]); 

    /* cepstrum */
    int minlag = min(in->Rate() / 1000, window/2); /* 1000 hz */
    int maxlag = min(in->Rate() / 50, window/2); /* 50 hz */

    vector<float> cepstrum(window/2);
    for (int i=0; i<window/2; i++) cepstrum[i] = sqrt(fft_abs2(&buf[0], window, i));

    /* max(cepstrum[minlag,maxlag]) */
    int maxind=-1; double max=-INFINITY;
    for (int i=minlag; i<maxlag; i++) {
      if (cepstrum[i] > max) { max=cepstrum[i]; maxind=i; }
    }

    /* cepstrum index to hz */
    return float(in->Rate()) / (maxind-1);
  }
  else if (method == F0EstmMethod::harmonicity) { /* harmonicity */
    vector<float> buf(window);
    FFT(in, 1, window, offset, window, &buf[0]);
  }
  return -1;
}

}; // namespace LFL
