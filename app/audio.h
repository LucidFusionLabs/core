/*
 * $Id: audio.h 1335 2014-12-02 04:13:46Z justin $
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

#ifndef LFL_CORE_APP_AUDIO_H__
#define LFL_CORE_APP_AUDIO_H__
namespace LFL {

DECLARE_int(sample_rate);
DECLARE_int(sample_secs);
DECLARE_int(chans_in);
DECLARE_int(chans_out);
DECLARE_int(audio_input_device); 
DECLARE_int(audio_output_device); 
DECLARE_bool(print_audio_devices);

struct MixFlag { enum { Reset=1, Mix=2, DontQueue=4 }; };
struct FreqDomain { enum { hz=1, mel=2, }; };
struct PowerDomain { enum { complex=1, abs=2, abs2=3, dB=4 }; };
struct F0EstmMethod { enum { fftbucket=1, xcorr=2, Default=2, cepstral=3, harmonicity=4 }; };

struct Sample {
  enum { U8 =1, S16 =2, S32 =3, FLOAT =4, DOUBLE =5,
         U8P=6, S16P=7, S32P=8, FLOATP=9, DOUBLEP=10 };
  int Size(int fmt);
};

struct Audio : public Module {
  mutex inlock, outlock;
  unique_ptr<RingBuf> IL, IR;
  RingBuf::Handle RL, RR;
  long long samples_read=0, samples_read_last=0;
  int outlast=0, mic_samples=0;
  SoundAsset *playing=0, *loop=0;
  deque<float> Out;
  unique_ptr<Module> impl;
  Audio() : Out(32768)  {}

  int Init ();
  int Start();
  int Frame(unsigned);
  int Free ();
  void QueueMix(SoundAsset *sa, int flag=MixFlag::Reset, int offset=-1, int len=-1);
  void QueueMixBuf(const RingBuf::Handle *L, int channels=1, int flag=0);
  int Snapshot(SoundAsset *sa);
  static double VisualDelay();
};

int Sinthesize(Audio *s, int hz1, int hz2, int hz3);
double LowPassFilter(int n, int i, int maxfreq);
double HighPassFilter(int n, int i, int minfreq);
inline vector<double> PreEmphasisFilter(double fv) { vector<double> v { 1, fv }; return v; }
float PseudoEnergy(const RingBuf::Handle *in, int window, int offset);
int ZeroCrossings(const RingBuf::Handle *in, int window, int offset);
RingBuf *Decimate(const RingBuf::Handle *in, int factor);
int CrossCorrelateTDOA(const RingBuf::Handle *a, const RingBuf::Handle *b, int window, int offset, int samps);

struct AudioResamplerInterface {
  RingBuf *out=0;
  int input_processed=0, input_chans=0, output_available=0, output_chans=0, output_rate=0;
  virtual ~AudioResamplerInterface() {}

  virtual bool Opened() const = 0;
  virtual int Open(RingBuf *out, int  in_channels, int  in_sample_rate, int in_sample_type,
                   int out_channels, int out_sample_rate, int out_sample_type) = 0;
  virtual int Update(int samples, const short *in) = 0;
  virtual int Update(int samples, RingBuf::Handle *L, RingBuf::Handle *R) = 0;
  virtual int Update(int samples, const short *const *inbuf, short *rsout, microseconds timestamp, int max_samples_out) = 0;

  static microseconds MonotonouslyIncreasingTimestamp(microseconds laststamp, microseconds stamp, microseconds *step, int steps);
};

AudioResamplerInterface *CreateAudioResampler();

struct StatefulFilter {
  double state[32];
  int size=0, next=0, filterLenB=0, filterLenA=0, samples=0;
  const double *filterB=0, *filterA=0;

  void Open(int FilterLenB, const double *FilterB, int FilterLenA, const double *FilterA);

  double Filter(double sample);
  int Filter(const RingBuf::Handle *in, RingBuf::Handle *out, int start, int length=0);
};

/* stateless filters */
double Filter(const RingBuf::Handle *in, int offset, int filterlen, const double *filter, bool nohistory=0);
double Filter(const RingBuf::Handle *in, int offset1, const RingBuf::Handle *out, int offset2, int filterlenB, const double *filterB, int filterlenA, const double *filterA, double initialcondition, bool nohistory);

int Filter(const RingBuf::Handle *in, RingBuf::Handle *out, int filterlen, const double *filter, bool nohistory=0);
int Filter(const RingBuf::Handle *in, RingBuf::Handle *out, int filterlenB, const double *filterB, int filterlenA, const double *filterA, int start, double *ic, double iclen, bool nohistory);

/* streaming overlap-add fft/ifft */
int FFT(const RingBuf::Handle *in, int i, int window, int hop, int fftlen, float *out,
        const vector<double> &preemph_filter=vector<double>(), bool hamming=true, int scale=1);
int IFFT(const Complex *in, int i, int window, int hop, int fftlen, RingBuf::Handle *out);

/* fft filter */
void FFTFilterCompile(int n, double *filter);
int FFTFilter(const RingBuf::Handle *in, RingBuf::Handle *out, int window, int hop, const double *filter);

Matrix *Spectogram(const RingBuf::Handle *in, Matrix *out, int window, int hop, int fftlen,
                   const vector<double> &preemph=vector<double>(), int pd=PowerDomain::dB, int scale=1);
RingBuf *ISpectogram(const Matrix *in, int window, int hop, int fftlen, int samplerate);

Matrix *F0Stream(const RingBuf::Handle *in, Matrix *out, int window, int hop, int method=F0EstmMethod::Default);
float FundamentalFrequency(const RingBuf::Handle *in, int window, int offset, int method=F0EstmMethod::Default);

}; // namespace LFL
#endif // LFL_CORE_APP_AUDIO_H__
