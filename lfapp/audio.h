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

#ifndef __LFL_LFAPP_AUDIO_H__
#define __LFL_LFAPP_AUDIO_H__
namespace LFL {

DECLARE_int(sample_secs);
DECLARE_int(sample_rate);
DECLARE_int(chans_out);
DECLARE_int(chans_in);
DECLARE_string(feat_type);
DECLARE_bool(feat_dither);
DECLARE_bool(feat_preemphasis);
DECLARE_double(feat_preemphasis_filter);
DECLARE_double(feat_minfreq);
DECLARE_double(feat_maxfreq);
DECLARE_int(feat_window);
DECLARE_int(feat_hop);
DECLARE_int(feat_melbands);
DECLARE_int(feat_cepcoefs);

extern const int feat_lpccoefs, feat_barkbands;
extern const double feat_progressbar_c;
extern const double feat_rastaB[], feat_rastaA[];

struct PowerDomain { enum { complex=1, abs=2, abs2=3, dB=4 }; };
struct FreqDomain { enum { hz=1, mel=2, }; };
struct F0EstmMethod { enum { fftbucket=1, xcorr=2, Default=2, cepstral=3, harmonicity=4 }; };
struct MixFlag { enum { Reset=1, Mix=2, DontQueue=4 }; };

struct Sample {
    enum { U8 =1, S16 =2, S32 =3, FLOAT =4, DOUBLE =5,
           U8P=6, S16P=7, S32P=8, FLOATP=9, DOUBLEP=10 };
    int Size(int fmt);
#ifdef LFL_FFMPEG
    static int FromFFMpegId(int fmt);
    static int ToFFMpegId(int fmt);
#endif
};

struct SystemAudio {
    static void PlaySoundEffect(SoundAsset *);
    static void PlayBackgroundMusic(SoundAsset *);
    static void SetVolume(int v);
    static int GetVolume();
    static int GetMaxVolume();
};

struct Audio : public Module {
    mutex inlock, outlock;
    RingBuf micL, micR;
    RingBuf *IL, *IR;
    RingBuf::Handle RL, RR;
    long long samples_read=0, samples_read_last=0;
    int outlast=0, mic_samples=0;
    SoundAsset *playing=0, *loop=0;
    deque<float> Out;
    Module *impl=0;
    Audio() : micL(FLAGS_sample_rate*FLAGS_sample_secs), micR(FLAGS_sample_rate*FLAGS_sample_secs),
    IL(&micL), IR(&micR), Out(32768)  {}

    int Init ();
    int Start();
    int Frame(unsigned);
    int Free ();
    void QueueMix(SoundAsset *sa, int flag=MixFlag::Reset, int offset=-1, int len=-1);
    void QueueMixBuf(const RingBuf::Handle *L, int channels=1, int flag=0);
    int Snapshot(SoundAsset *sa);
};

int Sinthesize(Audio *s, int hz1, int hz2, int hz3);
double LowPassFilter(int n, int i, int maxfreq);
double HighPassFilter(int n, int i, int minfreq);
float PseudoEnergy(const RingBuf::Handle *in, int window, int offset);
int ZeroCrossings(const RingBuf::Handle *in, int window, int offset);
RingBuf *Decimate(const RingBuf::Handle *in, int factor);
int CrossCorrelateTDOA(const RingBuf::Handle *a, const RingBuf::Handle *b, int window, int offset, int samps);
inline vector<double> PreEmphasisFilter() { vector<double> v { 1, FLAGS_feat_preemphasis_filter }; return v; }

struct AudioResampler {
    SwrContext *swr=0;
    int input_processed=0, input_chans=0, output_available=0, output_chans=0, output_rate=0;
    RingBuf *out=0;
    AudioResampler() { Clear(); }
    ~AudioResampler() { Close(); }

    void Clear();
    void Close();
    bool Opened() { return swr; }
    int Open(RingBuf *out, int  in_channels, int  in_sample_rate, int  in_sample_type,
                           int out_channels, int out_sample_rate, int out_sample_type);
    int Update(int samples, const short *in);
    int Update(int samples, RingBuf::Handle *L, RingBuf::Handle *R);
    int Update(int samples, const short **in, short *tmp, Time timestamp, int maxSamplesOut);

    static Time MonotonouslyIncreasingTimestamp(Time laststamp, Time stamp, double *step, int steps);
};

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
int FFT(const RingBuf::Handle *in, int i, int window, int hop, int fftlen, float *out, bool preemph=false, bool hamming=true, int scale=1);
int IFFT(const Complex *in, int i, int window, int hop, int fftlen, RingBuf::Handle *out);

/* fft filter */
void FFTFilterCompile(int n, double *filter);
int FFTFilter(const RingBuf::Handle *in, RingBuf::Handle *out, int window, int hop, const double *filter);

float FundamentalFrequency(const RingBuf::Handle *in, int window, int offset, int method=F0EstmMethod::Default);

Matrix *FFT2Mel(int outrows, double minfreq, double maxfreq, int fftlen, int samplerate);
Matrix *Mel2FFT(int outrows, double minfreq, double maxfreq, int fftlen, int samplerate);

Matrix *Spectogram(const RingBuf::Handle *in, Matrix *out, int window, int hop, int fftlen, bool preemph=false, int pd=PowerDomain::dB, int scale=1);
Matrix *F0Stream(const RingBuf::Handle *in, Matrix *out, int window, int hop, int method=F0EstmMethod::Default);

Matrix *PLP(const RingBuf::Handle *in, Matrix *out=0, vector<StatefulFilter> *rastaFilter=0, Allocator *alloc=0);
RingBuf *InvPLP(const Matrix *in, int samplerate, Allocator *alloc=0);

Matrix *MFCC(const RingBuf::Handle *in, Matrix *out=0, Allocator *alloc=0);
RingBuf *InvMFCC(const Matrix *in, int samplerate, const Matrix *f0=0);

}; // namespace LFL
#endif // __LFL_LFAPP_AUDIO_H__
