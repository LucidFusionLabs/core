/*
 * $Id: fs.cpp 1336 2014-12-08 09:29:59Z justin $
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
#include "lfapp/network.h"
#include "lfapp/ipc.h"
#include "ml/hmm.h"
#include "speech/speech.h"
#include "fs.h"

#include "ml/corpus.h"
#include "ml/counter.h"
#include "nlp/corpus.h"
#include "nlp/lm.h"
#include "speech/wfst.h"
#include "speech/recognition.h"

namespace LFL {
#ifdef _WIN32
DEFINE_bool  (install,               false,            "Win32 Register Server");
DEFINE_bool  (uninstall,             false,            "Win32 Unregister Server");
#endif                               

DEFINE_string(decode,                "",               "Filename to decode");
DEFINE_string(modeldir,              "assets/",        "Model directory");

DEFINE_bool  (fg,                    false,            "Run in foreground");
DEFINE_bool  (ssl,                   false,            "Run https server");
DEFINE_int   (WantIter,              -1,               "Model iteration");
DEFINE_int   (UseTransition,         1,                "Use transition probabilities in training");
DEFINE_double(BeamWidth,             256,              "Beam search width");
DEFINE_double(LanguageModelWeight,   2,                "Language model weight");
DEFINE_double(WordInsertionPenalty,  0,                "Word insertion penalty");

struct SpeechDecodeSession : public HTTPServer::Resource {
    static const int DeltaWindow=7, NBest=1, MaxResponseWords=128, MaxResponseTranscript=1024;

    RecognitionModel *model;
    FixedAlloc<65536*2> alloc;
    FixedAlloc<32*1024*1024> once;
    int feats_dim, feats_available, feats_normalized, feats_processed, feats_seqL, time_index;
    RingBuf inB, featB, varB, backtraceB, viterbiB;
    double *rollingMean, *rollingVariance;

    HMM::ObservationPtr obptr;
    HMM::TokenBacktracePtr<HMM::Token> btptr;
    RecognitionHMM::DynamicComposer transit;
    RecognitionHMM::Emission emit;
    RecognitionHMM::TokenPasser beam;

    SpeechDecodeSession(RecognitionModel *Model, int feature_rate, int sample_secs) : model(Model), feats_dim(Features::dimension()), feats_available(0), feats_normalized(0), feats_processed(0), feats_seqL(0), time_index(0),
        inB       (feature_rate, feature_rate*FLAGS_sample_secs, feats_dim*sizeof(double),           &once),
        featB     (feature_rate, feature_rate*FLAGS_sample_secs, feats_dim*3*sizeof(double),         &once),
        varB      (feature_rate, feature_rate*FLAGS_sample_secs, feats_dim*sizeof(double),           &once),
        backtraceB(feature_rate, feature_rate*FLAGS_sample_secs, FLAGS_BeamWidth*sizeof(HMM::Token), &once),
        viterbiB  (feature_rate, feature_rate*FLAGS_sample_secs, sizeof(HMM::Token),                 &once),
        rollingMean    ((double*)once.Malloc(sizeof(double)*feats_dim)),
        rollingVariance((double*)once.Malloc(sizeof(double)*feats_dim)),
        transit(model, FLAGS_UseTransition), emit(model, &obptr, &transit), beam(model, 1, NBest, FLAGS_BeamWidth, &btptr)
    {
        memset(rollingMean,     0, sizeof(double)*feats_dim);
        memset(rollingVariance, 0, sizeof(double)*feats_dim);
    }

    int FeatureRollingMeanSamples() { return min(feats_available, featB.ring.size); }

    void FeatureRollingMeanAndVariance(double *rmean, double *rvariance) {
        int samples = FeatureRollingMeanSamples();
        Vector::Assign(rmean, rollingMean, feats_dim);
        Vector::Assign(rvariance, rollingVariance, feats_dim);
        Vector::Div(rmean, samples, feats_dim);
        Vector::Div(rvariance, samples, feats_dim);
        for (int i=0; i<feats_dim; i++) rvariance[i] = sqrt(rvariance[i]);
    }

    void DeltaCoefficients(int D, int frame_n2, int frame, int frame_p2) {
        Features::deltaCoefficients(D, (double*)featB.Read(frame_n2), (double*)featB.Read(frame), (double*)featB.Read(frame_p2));
    }

    void DeltaDeltaCoefficients(int D, int frame_n3, int frame_n1, int frame, int frame_p1, int frame_p3) {
        Features::deltaDeltaCoefficients(D, (double*)featB.Read(frame_n3), (double*)featB.Read(frame_n1), (double*)featB.Read(frame),
                                         (double*)featB.Read(frame_p1), (double*)featB.Read(frame_p3));
    }

    void PatchDeltaCoefficients(int D, int frame_in, int frame_out1, int frame_out2) {
        Features::patchDeltaCoefficients(D, &((double*)featB.Read(frame_in))[D], &((double*)featB.Read(frame_out1))[D], &((double*)featB.Read(frame_out2))[D]);
    }
    
    void PatchDeltaDeltaCoefficients(int D, int frame_in, int frame_out1, int frame_out2, int frame_out3) {
        Features::patchDeltaDeltaCoefficients(D, &((double*)featB.Read(frame_in))[D*2], &((double*)featB.Read(frame_out1))[D*2],
                                              &((double*)featB.Read(frame_out2))[D*2], &((double*)featB.Read(frame_out3))[D*2]);
    }

    int DecodeFrame(int frame, bool init=false) {
        if (init) time_index = 0;
        int t = time_index++;
        obptr.v = (double*)featB.Read(frame); 
        btptr.v = (HMM::Token*)backtraceB.Read(frame);
        return HMM::forward(&beam, &transit, &emit, &beam, &beam, 0, init, t);
    }
    
    HTTPServer::Response Request(Connection *, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) {

        if (postlen < sizeof(AcousticEventHeader)) return HTTPServer::Response::_400;
        AcousticEventHeader *AEH = (AcousticEventHeader *)postdata;

        if (postlen != sizeof(AcousticEventHeader) + AEH->m * AEH->n * sizeof(float))
        if (AEH->m > featB.ring.size || AEH->n != feats_dim) return HTTPServer::Response::_400;        
        float *featureData = (float*)(AEH+1);

        StringPiece flushArg;
        if (args) HTTP::GrepURLArgs(args, 0, 1, "flush", &flushArg);

        return Request<float>(featureData, AEH->m, AEH->n, AEH->timestamp, !AEH->m || flushArg.data());
    }

    template <class X> HTTPServer::Response Request(X *featureData, int M, int N, long long timestamp, bool flush) {
        alloc.Reset();

        for (int i=0; i<M; i++) {
            double *feat = (double*)featB.Write(RingBuf::Stamp, microseconds(timestamp+i));
            double *in = (double*)inB.Write();
            double *var = (double*)varB.Write();
            int rollingMeanSamples = FeatureRollingMeanSamples();

            for (int j=0; j<N; j++) {
                float ov = in[j];
                float nv = featureData[i*N + j];

                in[j] = nv;
                feat[j] = nv;
                rollingMean[j] += nv - ov;

                float mean = rollingMean[j] / max(rollingMeanSamples, 1);
                float od = var[j];
                float nd = pow(mean - nv, 2);

                var[j] = nd;
                rollingVariance[j] += nd - od;
            }

            backtraceB.Write();
            *(HMM::Token*)viterbiB.Write() = HMM::Token();
            feats_available++;
        }

        if (Features::MeanNormalization || Features::VarianceNormalization) {
            double *rmean = (double*)alloc.Malloc(feats_dim*sizeof(double));
            double *rvariance = (double*)alloc.Malloc(feats_dim*sizeof(double));
            FeatureRollingMeanAndVariance(rmean, rvariance);

            for (/**/; feats_available > feats_normalized; feats_normalized++) {
                double *feat = (double*)featB.Read(-feats_available+feats_normalized);
                Features::meanAndVarianceNormalization(feats_dim, feat, rmean, rvariance);
            }
        }
        else feats_normalized = feats_available;

        AcousticEventHeader *response = (AcousticEventHeader*)alloc.Malloc(sizeof(AcousticEventHeader) + sizeof(float)*MaxResponseWords + MaxResponseTranscript);
        response->m = 0; response->n = 2;
        float *responseV = (float*)(response+1);
        int prior_feats_processed = feats_processed, decode_end, endseq=0, endindex;

        for (;;) {
            int ll_feats_processed = feats_processed, ls = min(DeltaWindow/2, feats_seqL);
            if (feats_available + flush < feats_processed + DeltaWindow - ls) break;

            int behind = feats_available - feats_processed, frame=-behind + DeltaWindow/2 - ls, D=feats_dim, lc=0, rc=0;
            for (int i=0; i<DeltaWindow/2+1 && featB.ReadTimestamp(frame-i) == featB.ReadTimestamp(frame-i-1) + microseconds(1); i++) lc++;
            for (int i=0; i<DeltaWindow/2   && featB.ReadTimestamp(frame+i) == featB.ReadTimestamp(frame+i+1) - microseconds(1); i++) rc++;

            if (lc < ls) { feats_seqL = lc; continue; }

            if (lc >= 3 && rc >= 3) {
                DeltaCoefficients(D, frame-2, frame, frame+2);
                DeltaDeltaCoefficients(D, frame-3, frame-1, frame, frame+1, frame+3);
            }

            if (lc == 3 && rc >= 3) {
                DeltaCoefficients(D, frame-3, frame-1, frame+1);
                PatchDeltaCoefficients(D, frame-1, frame-2, frame-3);
                PatchDeltaDeltaCoefficients(D, frame, frame-1, frame-2, frame-3);

                DecodeFrame(frame-3, true);
                for (int i=1; i<3; i++) endindex = DecodeFrame(frame-3+i);

                feats_processed += 3;
                feats_seqL = 3;
                decode_end = frame-1;
            }

            if (lc >= 3 && rc >= 3) {
                endindex = DecodeFrame(frame);

                feats_processed++;
                feats_seqL++;
                decode_end = frame;
            }

            if (rc == 2 && lc >= 4)  {
                DeltaCoefficients(D, frame-2, frame, frame+2);
                PatchDeltaCoefficients(D, frame, frame+1, frame+2);
                PatchDeltaDeltaCoefficients(D, frame-1, frame, frame+1, frame+2);

                for (int i=0; i<3; i++) endindex = DecodeFrame(frame+i);
                feats_processed += 3;
                feats_seqL += 3;
                decode_end = frame+2;
                endseq = 1;
                break;
            }

            if (feats_processed == ll_feats_processed) {
                ERROR("feat ", featB.ReadTimestamp(frame), " lacks context (lc=", lc, ", rc=", rc, "), skipping");
                feats_processed++; prior_feats_processed++;
            }
        }

        string transcript;
        if (feats_processed > prior_feats_processed) {
            int seql = min(feats_seqL, featB.ring.size+decode_end+1);
            response->timestamp = featB.ReadTimestamp(decode_end-seql+1).count();
            RingBuf::MatrixHandleT<HMM::Token> backtraceM(&backtraceB, backtraceB.ring.back+decode_end-seql+1, seql);
            RingBuf::MatrixHandleT<HMM::Token> viterbiM  (&  viterbiB,   viterbiB.ring.back+decode_end-seql+1, seql);
            int mergeind = HMM::Token::tracePath(&viterbiM, &backtraceM, endindex, seql, true);

            string ats;
            Recognizer::WordIter iter(model, &viterbiM);
            for (/**/; !iter.done(); iter.next()) {
                if (iter.end <= mergeind) continue;
                addResponseWord(response, responseV, iter, transcript, ats);
            }

            if (!transcript.size() && endseq) {
                int topn = FLAGS_BeamWidth * .2;
                for (int i=0; i<topn; i++) {
                    HMM::Token *end = &backtraceM.row(backtraceM.M-1)[endindex+i];
                    int source = end->ind / NBest, out;
                    if (!(out = model->predict(source))) continue;

                    iter.beg = seql - min(seql, end->steps);
                    iter.end = seql;
                    iter.word = model->recognitionNetwork.B->name(out);

                    addResponseWord(response, responseV, iter, transcript, ats, "PREDICTED");
                    break;
                }
                if (!ats.size()) StrAppend(&ats, "ZERO PRED ", endindex);
            }
            INFOf("-- %s (mergeind = %d, seql = %d, RM = %d, TS=%lld flush=%d, PTS=%lld fa=%d fp=%d)", ats.c_str(), mergeind, seql, response->m, response->timestamp, flush, timestamp, feats_available, feats_processed);
        }

        if (transcript.size() > MaxResponseTranscript) { ERRORf("overflow %d > %d", transcript.size(), MaxResponseTranscript); return HTTPServer::Response::_400; }
        memcpy(&responseV[response->m*response->n], transcript.c_str(), transcript.size()+1);

        return HTTPServer::Response("application/octet-stream", StringPiece((char*)response, sizeof(AcousticEventHeader) + response->m * response->n * sizeof(float) + transcript.size()+1));
    }

    static void addResponseWord(AcousticEventHeader *response, float *responseV, Recognizer::WordIter &iter, string &ts, string &ats, const char *annotation=0) {
        if (response->m >= MaxResponseWords) { ERROR("truncated response ", response->m); return; }
        responseV[response->m*2] = iter.beg;
        responseV[response->m++*2+1] = iter.end;

        ts += toconvert(iter.word.c_str(), tochar<' ', '-'>) + " ";

        long long ts1 = response->timestamp + iter.beg;
        StringAppendf(&ats, "%s-%lld (%d,%d) ", iter.word.c_str(), ts1, iter.beg, iter.end);
        if (annotation) ats += annotation;
        ats += " ";
    }
};

struct SpeechDecodeServer : public HTTPServer::SessionResource {
    RecognitionModel *model; int featureRate, sampleSecs;
    SpeechDecodeServer(RecognitionModel *Model, int FR, int SS) : model(Model), featureRate(FR), sampleSecs(SS) {}

    HTTPServer::Resource *open() { return new SpeechDecodeSession(model, featureRate, sampleSecs); }
    void close(HTTPServer::Resource *resource) { delete resource; }
};

int FusionServer(int argc, const char **argv) {
    RecognitionModel recognize;
    if (recognize.read("RecognitionNetwork", FLAGS_modeldir.c_str(), FLAGS_WantIter)) FATAL("open RecognitionNetwork ", FLAGS_modeldir);
    AcousticModel::toCUDA(&recognize.acousticModel);

    if (FLAGS_decode.size()) {
        SoundAsset sa;
        sa.filename = FLAGS_decode;
        sa.Load();
        if (!sa.wav) FATAL("load ", FLAGS_decode, " failed");

        RingBuf::Handle B(sa.wav);
        Matrix *feat = Features::fromBuf(&B);
        SpeechDecodeSession *decoder = new SpeechDecodeSession(&recognize, FLAGS_sample_rate/FLAGS_feat_hop, 3);
        HTTPServer::Response response = decoder->Request<double>(feat->row(0), feat->M, feat->N, 0, true);

        if (FLAGS_lfapp_debug) {
            INFO(FLAGS_decode, ": features(", feat->M, ")");
            for (int i=0; i<feat->M; i++) Vector::Print((double*)decoder->featB.Ind(i), Features::dimension()*3);
        }

        AcousticEventHeader *AEH = (AcousticEventHeader*)response.content;
        float *timestamp = (float*)(AEH+1);
        const char *transcript = (char *)(timestamp + AEH->m * AEH->n);
        INFO(FLAGS_decode, ": viterbi(", AEH->m, ") ", AEH->m ? transcript : "");
        return app->Free();
    }

    HTTPServer httpd(4044, FLAGS_ssl);
    if (app->network.Enable(&httpd)) return -1;
    httpd.AddURL("/favicon.ico", new HTTPServer::FileResource("./assets/icon.ico", "image/x-icon"));

    httpd.AddURL("/sink", new SpeechDecodeServer(&recognize, FLAGS_sample_rate/FLAGS_feat_hop, 3));

    httpd.AddURL("/flags", new HTTPServer::StringResource("text/html; charset=UTF-8", strdup((AcousticModel::flags() + "\r\n").c_str())));

    httpd.AddURL("/", new HTTPServer::StringResource("text/html; charset=UTF-8",
        "<html><h1>Fusion Server</h1>\r\n"
        "<a href=\"http://www.google.com\">google</a><br/>\r\n"
        "<form enctype=\"multipart/form-data\" method=\"post\" action=\"/sink\">\r\n"
        "<input type=\"file\" name=\"filename\">\r\n"
        "<input type=\"text\" name=\"input\">\r\n"
        "<input type=\"submit\">\r\n"
        "</form>\r\n"
        "</html>\r\n"));

    INFO("LFL fusion server initialized");
    return app->Main();
}

}; // namespace LFL
using namespace LFL;

extern "C" {
int main(int argc, const char **argv) {
    app->logfilename = StrCat(LFAppDownloadDir(), "fs.txt");
    static const char *service_name = "LFL Fusion Server";

    FLAGS_lfapp_network = 1;
#ifdef _WIN32
    if (argc>1) open_console = 1;
#endif

    if (app->Create(argc, argv, __FILE__)) { ERROR("lfapp init failed: ", strerror(errno)); return app->Free(); }
    if (app->Init()) { ERROR("lfapp open failed: ", strerror(errno)); return app->Free(); }

    bool exit=0;
#ifdef _WIN32
    if (install) { service_install(service_name, argv[0]); exit=1; }
    if (uninstall) { service_uninstall(service_name); exit=1; }
#endif
    if (FLAGS_fg) { return FusionServer(argc, argv); }
    if (exit) return app->Free();

#ifdef _WIN32
    string exedir(argv[0], DirNameLen(argv[0]));
    chdir(exedir.c_str());
#endif
    return NTService::WrapMain(service_name, FusionServer, argc, argv);
}
}
