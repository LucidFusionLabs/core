/*
 * $Id: fv.cpp 1336 2014-12-08 09:29:59Z justin $
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

#ifdef LFL_OPENCV
#include "opencv/cxcore.h"
#include "opencv/cv.h"
#endif

#include "lfapp/lfapp.h"
#include "lfapp/network.h"
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/gui.h"
#include "lfapp/camera.h"
#include "lfapp/audio.h"
#include "ml/hmm.h"
#include "speech/speech.h"   
#include "speech/voice.h"
#include "fs/fs.h"
#include "speech/aed.h"

namespace LFL {
DEFINE_float(clip, -30, "Clip audio under N decibels");
DEFINE_string(plot, "sg", "Audio GUI Plot [sg, zcr, pe]");
DEFINE_bool(camera_effects, true, "Render camera effects");
DEFINE_int(camera_orientation, 3, "Camera orientation");
DEFINE_string(speech_client, "auto", "Speech client send [manual, auto, flood]");

BindMap binds;
AssetMap asset;
SoundAssetMap soundasset;

Scene scene;
int myTab = 1;
bool myMonitor = 1, decoding = 0;
AcousticModel::Compiled *decodeModel = 0;
string transcript;
VoiceModel *voice = 0;
#ifdef LFL_FFMPEG
HTTPServer::StreamResource *stream = 0;
#endif
PhoneticSegmentationGUI *segments = 0;
AcousticEventDetector *AED = 0;
void ReloadAED(void *aed);
FeatureSink *serverSink = new SpeechDecodeClient(ReloadAED, &AED);
string speech_client_last;

void SetMyTab(int a) { myTab = a; }

void ReloadAED(void *aed) { 
    AcousticEventDetector **AED = (AcousticEventDetector**)aed;
    delete *AED;
    *AED = new AcousticEventDetector(FLAGS_sample_rate/FLAGS_feat_hop, serverSink);
    (*AED)->alwaysComputeFeatures();
}

struct LiveSpectogram {
    int feature_rate, feature_width;
    int samples_processed=0, samples_available=0, texture_slide=0;
    float vmax=35, scroll=0;
    RingBuf buf;
    RingBuf::RowMatHandle handle;
    Matrix *transform=0;

    LiveSpectogram() : feature_rate(FLAGS_sample_rate/FLAGS_feat_hop), feature_width(FLAGS_feat_window/2),
        buf(feature_rate, feature_rate*FLAGS_sample_secs, feature_width*sizeof(double)), handle(&buf) {
        asset("live")->tex.CreateBacked(feature_width, feature_rate*FLAGS_sample_secs);
        asset("snap")->tex.CreateBacked(feature_width, feature_rate*FLAGS_sample_secs);
    }

    int Behind() const { return samples_available - samples_processed; }

    void Resize(int width) {
        feature_width = width;
        buf.Resize(feature_rate, feature_rate*FLAGS_sample_secs, feature_width*sizeof(double));
        handle.Init();
        asset("live")->tex.Resize(feature_width, feature_rate*FLAGS_sample_secs);
        asset("snap")->tex.Resize(feature_width, feature_rate*FLAGS_sample_secs);
    }

    void XForm(const string &n) {
        if (n == "mel") {
            Typed::Replace<Matrix>(&transform, fft2mel(FLAGS_feat_melbands, FLAGS_feat_minfreq, FLAGS_feat_maxfreq, FLAGS_feat_window, FLAGS_sample_rate)->Transpose(mDelA));
            Resize(FLAGS_feat_melbands);
        } else {
            Typed::Replace<Matrix>(&transform, 0);
            Resize(FLAGS_feat_window/2);
        }
    }

    float Update(unsigned samples) {
        samples_available += samples;
        Asset *live = asset("live");

        while(samples_available >= samples_processed + FLAGS_feat_window) {
            const double *last = handle.Read(-1)->row(0);
            RingBuf::Handle L(app->audio.IL, app->audio.RL.next-Behind(), FLAGS_feat_window);

            Matrix *Frame, *FFT;
            Frame = FFT = spectogram(&L, transform ? 0 : handle.Write(), FLAGS_feat_window, FLAGS_feat_hop, FLAGS_feat_window, 0, PowerDomain::abs);
            if (transform) Frame = Matrix::Mult(FFT, transform, handle.Write());
            if (transform) delete FFT;

            int ind = texture_slide * live->tex.width * Pixel::size(live->tex.pf);
            glSpectogram(Frame, live->tex.buf+ind, 1, feature_width, feature_width, vmax, FLAGS_clip, 0, PowerDomain::abs);

            live->tex.Bind();
            live->tex.UpdateGL(0, texture_slide, live->tex.width, 1);

            samples_processed += FLAGS_feat_hop;
            scroll = (float)(texture_slide + 1) / (feature_rate * FLAGS_sample_secs);
            texture_slide = (texture_slide + 1) % (feature_rate * FLAGS_sample_secs);
        }
        return -scroll;
    }

    void Draw(Box win, bool onoff, bool fullscreen) {
        int orientation = fullscreen ? 3 : 5;
        StringWordIter plotIter(FLAGS_plot.c_str());
        for (const char *ploti = plotIter.next(); ploti; ploti = plotIter.next()) { 
            if (!strcmp(ploti, "sg")) {
                if (onoff) asset("live")->tex.DrawCrimped(win, orientation, 0, -scroll);
                else       asset("snap")->tex.DrawCrimped(win, orientation, 0, 0);
            }
            else if (!strcmp(ploti, "wf")) {
                static int decimate = 10;
                int delay = Behind(), samples = feature_rate*FLAGS_feat_hop*FLAGS_sample_secs;
                if (onoff) {
                    RingBuf::Handle B(app->audio.IL, app->audio.RL.next-delay, samples);
                    Waveform::Decimated(win.Dimension(), &Color::white, &B, decimate).Draw(win); 
                }
                else {
                    SoundAsset *snap = soundasset("snap");
                    RingBuf::Handle B(snap->wav, snap->wav->ring.back, samples);
                    Waveform::Decimated(win.Dimension(), &Color::white, &B, decimate).Draw(win); 
                }
            }
            else if (!strcmp(ploti, "pe") && onoff && AED) {
                RingBuf::Handle B(&AED->pe, AED->pe.ring.back);
                Waveform(win.Dimension(), &Color::white, &B).Draw(win);
            }
            else if (!strcmp(ploti, "zcr") && onoff && AED) {
                RingBuf::Handle B(&AED->zcr, AED->zcr.ring.back);
                Waveform(win.Dimension(), &Color::white, &B).Draw(win);
            }
            else if (!strcmp(ploti, "szcr") && onoff && AED) {
                RingBuf::DelayHandle B(&AED->szcr, AED->szcr.ring.back, AcousticEventDetector::szcr_shift);
                Waveform(win.Dimension(), &Color::white, &B).Draw(win);
            }
        }
    }

    void ProgressBar(Box win, float percent) {
        if (percent < 0 || percent > 1) return;

        Geometry *geom = new Geometry(GraphicsDevice::Lines, 2, (v2*)0, 0, 0, Color(1.0,1.0,1.0));
        v2 *vert = (v2*)&geom->vert[0];

        float xi = win.percentX(percent);
        vert[0] = v2(xi, win.y);
        vert[1] = v2(xi, win.y+win.h);

        screen->gd->DisableTexture();
        Scene::Select(geom);
        Scene::Draw(geom, 0);
        delete geom;
    }
} *liveSG = 0;

struct LiveCamera {
    LiveCamera() {
        asset("camera")->tex.CreateBacked(256, 256, Pixel::RGB24);
        asset("camfx" )->tex.CreateBacked(256, 256, Pixel::GRAY8);
    }

    Asset *Input() {
        Asset *cam = 0;
        if (!cam) cam = app->assets.movie_playing ? &app->assets.movie_playing->video : 0;
        if (!cam) cam = FLAGS_lfapp_camera ? asset("camera") : 0;
        if (!cam) cam = asset("browser");
        return cam;
    }

    void Update() {
        Asset *cam=Input(), *fx=asset("camfx");
        if (fx->tex.width != cam->tex.width || fx->tex.height != cam->tex.height) fx->tex.Resize(cam->tex.width, cam->tex.height);

        if (FLAGS_lfapp_camera && cam == asset("camera")) {
            /* update camera buffer */
            cam->tex.UpdateBuffer(app->camera.camera->image, FLAGS_camera_image_width, FLAGS_camera_image_height, app->camera.camera->image_format,
                                  app->camera.camera->image_linesize, Texture::Flag::Resample);

            /* flush camera buffer */
            cam->tex.Bind();
            cam->tex.UpdateGL();
        }

#ifdef LFL_OPENCV
        if (!FLAGS_camera_effects) return;

        /* copy greyscale image */
        IplImage camI, camfxI;
        fx->tex.ToIplImage(&camfxI);
        cam->tex.ToIplImage(&camI);
        cvCvtColor(&camI, &camfxI, CV_RGB2GRAY);

        if (0) {
            /* apply gaussian filter */
            cvSmooth(&camfxI, &camfxI, CV_GAUSSIAN, 11, 11);
        }

        /* perform canny edge detection */
        cvCanny(&camfxI, &camfxI, 10, 100, 3);

        /* flush camfx buffer */
        fx->tex.Bind();
        fx->tex.UpdateGL();
#endif
    }

    void Draw(Box camw, Box fxw) {
        Input()       ->tex.DrawCrimped(camw, FLAGS_camera_orientation, 0, 0);
        asset("camfx")->tex.DrawCrimped(fxw,  FLAGS_camera_orientation, 0, 0);
    }
} *liveCam = 0;

void MyDraw(const vector<string> &args) {
    Asset *a = asset(IndexOrDefault(args, 0));
    SoundAsset *sa = soundasset(IndexOrDefault(args, 0));
    if (!a || !sa) return;
    bool recalibrate=1;
    glSpectogram(sa, a, liveSG->transform, recalibrate?&liveSG->vmax:0, FLAGS_clip);
}

void MySpeechClient(const vector<string> &args) {
    if (args.empty() || args[0].empty()) return;
    FLAGS_speech_client = speech_client_last = args[0];
}

void MyMonitor(const vector<string>&) {
    FLAGS_speech_client = speech_client_last;
}

void MySnap(const vector<string> &args) {
    Asset *a = asset(IndexOrDefault(args, 0));
    SoundAsset *sa = soundasset(IndexOrDefault(args, 0));
    if (!FLAGS_lfapp_audio || !a || !sa) return;

    FLAGS_speech_client = "manual";
    transcript.clear();
    Typed::Replace(&segments, (PhoneticSegmentationGUI*)0);

    app->audio.Snapshot(sa);
    MyDraw(args);

    if (AED && AED->sink) {
        FeatureSink::DecodedWords decode;
        for (int i=0; i<AED->sink->decode.size(); i++) {
            FeatureSink::DecodedWord *word = &AED->sink->decode[i];
            double perc = AED->percent(word->beg);
            if (perc < 0) continue;
            if (perc > 1) break;
            int beg = perc * AED->feature_rate * FLAGS_sample_secs;
            int end = beg + (word->end - word->beg);
            decode.push_back(FeatureSink::DecodedWord(word->text.c_str(), beg, end));
            transcript += (!i ? "" : "  ") + word->text;
        }
        if (decode.size()) Typed::Replace
            (&segments, new PhoneticSegmentationGUI(screen, decode, AED->feature_rate * FLAGS_sample_secs, "snap"));
    }
}

void MyNetDecodeResponse(FeatureSink::DecodedWords &decode, int responselen) {
    transcript.clear();
    for (int i=0, l=decode.size(); i<l; i++) transcript += (!i ? "" : "  ") + decode[i].text;
    INFO("transcript: ", transcript);
    Typed::Replace(&segments, new PhoneticSegmentationGUI(screen, decode, responselen, "snap"));
    decoding = 0;
}

void MyNetDecode(const vector<string> &args) {
    SoundAsset *sa = soundasset(IndexOrDefault(args, 0));
    if (!sa) { INFO("decode <assset>"); return; }
    if (!AED || !AED->sink || !AED->sink->connected()) { INFO("not connected"); return; }

    Typed::Replace(&segments, (PhoneticSegmentationGUI*)0);
    Matrix *features = Features::fromAsset(sa, Features::Flag::Storable);
    AED->sink->decode.clear();
    int posted = AED->sink->Write(features, 0, true, MyNetDecodeResponse);
    delete features;
    INFO("posted decode len=", posted);
    decoding = 1;
}

void MyDecode(const vector<string> &args) {
    SoundAsset *sa = soundasset(IndexOrDefault(args, 0));
    if (!sa) { INFO("decode <assset>"); return; }
    if (AED && AED->sink && AED->sink->connected()) return MyNetDecode(args);

#if !defined(LFL_IPHONE) && !defined(LFL_ANDROID)
    if (!decodeModel)
#endif
    {
        ERROR("decode error: not connected to server: ", -1);
        return;
    }

    Matrix *features = Features::fromAsset(sa, Features::Flag::Full);
    Matrix *viterbi = Decoder::decodeFeatures(decodeModel, features, 1024);

    transcript = Decoder::transcript(decodeModel, viterbi);     
    Typed::Replace(&segments, new PhoneticSegmentationGUI(screen, decodeModel, viterbi, "snap"));

    delete features;
    delete viterbi;

    if (transcript.size()) INFO(transcript.c_str());
    else INFO("decode failed");
}

void MySynth(const vector<string> &args) {
    if (!voice) return;
    RingBuf *synth = voice->synth(Join(args, " ").c_str());
    if (!synth) { ERROR("synth ret 0"); return; }
    INFO("synth ", synth->ring.size);
    RingBuf::Handle B(synth);
    app->audio.QueueMixBuf(&B);
    delete synth;
}

void MyResynth(const vector<string> &args) {
    SoundAsset *sa = soundasset(args.size()?args[0]:"snap");
    if (sa) { resynthesize(&app->audio, sa); }
}

void MySpectogramTransform(const vector<string> &args) { liveSG->XForm(IndexOrDefault(args, 0).c_str()); }

void MyServer(const vector<string> &args) {
    if (args.empty() || args[0].empty()) { INFO("eg: server 192.168.2.188:4044/sink"); return; }
    ((SpeechDecodeClient*)AED->sink)->connect(args[0].c_str());
}

void MyStartCamera(const vector<string>&) {
    if (FLAGS_lfapp_camera) return INFO("camera already started");
    FLAGS_lfapp_camera = true;

    if (app->camera.Init()) { FLAGS_lfapp_camera = false; return INFO("camera init failed"); }
    INFO("camera started");
    liveCam = new LiveCamera();
}

struct AudioGUI : public GUI {
    bool monitor_hover=0, decode=0, last_decode=0, skip=0;
    int last_audio_count=1;
    Font *norm, *text;
    Widget::Button play_button, decode_button, record_button;
    AudioGUI(Window *W) : GUI(W),
        norm(Fonts::Get(FLAGS_default_font, 12, Color::grey70)),
        text(Fonts::Get(FLAGS_default_font, 8,  Color::grey80)),
        play_button  (this, DrawableNop(), norm, "play",    MouseController::CB([&](){ if (!app->audio.Out.size()) app->shell.play(vector<string>(1,"snap")); })),
        decode_button(this, DrawableNop(), norm, "decode",  MouseController::CB([&](){ decode = true; })),
        record_button(this, DrawableNop(), norm, "monitor", MouseController::CB([&](){ myMonitor = !myMonitor; })) {}

    int Frame(LFL::Window *W, unsigned clocks, unsigned samples, bool cam_sample, int flag) {
        if (!myMonitor && !W->events.gui && !app->audio.Out.size() && !last_audio_count && !decode && !screen->console->active) skip = 1;

        last_audio_count = app->audio.Out.size();
        if (0 && skip && !(flag & FrameFlag::DontSkip)) {
            if (segments) segments->mouse.Activate();
            mouse.Activate();
            return -1;
        }

        if (decode && last_decode) { if (!myMonitor) MyDecode(vector<string>(1, "snap")); decode=0; }
        last_decode = decode;

        if (auto b = play_button  .GetDrawBox()) b->drawable = &asset(app->audio.Out.size() ? "but1" : "but0")->tex;
        if (auto b = decode_button.GetDrawBox()) b->drawable = &asset((decode || decoding)  ? "but1" : "but0")->tex;
        if (auto b = record_button.GetDrawBox()) b->drawable = &asset(myMonitor ? (monitor_hover ? "onoff1hover" : "onoff1") :
                                                                                  (monitor_hover ? "onoff0hover" : "onoff0"))->tex;
        return DrawFrame();
    }

    int DrawFrame() {
        /* live spectogram */
        float yp=0.6, ys=0.4, xbdr=0.05, ybdr=0.07;
        Box sb = screen->Box(0, yp, 1, ys, xbdr, ybdr);
        Box si = screen->Box(0, yp, 1, ys, xbdr+0.04, ybdr+0.035);

        screen->gd->DrawMode(DrawMode::_2D);
        asset("sbg")->tex.Draw(sb);
        liveSG->Draw(si, myMonitor, false);
        asset("sgloss")->tex.Draw(sb);

        /* progress bar */
        if (app->audio.Out.size() && !myMonitor) {
            float percent=1-(float)(app->audio.Out.size()+feat_progressbar_c*FLAGS_sample_rate*FLAGS_chans_out)/app->audio.outlast;
            liveSG->ProgressBar(si, percent);
        }

        /* acoustic event frame */
        if (myMonitor && AED && (AED->words.size() || speech_client_flood()))
            AcousticEventGUI::Draw(AED, si);

        /* segmentation */
        if (!myMonitor && segments) segments->Frame(si, norm);

        /* transcript */
        if (transcript.size() && !myMonitor) text->Draw(StrCat("transcript: ", transcript), point(screen->width*.05, screen->height*.05));
        else text->Draw(StringPrintf("press tick for console - FPS = %.2f - CR = %.2f", FPS(), CamFPS()), point(screen->width*.05, screen->height*.05));

        /* f0 */
        if (0) {
            RingBuf::Handle f0in(app->audio.IL, app->audio.RL.next-FLAGS_feat_window, FLAGS_feat_window);
            text->Draw(StringPrintf("hz %.0f", fundamentalFrequency(&f0in, FLAGS_feat_window, 0)), point(screen->width*.85, screen->height*.05));
        }

        /* SNR */
        if (AED) text->Draw(StringPrintf("SNR = %.2f", AED->SNR()), point(screen->width*.75, screen->height*.05));

        GUI::Draw(screen->Box());
        return 0;
    }

    void Layout() {
        Flow flow(&box, 0, &child_box);

        play_button  .LayoutBox(&flow, screen->Box(0,  -.85, .5, .15, .16, .0001));
        decode_button.LayoutBox(&flow, screen->Box(.5, -.85, .5, .15, .16, .0001));
        record_button.LayoutBox(&flow, screen->Box(0,  -.65,  1, .2,  .38, .0001));

        play_button.AddHoverBox(record_button.box, MouseController::CB([&](){ monitor_hover = !monitor_hover; }));
        play_button.AddHoverBox(record_button.box, MouseController::CB([&](){ if ( myMonitor) MyMonitor(vector<string>(1,"snap")); }));
        play_button.AddClickBox(record_button.box, MouseController::CB([&](){ if (!myMonitor) MySnap   (vector<string>(1,"snap")); }));
        // static Widget::Button fullscreenButton(&gui, 0, 0, sb, 0, setMyTab, (void*)4);
    }
} *audio_gui;

int VideoFrame(LFL::Window *W, unsigned clocks, unsigned samples, bool cam_sample, int flag) {
    screen->gd->DrawMode(DrawMode::_2D);

    if (liveCam) {
        if (cam_sample || app->assets.movie_playing) liveCam->Update();

        float yp=0.5, ys=0.5, xbdr=0.05, ybdr=0.07;
        Box st = screen->Box(0, .5, 1, ys, xbdr+0.04, ybdr+0.035, xbdr+0.04, .01);
        Box sb = screen->Box(0, 0,  1, ys, xbdr+0.04, .01, xbdr+0.04, ybdr+0.035);
        liveCam->Draw(st, sb);
    }

    static Font *text = Fonts::Get(FLAGS_default_font, 9, Color::grey80);
    text->Draw(StringPrintf("press tick for console - FPS = %.2f - CR = %.2f", FPS(), CamFPS()), point(screen->width*.05, screen->height*.05));
    return 0;
}

int RoomModelFrame(LFL::Window *W, unsigned clicks, unsigned samples, bool cam_sample, int flag) {
    screen->camMain->look();
    scene.Get("arrow")->yawright((double)clicks/500);	
    scene.Draw(&asset.vec);
    return 0;
}

struct FullscreenGUI : public GUI {
    bool decode=0, lastdecode=0, close=0;
    int monitorcount=0;
    Font *norm, *text;
    Widget::Button play_button, close_button, decode_icon, fullscreen_button;

    FullscreenGUI(Window *W) : GUI(W),
    norm(Fonts::Get(FLAGS_default_font, 12, Color::grey70)),
    text(Fonts::Get(FLAGS_default_font, 12, Color::white, TTFFont::Flag::Outline)),
    play_button      (this, 0, 0, "", MouseController::CB([&](){ myMonitor=1; })),
    close_button     (this, 0, 0, "", MouseController::CB([&](){ close=1; })),
    decode_icon      (this, 0, 0, "", MouseController::CB()),
    fullscreen_button(this, 0, 0, "", MouseController::CB([&](){ decode=1; })) {}

    int Frame(LFL::Window *W, unsigned clicks, unsigned samples, bool cam_sample, int flag) {
        if (myMonitor) monitorcount++;
        else monitorcount = 0;

        if (decode && lastdecode) {
            if (monitorcount > 70) {
                myMonitor = false;
                MySnap(vector<string>(1, "snap"));
                MyDecode(vector<string>(1, "snap"));
            }
            decode = 0;
        }
        lastdecode = decode;

        screen->gd->DrawMode(DrawMode::_2D);
        liveSG->Draw(screen->Box(), myMonitor, true);

        if (myMonitor) {
            if (AED && (AED->words.size() || speech_client_flood())) AcousticEventGUI::Draw(AED, screen->Box(), true);
        } else if (segments) {
            segments->Frame(screen->Box(), norm, true);

            int total = AED->feature_rate * FLAGS_sample_secs;
            for (auto it = segments->segments.begin(); it != segments->segments.end(); it++)
                text->Draw((*it).name, point(box.centerX(), box.percentY((float)(*it).beg/total)), 0, Font::Flag::Orientation(3));
        }

        Draw(screen->Box().TopLeft());

        if (decode || decoding)
            decode_icon.drawable->Draw(screen->Box(.9, -.07, .09, .07));

        if (close) { SetMyTab(1); close = 0; }
        return 0;
    }

    void Layout() {
        Flow flow(&box, 0, &child_box);
        play_button      .LayoutBox(&flow, screen->Box(0,    -1, .09, .07));
        close_button     .LayoutBox(&flow, screen->Box(0,  -.07, .09, .07));
        fullscreen_button.LayoutBox(&flow, screen->Box());
        fullscreen_button.GetHitBox().run_only_if_first = true;
        play_button.AddClickBox(play_button.box, MouseController::CB([&](){ if (!myMonitor) MyMonitor(vector<string>(1,"snap")); }));
    }
} *fullscreen_gui;

struct FVGUI : public GUI {
    Font *font;
    Widget::Button tab1, tab2, tab3;
    FVGUI(Window *W) : GUI(W), font(Fonts::Get(FLAGS_default_font, 12, Color::grey70)),
    tab1(this, 0, font, "audio gui",  MouseController::CB(bind(&SetMyTab, 1))),
    tab2(this, 0, font, "video gui",  MouseController::CB(bind(&SetMyTab, 2))), 
    tab3(this, 0, font, "room model", MouseController::CB(bind(&SetMyTab, 3))) {}

    int Frame(LFL::Window *W, unsigned clicks, unsigned samples, bool cam_sample, int flag) {
#if !defined(LFL_IPHONE) && !defined(LFL_ANDROID)
        int orientation = NativeWindowOrientation();
        bool orientation_fs = orientation == 5 || orientation == 4 || orientation == 3;
        bool fullscreen = myTab == 4;
        if (orientation_fs && !fullscreen) myTab = 4;
        if (!orientation_fs && fullscreen) myTab = 1;
#endif
        int ret;
        if      (myTab == 1) ret = audio_gui->Frame(W, clicks, samples, cam_sample, flag);
        else if (myTab == 2) ret = VideoFrame(W, clicks, samples, cam_sample, flag);
        else if (myTab == 3) ret = RoomModelFrame(W, clicks, samples, cam_sample, flag);
        else if (myTab == 4) ret = fullscreen_gui->Frame(W, clicks, samples, cam_sample, flag);
        if (ret < 0) return ret;

        screen->gd->DrawMode(DrawMode::_2D);
        Draw(screen->Box().TopLeft());
        screen->DrawDialogs();
        return 0;
    }

    void Layout() {
        Flow flow(&box, 0, &child_box);
        tab1.Layout(&flow, point(screen->width/3, screen->height*.05));
        tab2.Layout(&flow, point(screen->width/3, screen->height*.05));
        tab3.Layout(&flow, point(screen->width/3, screen->height*.05));
    }
} *fv_gui;

int Frame(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) {

    if (AED) AED->update(mic_samples);
    if (liveSG) liveSG->Update(mic_samples);
#ifdef LFL_FFMPEG
    if (stream) stream->update(mic_samples, cam_sample);
#endif

    return fv_gui->Frame(W, clicks, mic_samples, cam_sample, flag);
}

}; // namespace LFL
using namespace LFL;

extern "C" int main(int argc, const char *argv[]) {

	app->frame_cb = Frame;
	app->logfilename = StrCat(dldir(), "fv.txt");
	screen->width = 640;
	screen->height = 480;
	screen->caption = "fusion viewer";
	FLAGS_lfapp_camera = true;

	if (app->Create(argc, argv, __FILE__)) { app->Free(); return -1; }
	if (app->Init()) { app->Free(); return -1; }

	//  asset.Add(Asset(name,          texture,          scale, translate, rotate, geometry     0, 0, 0));
	asset.Add(Asset("axis", "", 0, 0, 0, 0, 0, 0, 0, bind(&glAxis, _1, _2)));
	asset.Add(Asset("grid", "", 0, 0, 0, Grid::Grid3D(), 0, 0, 0));
	asset.Add(Asset("room", "", 0, 0, 0, 0, 0, 0, 0, bind(&glRoom, _1, _2)));
	asset.Add(Asset("arrow", "", .005, 1, -90, "arrow.obj", 0, 0));
	asset.Add(Asset("snap", "", 0, 0, 0, 0, 0, 0, 0));
	asset.Add(Asset("live", "", 0, 0, 0, 0, 0, 0, 0));
	asset.Add(Asset("browser", "", 0, 0, 0, 0, 0, 0, 0));
	asset.Add(Asset("camera", "", 0, 0, 0, 0, 0, 0, 0));
	asset.Add(Asset("camfx", "", 0, 0, 0, 0, 0, 0, 0));
	asset.Add(Asset("sbg", "spectbg.png", 0, 0, 0, 0, 0, 0, 0));
	asset.Add(Asset("sgloss", "spectgloss.png", 0, 0, 0, 0, 0, 0, 0));
	asset.Add(Asset("onoff0", "onoff0.png", 0, 0, 0, 0, 0, 0, 0));
	asset.Add(Asset("onoff0hover", "onoff0hover.png", 0, 0, 0, 0, 0, 0, 0));
	asset.Add(Asset("onoff1", "onoff1.png", 0, 0, 0, 0, 0, 0, 0));
	asset.Add(Asset("onoff1hover", "onoff1hover.png", 0, 0, 0, 0, 0, 0, 0));
	asset.Add(Asset("but0", "but0.png", 0, 0, 0, 0, 0, 0, 0));
	asset.Add(Asset("but1", "but1.png", 0, 0, 0, 0, 0, 0, 0));
	asset.Add(Asset("butplay", "play-icon.png", 0, 0, 0, 0, 0, 0, 0));
	asset.Add(Asset("butclose", "close-icon.png", 0, 0, 0, 0, 0, 0, 0));
	asset.Add(Asset("butinfo", "info-icon.png", 0, 0, 0, 0, 0, 0, 0));
	asset.Load();
	app->shell.assets = &asset;

	//  soundasset.Add(SoundAsset(name,   filename,   ringbuf,                                          channels, sample_rate,       seconds           ));
	soundasset.Add(SoundAsset("draw", "Draw.wav", 0, 0, 0, 0));
	soundasset.Add(SoundAsset("snap", "", new RingBuf(FLAGS_sample_rate*FLAGS_sample_secs), 1, FLAGS_sample_rate, FLAGS_sample_secs));
	soundasset.Load();
	app->shell.soundassets = &soundasset;

	app->shell.command.push_back(Shell::Command("speech_client", bind(&MySpeechClient, _1)));
	app->shell.command.push_back(Shell::Command("draw", bind(&MyDraw, _1)));
	app->shell.command.push_back(Shell::Command("snap", bind(&MySnap, _1)));
	app->shell.command.push_back(Shell::Command("decode", bind(&MyDecode, _1)));
	app->shell.command.push_back(Shell::Command("netdecode", bind(&MyNetDecode, _1)));
	app->shell.command.push_back(Shell::Command("synth", bind(&MySynth, _1)));
	app->shell.command.push_back(Shell::Command("resynth", bind(&MyResynth, _1)));
	app->shell.command.push_back(Shell::Command("sgtf", bind(&MySpectogramTransform, _1)));
	app->shell.command.push_back(Shell::Command("sgxf", bind(&MySpectogramTransform, _1)));
	app->shell.command.push_back(Shell::Command("server", bind(&MyServer, _1)));
	app->shell.command.push_back(Shell::Command("startcamera", bind(&MyStartCamera, _1)));

	//  binds.push_back(Bind(key,            callback,         arg));
	binds.push_back(Bind(Key::Backquote, Bind::CB(bind([&](){ screen->console->Toggle(); }))));
	binds.push_back(Bind(Key::Quote,     Bind::CB(bind([&](){ screen->console->Toggle(); }))));
    binds.push_back(Bind(Key::Escape,    Bind::CB(bind(&Shell::quit,            &app->shell, vector<string>()))));
    binds.push_back(Bind(Key::Return,    Bind::CB(bind(&Shell::grabmode,        &app->shell, vector<string>()))));
    binds.push_back(Bind(Key::LeftShift, Bind::TimeCB(bind(&Entity::RollLeft,   screen->camMain, _1))));
    binds.push_back(Bind(Key::Space,     Bind::TimeCB(bind(&Entity::RollRight,  screen->camMain, _1))));
    binds.push_back(Bind('w',            Bind::TimeCB(bind(&Entity::MoveFwd,    screen->camMain, _1))));
    binds.push_back(Bind('s',            Bind::TimeCB(bind(&Entity::MoveRev,    screen->camMain, _1))));
    binds.push_back(Bind('a',            Bind::TimeCB(bind(&Entity::MoveLeft,   screen->camMain, _1))));
    binds.push_back(Bind('d',            Bind::TimeCB(bind(&Entity::MoveRight,  screen->camMain, _1))));
    binds.push_back(Bind('q',            Bind::TimeCB(bind(&Entity::MoveDown,   screen->camMain, _1))));
    binds.push_back(Bind('e',            Bind::TimeCB(bind(&Entity::MoveUp,     screen->camMain, _1))));
    screen->binds = &binds;

#if !defined(LFL_IPHONE) && !defined(LFL_ANDROID)
    HTTPServer *httpd = new HTTPServer(4040, false);
    if (app->network.Enable(httpd)) return -1;
    httpd->AddURL("/favicon.ico", new HTTPServer::FileResource("./assets/icon.ico", "image/x-icon"));

#ifdef LFL_FFMPEG
    stream = new HTTPServer::StreamResource("flv", 32000, 300000);
    httpd->AddURL("/stream.flv", stream);
#endif

    httpd->AddURL("/test.flv", new HTTPServer::FileResource("./assets/test.flv", "video/x-flv"));
    httpd->AddURL("/mediaplayer.swf", new HTTPServer::FileResource("./assets/mediaplayer.swf", "application/x-shockwave-flash"));

    httpd->AddURL("/", new HTTPServer::StringResource("text/html; charset=UTF-8",
        "<html><h1>Web Page</h1>\r\n"
        "<a href=\"http://www.google.com\">google</a><br/>\r\n"
        "<h2>stream</h2>\r\n"
        "<embed src=\"/mediaplayer.swf\" width=\"320\" height=\"240\" \r\n"
        "       allowscriptaccess=\"always\" allowfullscreen=\"true\" \r\n"
        "       flashvars=\"width=320&height=240&file=/stream.flv\" />\r\n"
        "</html>\r\n"));

    AcousticModelFile *model = new AcousticModelFile();
    if (model->read("AcousticModel", "./assets/")<0) { ERROR("am read ./assets/"); return -1; }

    if (!(decodeModel = AcousticModel::fromModel1(model, true))) { ERROR("model create failed"); return -1; }
    AcousticModel::toCUDA(model);

    PronunciationDict::instance();
    voice = new VoiceModel();
    if (voice->read("./assets/")<0) { ERROR("voice read ./assets/"); return -1; }
#endif

    scene.Add(new Entity("axis",  asset("axis")));
    scene.Add(new Entity("grid",  asset("grid")));
    scene.Add(new Entity("room",  asset("room")));
    scene.Add(new Entity("arrow", asset("arrow"), v3(1,.24,1)));

    liveSG = new LiveSpectogram();
    liveSG->XForm("mel");

    speech_client_last = FLAGS_speech_client;
    ReloadAED(&AED);
    liveCam = new LiveCamera();
    audio_gui = new AudioGUI(screen);
    fv_gui = new FVGUI(screen);
    fullscreen_gui = new FullscreenGUI(screen);

    return app->Main();
}
