/*
 * $Id: input.cpp 1328 2014-11-04 09:35:46Z justin $
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
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/flow.h"
#include "lfapp/gui.h"

namespace LFL {
Shell::Shell(AssetMap *AM, SoundAssetMap *SAM, MovieAssetMap *MAM) : assets(AM), soundassets(SAM), movieassets(MAM) {
  command.push_back(Command("quit",       bind(&Shell::quit,         this, _1)));
  command.push_back(Command("cmds",       bind(&Shell::cmds,         this, _1)));
  command.push_back(Command("binds",      bind(&Shell::binds,        this, _1)));
  command.push_back(Command("flags",      bind(&Shell::flags,        this, _1)));
  command.push_back(Command("conscolor",  bind(&Shell::consolecolor, this, _1)));
  command.push_back(Command("clipboard",  bind(&Shell::clipboard,    this, _1)));
  command.push_back(Command("startcmd",   bind(&Shell::startcmd,     this, _1)));
  command.push_back(Command("dldir",      bind(&Shell::dldir,        this, _1)));
  command.push_back(Command("screenshot", bind(&Shell::screenshot,   this, _1)));
  command.push_back(Command("fillmode",   bind(&Shell::fillmode,     this, _1)));
  command.push_back(Command("texmode",    bind(&Shell::texmode,      this, _1)));
  command.push_back(Command("swapaxis",   bind(&Shell::swapaxis,     this, _1)));
  command.push_back(Command("campos",     bind(&Shell::campos,       this, _1)));
  command.push_back(Command("filter",     bind(&Shell::filter,       this, _1)));
  command.push_back(Command("fftfilter",  bind(&Shell::filter,       this, _1)));
  command.push_back(Command("f0",         bind(&Shell::f0,           this, _1)));
  command.push_back(Command("sinth",      bind(&Shell::sinth,        this, _1)));
  command.push_back(Command("play",       bind(&Shell::play,         this, _1)));
  command.push_back(Command("playmovie",  bind(&Shell::playmovie,    this, _1)));
  command.push_back(Command("loadsound",  bind(&Shell::loadsound,    this, _1)));
  command.push_back(Command("loadmovie",  bind(&Shell::loadmovie,    this, _1)));
  command.push_back(Command("copy",       bind(&Shell::copy,         this, _1)));
  command.push_back(Command("snap",       bind(&Shell::snap,         this, _1)));
  command.push_back(Command("writesnap",  bind(&Shell::writesnap,    this, _1)));
  command.push_back(Command("fps",        bind(&Shell::fps,          this, _1)));
  command.push_back(Command("wget",       bind(&Shell::wget,         this, _1)));
  command.push_back(Command("messagebox", bind(&Shell::MessageBox,   this, _1)));
  command.push_back(Command("texturebox", bind(&Shell::TextureBox,   this, _1)));
  command.push_back(Command("edit",       bind(&Shell::Edit,         this, _1)));
  command.push_back(Command("slider",     bind(&Shell::Slider,       this, _1)));
}

Asset      *Shell::asset     (const string &n) { return assets      ? (*     assets)(n) : 0; }
SoundAsset *Shell::soundasset(const string &n) { return soundassets ? (*soundassets)(n) : 0; }
MovieAsset *Shell::movieasset(const string &n) { return movieassets ? (*movieassets)(n) : 0; }

bool Shell::FGets() {
  char buf[1024];
  if (!LFL::FGets(buf, sizeof(buf))) return false;
  ChompNewline(buf, strlen(buf));
  Shell::Run(buf);
  return true;
}

void Shell::Run(const string &text) {
  if (!MainThread()) return RunInMainThread(new Callback(bind(&Shell::Run, this, text)));

  string cmd;
  vector<string> arg;
  Split(text, isspace, isquote, &arg);
  if (arg.size()) { cmd = arg[0]; arg.erase(arg.begin()); }
  if (cmd.empty()) return;

  for (auto i = command.begin(); i != command.end(); ++i) {
    if (StringEquals(i->name, cmd)) {
      i->cb(arg);
      return;
    }
  }

  FlagMap *flags = Singleton<FlagMap>::Get();
  for (auto i = flags->flagmap.begin(); i != flags->flagmap.end(); ++i) {
    Flag *flag = (*i).second;
    if (StringEquals(flag->name, cmd)) {
      flag->Update(arg.size() ? arg[0].c_str() : "");
      INFO(flag->name, " = ", flag->Get());
      return;
    }
  }
  INFO("unkown cmd '", cmd, "'");
}

void Shell::mousein (const vector<string>&) { Mouse::GrabFocus(); }
void Shell::mouseout(const vector<string>&) { Mouse::ReleaseFocus(); }

void Shell::quit(const vector<string>&) { app->run = false; }
void Shell::console(const vector<string>&) { if (screen->lfapp_console) screen->lfapp_console->Toggle(); }
void Shell::showkeyboard(const vector<string>&) { TouchDevice::OpenKeyboard(); }

void Shell::clipboard(const vector<string> &a) {
  if (a.empty()) INFO(Clipboard::Get());
  else Clipboard::Set(Join(a, " "));
}

void Shell::consolecolor(const vector<string>&) {
  if (!screen->lfapp_console) return;
  delete screen->lfapp_console->font;
  screen->lfapp_console->font = Fonts::Get(FLAGS_default_font, "", 9, Color::black);
}

void Shell::startcmd(const vector<string> &a) {
  if (a.empty() || !screen->lfapp_console) return;
  screen->lfapp_console->startcmd = Join(a, " ");
}

void Shell::dldir(const vector<string>&) { INFO(LFAppDownloadDir()); }

void Shell::screenshot(const vector<string> &a) {
  if (a.empty()) return INFO("usage: screenshot <file> [tex_id]");
  Texture tex;
  if (a.size() == 1) tex.Screenshot();
  else               tex.DumpGL(atoi(a[1]));
  LocalFile lf(a[0], "w");
  PngWriter::Write(&lf, tex);
}

void Shell::fillmode(const vector<string>&) {
#if !defined(LFL_IPHONE) && !defined(LFL_ANDROID)
  // glPolygonMode(GL_FRONT_AND_BACK, app->fillMode.next());
#endif
}

void Shell::grabmode(const vector<string> &a) { if (app->grab_mode.Next()) mousein(a); else mouseout(a); }
void Shell::texmode(const vector<string>&) { if (app->tex_mode.Next()) screen->gd->EnableTexture(); else screen->gd->DisableTexture(); }
void Shell::swapaxis(const vector<string>&) { screen->SwapAxis(); }

void Shell::campos(const vector<string>&) {
  INFO("camMain.pos=",  screen->cam->pos.DebugString(),
       " camMain.ort=", screen->cam->ort.DebugString(),
       " camMain.up=",  screen->cam->up .DebugString());
}

void Shell::snap(const vector<string> &arg) {
  Asset      *a  = asset     (arg.size() ? arg[0] : "snap"); 
  SoundAsset *sa = soundasset(arg.size() ? arg[0] : "snap");
  if (a && sa) { app->audio->Snapshot(sa); glSpectogram(sa, a); }
}

void Shell::play(const vector<string> &arg) {
  SoundAsset *sa     = arg.size() > 0 ? soundasset(arg[0]) : soundasset("snap");
  int         offset = arg.size() > 1 ?       atoi(arg[1]) : -1;
  int         len    = arg.size() > 2 ?       atoi(arg[2]) : -1;
  if (sa) app->audio->QueueMix(sa, MixFlag::Reset, offset, len);
}

void Shell::playmovie(const vector<string> &arg) {
  MovieAsset *ma = arg.size() ? movieasset(arg[0]) : 0;
  if (ma) ma->Play(0);
}

void Shell::loadsound(const vector<string> &arg) {
  static int id = 1;
  if (arg.empty()) return;
  SoundAsset *a = new SoundAsset();
  a->filename = arg[0];
  a->name = StrCat("sa", id++);
  a->Load();
  INFO("loaded ", a->name);
}

void Shell::loadmovie(const vector<string> &arg) {
  static int id = 1;
  if (arg.empty()) return;
  MovieAsset *ma = new MovieAsset();
  SoundAsset *a = &ma->audio;
  ma->audio.name = ma->video.name = StrCat("ma", id++);
  ma->Load(arg[0].c_str());
  INFO("loaded ", ma->name);
  if (a->wav) INFO("loaded ", a->name, " : ", a->filename, " chans=", a->channels, " sr=", a->sample_rate, " ", a->wav->ring.size);
}

void Shell::copy(const vector<string> &arg) {
  SoundAsset *src = 0, *dst = 0;
  if (!(src = app->shell.soundasset(arg.size() > 0 ? arg[0] : "")) ||
      !(dst = app->shell.soundasset(arg.size() > 1 ? arg[1] : ""))) { INFO("copy <src> <dst>"); return; }

  INFOf("copy %s %d %d %d %s %d %d %d",
        src->name.c_str(), src->sample_rate, src->channels, src->seconds,
        dst->name.c_str(), dst->sample_rate, dst->channels, dst->seconds);

  RingBuf::Handle srch(src->wav), dsth(dst->wav);
  dsth.CopyFrom(&srch);
}

void shell_filter(const vector<string> &arg, bool FFTfilter, int taps, int hop=0) {
  SoundAsset *sa=0; vector<double> filter; double cutoff=0;

  if (arg.size() > 0) sa     = app->shell.soundasset(arg[0]);
  if (arg.size() > 2) cutoff = atof(arg[2]);
  if (arg.size() > 1) {
    filter.resize(taps);
    if        (arg[1] == "low") {
      for (int i=0; i<taps; i++) filter[i] = LowPassFilter(taps, i, (int)cutoff);
    } else if (arg[1] == "high") {
      for (int i=0; i<taps; i++) filter[i] = HighPassFilter(taps, i, (int)cutoff);
    } else if (arg[1] == "preemph") {
      taps = 2;
      filter = PreEmphasisFilter();
    }
  }
  if (arg.size() > 3) { taps = atoi(arg[3]); hop = taps/2; }

  if (!sa || filter.empty() || !taps) {
    INFO("filter <asset> <low,high> <cutoff> [taps]");
    return;
  }

  RingBuf filtered(sa->wav->samples_per_sec, sa->wav->ring.size);
  RingBuf::Handle I(sa->wav), O(&filtered);

  if (FFTfilter) {
    FFTFilterCompile(taps, &filter[0]);
    if (FFTFilter(&I, &O, taps, hop, &filter[0])) return;
  }
  else {
    if (LFL::Filter(&I, &O, taps, &filter[0])) return;
  }

  if (1) {
    int N=20; string b="input = ";
    for (int i=0; i<N; i++) StringAppendf(&b, "x[%d]=%f, ", i, I.Read(i));
    INFO(b);

    b = "output = ";
    for (int i=0; i<N; i++) StringAppendf(&b, "y[%d]=%f, ", i, O.Read(i));
    INFO(b);
  }

  app->audio->QueueMixBuf(&O);
}

void Shell::filter   (const vector<string> &arg) { shell_filter(arg, false, 16); }
void Shell::fftfilter(const vector<string> &arg) { shell_filter(arg, true, FLAGS_feat_window, FLAGS_feat_hop); }

void Shell::f0(const vector<string> &arg) {
  SoundAsset *sa=0; int offset=0; int method=F0EstmMethod::Default;

  if (arg.size() > 0) sa = app->shell.soundasset(arg[0]);
  if (arg.size() > 1) offset = atoi(arg[1]);
  if (arg.size() > 2) method = atoi(arg[2]);

  if (!sa || !sa->wav || sa->wav->ring.size < offset+FLAGS_feat_window) {
    INFO("f0 <asset> <offset>");
    return;
  }

  if (offset) {
    RingBuf::Handle I(sa->wav, offset);
    float f0 = FundamentalFrequency(&I, FLAGS_feat_window, 0, method);
    INFO("f0 = (", sa->name, ":", offset, ") = ", f0);    
  }
  else {
    RingBuf::Handle I(sa->wav, offset);
    Matrix *f0 = F0Stream(&I, 0, FLAGS_feat_window, FLAGS_feat_hop, method);
    for (int i=0; i<f0->M; /**/) {
      char buf[1024]; int len=0;
      for (int j=0; j<20 && i<f0->M; j++,i++) len += sprint(buf+len, sizeof(buf)-len, "%.2f, ", f0->row(i)[0]);
      INFO(buf);
    }
  }
}

void Shell::sinth(const vector<string> &a) { 
  int hz[3] = { 440, 0, 0};
  for (int i=0; i<sizeofarray(hz) && i<a.size(); i++) hz[i] = atof(a[i]);
  Sinthesize(app->audio, hz[0], hz[1], hz[2]);
}

void Shell::writesnap(const vector<string> &a) {
  SoundAsset *sa = app->shell.soundasset(a.size() ? a[0] : "snap");
  if (sa) {
    string filename = StrCat(LFAppDownloadDir(), "snap.wav"); 
    RingBuf::Handle B(sa->wav);
    LocalFile lf(filename, "r");
    WavWriter w(&lf);
    int ret = w.Write(&B);
    INFO("wrote ", filename, " ret ", ret);
  }
}

void Shell::fps(const vector<string>&) { INFO("FPS ", FPS()); }

void Shell::wget(const vector<string> &a) {
  if (a.empty()) return;
  Singleton<HTTPClient>::Get()->WGet(a[0]);
}

void Shell::MessageBox(const vector<string> &a) { Dialog::MessageBox(Join(a, " ")); }
void Shell::TextureBox(const vector<string> &a) { Dialog::TextureBox(a.size() ? a[0] : ""); }

void Shell::Slider(const vector<string> &a) {
  if (a.empty()) { INFO("slider <flag_name> [total] [inc]"); return; }
  string flag_name = a[0];
  float total = a.size() >= 1 ? atof(a[1]) : 0;
  float inc   = a.size() >= 2 ? atof(a[2]) : 0;
  new SliderFlagDialog(flag_name, total ? total : 100, inc ? inc : 1);
}

void Shell::Edit(const vector<string> &a) {
  string s = Asset::FileContents("lfapp_vertex.glsl");
  new EditorDialog(screen, Fonts::Default(), new BufferFile(s));
}

void Shell::cmds(const vector<string>&) {
  for (auto i = command.begin(); i != command.end(); ++i) INFO(i->name);
}

void Shell::flags(const vector<string>&) { Singleton<FlagMap>::Get()->Print(); }

void Shell::binds(const vector<string>&) { }

}; // namespace LFL