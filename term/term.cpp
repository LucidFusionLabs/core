/*
 * $Id: term.cpp 1336 2014-12-08 09:29:59Z justin $
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
#include "lfapp/gui.h"

using namespace LFL;

DEFINE_int(peak_fps,   60,    "Peak FPS");
DEFINE_int(normal_fps, 15,    "Normal peak FPS");
DEFINE_bool(draw_fps,  false, "Draw FPS");

Scene scene;
BindMap *binds;
Shader warpershader;
AnyBoolSet effects_mode;
SimpleBrowser *image_browser;

void MyNewLinkCB(TextArea::Link *link) {
    string image_url = link->widget.link;
    if (!FileSuffix::Image(image_url)) {
        string prot, host, port, path;
        if (HTTP::URL(image_url.c_str(), &prot, &host, &port, &path) &&
            SuffixMatch(host, "imgur.com") && !FileSuffix::Image(path)) {
            image_url += ".jpg";
        } else { 
            return;
        }
    }
    link->image_src.SetNameValue("src", image_url);
    link->image.setAttributeNode(&link->image_src);
    image_browser->Open(image_url, &link->image);
}

void MyHoverLinkCB(TextArea::Link *link) {
    Asset *a = link ? link->image.asset : 0;
    if (!a) return;
    a->tex.Bind();
    screen->gd->SetColor(Color::white - Color::Alpha(0.2));
    Box::DelBorder(screen->Box(), screen->width*.2, screen->height*.2).Draw();
}

struct MyTerminalWindow {
    Process process;
    Terminal *terminal=0;
    Shader *activeshader;
    int font_size;
    AnyBoolElement effects_mode;

    MyTerminalWindow() : activeshader(&app->video.shader_default), font_size(FLAGS_default_font_size), effects_mode(&::effects_mode) {}
    ~MyTerminalWindow() { if (process.in) app->scheduler.DelWaitForeverSocket(fileno(process.in)); }

    void Open() {
        setenv("TERM", "screen", 1);
        string shell = BlankNull(getenv("SHELL"));
        CHECK(!shell.empty());
        const char *av[] = { shell.c_str(), 0 };
        CHECK_EQ(process.OpenPTY(av), 0);
        app->scheduler.AddWaitForeverSocket(fileno(process.in), SocketSet::READABLE, 0);

        terminal = new Terminal(fileno(process.out), screen, Fonts::Get(FLAGS_default_font, font_size, Color::white));
        terminal->new_link_cb = MyNewLinkCB;
        terminal->hover_link_cb = MyHoverLinkCB;
        terminal->Draw(screen->Box(), false);
        terminal->active = true;
    }
};

void UpdateTargetFPS() {
    FLAGS_lfapp_wait_forever = !effects_mode.Get();
    FLAGS_target_fps = FLAGS_lfapp_wait_forever ? FLAGS_normal_fps : FLAGS_peak_fps;
}

int Frame(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) {
    MyTerminalWindow *tw = (MyTerminalWindow*)W->user1;
    Box root = screen->Box();

    bool font_changed = tw->terminal->font->size != tw->terminal->line_fb.font_size, terminal_updated = false;
    bool resized = tw->terminal->line_fb.w != root.w || tw->terminal->line_fb.h != root.h;
    bool custom_shader = tw->activeshader != &app->video.shader_default, dont_skip = flag & FrameFlag::DontSkip;

    string terminal_output = NBRead(fileno(tw->process.in), 4096);
    if (!terminal_output.empty()) tw->terminal->Write(terminal_output);
    if (!terminal_output.empty() || resized || font_changed || (custom_shader && dont_skip) ||
        tw->terminal->mouse_gui.events.hover) {
        tw->terminal->mouse_gui.Deactivate();
        tw->terminal->Draw(root, custom_shader);
        terminal_updated = true;
    }
    // else return -1;

    tw->effects_mode.Set(custom_shader || W->console->animating);
    UpdateTargetFPS();
    if (0 && !terminal_updated && !tw->effects_mode.Get() && !W->events.bind && !W->events.mouse_click &&
        !W->console->events.total && !tw->terminal->selection_changing && !dont_skip) return -1;

    W->gd->DrawMode(DrawMode::_2D);
    tw->terminal->Draw(root, tw->activeshader);

    if (!custom_shader) {
        // ((TextGUI*)tw->terminal)->Draw(0, root.x + tw->terminal->cursor.p.x, root.y + tw->terminal->cursor.p.y);
        // tw->terminal->DrawOrCopySelection();
    }
    screen->DrawDialogs();
    if (FLAGS_draw_fps) Fonts::Default()->Draw(StringPrintf("FPS = %.2f", FPS()), point(W->width*.85, 0));
    return 0;
}

void SetFontSize(int n) {
    MyTerminalWindow *tw = (MyTerminalWindow*)screen->user1;
    tw->font_size = n;
    tw->terminal->font = Fonts::Get(FLAGS_default_font, tw->font_size, Color::white);
    screen->Reshape(tw->terminal->font->fixed_width * tw->terminal->term_width,
                    tw->terminal->font->height      * tw->terminal->term_height);
}
void MyIncreaseFontCmd(const vector<string>&) { SetFontSize(((MyTerminalWindow*)screen->user1)->font_size + 1); }
void MyDecreaseFontCmd(const vector<string>&) { SetFontSize(((MyTerminalWindow*)screen->user1)->font_size - 1); }
void MyConsole(const vector<string>&) {
    MyTerminalWindow *tw = (MyTerminalWindow*)screen->user1;
    tw->effects_mode.Set(true);
    UpdateTargetFPS();
    app->shell.console(vector<string>());
}
void MyColorsCmd(const vector<string> &arg) {
    string colors_name = arg.size() ? arg[0] : "";
    MyTerminalWindow *tw = (MyTerminalWindow*)screen->user1;
    if      (colors_name ==       "vga") tw->terminal->SetColors(Singleton<Terminal::StandardVGAColors>::Get());
    else if (colors_name == "solarized") tw->terminal->SetColors(Singleton<Terminal::SolarizedColors>  ::Get());
}
void MyShaderCmd(const vector<string> &arg) {
    string shader_name = arg.size() ? arg[0] : "";
    MyTerminalWindow *tw = (MyTerminalWindow*)screen->user1;
    if (shader_name == "warper") tw->activeshader = &warpershader;
    else                         tw->activeshader = &app->video.shader_default;
}
void MyScrollRegionCmd(const vector<string> &arg) {
    if (arg.size() < 2) { ERROR("scroll_region b e"); return; }
    INFO("set scroll region ", arg[0], " ", arg[1]);
    MyTerminalWindow *tw = (MyTerminalWindow*)screen->user1;
    tw->terminal->SetScrollRegion(atoi(arg[0]), atoi(arg[1]), true);
}
void MyTermDebugCmd(const vector<string> &arg) {
    string out;
    MyTerminalWindow *tw = (MyTerminalWindow*)screen->user1;
    for (int i=0; i<tw->terminal->term_height; i++) {
        TextGUI::Line *l = &tw->terminal->line[-1-i];
        StrAppend(&out, -1-i, " ", l->p.DebugString(), " ", l->Text(), "\n");
    }
}

void MyWindowDefaults(LFL::Window *W) {
    W->width = 80*10;
    W->height = 25*17;
    W->caption = "Terminal";
    W->binds = binds;
    W->user1 = new MyTerminalWindow();
}
void MyNewWindow(const vector<string>&) {
    LFL::Window *new_window = new LFL::Window();
    MyWindowDefaults(new_window);
    CHECK(LFL::Window::Create(new_window));
    LFL::Window::MakeCurrent(new_window);
    app->video.CreateGraphicsDevice();
    screen->InitConsole();
    MyTerminalWindow *tw = (MyTerminalWindow*)screen->user1;
    tw->Open();
}
void MyWindowClosedCB() {
    delete (MyTerminalWindow*)screen->user1;
    delete screen;
}

extern "C" int main(int argc, const char *argv[]) {

    app->logfilename = StrCat(LFAppDownloadDir(), "term.txt");
    app->frame_cb = Frame;
    binds = new BindMap();
    MyWindowDefaults(screen);
    FLAGS_lfapp_wait_forever = true;
    FLAGS_target_fps = FLAGS_normal_fps;
    FLAGS_lfapp_video = FLAGS_lfapp_input = 1;
    // FLAGS_font_engine = "coretext";
    // FLAGS_default_font = "Monaco"; // "DejaVuSansMono-Bold.ttf"; // "Monaco";
    FLAGS_default_font = "VeraMono.ttf";
    FLAGS_default_font_size = 16;
    // FLAGS_default_font_flag = FontDesc::Mono;
    FLAGS_atlas_font_sizes = "32";
    FLAGS_default_missing_glyph = 42;

    app->scheduler.AddWaitForeverService(Singleton<HTTPClient>::Get());
    app->scheduler.AddWaitForeverService(Singleton <UDPClient>::Get());

    if (app->Create(argc, argv, __FILE__)) { app->Free(); return -1; }
    if (app->Init()) { app->Free(); return -1; }

    app->window_closed_cb = MyWindowClosedCB;
    app->shell.command.push_back(Shell::Command("colors", bind(&MyColorsCmd, _1)));
    app->shell.command.push_back(Shell::Command("shader", bind(&MyShaderCmd, _1)));
    app->shell.command.push_back(Shell::Command("scroll_region", bind(&MyScrollRegionCmd, _1)));
    app->shell.command.push_back(Shell::Command("term_debug", bind(&MyTermDebugCmd, _1)));

    binds->Add(Bind('=', Key::Modifier::Cmd, Bind::CB(bind(&MyIncreaseFontCmd, vector<string>()))));
    binds->Add(Bind('-', Key::Modifier::Cmd, Bind::CB(bind(&MyDecreaseFontCmd, vector<string>()))));
    binds->Add(Bind('n', Key::Modifier::Cmd, Bind::CB(bind(&MyNewWindow,       vector<string>()))));
    binds->Add(Bind('6', Key::Modifier::Cmd, Bind::CB(bind(&MyConsole,         vector<string>()))));

    string lfapp_vertex_shader = LocalFile::FileContents(StrCat(ASSETS_DIR, "lfapp_vertex.glsl"));
    string warper_shader = LocalFile::FileContents(StrCat(ASSETS_DIR, "warper.glsl"));
    Shader::Create("warpershader", lfapp_vertex_shader.c_str(), warper_shader.c_str(),
                   "#define TEX2D\n#define VERTEXCOLOR\n", &warpershader);

    image_browser = new SimpleBrowser();
    MyTerminalWindow *tw = (MyTerminalWindow*)screen->user1;
    tw->Open();

    INFO("Starting Terminal ", FLAGS_default_font, " (w=", tw->terminal->font->fixed_width,
                                                   ", h=", tw->terminal->font->height, ")");

    app->scheduler.Start();
    return app->Main();
}
