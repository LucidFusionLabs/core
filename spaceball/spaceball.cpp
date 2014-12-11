/*
 * $Id: spaceball.cpp 1336 2014-12-08 09:29:59Z justin $
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
#include "lfapp/video.h"
#include "lfapp/network.h"
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/gui.h"
#include "lfapp/game.h"

#include "spaceballserv.h"

namespace LFL {
struct Editor : public TextArea {
    struct LineOffset { 
        int offset, size, font_size, wrapped_line_number; float width; 
        LineOffset(int O=0, int S=0, int FS=0, int WLN=0, float W=0) :
            offset(O), size(S), font_size(FS), wrapped_line_number(WLN), width(W) {}
        bool operator<(const LineOffset &l) const { return wrapped_line_number < l.wrapped_line_number; }
    };
    typedef pair<int, int> LineOffsetSegment, WrappedLineOffset;
    WrappedLineOffset first_line, last_line;
    int last_fb_lines=0, wrapped_lines=0;
    float last_v_scrolled=0;
    shared_ptr<File> file;
    vector<LineOffset> file_line;
    Editor(Window *W, Font *F, File *I) : TextArea(W, F), file(I) { /*line_fb.wrap=1;*/ BuildLineMap(); }

    bool Wrap() const { return line_fb.wrap; }
    void BuildLineMap() {
        int ind=0, offset=0;
        for (const char *l = file->nextlineraw(&offset); l; l = file->nextlineraw(&offset))
            file_line.push_back(LineOffset(offset, file->nr.record_len, TextArea::font->size,
                                           ind++, TextArea::font->Width(l)));
    }
    void UpdateWrappedLines(int cur_font_size, int box_width) {
        wrapped_lines = 0;
        for (auto &l : file_line) {
            wrapped_lines += 1 + l.width * cur_font_size / l.font_size / box_width;
            l.wrapped_line_number = wrapped_lines;
        }
    }
    void GetWrappedLineOffset(float percent, WrappedLineOffset *out) const {
        if (!Wrap()) { *out = WrappedLineOffset(percent * file_line.size(), 0); return; }
        int wrapped_line = percent * wrapped_lines;
        auto it = lower_bound(file_line.begin(), file_line.end(), LineOffset(0,0,0,wrapped_line));
        if (it == file_line.end()) { *out = WrappedLineOffset(file_line.size(), 0); return; }
        int wrapped_line_index = it - file_line.begin();
        *out = WrappedLineOffset(wrapped_line_index, wrapped_line - wrapped_line_index - 1);
    }
    int Distance(const WrappedLineOffset &o, bool reverse) {
        int dist = 0;
  //      for (int i=a_first_line; i<=b_first_line; i++) {
//            dist += 
        return abs(o.first - first_line.first);
    }
    void UpdateLines(float v_scrolled, float h_scrolled) {
        bool resized = last_fb_lines != line_fb.lines;
        if (resized) { line.Clear(); if (Wrap()) UpdateWrappedLines(TextArea::font->size, line_fb.w); }
        else if (Equal(last_v_scrolled, v_scrolled)) return;

        LineOffsetSegment read_lines;
        WrappedLineOffset new_first_line, new_last_line;
        GetWrappedLineOffset(v_scrolled, &new_first_line);
        bool reverse = new_first_line < first_line && !resized;
        int dist = Distance(new_first_line, reverse);
        if (dist < line_fb.lines && !resized) {
            if (reverse) read_lines = LineOffsetSegment(new_first_line.first, dist);
            else         read_lines = LineOffsetSegment(new_first_line.first + line_fb.lines - dist, dist);
        } else           read_lines = LineOffsetSegment(new_first_line.first, line_fb.lines);
        // can reduce read_lines here

        int add_blank_lines = Typed::Max<int>(0, Typed::Min<int>(dist, read_lines.first + read_lines.second - file_line.size())), read_len=0;
        read_lines.second = Typed::Max<int>(0, read_lines.second - add_blank_lines);
        for (int i=read_lines.first, n=i+read_lines.second; i<n; i++) read_len += file_line[i].size + (i<(n-1));

        string buf(read_len, 0);
        file->seek(file_line[read_lines.first].offset, File::Whence::SET);
        CHECK_EQ(read_len, file->read((char*)buf.data(), read_len));
        Line *L = 0;
        line_fb.fb.Attach();

        for (int i=0, wl=0, tl=read_lines.second, bo=0, l; i<tl && wl<tl; i++, bo += l+(!reverse || i)) {
            l = file_line[read_lines.first + (reverse ? (read_lines.second-1-i) : i)].size;
            if (reverse) (L = line.PushFront())->AssignText(buf.substr(read_len - bo - l, l));
            else         (L = line.PushBack ())->AssignText(buf.substr(bo,                l));
            if (reverse && !resized) line.PopBack (1);
            else if (      !resized) line.PopFront(1);
            if (reverse) wl += line_fb.PushFrontAndUpdate(L);
            else         wl += line_fb. PushBackAndUpdate(L);
        }
        for (int i=0; !reverse && i<add_blank_lines; i++) { 
            (L = line.PushBack())->AssignText(" ");
            if (!resized) line.PopFront(1);
            line_fb.PushBackAndUpdate(L);
        }
        line_fb.fb.Release();
        first_line = new_first_line;
        last_line = new_last_line;
        last_fb_lines = line_fb.lines;
        last_v_scrolled = v_scrolled;
    }
    void Draw(const Box &box, float v_scrolled, float h_scrolled) {
        TextArea::Draw(box, true);
        UpdateLines(v_scrolled, h_scrolled);
    }
};

struct EditorDialog : public Dialog {
    Editor editor;
    Widget::Scrollbar v_scrollbar, h_scrollbar;
    EditorDialog(Window *W, Font *F, File *I) : Dialog(.5, .5), editor(W, F, I),
    v_scrollbar(this), h_scrollbar(this, Box(), Widget::Scrollbar::Flag::AttachedHorizontal) {}

    void Layout() {
        Dialog::Layout();
        if (1)              v_scrollbar.LayoutAttached(box.Dimension());
        if (!editor.Wrap()) h_scrollbar.LayoutAttached(box.Dimension());
    }
    void Draw() {
        Dialog::Draw();
        editor.Draw(box, v_scrollbar.scrolled, h_scrollbar.scrolled);
        GUI::Draw();
        if (1)              v_scrollbar.Update();
        if (!editor.Wrap()) h_scrollbar.Update();
    }
};

BindMap binds;
AssetMap asset;
SoundAssetMap soundasset;
SpaceballSettings sbsettings;
SpaceballMap *sbmap;
Scene scene;
Game *world;
GameMenuGUI *menubar;
GameChatGUI *chat;
GamePlayerListGUI *playerlist;
GameMultiTouchControls *touchcontrols;
vector<string> save_settings;
unsigned fb_tex1, fb_tex2;
int map_transition;
FrameBuffer framebuffer;
Shader fadershader, warpershader, explodeshader;
HelperGUI *helper;

DEFINE_bool  (draw_fps,      true,                                  "Draw FPS");
DEFINE_int   (default_port,  27640,                                 "Default port");
DEFINE_string(master,        "lucidfusionlabs.com:27994/spaceball", "Master server list");
DEFINE_string(player_name,   "",                                    "Player name");
DEFINE_bool  (first_run,     true,                                  "First run of program");

#define LFL_BUILTIN_SERVER
#ifdef  LFL_BUILTIN_SERVER
SpaceballServer *builtin_server;
bool             builtin_server_enabled;
#endif

// Rippling caustics
TexSeq caust;

// Trails
typedef Particles<256, 1, true> BallTrails;
BallTrails ball_trail("BallTrails", true, .05, .05, 0, 0);
Entity *ball_particles;

// Shooting stars
typedef Particles<256, 1, true> ShootingStars;
ShootingStars shooting_stars("ShootingStars", true, .1, .2, 0, 0);
Entity *star_particles;

// Fireworks
typedef Particles<1024, 1, true> Fireworks;
Fireworks fireworks("Fireworks", true);
vector<v3> fireworks_positions;

#define MyShip SpaceballGame::Ship
#define MyBall SpaceballGame::Ball

int Frame(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag);

Geometry *FieldGeometry(const Color &rg, const Color &bg, const Color &fc) {
    vector<v3> verts, norm; vector<v2> tex; vector<Color> col; int ci=0;
    SpaceballGame::FieldDefinition *fd = SpaceballGame::FieldDefinition::get();
    v3 goals = SpaceballGame::Goals::get(), up=v3(0,1,0), fwd=v3(0,0,1), rev(0,0,-1);
    float tx=10, ty=10;

    /* field */
    verts.push_back(fd->B); norm.push_back(up); tex.push_back(v2(0,  0));  col.push_back(fc.a(10 + ci++ * 6.0/255));
    verts.push_back(fd->C); norm.push_back(up); tex.push_back(v2(tx, 0));  col.push_back(fc.a(10 + ci++ * 6.0/255));
    verts.push_back(fd->G); norm.push_back(up); tex.push_back(v2(tx, ty)); col.push_back(fc.a(10 + ci++ * 6.0/255));
    verts.push_back(fd->B); norm.push_back(up); tex.push_back(v2(0,  0));  col.push_back(fc.a(10 + ci++ * 6.0/255));
    verts.push_back(fd->G); norm.push_back(up); tex.push_back(v2(tx, ty)); col.push_back(fc.a(10 + ci++ * 6.0/255));
    verts.push_back(fd->F); norm.push_back(up); tex.push_back(v2(0,  ty)); col.push_back(fc.a(10 + ci++ * 6.0/255));
    tx *= .15;
    ty *= .15;

    /* red goal */
    ci = 0;
    verts.push_back(fd->B * goals); norm.push_back(rev); tex.push_back(v2(0,  0));  col.push_back(rg.a(10 + ci++ * 6.0/255));
    verts.push_back(fd->C * goals); norm.push_back(rev); tex.push_back(v2(tx, 0));  col.push_back(rg.a(10 + ci++ * 6.0/255));
    verts.push_back(fd->D * goals); norm.push_back(rev); tex.push_back(v2(tx, ty)); col.push_back(rg.a(10 + ci++ * 6.0/255));
    verts.push_back(fd->B * goals); norm.push_back(rev); tex.push_back(v2(0,  0));  col.push_back(rg.a(10 + ci++ * 6.0/255));
    verts.push_back(fd->D * goals); norm.push_back(rev); tex.push_back(v2(tx, ty)); col.push_back(rg.a(10 + ci++ * 6.0/255));
    verts.push_back(fd->A * goals); norm.push_back(rev); tex.push_back(v2(0,  ty)); col.push_back(rg.a(10 + ci++ * 6.0/255));

    /* blue goal */
    ci = 0;
    verts.push_back(fd->F * goals); norm.push_back(fwd); tex.push_back(v2(0,  0));  col.push_back(bg.a(10 + ci++ * 6.0/255));
    verts.push_back(fd->G * goals); norm.push_back(fwd); tex.push_back(v2(tx, 0));  col.push_back(bg.a(10 + ci++ * 6.0/255));
    verts.push_back(fd->H * goals); norm.push_back(fwd); tex.push_back(v2(tx, ty)); col.push_back(bg.a(10 + ci++ * 6.0/255));
    verts.push_back(fd->F * goals); norm.push_back(fwd); tex.push_back(v2(0,  0));  col.push_back(bg.a(10 + ci++ * 6.0/255));
    verts.push_back(fd->H * goals); norm.push_back(fwd); tex.push_back(v2(tx, ty)); col.push_back(bg.a(10 + ci++ * 6.0/255));
    verts.push_back(fd->E * goals); norm.push_back(fwd); tex.push_back(v2(0,  ty)); col.push_back(bg.a(10 + ci++ * 6.0/255));

    return new Geometry(GraphicsDevice::Triangles, verts.size(), &verts[0], 0, &tex[0], &col[0]);
}

Geometry *FieldLines(float tx, float ty) {
    SpaceballGame::FieldDefinition *fd = SpaceballGame::FieldDefinition::get();
    vector<v3> verts; vector<v2> tex;
    verts.push_back(fd->B); tex.push_back(v2(0,  0));
    verts.push_back(fd->C); tex.push_back(v2(tx, 0));
    verts.push_back(fd->G); tex.push_back(v2(tx, ty));
    verts.push_back(fd->B); tex.push_back(v2(0,  0));
    verts.push_back(fd->G); tex.push_back(v2(tx, ty));
    verts.push_back(fd->F); tex.push_back(v2(0,  ty));
    return new Geometry(GraphicsDevice::Triangles, verts.size(), &verts[0], 0, &tex[0], (Color*)0);
}

void SpaceballMap::Draw(const Entity &camera) { skybox.Draw(); }
void SpaceballMap::Load(const string &home_name, const string &away_name) {
    home = SpaceballTeam::Get(home_name);
    away = SpaceballTeam::Get(away_name);
    if (!home || !away) { ERROR("unknown team: ", home_name, " or ", away_name); return; }
    skybox.Load(home->skybox_name);

    asset("shipred" )->col = home->ship_color.diffuse;
    asset("shipblue")->col = away->ship_color.diffuse;

    Asset *field = asset("field");
    Typed::Replace<Geometry>(&field->geometry, FieldGeometry(home->goal_color, away->goal_color, home->field_color));

    screen->gd->EnableLight(0);
    screen->gd->Light(0, GraphicsDevice::Ambient,  home->light.color.ambient.x);
    screen->gd->Light(0, GraphicsDevice::Diffuse,  home->light.color.diffuse.x);
    screen->gd->Light(0, GraphicsDevice::Specular, home->light.color.specular.x);
}

void ShipDraw(Asset *a, Entity *e) {
    static Geometry *stripes = Geometry::LoadOBJ(StrCat(ASSETS_DIR, "ship_stripes.obj"));
    scene.Select(stripes);
    screen->gd->SetColor(e->color1);
    Scene::Draw(stripes, e);

    scene.Select(a);
    Shader *anim_shader = 0;
    if (e->animation.ShaderActive()) {
        anim_shader = e->animation.shader;
        screen->gd->UseShader(anim_shader);
        anim_shader->setUniform1f("time", e->animation.Percent());
    }
    Scene::Draw(a->geometry, e);
    if (anim_shader) screen->gd->UseShader(0);

    static Timer lightning_timer;
    static float last_lightning_offset = 0;
    static int lightning_texcoord_min_int_x = 0;
    static Font *lightning_font = Fonts::Get("lightning", 0, Color::black);
    static Font::Glyph *lightning_glyph = &lightning_font->glyph->table[2];
    static Geometry *lightning_obj = 0;
    if (!lightning_obj) {
        float lightning_glyph_texcoord[4];
        memcpy(lightning_glyph_texcoord, lightning_glyph->tex.coord, sizeof(lightning_glyph_texcoord));
        lightning_glyph_texcoord[Texture::CoordMaxX] *= .1;
        lightning_obj = Geometry::LoadOBJ(StrCat(ASSETS_DIR, "ship_lightning.obj"), lightning_glyph_texcoord);
    }
    screen->gd->BindTexture(GraphicsDevice::Texture2D, lightning_glyph->tex.ID);

    float lightning_offset = (e->namehash % 11) / 10.0;
    lightning_obj->ScrollTexCoord(-.4 * ToSeconds(lightning_timer.time(true)),
                                  lightning_offset - last_lightning_offset,
                                  &lightning_texcoord_min_int_x);
    last_lightning_offset = lightning_offset;

    scene.Select(lightning_obj);
    screen->gd->EnableBlend();
    Color c = e->color1;
    c.scale(2.0);
    screen->gd->SetColor(c);
    Scene::Draw(lightning_obj, e);
}

void SetInitialCameraPosition() {
    // position camera for a nice earth shot; from command 'campos'
    screen->camMain->pos = v3(5.54,1.70,4.39);
    screen->camMain->ort = v3(-0.14,0.02,-0.69);
    screen->camMain->up = v3(0.01,1.00,0.02);
}

struct SpaceballClient : public GameClient {
    Entity *ball=0;
    Time map_started;
    int last_scored_team=0;
    string last_scored_PlayerName;
    typedef Particles<16, 1, true> Thrusters;
    vector<v3> thrusters_transform;
    SpaceballClient(Game *w, GUI *PlayerList, TextArea *Chat) : GameClient(w, PlayerList, Chat), map_started(Now()) {
        thrusters_transform.push_back(v3(-.575, -.350, -.1));
        thrusters_transform.push_back(v3( .525, -.350, -.1));
        thrusters_transform.push_back(v3(-.025,  .525, -.1));
    }

    void moveboost(unsigned) { SpaceballGame::Ship::set_boost(&control); }
    void NewEntityCB(Entity *e) {
        Asset *a = e->asset;
        if (!a) return;
        if (a->name == string("ball")) ball = e;
        if (a->particleTexID) {
            Thrusters *thrusters = new Thrusters("Thrusters", true, .1, .25, 0, 0.1);
            thrusters->emitter_type = Thrusters::Emitter::GlowFade;
            thrusters->texture = a->particleTexID;
            thrusters->ticks_step = 10;
            thrusters->gravity = -0.1;
            thrusters->age_min = .2;
            thrusters->age_max = .6;
            thrusters->radius_decay = true;
            thrusters->billboard = true;
            thrusters->pos_transform = &thrusters_transform;
            thrusters->move_with_pos = true;
            e->particles = thrusters;
        }
    }
    void DelEntityCB(Entity *e) {
        if (!e) return;
        if (ball == e) ball = 0;
        if (e->particles) delete (Thrusters*)e->particles;
    }
    void AnimationChange(Entity *e, int NewID, int NewSeq) {
        static SoundAsset *bounce = soundasset("bounce");
        if      (NewID == SpaceballGame::AnimShipBoost) SystemAudio::PlaySoundEffect(bounce);
        else if (NewID == SpaceballGame::AnimExplode)   e->animation.Start(&explodeshader);
    }
    void RconRequestCB(const string &cmd, const string &arg, int seq) { 
        // INFO("cmd: ", cmd, " ", arg);
        if (cmd == "goal") {
            last_scored_team = ::atoi(arg.c_str()); 
            const char *scoredby = strchr(arg.c_str(), ' ');
            last_scored_PlayerName = scoredby ? scoredby+1 : "";
            INFO(last_scored_PlayerName, " scores for ", last_scored_team == Game::Team::Red ? "red" : "blue");

            unsigned updateInterval = last.time_recv_WorldUpdate[0] - last.time_recv_WorldUpdate[1];
            replay.start_ind = Typed::Max(1, (int)(last.WorldUpdate.size() - (SpaceballGame::ReplaySeconds-1) * 1000.0 / updateInterval));
            replay.while_seq = last.seq_WorldUpdate = seq;
            replay.start = Now();

            screen->camMain->pos = v3(1.73,   2.53, 16.83);
            screen->camMain->up  = v3(-0.01,  0.98, -0.19);
            screen->camMain->ort = v3(-0.03, -0.13, -0.69);

            if (last_scored_team == Game::Team::Blue) {
                screen->camMain->pos.z *= -1;
                screen->camMain->ort.z *= -1;
                screen->camMain->up.z *= -1;
            }
            screen->camMain->ort.norm();
            screen->camMain->up.norm();
        }
        else if (cmd == "win") {
            gameover.start_ind = ::atoi(arg.c_str());
            gameover.while_seq = last.seq_WorldUpdate = seq;
            gameover.start = Now();
        } else if (cmd == "map") {
            vector<string> args;
            Split(arg, isspace, &args);
            if (args.size() != 3) { ERROR("map ", arg); return; }
            framebuffer.Attach(fb_tex1);
            framebuffer.Render(Frame);
            sbmap->Load(args[0], args[1]);
            framebuffer.Attach(fb_tex2);
            framebuffer.Render(Frame);
            map_started = Now() - ::atoi(args[2].c_str());
            map_transition = Seconds(3);
        } else {
            ERROR("unknown rcon: ", cmd, " ", arg); return;
        }
    }
} *server;

struct TeamSelectGUI : public GUI {
    vector<SpaceballTeam> *teams;
    Font *font, *team_font;
    Widget::Button start_button;
    vector<Widget::Button> team_buttons;
    int home_team=0, away_team=0;

    TeamSelectGUI(LFL::Window *w) : GUI(w), teams(SpaceballTeam::GetList()),
    font(Fonts::Get("Origicide.ttf", 8, Color::white)), team_font(Fonts::Get("sbmaps", 0, Color::black)),
    start_button(this, 0, font, "start", MouseController::CB(bind(&TeamSelectGUI::Start, this))) {
        start_button.outline = &font->fg;
        team_buttons.resize(teams->size());
        for (int i=0; i<team_buttons.size(); i++) team_buttons[i] =
            Widget::Button(this, team_font->FindGlyph((*teams)[i].font_index), font, (*teams)[i].name,
                           MouseController::CB(bind(&TeamSelectGUI::SetHomeTeamIndex, this, i)));
    }

    void SetHomeTeamIndex(int n) { home_team = n; }
    void Start() { ShellRun("local_server"); }

    void Draw(Shader *MyShader) {
        glTimeResolutionShaderWindows(MyShader, Color(25, 60, 130, 120), box);
        GUI::Draw();
        screen->gd->SetColor(Color::white);
        BoxOutline().Draw(team_buttons[home_team].GetHitBoxBox());
    }

    void Layout() {
        box = Box::FromScreen(.1, .1, .8, .8);
        int bw=50, bh=50, sx=25, px=(box.w - (bw*4 + sx*3))/2;
        Flow flow(&box, font, Reset());
        flow.AppendNewlines(1);
        flow.p.x += px;
        for (int i = 0; i < team_buttons.size(); i++) {
            team_buttons[i].box = Box(bw, bh);
            team_buttons[i].Layout(&flow);
            flow.p.x += sx;

            if ((i+1) % 4 != 0) continue;
            flow.AppendNewlines(2);
            if (i+1 < team_buttons.size()) flow.p.x += px;
        }
        flow.layout.align_center = 1;
        start_button.box = Box::FromScreen(.4, .05);
        start_button.Layout(&flow);
        flow.Complete();
    }
} *team_select;

void MyLocalServerDisable() {
#ifdef LFL_BUILTIN_SERVER
    if (builtin_server_enabled) {
        builtin_server_enabled = false;
        app->network.Shutdown(builtin_server->svc);
        app->network.Disable(builtin_server->svc);
    }
#endif
}

void MyGameFinished(SpaceballGame *world) {
    if (world->game_players == SpaceballSettings::PLAYERS_MULTIPLE) return world->StartNextGame(builtin_server);

    MyLocalServerDisable();
    server->reset();

    if (world->game_type == SpaceballSettings::TYPE_TOURNAMENT) team_select->display = true;
    else { menubar->ToggleDisplay(); menubar->selected = 1; }

    SetInitialCameraPosition();
}

void MyLocalServerEnable(int game_type) {
    MyLocalServerDisable();

#ifdef LFL_BUILTIN_SERVER
    if (!builtin_server) {
        builtin_server = new SpaceballServer(StrCat(FLAGS_player_name, "'s server"), FLAGS_default_port, 20, &asset.vec);
        builtin_server->bots = new SpaceballBots(builtin_server->world);
        builtin_server->World()->game_finished_cb = bind(&MyGameFinished, builtin_server->World());
    }

    if (!builtin_server_enabled) {
        builtin_server_enabled = true;
        app->network.Enable(builtin_server->svc);
    }

    SpaceballGame *world = builtin_server->World();
    if (menubar->selected == 1) world->game_players = SpaceballSettings::PLAYERS_SINGLE;
    else                        world->game_players = SpaceballSettings::PLAYERS_MULTIPLE;
    world->game_type = sbsettings.GetIndex(SpaceballSettings::GAME_TYPE);
    world->game_limit = sbsettings.GetIndex(SpaceballSettings::GAME_LIMIT);
    world->game_control = sbsettings.GetIndex(SpaceballSettings::GAME_CONTROL);
    bool tourny = world->game_type == SpaceballSettings::TYPE_TOURNAMENT;
    if (!tourny) world->RandomTeams();
    else { 
        world->home = &(*team_select->teams)[team_select->home_team];
        world->away = SpaceballTeam::GetRandom(world->home);
    }

    builtin_server->bots->Clear();
    bool empty = game_type == SpaceballSettings::TYPE_EMPTYCOURT, spectator = game_type == SpaceballSettings::TYPE_SIMULATION;
    if (!empty) {
        builtin_server->bots->Insert(SpaceballGame::PlayersPerTeam*2);
        if (!spectator) builtin_server->bots->RemoveFromTeam(tourny ? Game::Team::Home : Game::Team::Random());
    }

    world->Reset();
#endif    
}

void MyLocalServerCmd(const vector<string>&) {
    int game_type = sbsettings.GetIndex(SpaceballSettings::GAME_TYPE);
    if (game_type == SpaceballSettings::TYPE_TOURNAMENT) {
        if (!team_select->display) { team_select->display = true; return; }
        team_select->display = false;
    }
    MyLocalServerEnable(game_type);
    server->connect("127.0.0.1", FLAGS_default_port);
}

void MyServerCmd(const vector<string> &arg) {
    MyLocalServerDisable();
    if (arg.empty()) { INFO("eg: server 192.168.1.144:", FLAGS_default_port); return; }
    server->connect(arg[0], FLAGS_default_port);
}

void MyGPlusClientCmd(const vector<string> &arg) {
#ifdef LFL_ANDROID
    menubar->ToggleDisplay();
    MyLocalServerDisable();
    if (arg.empty()) { INFO("eg: gplus_client participant_id"); return; }
    INFO("GPlusClient ", arg[0]);
    android_gplus_service(Singleton<GPlusClient>::Get());
    server->connectGPlus(arg[0]);
#endif
}

void MyGPlusServerCmd(const vector<string> &arg) {
#ifdef LFL_ANDROID
    menubar->ToggleDisplay();
    MyLocalServerEnable(SpaceballSettings::TYPE_EMPTYCOURT);
    if (arg.empty()) { INFO("eg: gplus_server participant_id"); return; }
    INFO("GPlusServer ", arg[0]);
    android_gplus_service(builtin_server->gplus_transport);
    server->connect("127.0.0.1", FLAGS_default_port);
#endif
}

void MySwitchPlayerCmd(const vector<string> &) {
    if (server) server->rcon("player_switch");
}

void MyFieldColorCmd(const vector<string> &arg) {
    Color fc(arg.size() ? arg[0] : "");
    Asset *field = asset("field");
    Typed::Replace<Geometry>(&field->geometry, FieldGeometry(sbmap->home->goal_color, sbmap->away->goal_color, fc));
    INFO("field_color = ", fc.HexString());
}

// LFL::Application FrameCB
int Frame(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) {
    if (Singleton<FlagMap>::Get()->dirty) {
        Singleton<FlagMap>::Get()->dirty = false;
        SettingsFile::write(save_settings, dldir(), "settings");
    }

    if (FLAGS_multitouch) {
        static int last_rpad_down = 0;
        bool changed = touchcontrols->rpad_down != last_rpad_down;
        last_rpad_down = touchcontrols->rpad_down;
        if (changed && touchcontrols->rpad_down == GameMultiTouchControls::DOWN) { MySwitchPlayerCmd(vector<string>()); }
        if (changed && touchcontrols->rpad_down == GameMultiTouchControls::UP)   { /* release = fire! */ }
        else                                                                     { server->moveboost(0); }
    }

#ifdef LFL_BUILTIN_SERVER
    if (builtin_server_enabled) builtin_server->frame();
#endif

    if (map_transition > 0) {
        map_transition -= clicks;
        screen->gd->DrawMode(DrawMode::_2D);
        glTimeResolutionShaderWindows(&warpershader, Color::black, Box::FromScreen(), &framebuffer.tex);

    } else {
        screen->camMain->look();
        shooting_stars.Update(clicks, 0, 0, 0);
        ball_trail.Update(clicks, 0, 0, 0);
        if (server->ball) ball_particles->pos = server->ball->pos;

        Scene::EntityVector deleted;
        Scene::LastUpdatedFilter scene_filter_deleted(0, server->last.time_frame, &deleted);

        screen->gd->Light(0, GraphicsDevice::Position, &sbmap->home->light.pos.x);
        sbmap->Draw(*screen->camMain);

        // Custom Scene::Draw();
        for (vector<Asset>::iterator a = asset.vec.begin(); a != asset.vec.end(); ++a) {
            if (a->zsort) continue;
            if (a->name == "lines") {
                scene.Draw(&(*a));
                screen->gd->EnableDepthTest();
            } else if (a->name == "ball") {
                scene.Draw(&(*a), &scene_filter_deleted);
            } else {
                scene.Draw(&(*a));
            }
        }

        scene.ZSort(asset.vec);
        scene.ZSortDraw(&scene_filter_deleted, clicks);
        server->WorldDeleteEntity(deleted);
    }

    screen->gd->DrawMode(DrawMode::_2D);
    chat->Draw();

    if (FLAGS_multitouch) {
        touchcontrols->Update(clicks);
        touchcontrols->Draw();

        // Game menu and player list buttons
        GUI *root = screen->gui_root;
        static Font *mobile_font = Fonts::Get("MobileAtlas", 0, Color::black);
        // static Widget::Button gamePlayerListButton(root, 0, 0, Box::FromScreen(.465, .05, .07, .05), MouseController::CB(bind(&GUI::ToggleDisplay, (GUI*)playerlist)));
        // static Widget::Button           helpButton(root, 0, 0, Box::FromScreen(.56,  .05, .07, .05), MouseController::CB(bind(&GUI::ToggleDisplay, (GUI*)helper)));

        // if (helper && gamePlayerListButton.init) helper->AddLabel(gamePlayerListButton.win, "player list", HelperGUI::Hint::UP, .08);
        // if (helper &&           helpButton.init) helper->AddLabel(          helpButton.win, "help",        HelperGUI::Hint::UPRIGHT);
        // gamePlayerListButton.Draw(mobile_font, 4);
        // helpButton.Draw(mobile_font, 6);
    }

    if (server->replay.enabled()) {
        if (server->ball) {
            // Replay camera tracks ball
            v3 targ = server->ball->pos + server->ball->vel;
            v3 yaw_delta = v3::norm(targ - v3(screen->camMain->pos.x, targ.y, screen->camMain->pos.z));
            v3 ort = v3::norm(targ - screen->camMain->pos);
            v3 delta = ort - yaw_delta;
            screen->camMain->up = v3(0,1,0) + delta;
            screen->camMain->ort = ort;
        }

        Box win(screen->width*.4, screen->height*.8, screen->width*.2, screen->height*.1, false);
        Asset *goal = asset("goal");
        goal->tex.Bind();
        win.Draw(goal->tex.coord);

        static Font *font = Fonts::Get("Origicide.ttf", 16, Color::white);
        font->Draw(StrCat(server->last_scored_PlayerName, " scores"),
                   Box(win.x, win.y - screen->height*.1, screen->width*.2, screen->height*.1, false), 
                   0, Font::Flag::AlignCenter | Font::Flag::NoWrap);

        SpaceballTeam *scored_team = server->last_scored_team == Game::Team::Home ? sbmap->home : sbmap->away;
        fireworks.rand_color_min = fireworks.rand_color_max = scored_team->goal_color;
        fireworks.rand_color_min.scale(.6);
        for (int i=0; i<Fireworks::MaxParticles; i++) {
            if (fireworks.particles[i].dead) continue;
            fireworks.particles[i].InitColor();
        }
        fireworks_positions[0].set(-screen->width*.2, screen->height*.8, 0);
        fireworks_positions[1].set(-screen->width*.8, screen->height*.8, 0);
        fireworks.Update(clicks, 0, 0, 0);
        fireworks.Draw();
        scene.Select();
    }

    if (server->gameover.enabled()) {
        Box win(screen->width*.4, screen->height*.9, screen->width*.2, screen->height*.1, false);
        static Font *font = Fonts::Get("Origicide.ttf", 16, Color::white);
        font->Draw(StrCat(server->gameover.start_ind == SpaceballGame::Team::Home ? sbmap->home->name : sbmap->away->name, " wins"),
                   win, 0, Font::Flag::AlignCenter);
    }

    if (server->replay.just_ended) {
        server->replay.just_ended = false;
        screen->camMain->ort = SpaceballGame::StartOrientation(server->team);
        screen->camMain->up  = v3(0, 1, 0);
    }

    if (team_select->display) team_select->Draw(fadershader.ID > 0 ? &fadershader : 0);

    // Press escape for menubar
    else if (menubar->display) menubar->Draw(clicks, fadershader.ID > 0 ? &fadershader : 0);

    // Press 't' to talk
    else if (chat->active) {}

    // Press tab for playerlist
    else if (playerlist->display || server->gameover.enabled()) {
        server->control.set_playerlist();
        playerlist->Draw(fadershader.ID > 0 ? &fadershader : 0);
    }

    // Press tick for console
    else screen->DrawDialogs();

    Scene::Select();

    if (helper && helper->display) {
        screen->gd->SetColor(helper->font->fg);
        BoxOutline().Draw(menubar->topbar.box);
        helper->Draw();
    }

    static Font *text = Fonts::Get(FLAGS_default_font, 8, Color::white);
    if (FLAGS_draw_fps)    text->Draw(StringPrintf("FPS = %.2f", FPS()),                    point(screen->width*.05, screen->height*.05));
    if (!menubar->display) text->Draw(intervalminutes(Now() - server->map_started),         point(screen->width*.93, screen->height*.97));
    if (!menubar->display) text->Draw(StrCat(sbmap->home->name, " vs ", sbmap->away->name), point(screen->width*.01, screen->height*.97));

    return 0;
}

void MyEditorCmd(const vector<string> &) {
    string s = LocalFile::filecontents(StrCat(ASSETS_DIR, "lfapp_vertex.glsl"));
    new EditorDialog(screen, Fonts::Get(FLAGS_default_font, 8, Color::white), new BufferFile(s.c_str(), s.size()));
}

}; // namespace LFL
using namespace LFL;

extern "C" int main(int argc, const char *argv[]) {

    app->logfilename = StrCat(dldir(), "spaceball.txt");
    app->frame_cb = Frame;
    screen->binds = &binds;
#if defined(LFL_ANDROID) || defined(LFL_IPHONE)
    FLAGS_target_fps = 30;
    screen->width = 420;
    screen->height = 380;
#else
    FLAGS_lfapp_multithreaded = true;
    FLAGS_target_fps = 50;
    screen->width = 620;
    screen->height = 480;
#endif
    FLAGS_far_plane = 1000;
    FLAGS_soundasset_seconds = 1;
    FLAGS_scale_font_height = 320;
    FLAGS_default_font = "Origicide.ttf";
    screen->caption = "Spaceball 6006";
    screen->multitouch_keyboard_x = .37;

    if (app->Create(argc, argv, __FILE__)) { app->Free(); return -1; }
    if (app->Init())                       { app->Free(); return -1; }
    INFO("BUILD Version ", "1.02.1");

    Fonts::InsertAtlas("MobileAtlas", "", 0, Color::black, 0);
    Fonts::InsertAtlas("dpad_atlas",  "", 0, Color::black, 0);
    Fonts::InsertAtlas("sbmaps",      "", 0, Color::black, 0);
    Fonts::InsertAtlas("lightning",   "", 0, Color::black, 0);

    save_settings.push_back("player_name");
    save_settings.push_back("first_run");
    save_settings.push_back("msens");
    SettingsFile::read(dldir(), "settings");
    Singleton<FlagMap>::Get()->dirty = false;

    if (FLAGS_player_name.empty()) {
#if defined(LFL_ANDROID)
        char buf[40];
        if (android_device_name(buf, sizeof(buf))) FLAGS_player_name = buf;
#endif
        if (FLAGS_player_name.empty()) FLAGS_player_name = "n00by";
    }

    if (FLAGS_first_run) {
        INFO("Welcome to Spaceball 6006, New Player.");
        Singleton<FlagMap>::Get()->Set("first_run", "0");
    }

    // assets.Add(Asset(name,     texture,          scale,            trans, rotate, geometry      hull,                            cubemap,     texgen, callback));
    asset.Add(Asset("particles",  "particle.png",   1,                0,     0,      0,            0,                               0,           0,      Asset::DrawCB(bind(&BallTrails::AssetDrawCB,    &ball_trail,     _1, _2))));
    asset.Add(Asset("stars",      "",               1,                0,     0,      0,            0,                               0,           0,      Asset::DrawCB(bind(&ShootingStars::AssetDrawCB, &shooting_stars, _1, _2))));
    asset.Add(Asset("glow",       "glow.png",       1,                0,     0,      0,            0,                               0,           0));
    asset.Add(Asset("field",      "",               1,                0,     0,      0,            0,                               0,           0,      Asset::DrawCB(bind(&TexSeq::draw, &caust, _1, _2))));
    asset.Add(Asset("lines",      "lines.png",      1,                0,     0,      0,            0,                               0,           0));
    asset.Add(Asset("ball",       "",               MyBall::radius(), 1,     0,      "sphere.obj", 0,                               0));
    asset.Add(Asset("ship",       "ship.png",       .05,              1,     0,      "ship.obj",   Cube::Create(MyShip::radius()),  0,                   Asset::DrawCB(bind(&ShipDraw, _1, _2))));
    asset.Add(Asset("shipred",    "",               0,                0,     0,      0,            0,                               0,           0));
    asset.Add(Asset("shipblue",   "",               0,                0,     0,      0,            0,                               0,           0));
    asset.Add(Asset("title",      "title.png",      1,                0,     0,      0,            0,                               0,           0));
    asset.Add(Asset("goal",       "goal.png",       1,                0,     0,      0,            0,                               0,           0));
    asset.Load();
    app->shell.assets = &asset;

    // soundasset.Add(SoundAsset(name,  filename,              ringbuf, channels, sample_rate, seconds ));
    soundasset.Add(SoundAsset("music",  "dstsecondballad.mp3", 0,       0,        0,           0       ));
    soundasset.Add(SoundAsset("bounce", "scififortyfive.wav",  0,       0,        0,           0       ));
    soundasset.Load();
    app->shell.soundassets = &soundasset;

    caust.load("%s%02d.%s", "caust", "png", 32);
    scene.Add(new Entity("field", asset("field")));
    asset("field")->blendt = GraphicsDevice::SrcAlpha;

    Asset *lines = asset("lines");
    lines->geometry = FieldLines(lines->tex.coord[2], lines->tex.coord[3]);
    scene.Add(new Entity("lines", lines));

    if (screen->gd->ShaderSupport()) {
        string lfapp_vertex_shader = LocalFile::filecontents(StrCat(ASSETS_DIR, "lfapp_vertex.glsl"));
        string lfapp_pixel_shader = LocalFile::filecontents(StrCat(ASSETS_DIR, "lfapp_pixel.glsl"));
        string vertex_shader = LocalFile::filecontents(StrCat(ASSETS_DIR, "vertex.glsl"));
        string fader_shader  = LocalFile::filecontents(StrCat(ASSETS_DIR, "fader.glsl"));
        string warper_shader = LocalFile::filecontents(StrCat(ASSETS_DIR, "warper.glsl"));
        string explode_shader = lfapp_vertex_shader;
        CHECK(StringReplace(&explode_shader, "// LFLPositionShaderMarker",
                                             LocalFile::filecontents(StrCat(ASSETS_DIR, "explode.glsl"))));

        Shader::create("fadershader",         vertex_shader.c_str(),       fader_shader.c_str(), "",                                          &fadershader);
        Shader::create("warpershader",  lfapp_vertex_shader.c_str(),      warper_shader.c_str(), "#define TEX2D \r\n#define VERTEXCOLOR\r\n", &warpershader);
        Shader::create("explodeshader",      explode_shader.c_str(), lfapp_pixel_shader.c_str(), "#define TEX2D \r\n#define NORMALS\r\n",     &explodeshader);
    }

    // ball trail
    ball_trail.emitter_type = BallTrails::Emitter::Sprinkler | BallTrails::Emitter::RainbowFade;
#if 0
    ball_trail.emitter_type = BallTrails::Emitter::Sprinkler | BallTrails::Emitter::GlowFade;
    ball_trail.floor = true;
    ball_trail.floorval = Spaceball::FieldDefinition::get()->B.y;
#endif
    ball_trail.ticks_step = 10;
    ball_trail.gravity = -3;
    ball_trail.age_min = 1;
    ball_trail.age_max = 3;
    ball_trail.radius_decay = false;
    ball_trail.texture = asset("particles")->tex.ID;
    ball_trail.billboard = true;
    ball_particles = new Entity("ball_particles", asset("particles"));
    scene.Add(ball_particles);

    // shooting stars
    asset("stars")->tex.ID = asset("particles")->tex.ID;
    shooting_stars.texture = asset("particles")->tex.ID;
    shooting_stars.burst = 16;
    shooting_stars.color = Color(1.0, 1.0, 1.0, 1.0);
    shooting_stars.ticks_step = 10;
    shooting_stars.gravity = 0;
    shooting_stars.age_min = .2;
    shooting_stars.age_max = .6;
    shooting_stars.radius_decay = true;
    shooting_stars.billboard = true;
    shooting_stars.vel = v3(0, 0, -1);
    shooting_stars.always_on = false;

    star_particles = new Entity("star_particles", asset("stars"));
    star_particles->pos = v3(0, -4, 5);
    // scene.add(star_particles);

    // fireworks
    fireworks_positions.resize(2);
    fireworks.texture = asset("particles")->tex.ID;
    fireworks.pos_transform = &fireworks_positions;
    fireworks.rand_color = true;

    menubar = new GameMenuGUI(screen, FLAGS_master.c_str(), FLAGS_default_port, asset("title"), asset("glow"));
    menubar->tab3_player_name.cmd_line.AssignText(FLAGS_player_name);
    menubar->settings = &sbsettings;
    menubar->display = true;

    playerlist = new GamePlayerListGUI(screen, "Spaceball 6006", "Team 1: silver", "Team 2: ontario");
    chat = new GameChatGUI(screen, 't', (GameClient**)&server);
    team_select = new TeamSelectGUI(screen);

    world = new SpaceballGame(&scene);
    server = new SpaceballClient(world, playerlist, chat);

    // init frame buffer
    framebuffer.Create(screen->width, screen->height, FrameBuffer::Flag::ReleaseFB);
    fb_tex1 = framebuffer.tex.ID;
    framebuffer.AllocTexture(&fb_tex2);

    // Color ships red and blue
    Asset *ship = asset("ship"), *shipred = asset("shipred"), *shipblue = asset("shipblue");
    ship->particleTexID = asset("glow")->tex.ID;
    ship->zsort = true;
    Asset::Copy(ship, shipred);
    Asset::Copy(ship, shipblue);
    shipred->color = shipblue->color = true;

    sbmap = new SpaceballMap();
    SpaceballTeam *home = SpaceballTeam::GetRandom();
    sbmap->Load(home->name, SpaceballTeam::GetRandom(home)->name);
    SetInitialCameraPosition();

    // add reflection to ball
    Asset *ball = asset("ball"), *sky = sbmap->SkyboxAsset();
    if (sky->tex.cubemap) {
        ball->tex.ID = sky->tex.ID;
        ball->tex.cubemap = sky->tex.cubemap;
        ball->texgen = TexGen::REFLECTION;
        ball->geometry->material = 0;
    }

    Game::Credits *credits = Singleton<Game::Credits>::Get();
    credits->push_back(Game::Credit("Game",         "koldfuzor",    "http://www.lucidfusionlabs.com/", ""));
    credits->push_back(Game::Credit("Skyboxes",     "3delyvisions", "http://www.3delyvisions.com/",    ""));
    credits->push_back(Game::Credit("Physics",      "Box2D",        "http://box2d.org/",               ""));
    credits->push_back(Game::Credit("Fonts",        "Cpr.Sparhelt", "http://www.facebook.com/pages/Magique-Fonts-Koczman-B%C3%A1lint/110683665690882", ""));
    credits->push_back(Game::Credit("Image format", "Libpng",       "http://www.libpng.org/",          ""));

    // start music
    SystemAudio::PlayBackgroundMusic(soundasset("music"));

    if (FLAGS_multitouch) {
        touchcontrols = new GameMultiTouchControls(server);
        helper = new HelperGUI(screen);
        const Box &lw = touchcontrols->lpad_win, &rw = touchcontrols->rpad_win;
        helper->AddLabel(Box(lw.x + lw.w*.15, lw.y + lw.h*.5,  1, 1), "move left",     HelperGUI::Hint::UPLEFT);
        helper->AddLabel(Box(lw.x + lw.w*.85, lw.y + lw.h*.5,  1, 1), "move right",    HelperGUI::Hint::UPRIGHT);
        helper->AddLabel(Box(lw.x + lw.w*.5,  lw.y + lw.h*.85, 1, 1), "move forward",  HelperGUI::Hint::UP);
        helper->AddLabel(Box(lw.x + lw.w*.5,  lw.y + lw.h*.15, 1, 1), "move back",     HelperGUI::Hint::DOWN);
        helper->AddLabel(Box(rw.x + rw.w*.15, rw.y + rw.h*.5,  1, 1), "turn left",     HelperGUI::Hint::UPLEFT);
        helper->AddLabel(Box(rw.x + rw.w*.85, rw.y + rw.h*.5,  1, 1), "turn right",    HelperGUI::Hint::UPRIGHT);
        helper->AddLabel(Box(rw.x + rw.w*.5,  rw.y + rw.h*.85, 1, 1), "burst forward", HelperGUI::Hint::UP);
        helper->AddLabel(Box(rw.x + rw.w*.5,  rw.y + rw.h*.15, 1, 1), "change player", HelperGUI::Hint::DOWN);
        helper->AddLabel(Box(screen->width*(screen->multitouch_keyboard_x + .035), screen->height*.025, 1, 1), "keyboard", HelperGUI::Hint::UPLEFT);
        helper->AddLabel(menubar->topbar.box, "options menu", HelperGUI::Hint::DOWN, .15);
    }

    app->shell.command.push_back(Shell::Command("server",       bind(&MyServerCmd,      _1)));
    app->shell.command.push_back(Shell::Command("field_color",  bind(&MyFieldColorCmd,  _1)));
    app->shell.command.push_back(Shell::Command("local_server", bind(&MyLocalServerCmd, _1)));
    app->shell.command.push_back(Shell::Command("gplus_client", bind(&MyGPlusClientCmd, _1)));
    app->shell.command.push_back(Shell::Command("gplus_server", bind(&MyGPlusServerCmd, _1)));
    app->shell.command.push_back(Shell::Command("edit",         bind(&MyEditorCmd, _1)));
    app->shell.command.push_back(Shell::Command("rcon",         bind(&GameClient::rcon_cmd,     server, _1)));
    app->shell.command.push_back(Shell::Command("name",         bind(&GameClient::setname,      server, _1)));
    app->shell.command.push_back(Shell::Command("team",         bind(&GameClient::setteam,      server, _1)));
    app->shell.command.push_back(Shell::Command("me",           bind(&GameClient::myentityname, server, _1)));

#if 0                                   
    binds.push_back(Bind('w',             Bind::TimeCB(bind(&Entity::MoveFwd,       screen->camMain, _1))));
    binds.push_back(Bind('s',             Bind::TimeCB(bind(&Entity::MoveRev,       screen->camMain, _1))));
    binds.push_back(Bind('a',             Bind::TimeCB(bind(&Entity::MoveLeft,      screen->camMain, _1))));
    binds.push_back(Bind('d',             Bind::TimeCB(bind(&Entity::MoveRight,     screen->camMain, _1))));
    binds.push_back(Bind('q',             Bind::TimeCB(bind(&Entity::MoveDown,      screen->camMain, _1))));
    binds.push_back(Bind('e',             Bind::TimeCB(bind(&Entity::MoveUp,        screen->camMain, _1))));
#else
    binds.push_back(Bind('w',             Bind::TimeCB(bind(&GameClient::movefwd,   server, _1))));
    binds.push_back(Bind('s',             Bind::TimeCB(bind(&GameClient::moverev,   server, _1))));
    binds.push_back(Bind('a',             Bind::TimeCB(bind(&GameClient::moveleft,  server, _1))));
    binds.push_back(Bind('d',             Bind::TimeCB(bind(&GameClient::moveright, server, _1))));
    binds.push_back(Bind('q',             Bind::TimeCB(bind(&GameClient::movedown,  server, _1))));
    binds.push_back(Bind('e',             Bind::TimeCB(bind(&GameClient::moveup,    server, _1))));
#endif
#if !defined(LFL_IPHONE) && !defined(LFL_ANDROID)
    binds.push_back(Bind(Bind::MOUSE1,    Bind::TimeCB(bind(&SpaceballClient::moveboost, server, _1))));
#endif
    binds.push_back(Bind(Key::LeftShift,  Bind::TimeCB(bind(&Entity::RollLeft,   screen->camMain, _1))));
    binds.push_back(Bind(Key::Space,      Bind::TimeCB(bind(&Entity::RollRight,  screen->camMain, _1))));
    binds.push_back(Bind(Key::Tab,        Bind::TimeCB(bind(&GUI::EnableDisplay, playerlist))));
    binds.push_back(Bind(Key::F1,         Bind::CB(bind(&GameClient::setcamera,  server,          vector<string>(1, string("1"))))));
    binds.push_back(Bind(Key::F2,         Bind::CB(bind(&GameClient::setcamera,  server,          vector<string>(1, string("2"))))));
    binds.push_back(Bind(Key::F3,         Bind::CB(bind(&GameClient::setcamera,  server,          vector<string>(1, string("3"))))));
    binds.push_back(Bind(Key::F4,         Bind::CB(bind(&GameClient::setcamera,  server,          vector<string>(1, string("4"))))));
    binds.push_back(Bind(Key::Return,     Bind::CB(bind(&Shell::grabmode,        &app->shell,     vector<string>()))));
    binds.push_back(Bind('r',             Bind::CB(bind(&MySwitchPlayerCmd,                       vector<string>())))); 
	binds.push_back(Bind('t',             Bind::CB(bind([&](){ chat->Toggle(); }))));
	binds.push_back(Bind(Key::Escape,     Bind::CB(bind([&](){ menubar->ToggleDisplay(); }))));
    binds.push_back(Bind(Key::Backquote,  Bind::CB(bind(&GUI::ToggleConsole,     menubar))));
    binds.push_back(Bind(Key::Quote,      Bind::CB(bind(&GUI::ToggleConsole,     menubar))));

    // start our engine
    return app->Main();
}
