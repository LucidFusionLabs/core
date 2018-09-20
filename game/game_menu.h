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

#ifndef LFL_CORE_GAME_GAME_MENU_H__
#define LFL_CORE_GAME_GAME_MENU_H__
namespace LFL {

struct GameMenuGUI : public View, public Connection::Handler {
  struct Server { string addr, name, players; };
  typedef Particles<1024, 1, true> MenuParticles;

  GameSettings *settings=0;
  SocketServices *net;
  IPV4::Addr ip, broadcast_ip;
  UDPServer pinger;
  string master_get_url;
  vector<Server> master_server_list;

  View topbar;
  Texture *title;
  FontRef font, bright_font, glow_font;
  Box titlewin, menuhdr, menuftr1, menuftr2;
  int default_port, last_selected=-1, last_sub_selected=-1, master_server_selected=-1;
  unique_ptr<ToolbarViewInterface> toplevel, sublevel;
  unique_ptr<TableViewInterface> singleplayer, startserver, serverlist, options, gplus_invite;
  unique_ptr<AdvertisingViewInterface> ads;
  unique_ptr<NavigationViewInterface> nav;
#ifdef LFL_ANDROID
  unique_ptr<ToolbarViewInterface> gplus_login, gplus_logout, gplus_menu;
  GPlus *gplus=0;
#endif
  MenuParticles particles;
  Entity *cam=0;

  GameMenuGUI(Window *W, SocketServices *N, ToolkitInterface *TK, Audio *audio,
              const string &master_url, int port, GameSettings *Settings=0, Texture *Title=0) :
    View(W, "GameMenuGUI"), settings(Settings), net(N), pinger(N, -1), master_get_url(master_url), topbar(W, "TopbarView"), title(Title),
    font       (W, FontDesc(FLAGS_font,                 "", 12, Color::grey80)),
    bright_font(W, FontDesc(FLAGS_font,                 "", 12, Color::white)),
    glow_font  (W, FontDesc(StrCat(FLAGS_font, "Glow"), "", 12, Color::white)), default_port(port),
    nav(make_unique<NavigationView>(W, "", "")), particles("GameMenuParticles") {

    pinger.handler = this;
    N->Enable(&pinger);
    SystemNetwork::SetSocketBroadcastEnabled(pinger.GetListener()->socket, true);
    Sniffer::GetIPAddress(&ip);
    Sniffer::GetBroadcastAddress(&broadcast_ip);

    toplevel = make_unique<ToolbarView>(root, "Clear", MenuItemVec{
      { "single player", "", [=](){ if (!Changed(&toplevel->selected, 0)) Deactivate(); else { nav->PopAll(); nav->PushTableView(singleplayer.get()); Layout(); }} },
      { "multi player",  "", [=](){ if (!Changed(&toplevel->selected, 1)) Deactivate(); else { nav->PopAll(); nav->PushTableView(serverlist.get());   Layout(); }} }, 
      { "options",       "", [=](){ if (!Changed(&toplevel->selected, 2)) Deactivate(); else { nav->PopAll(); nav->PushTableView(options.get());      Layout(); }} },
      { "quit",          "", bind(&GameMenuGUI::MenuQuit, this) },
      }, font, glow_font);
    toplevel->selected = 0;

    sublevel = make_unique<ToolbarView>(root, "Clear", MenuItemVec{
      { "g+",    "", [=]() { sublevel->selected=0; Layout(); }},
      { "join",  "", [=]() { sublevel->selected=1; Layout(); }},
      { "start", "", [=]() { sublevel->selected=2; Layout(); }},
      }, font, glow_font, &Color::white);
    sublevel->selected = 1;

    TableItemVec settings_items;
    for (auto &i : settings->vec)
      settings_items.emplace_back(i.key, TableItem::Selector, Join(i.value->data, ","));
    settings_items.emplace_back("Start", TableItem::Button,  "", "", 0, 0, 0, bind(&GameMenuGUI::MenuServerStart, this)); 
    singleplayer = TK->CreateTableView(root, "Single Player", "", "Clear", move(settings_items));

    startserver = TK->CreateTableView(root, "Start Server", "", "Clear", TableItemVec{});
    serverlist = TK->CreateTableView(root, "Multiplayer", "", "Clear", TableItemVec{
      TableItem("", TableItem::Separator),
      TableItem("[ add server ]", TableItem::TextInput, "", "", 0, 0, 0, Callback(), bind(&GameMenuGUI::MenuAddServer, this, _1)),
      TableItem("Join",           TableItem::Button,    "", "", 0, 0, 0, bind(&GameMenuGUI::MenuServerJoin, this))
    });

    TableItemVec options_items{
#ifdef LFL_ANDROID
      // LayoutGPlusSigninButton(&menuflow, gplus->GetSignedIn());
      // gplus_signin_button.EnableHover();
#endif
      TableItem("Player Name",         TableItem::TextInput, Singleton<FlagMap>::Get()->Get("player_name"), "", 0, 0, 0, Callback()),
      TableItem("Control Sensitivity", TableItem::Slider,    ""),
      TableItem("Volume",              TableItem::Slider,    ""),
      TableItem("Move Forward:",       TableItem::Label,     "W"),
      TableItem("Move Left:",          TableItem::Label,     "A"),
      TableItem("Move Reverse:",       TableItem::Label,     "S"),
      TableItem("Move Right:",         TableItem::Label,     "D"),
      TableItem("Change Player:",      TableItem::Label,     "R"),
      TableItem("Charge Speed Boost:", TableItem::Label,     "Hold Click"),
      TableItem("Grab Mouse:",         TableItem::Label,     "Enter"),
      TableItem("Topbar Menu:",        TableItem::Label,     "Escape"),
      TableItem("Player List:",        TableItem::Label,     "Tab"),
      TableItem("Console:",            TableItem::Label,     "~"),
      TableItem("",                    TableItem::Separator),
      TableItem("Promotions",          TableItem::WebView,   "http://lucidfusionlabs.com/apps.html"),
    };
    float max_volume = audio->GetMaxVolume();
    options_items[1].CheckAssignValues("Control Sensitivity", FLAGS_msens, 0, 5, .1);
    options_items[2].CheckAssignValues("Volume", audio->GetVolume(), 0, max_volume, .5);
    options = TK->CreateTableView(root, "Options", "", "Clear", move(options_items));

#ifdef LFL_ANDROID
    gplus = Singleton<GPlus>::Get();
    gplus_login = make_unique<ToolbarView>(root, "Clear", MenuItemVec{
      { "", "", [=](){ gplus->SignIn(); /* gplus_signin_button.decay = 10; */ } },
      }, font);

    gplus_logout = make_unique<ToolbarView>(root, "Clear", MenuItemVec{
      { "g+ Signout", "", [=](){ gplus->SignOut(); } },
    }, font);

    gplus_menu = make_unique<ToolbarView>(root, "Clear", MenuItemVec{
      { "match",  "", [=](){ gplus->QuickGame(); } },
      { "invite", "", [=](){ gplus->Invite(); } },
      { "accept", "", [=](){ gplus->Accept(); } },
    }, font);

    gplus_invite = TK->CreateTableView(root, "GPlus", "", "Clear", TableItemVec{});
#endif

    nav->PushTableView(singleplayer.get());
  }

  void EnableParticles(Entity *c, Texture *parts) {
    cam = c;
    particles.emitter_type = MenuParticles::Emitter::Mouse | MenuParticles::Emitter::GlowFade;
    particles.texture = parts->ID;
  }

  bool Activate  () override {                active=1; topbar.active=1; toplevel->selected=last_selected=-1; root->shell->mouseout(vector<string>()); if (ads) ads->Show(false); return 1; }
  bool Deactivate() override { nav->PopAll(); active=0; topbar.active=0; UpdateSettings(); if (ads) ads->Show(true); return 1; }

  void UpdateSettings() {
    auto settings = options->GetSectionText(0);
    root->shell->Run(StrCat("name ", settings[0].second));
    root->shell->Run(StrCat("msens ", settings[1].second));
    root->parent->audio->SetVolume(atoi(settings[2].second));
  }

  void MenuQuit() { toplevel->selected=3; root->parent->Shutdown(); }
  void MenuServerStart() {
    if (toplevel->selected != 0 && !(toplevel->selected == 1 && sublevel->selected == 2)) return;
    Deactivate();
    root->shell->Run("local_server");
  }

  void MenuServerJoin() {
    if (toplevel->selected != 1 || master_server_selected < 0 || master_server_selected >= master_server_list.size()) return;
    Deactivate();
    root->shell->Run(StrCat("server ", master_server_list[master_server_selected].addr));
  }

  void MenuAddServer(const string &text) {
    int delim = text.find(':');
    if (delim != string::npos) SystemNetwork::SendTo(pinger.GetListener()->socket, SystemNetwork::GetHostByName(text.substr(0, delim)),
                                                     atoi(text.c_str()+delim+1), "ping\n", 5);
  }

  void Refresh() { 
    if (broadcast_ip) SystemNetwork::SendTo(pinger.GetListener()->socket, broadcast_ip, default_port, "ping\n", 5);
    if (!master_get_url.empty()) HTTPClient::WGet(net, nullptr, master_get_url, 0, bind(&GameMenuGUI::MasterGetResponseCB, this, _1, _2, _3, _4, _5));
    master_server_list.clear();
    master_server_selected=-1;
  }

  int MasterGetResponseCB(Connection *c, const char *h, const string &ct, const char *cb, int cl) {
    if (!cb || !cl) return 0;
    const char *p;
    string servers(cb, cl);
    StringLineIter lines(servers);
    for (string l = lines.NextString(); !lines.Done(); l = lines.NextString()) {
      if (!(p = strchr(l.c_str(), ':'))) continue;
      SystemNetwork::SendTo(pinger.GetListener()->socket, IPV4::Parse(string(l.c_str(), p-l.c_str())), atoi(p+1), "ping\n", 5);
    }
    return 0;
  }

  void Close(Connection *c) override { c->handler.release(); }
  int Read(Connection *c) override {
    for (int i=0; i<c->packets.size(); i++) {
      string reply(c->rb.begin() + c->packets[i].offset, c->packets[i].len);
      PingResponseCB(c, reply);
    }
    return 0;
  }

  void PingResponseCB(Connection *c, const string &reply) {
    IPV4Endpoint remote = c->RemoteIPV4();
    if (ip && ip == remote.addr) return;
    const char *p;
    string name, players;
    StringLineIter lines(reply);
    for (string l = lines.NextString(); !lines.Done(); l = lines.NextString()) {
      if (!(p = strchr(l.c_str(), '='))) continue;
      string k(l, p-l.c_str()), v(p+1);
      if      (k == "name")    name    = v;
      else if (k == "players") players = v;
    }
    Server s = { c->Name(), StrCat(name, (remote.addr & broadcast_ip) == remote.addr ? " (local)" : ""), players };
    master_server_list.push_back(s);
  }

  View *Layout(Flow *flow=nullptr) override {
    topbar.box = Box::DelBorder(root->ViewBox().Scale(0, .95, 1, .05), Border(1,1,1,1));
    titlewin   = Box::DelBorder(root->ViewBox().Scale(.15, .9, .7, .05), Border(1,1,1,1)); 
    box        = Box::DelBorder(root->ViewBox().Scale(.15, .4, .7, .5), Border(1,1,1,1));
    menuhdr    = Box (box.x, box.y+box.h-font->Height(), box.w, font->Height());
    menuftr1   = Box (box.x, box.y+font->Height()*4, box.w, box.h-font->Height()*5);
    menuftr2   = Box (box.x, box.y+font->Height()*4, box.w, box.h-font->Height()*6);
    LayoutTopbar();
    LayoutMenu();
    return this;
  }

  void LayoutTopbar() {
    Flow topbarflow(&topbar.box, font, topbar.ResetView());
    if (auto v = toplevel->Layout(&topbarflow)) topbar.AppendChildView(v);
  }

  void LayoutMenu() {
    Box b;
    int fh = font->Height();
    Flow menuflow(&box, bright_font, ResetView());

    int my_selected = toplevel->selected;
    if (my_selected == 1) {
      if (auto v = sublevel->Layout(&menuflow)) AppendChildView(v);
      menuflow.SetFont(bright_font);
      menuflow.AppendNewline();

      if (sublevel->selected == 0) {
#ifdef LFL_ANDROID
        bool gplus_signedin = gplus->GetSignedIn();
        if (!gplus_signedin) LayoutGPlusSigninButton(&menuflow, gplus_signedin);
        else {
          if (auto v = gplus_menu->AppendFlow(&menuflow)) AppendChildView(v);
          menuflow.AppendNewlines(1);
        }
#endif
      } else if (sublevel->selected == 1) {
        if (last_selected != 1 || last_sub_selected != 1) Refresh();
        {
          TableItemVec items = { TableItem("Server List:", TableItem::Label, "Players") };
          for (int i = 0; i < master_server_list.size(); ++i)
            items.emplace_back(master_server_list[i].name, TableItem::Label, master_server_list[i].players);
          serverlist->ReplaceSection(0, TableItem(), 0, move(items));
        }
      } else if (sublevel->selected == 2) {
        my_selected = 0;
      }
    }
    if (my_selected == 0) {
      if (auto v = nav->Layout(&menuflow)) AppendChildView(v);
    }
    else if (my_selected == 2) {
      if (auto v = nav->Layout(&menuflow)) AppendChildView(v);
      menuflow.AppendNewlines(1);
    }
  }

  void LayoutGPlusSigninButton(Flow *menuflow, bool signedin) {
#if 0
      // def LFL_ANDROID
    int bh = menuflow->cur_attr.font->Height()*2, bw = bh * 41/9.0;
    menuflow->AppendBox((.95 - (float)bw/menuflow->container->w)/2, bw, bh, &gplus_signin_button.box);
    if (!signedin) { 
      // mobile_font->Select(root->gd);
      // gplus_signin_button.Draw(mobile_font, gplus_signin_button.decay ? 2 : (gplus_signin_button.hover ? 1 : 0));
    } else {
      // gplus_signout_button.box = gplus_signin_button.box;
      // gplus_signout_button.Draw(true);
    }
    menuflow->AppendNewlines(1);
#endif
  }

  void Draw(const point &p, unsigned clicks, Shader *MyShader) {
    GraphicsContext gc(root->gd);
    gc.gd->EnableBlend();
    vector<const Box*> bgwins;
    bgwins.push_back(&topbar.box);
    if (toplevel->selected >= 0) bgwins.push_back(&box);
    glShadertoyShaderWindows(gc.gd, MyShader, Color(25, 60, 130, 120), bgwins, nullptr, p);

    if (title && toplevel->selected >= 0) {
      gc.gd->DisableBlend();
      title->Draw(&gc, titlewin + p);
      gc.gd->EnableBlend();
      gc.gd->SetColor(font->fg);
      gc.gd->SetColor(Color::grey80); BoxTopLeftOutline    ().Draw(&gc, titlewin);
      gc.gd->SetColor(Color::grey40); BoxBottomRightOutline().Draw(&gc, titlewin);
    }

    View::Draw(p);
    topbar.Draw(p);

    if (particles.texture) {
      particles.Update(cam, clicks, root->mouse.x, root->mouse.y, root->parent->input->MouseButton1Down());
      particles.Draw(gc.gd);
    }

    if (toplevel->selected >= 0) {
      gc.gd->SetColor(Color::grey80); BoxTopLeftOutline    ().Draw(&gc, box + p);
      gc.gd->SetColor(Color::grey40); BoxBottomRightOutline().Draw(&gc, box + p);
    }

    last_selected = toplevel->selected;
    last_sub_selected = sublevel->selected;
  }
};

struct GamePlayerListGUI : public View {
  typedef vector<string> Player;
  typedef vector<Player> PlayerList;

  FontRef font;
  bool toggled=0;
  string titlename, titletext, team1, team2;
  PlayerList playerlist;
  int winning_team=0;
  GamePlayerListGUI(Window *W, const char *TitleName, const char *Team1, const char *Team2) : View(W, "GamePlayerListGUI"),
    font(W, FontDesc(FLAGS_font, "", 12, Color::white)), titlename(TitleName), team1(Team1), team2(Team2) {}

  void HandleTextMessage(const string &in) {
    playerlist.clear();
    StringLineIter lines(in);
    const char *hdr = lines.Next();
    string red_score_text, blue_score_text;
    Split(hdr, iscomma, &red_score_text, &blue_score_text);
    int red_score=atoi(red_score_text), blue_score=atoi(blue_score_text);
    winning_team = blue_score > red_score ? Game::Team::Blue : Game::Team::Red;
    titletext = StrCat(titlename, " ", red_score, "-", blue_score);

    for (const char *line = lines.Next(); line; line = lines.Next()) {
      StringWordIter words(line, lines.cur_len, iscomma);
      Player player;
      for (string word = words.NextString(); !words.Done(); word = words.NextString()) player.push_back(word);
      if (player.size() < 5) continue;
      playerlist.push_back(player);
    }
    sort(playerlist.begin(), playerlist.end(), PlayerCompareScore);
  }

  void Layout() {
    CHECK(font.Load(root));
    if (!child_box.Size()) child_box.PushNop();
  }

  void Draw(Shader *MyShader) {
    View::Draw(point());
    if (!toggled) Deactivate();
    GraphicsContext gc(root->gd);
    gc.gd->EnableBlend();
    Box win = root->Box().Scale(.1, .1, .8, .8, false);
    glShadertoyShaderWindows(gc.gd, MyShader, Color(255, 255, 255, 120), win);

    int fh = win.h/2-font->Height()*2;
    DrawableBoxArray outgeom1, outgeom2;
    Box out1(win.x, win.centerY(), win.w, fh), out2(win.x, win.y, win.w, fh);
    Flow menuflow1(&out1, font, &outgeom1), menuflow2(&out2, font, &outgeom2);
    for (auto &p : playerlist) {
      int team = atoi(p[2]);
      bool winner = team == winning_team;
      LayoutLine(winner ? &menuflow1 : &menuflow2, PlayerName(p), PlayerScore(p), PlayerPing(p));
    }
    outgeom1.Draw(gc.gd, out1.TopLeft());
    outgeom2.Draw(gc.gd, out2.TopLeft());
    font->Draw(gc.gd, titletext, Box(win.x, win.top()-font->Height(), win.w, font->Height()), 0, Font::DrawFlag::AlignCenter);
    gc.gd->SetColor(Color::grey60); BoxTopLeftOutline    ().Draw(&gc, win);
    gc.gd->SetColor(Color::grey20); BoxBottomRightOutline().Draw(&gc, win);
  }

  void LayoutLine(Flow *flow, const string &name, const string &score, const string &ping) {
    flow->AppendText(name);
    flow->AppendText(.6, score);
    flow->AppendText(.85, ping);
    flow->AppendNewlines(1);
  }

  static bool PlayerCompareScore(const Player &l, const Player &r) { return atoi(PlayerScore(l)) > atoi(PlayerScore(r)); }
  static string PlayerName (const Player &p) { return p.size() < 5 ? "" : p[1]; }
  static string PlayerScore(const Player &p) { return p.size() < 5 ? "" : p[3]; }
  static string PlayerPing (const Player &p) { return p.size() < 5 ? "" : p[4]; }
};

struct GameChatGUI : public TextArea {
  GameClient **server;
  GameChatGUI(Window *W, int key, GameClient **s) :
    TextArea(W, FontRef(W, FontDesc(FLAGS_font, "", 10, Color::grey80)), 100, 10), server(s) { 
    write_timestamp = deactivate_on_enter = true;
    line_fb.align_top_or_bot = false;
    SetToggleKey(key, true);
    bg_color = Color::clear;
    cursor.type = Cursor::Underline;
  }

  void Run(string text) { if (server && *server) (*server)->Rcon(StrCat("say ", text)); }
  void Draw() {
    if (!Active() && Now() - write_last >= Seconds(5)) return;
    root->gd->EnableBlend(); 
    {
      int h = int(root->gl_h/1.6);
      Scissor scissor(root->gd, Box(1, root->gl_h-h+1, root->gl_w, root->gl_h*.15, false));
      TextArea::Draw(Box(0, root->gl_h-h, root->gl_w, int(root->gl_h*.15)));
    }
  }
};

}; // namespace LFL
#endif /* LFL_CORE_GAME_GAME_MENU_H__ */
