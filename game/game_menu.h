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
  IPV4::Addr ip, broadcast_ip;
  UDPServer pinger;
  string master_get_url;
  vector<Server> master_server_list;

  View topbar;
  Texture *title;
  FontRef font, bright_font, glow_font, mobile_font;
  Box titlewin, menuhdr, menuftr1, menuftr2;
  int default_port, selected=1, last_selected=0, sub_selected=0, last_sub_selected=0, master_server_selected=-1;
  int line_clicked=-1;
  Widget::Button tab1, tab2, tab3, tab4, tab1_server_start, tab2_server_join, sub_tab1, sub_tab2, sub_tab3;
#ifdef LFL_ANDROID
  GPlus *gplus;
  Widget::Button gplus_signin_button, gplus_signout_button, gplus_quick, gplus_invite, gplus_accept;
#endif
  unique_ptr<TableViewInterface> singleplayer, startserver, serverlist, gplusinvite, options;
  unique_ptr<AdvertisingViewInterface> ads;
  unique_ptr<NavigationViewInterface> nav;
  Browser browser;
  MenuParticles particles;
  Entity *cam=0;

  GameMenuGUI(Window *W, const string &master_url, int port, GameSettings *Settings=0, Texture *Title=0) :
    View(W), settings(Settings), pinger(-1), master_get_url(master_url), topbar(W), title(Title),
    font       (FontDesc(FLAGS_font,                 "", 12, Color::grey80)),
    bright_font(FontDesc(FLAGS_font,                 "", 12, Color::white)),
    glow_font  (FontDesc(StrCat(FLAGS_font, "Glow"), "", 12)),
    default_port(port),
    tab1(&topbar, 0, "single player",   MouseController::CB([&](){ if (!Changed(&selected, 1)) Deactivate(); else { nav->PopAll(); nav->PushTableView(singleplayer.get()); Layout(); } })),
    tab2(&topbar, 0, "multi player",    MouseController::CB([&](){ if (!Changed(&selected, 2)) Deactivate(); else { nav->PopAll(); nav->PushTableView(serverlist.get());   Layout(); } })), 
    tab3(&topbar, 0, "options",         MouseController::CB([&](){ if (!Changed(&selected, 3)) Deactivate(); else { nav->PopAll(); nav->PushTableView(options.get());      Layout(); } })),
    tab4(&topbar, 0, "quit",            MouseController::CB(bind(&GameMenuGUI::MenuQuit, this))),
    tab1_server_start(this, 0, "start", MouseController::CB(bind(&GameMenuGUI::MenuServerStart, this))),
    tab2_server_join (this, 0, "join",  MouseController::CB(bind(&GameMenuGUI::MenuServerJoin,  this))),
    sub_tab1(this, 0, "g+",             MouseController::CB([&]() { sub_selected=1; })),
    sub_tab2(this, 0, "join",           MouseController::CB([&]() { sub_selected=2; })),
    sub_tab3(this, 0, "start",          MouseController::CB([&]() { sub_selected=3; })),
#ifdef LFL_ANDROID
    gplus(Singleton<GPlus>::Get()),
    gplus_signin_button (this, 0, "",           MouseController::CB([&](){ gplus->SignIn(); gplus_signin_button.decay = 10; })),
    gplus_signout_button(this, 0, "g+ Signout", MouseController::CB([&](){ gplus->SignOut(); })),
    gplus_quick         (this, 0, "match" ,     MouseController::CB([&](){ gplus->QuickGame(); })),
    gplus_invite        (this, 0, "invite",     MouseController::CB([&](){ gplus->Invite(); })),
    gplus_accept        (this, 0, "accept",     MouseController::CB([&](){ gplus->Accept(); })),
#endif
    nav(make_unique<NavigationView>(W, "", "")), browser(this, box), particles("GameMenuParticles") {
    tab1.outline_topleft = tab2.outline_topleft = tab3.outline_topleft = tab4.outline_topleft = tab1_server_start.outline_topleft = tab2_server_join.outline_topleft = sub_tab1.outline_topleft = sub_tab2.outline_topleft = sub_tab3.outline_topleft = &Color::grey80;
    tab1.outline_bottomright = tab2.outline_bottomright = tab3.outline_bottomright = tab4.outline_bottomright = tab1_server_start.outline_bottomright = tab2_server_join.outline_bottomright = sub_tab1.outline_bottomright = sub_tab2.outline_bottomright = sub_tab3.outline_bottomright = &Color::grey40;
    tab1_server_start.solid = tab2_server_join.solid = &Color::grey60;

#if 0
    tab3_sensitivity.increment = .1;
    tab3_sensitivity.doc_height = 10;
    tab3_sensitivity.scrolled = FLAGS_msens / 10.0;
    tab3_volume.increment = .5;
    tab3_volume.doc_height = app->GetMaxVolume();
    tab3_volume.scrolled = float(app->GetVolume()) / tab3_volume.doc_height;
#endif
    sub_selected = 2;
    pinger.handler = this;
    app->net->Enable(&pinger);
    SystemNetwork::SetSocketBroadcastEnabled(pinger.GetListener()->socket, true);
    Sniffer::GetIPAddress(&ip);
    Sniffer::GetBroadcastAddress(&broadcast_ip);
#ifdef LFL_MOBILE
    auto toolkit = app->toolkit;
#else
    auto toolkit = Singleton<Toolkit>::Get();
#endif

    {
      TableItemVec items;
      for (auto &i : settings->vec)
        items.emplace_back(i.key, TableItem::Selector, Join(i.value->data, ","));
      singleplayer = toolkit->CreateTableView("Single Player", "", "Clear", move(items));
    }

    {
      TableItemVec items;
      startserver = toolkit->CreateTableView("Start Server", "", "Clear", move(items));
    }

    {
      TableItemVec items = {
        TableItem("", TableItem::Separator),
        TableItem("[ add server ]", TableItem::TextInput, "", "", 0, 0, 0, Callback(), bind(&GameMenuGUI::MenuAddServer, this, _1)),
      };
      serverlist = toolkit->CreateTableView("Multiplayer", "", "Clear", move(items));
    }

    {
      TableItemVec items = {
        TableItem("Player Name",         TableItem::TextInput, "", "", 0, 0, 0, Callback()),
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
      };
      options = toolkit->CreateTableView("Options", "", "Clear", move(items));
    }

    {
      TableItemVec items;
      gplusinvite = toolkit->CreateTableView("GPlus", "", "Clear", move(items));
    }

    nav->PushTableView(singleplayer.get());
  }

  void EnableParticles(Entity *c, Texture *parts) {
    cam = c;
    particles.emitter_type = MenuParticles::Emitter::Mouse | MenuParticles::Emitter::GlowFade;
    particles.texture = parts->ID;
  }

  bool Activate  () { active=1; topbar.active=1; selected=last_selected=0; root->shell->mouseout(vector<string>()); if (ads) ads->Show(false); return 1; }
  bool Deactivate() { active=0; topbar.active=0; UpdateSettings(); if (ads) ads->Show(true); return 1; }

  void UpdateSettings() {
    // root->shell->Run(StrCat("name ", String::ToUTF8(tab3_player_name.Text16())));
    // root->shell->Run(StrCat("msens ", StrCat(tab3_sensitivity.scrolled * tab3_sensitivity.doc_height)));
  }

  void MenuQuit() { selected=4; app->run=0; }
  void MenuServerStart() {
    if (selected != 1 && !(selected == 2 && sub_selected == 3)) return;
    Deactivate();
    root->shell->Run("local_server");
  }

  void MenuServerJoin() {
    if (selected != 2 || master_server_selected < 0 || master_server_selected >= master_server_list.size()) return;
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
    if (!master_get_url.empty()) HTTPClient::WGet(master_get_url, 0, bind(&GameMenuGUI::MasterGetResponseCB, this, _1, _2, _3, _4, _5));
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

  void Close(Connection *c) { c->handler.release(); }
  int Read(Connection *c) {
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

  void Layout() {
    CHECK(font.Load() && bright_font.Load() && glow_font.Load());
#ifdef LFL_ANDROID
    gplus_signin_button.EnableHover();
    mobile_font.desc = FontDesc("MobileAtlas", "", 0, Color::white);
    CHECK(mobile_font.Load());
#endif
    topbar.box = Box::DelBorder(root->Box(0, .95, 1, .05), Border(1,1,1,1));
    titlewin   = Box::DelBorder(root->Box(.15, .9, .7, .05), Border(1,1,1,1)); 
    box        = Box::DelBorder(root->Box(.15, .4, .7, .5), Border(1,1,1,1));
    menuhdr    = Box (box.x, box.y+box.h-font->Height(), box.w, font->Height());
    menuftr1   = Box (box.x, box.y+font->Height()*4, box.w, box.h-font->Height()*5);
    menuftr2   = Box (box.x, box.y+font->Height()*4, box.w, box.h-font->Height()*6);
    LayoutTopbar();
    LayoutMenu();
  }

  void LayoutTopbar() {
    Flow topbarflow(&topbar.box, font, topbar.ResetView());
    tab1.box = tab2.box = tab3.box = tab4.box = Box(topbar.box.w/4, topbar.box.h);
    tab1.Layout(&topbarflow, (selected == 1) ? glow_font : font);
    tab2.Layout(&topbarflow, (selected == 2) ? glow_font : font);
    tab3.Layout(&topbarflow, (selected == 3) ? glow_font : font);
    tab4.Layout(&topbarflow, (selected == 4) ? glow_font : font);
  }

  void LayoutMenu() {
    Box b;
    int fh = font->Height();
    Flow menuflow(&box, bright_font, ResetView());

    int my_selected = selected;
    if (my_selected == 2) {
      sub_tab1.box = sub_tab2.box = sub_tab3.box = Box(box.w/3, fh);
      sub_tab1.outline = (sub_selected == 1) ? 0 : &Color::white;
      sub_tab2.outline = (sub_selected == 2) ? 0 : &Color::white;
      sub_tab3.outline = (sub_selected == 3) ? 0 : &Color::white;
      sub_tab1.Layout(&menuflow, (sub_selected == 1) ? glow_font : font);
      sub_tab2.Layout(&menuflow, (sub_selected == 2) ? glow_font : font);
      sub_tab3.Layout(&menuflow, (sub_selected == 3) ? glow_font : font);
      menuflow.SetFont(bright_font);
      menuflow.AppendNewline();

      if (sub_selected == 1) {
#ifdef LFL_ANDROID
        child_view = nullptr;
        Scissor s(root->gd, *menuflow.container);
        bool gplus_signedin = gplus->GetSignedIn();
        if (!gplus_signedin) LayoutGPlusSigninButton(&menuflow, gplus_signedin);
        else {
          int fw = menuflow.container->w, bh = font->Height();
          menuflow.AppendBox(0/3.0, fw/3.0, bh, &gplus_quick.box);  gplus_quick. LayoutBox(&menuflow, font, gplus_quick.box);
          menuflow.AppendBox(1/3.0, fw/3.0, bh, &gplus_invite.box); gplus_invite.LayoutBox(&menuflow, font, gplus_invite.box);
          menuflow.AppendBox(2/3.0, fw/3.0, bh, &gplus_accept.box); gplus_accept.LayoutBox(&menuflow, font, gplus_accept.box);
          menuflow.AppendNewlines(1);
        }
#endif
      } else if (sub_selected == 2) {
        child_view = nullptr;
        if (last_selected != 2 || last_sub_selected != 2) Refresh();
        {
          TableItemVec items = { TableItem("Server List:", TableItem::Label, "Players") };
          for (int i = 0; i < master_server_list.size(); ++i)
            items.emplace_back(master_server_list[i].name, TableItem::Label, master_server_list[i].players);
          serverlist->ReplaceSection(0, TableItem(), 0, move(items));
        }
        tab2_server_join.LayoutBox(&menuflow, bright_font, Box(box.w*.2, -box.h*.8, box.w*.6, box.h*.1));
      } else if (sub_selected == 3) {
        child_view = nullptr;
        my_selected = 1;
      }
    }
    if (my_selected == 1) {
      if ((child_view = nav->AppendFlow(&menuflow)))
        child_box.PushBack(box, menuflow.out->attr.GetAttrId(menuflow.cur_attr), child_view);
      tab1_server_start.LayoutBox(&menuflow, bright_font, Box(box.w*.2, -box.h*.8, box.w*.6, box.h*.1));
    }
    else if (my_selected == 3) {
      Scissor s(root->gd, *menuflow.container);
#ifdef LFL_ANDROID
      LayoutGPlusSigninButton(&menuflow, gplus->GetSignedIn());
#endif
      if ((child_view = nav->AppendFlow(&menuflow)))
        child_box.PushBack(box, menuflow.out->attr.GetAttrId(menuflow.cur_attr), child_view);

      menuflow.AppendNewlines(1);
      if (last_selected != 3) browser.Open("http://lucidfusionlabs.com/apps.html");
      browser.Paint(&menuflow, box.TopLeft() + point(0, 0 /*scrolled*/));
      // browser.doc.gui.Draw();
      browser.UpdateScrollbar();
    } else child_view = nullptr;
  }

  void LayoutGPlusSigninButton(Flow *menuflow, bool signedin) {
#ifdef LFL_ANDROID
    int bh = menuflow->cur_attr.font->Height()*2, bw = bh * 41/9.0;
    menuflow->AppendBox((.95 - (float)bw/menuflow->container->w)/2, bw, bh, &gplus_signin_button.box);
    if (!signedin) { 
      mobile_font->Select(root->gd);
      // gplus_signin_button.Draw(mobile_font, gplus_signin_button.decay ? 2 : (gplus_signin_button.hover ? 1 : 0));
    } else {
      gplus_signout_button.box = gplus_signin_button.box;
      // gplus_signout_button.Draw(true);
    }
    menuflow->AppendNewlines(1);
#endif
  }

  void Draw(unsigned clicks, Shader *MyShader) {
    GraphicsContext gc(root->gd);
    gc.gd->EnableBlend();
    vector<const Box*> bgwins;
    bgwins.push_back(&topbar.box);
    if (selected) bgwins.push_back(&box);
    glShadertoyShaderWindows(gc.gd, MyShader, Color(25, 60, 130, 120), bgwins);

    if (title && selected) {
      gc.gd->DisableBlend();
      title->Draw(&gc, titlewin);
      gc.gd->EnableBlend();
      gc.gd->SetColor(font->fg);
      gc.gd->SetColor(Color::grey80); BoxTopLeftOutline    ().Draw(&gc, titlewin);
      gc.gd->SetColor(Color::grey40); BoxBottomRightOutline().Draw(&gc, titlewin);
    }

    {
      Scissor s(gc.gd, box);
      View::Draw();
    }
    topbar.Draw();

    if (particles.texture) {
      particles.Update(cam, clicks, root->mouse.x, root->mouse.y, app->input->MouseButton1Down());
      particles.Draw(gc.gd);
    }

    if (selected) {
      gc.gd->SetColor(Color::grey80); BoxTopLeftOutline    ().Draw(&gc, box);
      gc.gd->SetColor(Color::grey40); BoxBottomRightOutline().Draw(&gc, box);
    }

    line_clicked = -1;
    last_selected = selected;
    last_sub_selected = sub_selected;
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
  GamePlayerListGUI(Window *W, const char *TitleName, const char *Team1, const char *Team2) : View(W),
    font(FontDesc(FLAGS_font, "", 12, Color::black)), titlename(TitleName), team1(Team1), team2(Team2) {}

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
    CHECK(font.Load());
    if (!child_box.Size()) child_box.PushNop();
  }

  void Draw(Shader *MyShader) {
    View::Draw();
    if (!toggled) Deactivate();
    GraphicsContext gc(root->gd);
    gc.gd->EnableBlend();
    Box win = root->Box(.1, .1, .8, .8, false);
    glShadertoyShaderWindows(gc.gd, MyShader, Color(255, 255, 255, 120), win);

    int fh = win.h/2-font->Height()*2;
    DrawableBoxArray outgeom1, outgeom2;
    Box out1(win.x, win.centerY(), win.w, fh), out2(win.x, win.y, win.w, fh);
    Flow menuflow1(&out1, font, &outgeom1), menuflow2(&out2, font, &outgeom2);
    for (PlayerList::iterator it = playerlist.begin(); it != playerlist.end(); it++) {
      const Player &p = (*it);
      int team = atoi(p[2]);
      bool winner = team == winning_team;
      LayoutLine(winner ? &menuflow1 : &menuflow2, PlayerName(p), PlayerScore(p), PlayerPing(p));
    }
    outgeom1.Draw(gc.gd, out1.TopLeft());
    outgeom2.Draw(gc.gd, out2.TopLeft());
    font->Draw(titletext, Box(win.x, win.top()-font->Height(), win.w, font->Height()), 0, Font::DrawFlag::AlignCenter);
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
    TextArea(W, FontDesc(FLAGS_font, "", 10, Color::grey80), 100, 10), server(s) { 
    write_timestamp = deactivate_on_enter = true;
    line_fb.align_top_or_bot = false;
    SetToggleKey(key, true);
    bg_color = &Color::clear;
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
