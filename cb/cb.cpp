/*
 * $Id: cb.cpp 1336 2014-12-08 09:29:59Z justin $
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
#include "lfapp/network.h"

namespace LFL {
DEFINE_FLAG(sniff_device, int, 0, "Network interface index");

AssetMap asset;
SoundAssetMap soundasset;

Scene scene;
Sniffer *sniffer;
GeoResolution *geo;

// engine callback driven by LFL::Application
int Frame(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) {
    scene.Get("arrow")->YawRight((double)clicks/500);
    scene.Draw(&asset.vec);

    // Press tick for console
    screen->gd->DrawMode(DrawMode::_2D);
    screen->DrawDialogs();
    return 0;
}

void sniff(const char *packet, int avail, int size) {
    if (avail < Ethernet::Header::Size + IPV4::Header::MinSize) return;
    IPV4::Header *ip = (IPV4::Header*)(packet + Ethernet::Header::Size);
    int iphdrlen = ip->hdrlen() * 4;
    if (iphdrlen < IPV4::Header::MinSize || avail < Ethernet::Header::Size + iphdrlen) return;
    string src_ip = IPV4Endpoint::name(ip->src), dst_ip = IPV4Endpoint::name(ip->dst), src_city, dst_city;
    float src_lat, src_lng, dst_lat, dst_lng;
    geo->resolve(src_ip, 0, 0, &src_city, &src_lat, &src_lng);
    geo->resolve(dst_ip, 0, 0, &dst_city, &dst_lat, &dst_lng);

    if (ip->prot == 6 && avail >= Ethernet::Header::Size + iphdrlen + TCP::Header::MinSize) {
        TCP::Header *tcp = (TCP::Header*)((char*)ip + iphdrlen);
        INFO("TCP ", src_ip, ":", ntohs(tcp->src), " (", src_city, ") -> ", dst_ip, ":", ntohs(tcp->dst), " (", dst_city, ")");
    }
    else if (ip->prot == 17 && avail >= Ethernet::Header::Size + iphdrlen + UDP::Header::Size) {
        UDP::Header *udp = (UDP::Header*)((char*)ip + iphdrlen);
        INFO("UDP ", src_ip, ":", ntohs(udp->src), " -> ", dst_ip, ":", ntohs(udp->dst));
    }
    else INFO("ip ver=", ip->version(), " prot=", ip->prot, " ", src_ip, " -> ", dst_ip);
}

}; // namespace LFL
using namespace LFL;

extern "C" int main(int argc, const char *argv[]) {

    app->frame_cb = Frame;
    app->logfilename = StrCat(LFAppDownloadDir(), "cb.txt");
    screen->width = 420;
    screen->height = 380;
    screen->caption = "CrystalBawl";
    FLAGS_lfapp_video = FLAGS_lfapp_input = FLAGS_lfapp_network = 1;
    FLAGS_target_fps = 50;
    FLAGS_threadpool_size = 1;

    if (app->Create(argc, argv, __FILE__)) { app->Free(); return -1; }
    if (app->Init()) { app->Free(); return -1; }

    // for dynamic ::load(Asset *); 
    // asset.Add(Asset(name, texture,     scale, translate, rotate, geometry,       0, 0, 0));
    asset.Add(Asset("axis",  "",          0,     0,         0,      0,              0, 0, 0, Asset::DrawCB(bind(&glAxis, _1, _2))));
    asset.Add(Asset("grid",  "",          0,     0,         0,      Grid::Grid3D(), 0, 0, 0));
    asset.Add(Asset("room",  "",          0,     0,         0,      0,              0, 0, 0, Asset::DrawCB(bind(&glRoom, _1, _2))));
    asset.Add(Asset("arrow", "",         .005,   1,        -90,     "arrow.obj",    0, 0));
    asset.Load();
    app->shell.assets = &asset;

    // for dynamic ::load(SoundAsset *);
    // soundassets.push_back(SoundAsset(name, filename,   ringbuf, channels, sample_rate, seconds));
    soundasset.Add(SoundAsset("draw",  "Draw.wav", 0,       0,        0,           0      ));
    soundasset.Load();
    app->shell.soundassets = &soundasset;

    BindMap *binds = screen->binds = new BindMap();
//  binds->Add(Bind(key,            callback));
    binds->Add(Bind(Key::Backquote, Bind::CB(bind([&](){ screen->console->Toggle(); }))));
    binds->Add(Bind(Key::Quote,     Bind::CB(bind([&](){ screen->console->Toggle(); }))));
    binds->Add(Bind(Key::Escape,    Bind::CB(bind(&Shell::quit,            &app->shell, vector<string>()))));
    binds->Add(Bind(Key::Return,    Bind::CB(bind(&Shell::grabmode,        &app->shell, vector<string>()))));
    binds->Add(Bind(Key::LeftShift, Bind::TimeCB(bind(&Entity::RollLeft,   screen->cam, _1))));
    binds->Add(Bind(Key::Space,     Bind::TimeCB(bind(&Entity::RollRight,  screen->cam, _1))));
    binds->Add(Bind('w',            Bind::TimeCB(bind(&Entity::MoveFwd,    screen->cam, _1))));
    binds->Add(Bind('s',            Bind::TimeCB(bind(&Entity::MoveRev,    screen->cam, _1))));
    binds->Add(Bind('a',            Bind::TimeCB(bind(&Entity::MoveLeft,   screen->cam, _1))));
    binds->Add(Bind('d',            Bind::TimeCB(bind(&Entity::MoveRight,  screen->cam, _1))));
    binds->Add(Bind('q',            Bind::TimeCB(bind(&Entity::MoveDown,   screen->cam, _1))));
    binds->Add(Bind('e',            Bind::TimeCB(bind(&Entity::MoveUp,     screen->cam, _1))));

    scene.Add(new Entity("axis",  asset("axis")));
    scene.Add(new Entity("grid",  asset("grid")));
    scene.Add(new Entity("room",  asset("room")));
    scene.Add(new Entity("arrow", asset("arrow"), v3(1, .24, 1)));

    vector<string> devices;
    Sniffer::PrintDevices(&devices);
    if (FLAGS_sniff_device < 0 || FLAGS_sniff_device >= devices.size()) FATAL(FLAGS_sniff_device, " oob ", devices.size(), ", are you running as root?");
    if (!(sniffer = Sniffer::Open(devices[FLAGS_sniff_device], "", 1024, sniff))) FATAL("sniffer Open failed");
    if (!(geo = GeoResolution::Open(StrCat(ASSETS_DIR, "GeoLiteCity.dat").c_str()))) FATAL("geo Open failed");

    // start our engine
    return app->Main();
}
