/*
 * $Id: image.cpp 1336 2014-12-08 09:29:59Z justin $
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
bool input_3D = false;
DEFINE_string(output, "", "Output");
DEFINE_string(input, "", "Input");
DEFINE_float(input_scale, 0, "Input scale");
DEFINE_string(input_prims, "", "Comma separated list of .obj primitives to highlight");
DEFINE_string(input_filter, "", "Filter type [dark2alpha]");
DEFINE_bool(visualize, false, "Display");
DEFINE_string(shader, "", "Apply this shader");
DEFINE_string(make_png_atlas, "", "Build PNG atlas with files in directory");
DEFINE_int(make_png_atlas_size, 256, "Build PNG atlas with this size");
DEFINE_string(split_png_atlas, "", "Split PNG atlas back into individual files");
DEFINE_string(filter_png_atlas, "", "Filter PNG atlas");

AssetMap asset;
SoundAssetMap soundasset;

Scene scene;
Shader MyShader;

void DrawInput3D(Asset *a, Entity *e) {
    if (!FLAGS_input_prims.empty() && a && a->geometry) {
        vector<int> highlight_input_prims;
        Split(FLAGS_input_prims, isint2<',', ' '>, &highlight_input_prims);

        screen->gd->DisableLighting();
        screen->gd->Color4f(1.0, 0.0, 0.0, 1.0);
        int vpp = GraphicsDevice::VertsPerPrimitive(a->geometry->primtype);
        for (auto i = highlight_input_prims.begin(); i != highlight_input_prims.end(); ++i) {
            CHECK_LT(*i, a->geometry->count / vpp);
            scene.Draw(a->geometry, e, *i * vpp, vpp);
        }
        screen->gd->Color4f(1.0, 1.0, 1.0, 1.0);
    }
}

void Frame3D(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) {
    screen->cam->Look();
    scene.Draw(&asset.vec);
}

void Frame2D(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) {
    Asset *a = asset("input");

    if (MyShader.ID) {
        screen->gd->ActiveTexture(0);
        screen->gd->BindTexture(GraphicsDevice::Texture2D, a->tex.ID);
        screen->gd->UseShader(&MyShader);
        MyShader.SetUniform1f("xres", screen->width);

        // mandelbox params
        float par[20][3] = { 0.25, -1.77 };
        MyShader.SetUniform3fv("par", sizeofarray(par), &par[0][0]);
        MyShader.SetUniform1f("fov_x", FLAGS_field_of_view);
        MyShader.SetUniform1f("fov_y", FLAGS_field_of_view);
        MyShader.SetUniform1f("min_dist", .000001);
        MyShader.SetUniform1i("max_steps", 128);
        MyShader.SetUniform1i("iters", 14);
        MyShader.SetUniform1i("color_iters", 10);
        MyShader.SetUniform1f("ao_eps", .0005);
        MyShader.SetUniform1f("ao_strength", .1);
        MyShader.SetUniform1f("glow_strength", .5);
        MyShader.SetUniform1f("dist_to_color", .2);
        MyShader.SetUniform1f("x_scale", 1);
        MyShader.SetUniform1f("x_offset", 0);
        MyShader.SetUniform1f("y_scale", 1);
        MyShader.SetUniform1f("y_offset", 0);

        v3 up = screen->cam->up, ort = screen->cam->ort, pos = screen->cam->pos;
        v3 right = v3::Cross(ort, up);
        float m[16] = { right.x, right.y, right.z, 0,
                        up.x,    up.y,    up.z,    0,
                        ort.x,   ort.y,   ort.z,   0,
                        pos.x,   pos.y,   pos.z,   0 };
        screen->gd->LoadIdentity();
        screen->gd->Mult(m);

        glTimeResolutionShaderWindows
            (&MyShader, Color::black, Box(-screen->width/2, -screen->height/2, screen->width, screen->height));
    } else {
        screen->gd->EnableLayering();
        a->tex.Draw(screen->Box());
    }
}

int Frame(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) {

    if (input_3D)  Frame3D(W, clicks, mic_samples, cam_sample, flag);

    screen->gd->DrawMode(DrawMode::_2D);

    if (!input_3D) Frame2D(W, clicks, mic_samples, cam_sample, flag);

    // Press tick for console
    screen->DrawDialogs();

    return 0;
}

}; // namespace LFL
using namespace LFL;

extern "C" int main(int argc, const char *argv[]) {

    app->logfilename = StrCat(LFAppDownloadDir(), "image.txt");
    screen->frame_cb = Frame;
    screen->width = 420;
    screen->height = 380;
    screen->caption = "Image";
    FLAGS_near_plane = 0.1;
    FLAGS_lfapp_video = FLAGS_lfapp_input = true;

    if (app->Create(argc, argv, __FILE__)) { app->Free(); return -1; }
    if (app->Init()) { app->Free(); return -1; }

    // asset.Add(Asset(name,  texture,     scale, translate, rotate, geometry,       0, 0, 0));
    asset.Add(Asset("axis",   "",          0,     0,         0,      0,              0, 0, 0, Asset::DrawCB(bind(&glAxis, _1, _2))));
    asset.Add(Asset("grid",   "",          0,     0,         0,      Grid::Grid3D(), 0, 0, 0));
    asset.Add(Asset("input",  "",          0,     0,         0,      0,              0, 0, 0));
    asset.Load();
    app->shell.assets = &asset;

    // soundasset.Add(SoundAsset(name, filename,   ringbuf, channels, sample_rate, seconds));
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

    if (!FLAGS_make_png_atlas.empty()) {
        FLAGS_atlas_dump=1;
        vector<string> png;
        DirectoryIter d(FLAGS_make_png_atlas, 0, 0, ".png");
        for (const char *fn = d.Next(); fn; fn = d.Next()) png.push_back(FLAGS_make_png_atlas + fn);
        AtlasFontEngine::MakeFromPNGFiles("png_atlas", png, FLAGS_make_png_atlas_size, NULL);
    }

    if (!FLAGS_split_png_atlas.empty()) {
        Singleton<AtlasFontEngine>::Get()->Init(FontDesc(FLAGS_split_png_atlas, "", 0, Color::black));
        Font *font = Fonts::Get(FLAGS_split_png_atlas, "", 0, Color::black);
        CHECK(font);
        map<v4, int> glyph_index;
        for (int i = 255; i >= 0; i--) {
            if (!font->glyph->table[i].tex.width && !font->glyph->table[i].tex.height) continue;
            glyph_index[v4(font->glyph->table[i].tex.coord)] = i;
        }
        map<int, v4> glyphs;
        for (map<v4, int>::const_iterator i = glyph_index.begin(); i != glyph_index.end(); ++i) glyphs[i->second] = i->first;

        string outdir = StrCat(app->assetdir, FLAGS_split_png_atlas);
        LocalFile::mkdir(outdir, 0755);
        string atlas_png_fn = StrCat(app->assetdir, FLAGS_split_png_atlas, "0,0,0,0,0", "00.png");
        AtlasFontEngine::SplitIntoPNGFiles(atlas_png_fn, glyphs, outdir + LocalFile::Slash);
    }

    if (!FLAGS_filter_png_atlas.empty()) {
        if (Font *f = AtlasFontEngine::OpenAtlas(FontDesc(FLAGS_filter_png_atlas, "", 0, Color::white))) {
            AtlasFontEngine::WriteGlyphFile(FLAGS_filter_png_atlas, f);
            INFO("filtered ", FLAGS_filter_png_atlas);
        }
    }

    if (FLAGS_input.empty()) FATAL("no input supplied");
    Asset *asset_input = asset("input");

    if (!FLAGS_shader.empty()) {
        string vertex_shader = LocalFile::FileContents(StrCat(app->assetdir, "vertex.glsl"));
        string fragment_shader = LocalFile::FileContents(FLAGS_shader);
        Shader::Create("my_shader", vertex_shader, fragment_shader, "", &MyShader);
    }

    if (SuffixMatch(FLAGS_input, ".obj", false)) {
        input_3D = true;
        asset_input->cb = bind(&DrawInput3D, _1, _2);
        asset_input->scale = FLAGS_input_scale;
        asset_input->geometry = Geometry::LoadOBJ(FLAGS_input);
        scene.Add(new Entity("axis",  asset("axis")));
        scene.Add(new Entity("grid",  asset("grid")));
        scene.Add(new Entity("input", asset_input));

        if (!FLAGS_output.empty()) {
            set<int> filter_prims;
            bool filter_invert = FLAGS_input_filter == "notprims", filter = false;
            if (filter_invert || FLAGS_input_filter == "prims")    filter = true;
            if (filter) Split(FLAGS_input_prims, isint2<',', ' '>, &filter_prims);
            string out = Geometry::ExportOBJ(asset_input->geometry, filter ? &filter_prims : 0, filter_invert);
            int ret = LocalFile::WriteFile(FLAGS_output, out);
            INFO("write ", FLAGS_output, " = ", ret);
        }
    } else {
        input_3D = false;
        FLAGS_draw_grid = true;
        Texture pb;
        app->assets.default_video_loader->LoadVideo
            (app->assets.default_video_loader->LoadVideoFile(FLAGS_input), &asset_input->tex, false);
        pb.AssignBuffer(&asset_input->tex, true);

        if (pb.width && pb.height) screen->Reshape(FLAGS_input_scale ? pb.width *FLAGS_input_scale : pb.width,
                                                   FLAGS_input_scale ? pb.height*FLAGS_input_scale : pb.height);
        INFO("input dim = (", pb.width, ", ", pb.height, ") pf=", pb.pf);

        if (FLAGS_input_filter == "dark2alpha") {
            for (int i=0; i<pb.height; i++)
                for (int j=0; j<pb.width; j++) {
                    int ind = (i*pb.width + j) * Pixel::size(pb.pf);
                    unsigned char *b = pb.buf + ind;
                    float dark = b[0]; // + b[1] + b[2] / 3.0;
                    // if (dark < 256*1/3.0) b[3] = 0;
                    b[3] = dark;
                }
        }

        if (!FLAGS_output.empty()) {
            int ret = PngWriter::Write(FLAGS_output, pb);
            INFO("write ", FLAGS_output, " = ", ret);
        }
    }

    if (!FLAGS_visualize) return 0;

    // start our engine
    return app->Main();
}
