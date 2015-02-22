/*
 * $Id: video.cpp 1336 2014-12-08 09:29:59Z justin $
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
#endif

#include "lfapp/lfapp.h"
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/gui.h"

#if defined(LFL_GLEW) && !defined(LFL_HEADLESS)
#define glewGetContext() ((GLEWContext*)screen->glew_context)
#include <GL/glew.h>
#endif

#define LFL_GLSL_SHADERS
#if (defined(LFL_ANDROID) || defined(LFL_IPHONE)) && !defined(LFL_GLES2)
#undef LFL_GLSL_SHADERS
#endif

#if defined(LFL_HEADLESS)
#define GLint int
#define GLuint unsigned 
#define glFlush(x)
#define glGetError() 0
#define glGetString(a) ""
#define glReadPixels(a,b,c,d,e,f,g)
#define glTexImage2D(a,b,c,d,e,f,g,h,i)
#define glTexSubImage2D(a,b,c,d,e,f,g,h,i)
#define glTexParameteri(a,b,c)
#define glGenRenderbuffers(a,b)
#define glGenFramebuffers(a,b)
#define glBindRenderbuffer(a,b)
#define glBindFramebuffer(a,b)
#define glRenderbufferStorage(a,b,c,d)
#define glFramebufferRenderbuffer(a,b,c,d)
#define glFramebufferTexture2D(a,b,c,d,e)
#define glCheckFramebufferStatus(a) 0
#define GL_FRAMEBUFFER 0
#define GL_LUMINANCE 0
#define GL_LUMINANCE_ALPHA 0
#define GL_RGB 0
#define GL_RGBA 0
#define GL_TEXTURE_2D 0
#define GL_TEXTURE_CUBE_MAP 0
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_X 0
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Y 0
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Z 0
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X 0
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Y 0
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Z 0
#define GL_UNSIGNED_BYTE 0
#define GL_FRAGMENT_SHADER 0
#define GL_ACTIVE_UNIFORMS 0
#define GL_ACTIVE_ATTRIBUTES 0
#define GL_FRAMEBUFFER_COMPLETE 0
#define GL_VERTEX_SHADER 0
#define GL_MAX_VERTEX_UNIFORM_COMPONENTS 0
#define GL_MAX_VERTEX_ATTRIBS 0
#define GL_DEPTH_COMPONENT16 0

#elif defined(LFL_IPHONE)
#ifdef LFL_GLES2
#include <OpenGLES/ES2/gl.h>
#include <OpenGLES/ES2/glext.h>
#endif
#include <OpenGLES/ES1/gl.h>
#include <OpenGLES/ES1/glext.h>
#define glOrtho glOrthof 
#define glFrustum glFrustumf 
char *NSFMDocumentPath();
int NSFMreaddir(const char *path, int dirs, void *DirectoryIter, void (*DirectoryIterAdd)(void *di, const char *k, int));

#elif defined(LFL_ANDROID)
#ifdef LFL_GLES2
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#endif
#include <GLES/gl.h>
#include <GLES/glext.h>
#define glOrtho glOrthof 
#define glFrustum glFrustumf 

#elif defined(__APPLE__)
#include <OpenGL/glu.h>

#else
#include <GL/glu.h>
#endif

#ifdef LFL_FREETYPE
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_LCD_FILTER_H
FT_Library ft_library;
#endif

extern "C" {
#ifdef LFL_FFMPEG
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#endif
};

#ifdef __APPLE__
#import <CoreText/CTFont.h>
#import <CoreText/CTLine.h>
#import <CoreText/CTRun.h>
#import <CoreText/CTStringAttributes.h>
#import <CoreFoundation/CFAttributedString.h>
#import <CoreGraphics/CGBitmapContext.h> 
#endif

#ifdef LFL_QT
#include <QtOpenGL>
#include <QApplication>
#undef main
static QApplication *lfl_qapp;
static vector<string> lfl_argv;  
extern "C" int LFLQTMain(int argc, const char *argv[]);
extern "C" int      main(int argc, const char *argv[]) {
    for (int i=0; i<argc; i++) lfl_argv.push_back(argv[i]);
    QApplication app(argc, (char**)argv);
    lfl_qapp = &app;
    LFL::Window::Create(LFL::screen);
    return app.exec();
}
#endif // LFL_QT

#ifdef LFL_GLFWVIDEO
#include "GLFW/glfw3.h"
#endif

#ifdef LFL_SDLVIDEO
#include "SDL.h"
#if defined(LFL_ANDROID)
extern "C" {
#include "SDL_androidvideo.h"
};
#endif
#endif

namespace LFL {
const int Texture::CoordMinX = 0;
const int Texture::CoordMinY = 1;
const int Texture::CoordMaxX = 2;
const int Texture::CoordMaxY = 3;

Color Color::white (1.0, 1.0, 1.0);
Color Color::black (0.0, 0.0, 0.0);
Color Color::red   (1.0, 0.0, 0.0);
Color Color::green (0.0, 1.0, 0.0);
Color Color::blue  (0.0, 0.0, 1.0);
Color Color::cyan  (0.0, 1.0, 1.0);
Color Color::yellow(1.0, 1.0, 0.0);
Color Color::magenta(1.0, 0.0, 1.0);
Color Color::grey90(.9, .9, .9);
Color Color::grey80(.8, .8, .8);
Color Color::grey70(.7, .7, .7);
Color Color::grey60(.6, .6, .6);
Color Color::grey50(.5, .5, .5);
Color Color::grey40(.4, .4, .4);
Color Color::grey30(.3, .3, .3);
Color Color::grey20(.2, .2, .2);
Color Color::grey10(.1, .1, .1);

#ifdef LFL_GLES2
DEFINE_int(request_gles_version, 2, "OpenGLES version");
#else
DEFINE_int(request_gles_version, 1, "OpenGLES version");
#endif

DEFINE_bool(gd_debug, false, "Debug graphics device");
DEFINE_float(rotate_view, 0, "Rotate view by angle");
DEFINE_float(field_of_view, 45, "Field of view");
DEFINE_float(near_plane, 1, "Near clipping plane");
DEFINE_float(far_plane, 100, "Far clipping plane");
DEFINE_int(dots_per_inch, 75, "Screen DPI");
DEFINE_bool(swap_axis, false," Swap x,y axis");
DEFINE_string(font_engine, "atlas", "[atlas,freetype,coretext]");
DEFINE_string(default_font, "Nobile.ttf", "Default font");
DEFINE_string(default_font_family, "sans-serif", "Default font family");
DEFINE_int(default_font_size, 16, "Default font size");
DEFINE_int(default_font_flag, 0, "Default font flag");
DEFINE_int(default_missing_glyph, 127, "Default glyph returned for missing requested glyph");
DEFINE_bool(atlas_dump, false, "Dump .png files for every font");
DEFINE_string(atlas_font_sizes, "8,16,32,64", "Load font atlas CSV sizes");
DEFINE_float(atlas_pad_top, 0, "Pad top of each glyph with size*atlas_pad_top pixels");
DEFINE_int(glyph_table_size, 128, "Load lowest glyph_table_size unicode code points");
DEFINE_bool(subpixel_fonts, false, "Treat RGB components as subpixels, tripling width");
DEFINE_bool(font_dont_reopen, false, "Scale atlas to font size instead of re-raster");
DEFINE_int(scale_font_height, 0, "Scale font when height != scale_font_height");
DEFINE_int(add_font_size, 0, "Increase all font sizes by add_font_size");

#ifndef LFL_HEADLESS
#define GDDebug(...) if (FLAGS_gd_debug) INFO(__VA_ARGS__)
const int GraphicsDevice::Float            = GL_FLOAT;
const int GraphicsDevice::Points           = GL_POINTS;
const int GraphicsDevice::Lines            = GL_LINES;
const int GraphicsDevice::LineLoop         = GL_LINE_LOOP;
const int GraphicsDevice::Triangles        = GL_TRIANGLES;
const int GraphicsDevice::TriangleStrip    = GL_TRIANGLE_STRIP;
const int GraphicsDevice::Texture2D        = GL_TEXTURE_2D;
const int GraphicsDevice::UnsignedInt      = GL_UNSIGNED_INT;
const int GraphicsDevice::Ambient          = GL_AMBIENT;
const int GraphicsDevice::Diffuse          = GL_DIFFUSE;
const int GraphicsDevice::Specular         = GL_SPECULAR;
const int GraphicsDevice::Position         = GL_POSITION;
const int GraphicsDevice::Emission         = GL_EMISSION;
const int GraphicsDevice::One              = GL_ONE;
const int GraphicsDevice::SrcAlpha         = GL_SRC_ALPHA;
const int GraphicsDevice::OneMinusSrcAlpha = GL_ONE_MINUS_SRC_ALPHA;
const int GraphicsDevice::OneMinusDstColor = GL_ONE_MINUS_DST_COLOR;
const int GraphicsDevice::Fill             = GL_FILL;
const int GraphicsDevice::Line             = GL_LINE;
const int GraphicsDevice::Point            = GL_POINT;

struct OpenGLES1 : public GraphicsDevice {
    int target_matrix;
    OpenGLES1() : target_matrix(-1) { default_color.push_back(Color(1.0, 1.0, 1.0, 1.0)); }
    void Init() {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glEnableClientState(GL_VERTEX_ARRAY);
#if !defined(LFL_IPHONE) && !defined(LFL_ANDROID)
        float black[]={0,0,0,1};
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);
        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, 1);
        // glLightModelf(GL_LIGHT_MODEL_COLOR_CONTROL, GL_SEPARATE_SPECULAR_COLOR);
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
#endif
    }
    void UpdateColor() { const Color &c = default_color.back(); glColor4f(c.r(), c.g(), c.b(), c.a()); }
    bool ShaderSupport() {
#if defined(LFL_ANDROID) || defined(LFL_IPHONE)
        return false;
#endif
        const char *ver = (const char*)glGetString(GL_VERSION);
        return ver && *ver == '2';
    }
    void  EnableTexture() { GDDebug("Texture=1");  glEnable(GL_TEXTURE_2D);  glEnableClientState(GL_TEXTURE_COORD_ARRAY); }
    void DisableTexture() { GDDebug("Texture=0"); glDisable(GL_TEXTURE_2D); glDisableClientState(GL_TEXTURE_COORD_ARRAY); }
    void  EnableLighting() { GDDebug("Lighting=1");  glEnable(GL_LIGHTING);  glEnable(GL_COLOR_MATERIAL); }
    void DisableLighting() { GDDebug("Lighting=0"); glDisable(GL_LIGHTING); glDisable(GL_COLOR_MATERIAL); }
    void  EnableVertexColor() { GDDebug("VertexColor=1");  glEnableClientState(GL_COLOR_ARRAY); }
    void DisableVertexColor() { GDDebug("VertexColor=0"); glDisableClientState(GL_COLOR_ARRAY); }
    void  EnableNormals() { GDDebug("Normals=1");  glEnableClientState(GL_NORMAL_ARRAY); }
    void DisableNormals() { GDDebug("Normals=0"); glDisableClientState(GL_NORMAL_ARRAY); }
    //void TextureEnvReplace()  { GDDebug("TextureEnv=R"); glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE); }
    //void TextureEnvModulate() { GDDebug("TextureEnv=M"); glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE); }
    void  EnableLight(int n) { GDDebug("Light", n, "=1"); if (n)  glEnable(GL_LIGHT1); else  glEnable(GL_LIGHT0); }
    void DisableLight(int n) { GDDebug("Light", n, "=0"); if (n) glDisable(GL_LIGHT1); else glDisable(GL_LIGHT0); }
    void Material(int t, float *color) { glMaterialfv(GL_FRONT_AND_BACK, t, color); }
    void Light(int n, int t, float *color) { glLightfv(((n) ? GL_LIGHT1 : GL_LIGHT0), t, color); }
#if defined(LFL_IPHONE) || defined(LFL_ANDROID)
    void TextureGenLinear() {}
    void TextureGenReflection() {}
    void DisableTextureGen() {}
#else
    void  EnableTextureGen() { GDDebug("TextureGen=1");  glEnable(GL_TEXTURE_GEN_S);  glEnable(GL_TEXTURE_GEN_T);  glEnable(GL_TEXTURE_GEN_R); }
    void DisableTextureGen() { GDDebug("TextureGen=0"); glDisable(GL_TEXTURE_GEN_S); glDisable(GL_TEXTURE_GEN_T); glDisable(GL_TEXTURE_GEN_R); }
    void TextureGenLinear() {
        static float X[4] = { -1,0,0,0 }, Y[4] = { 0,-1,0,0 }, Z[4] = { 0,0,-1,0 };
        EnableTextureGen();
        glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
        glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
        glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
        glTexGenfv(GL_S, GL_OBJECT_PLANE, X);
        glTexGenfv(GL_T, GL_OBJECT_PLANE, Y);
        glTexGenfv(GL_R, GL_OBJECT_PLANE, Z);
        GDDebug("TextureGen=L");
    }
    void TextureGenReflection() {
        EnableTextureGen();
        glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
        glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
        glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
        GDDebug("TextureGen=R");
    }
#endif
    void DisableCubeMap()   { GDDebug("CubeMap=", 0); glDisable(GL_TEXTURE_CUBE_MAP); DisableTextureGen(); }
    void BindCubeMap(int n) { GDDebug("CubeMap=", n);  glEnable(GL_TEXTURE_CUBE_MAP); glBindTexture(GL_TEXTURE_CUBE_MAP, n); }
    void ActiveTexture(int n) {
        GDDebug("ActiveTexture=", n);
        glClientActiveTexture(GL_TEXTURE0 + n);
        glActiveTexture(GL_TEXTURE0 + n);
        // glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_COMBINE);
        // glTexEnvf (GL_TEXTURE_ENV, GL_COMBINE_RGB_EXT, GL_MODULATE);
    }
    void BindTexture(int t, int n) { GDDebug("BindTexture=", t, ",", n); glBindTexture(t, n); }
    void VertexPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool ud) { glVertexPointer(m, t, w, verts + o/sizeof(float)); }
    void TexPointer(int m, int t, int w, int o, float *tex, int l, int *out, bool ud) { glTexCoordPointer(m, t, w, tex + o/sizeof(float)); }
    void ColorPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool ud) { glColorPointer(m, t, w, verts + o/sizeof(float)); }
    void NormalPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool ud) { glNormalPointer(t, w, verts + o/sizeof(float)); }
    void Color4f(float r, float g, float b, float a) { default_color.back() = Color(r,g,b,a); UpdateColor(); }
    void MatrixProjection() { target_matrix=2; glMatrixMode(GL_PROJECTION); }
    void MatrixModelview() { target_matrix=1; glMatrixMode(GL_MODELVIEW); }
    void LoadIdentity() { glLoadIdentity(); }
    void PushMatrix() { glPushMatrix(); }
    void PopMatrix() { glPopMatrix(); }
    void GetMatrix(m44 *out) { glGetFloatv(target_matrix == 2 ? GL_PROJECTION_MATRIX : GL_MODELVIEW_MATRIX, &(*out)[0][0]); }
    void PrintMatrix() {}
    void Scalef(float x, float y, float z) { glScalef(x, y, z); }
    void Rotatef(float angle, float x, float y, float z) { glRotatef(angle, x, y, z); }
    void Ortho(float l, float r, float b, float t, float nv, float fv) { glOrtho(l,r, b,t, nv,fv); }
    void Frustum(float l, float r, float b, float t, float nv, float fv) { glFrustum(l,r, b,t, nv,fv); }
    void Mult(const float *m) { glMultMatrixf(m); }
    void Translate(float x, float y, float z) { glTranslatef(x, y, z); }
    void Draw(int pt, int np, int it, int o, void *index, int l, int *out, bool dirty) {
        GDDebug("Draw(", pt, ", ", np, ", ", it, ", ", o, ", ", index, ", ", l, ", ", dirty, ")");
        glDrawElements(pt, np, it, (char*)index + o);
    }
    void DrawArrays(int type, int o, int n) {
        GDDebug("DrawArrays(", type, ", ", o, ", ", n, ")");
        glDrawArrays(type, o, n);
    }
    void UseShader(Shader *S) { if (!S) S = &app->video.shader_default; GDDebug("Shader=", S->name); glUseProgram(S->ID); }
};

#ifdef LFL_GLES2
#ifdef LFL_QT
class OpenGLES2 : public QWindow, protected QOpenGLFunctions, public GraphicsDevice {
  public:
    bool QT_init, QT_grabbed;
    point QT_mp;

    void render_request() { QCoreApplication::postEvent(this, new QEvent(QEvent::UpdateRequest)); }
    bool event(QEvent *event) {
        if (event->type() != QEvent::UpdateRequest) return QWindow::event(event); 
        QCoreApplication::processEvents();

        if (!QT_init) {
            CHECK(!LFL::screen->gl);
            LFL::screen->gl = new QOpenGLContext(this);
            ((QOpenGLContext*)LFL::screen->gl)->setFormat(requestedFormat());
            ((QOpenGLContext*)LFL::screen->gl)->create();
            Window::MakeCurrent(LFL::screen);
            initializeOpenGLFunctions();

            vector<const char *> av;
            for (int i=0; i<lfl_argv.size(); i++) av.push_back(lfl_argv[i].c_str());
            av.push_back(0);
            LFLQTMain(lfl_argv.size(), &av[0]);
            QT_init = true;
            if (!app->run) { app->Free(); lfl_qapp->exit(); return true; }
        }

        app->Frame();

        if (!app->run) { app->Free(); lfl_qapp->exit(); return true; }
        if (!app->scheduler.wait_forever) render_request();
        return true;
    }
    void resizeEvent(QResizeEvent *ev) { QWindow::resizeEvent(ev); if (!QT_init) return; LFL::screen->Reshaped(ev->size().width(), ev->size().height()); }
    void keyPressEvent  (QKeyEvent *ev) { if (!QT_init) return; app->input.QueueKey(QT_key(ev->key()), 1); ev->accept(); }
    void keyReleaseEvent(QKeyEvent *ev) { if (!QT_init) return; app->input.QueueKey(QT_key(ev->key()), 0); ev->accept(); }
    void mousePressEvent  (QMouseEvent *ev) { if (!QT_init) return; app->input.QueueMouseClick(QT_mouse_button(ev->button()), 1, point(ev->x(), ev->y())); }
    void mouseReleaseEvent(QMouseEvent *ev) { if (!QT_init) return; app->input.QueueMouseClick(QT_mouse_button(ev->button()), 0, point(ev->x(), ev->y())); }
    void mouseMoveEvent   (QMouseEvent *ev) {
        point p(ev->x(), ev->y()), dx = p - QT_mp;
        if (!QT_init || (!p.x && !p.y)) return;
        app->input.QueueMouseMovement(p, dx);
        if (!QT_grabbed) {
            QT_mp = p;
        } else {
            QT_mp = point(width()/2, height()/2);
            QCursor::setPos(mapToGlobal(QPoint(QT_mp.x, QT_mp.y)));
        }
    }
    static unsigned QT_key(unsigned k) { return k < 256 && isalpha(k) ? ::tolower(k) : k; }
    static unsigned QT_mouse_button(int b) {
        if      (b == Qt::LeftButton)  return 1;
        else if (b == Qt::RightButton) return 2;
        return 0;
    }
#else /* LFL_QT */
struct OpenGLES2 : public GraphicsDevice {
    void *QT_init, *QT_grabbed, *QT_mx, *QT_my, *QT_GL_context;
#endif /* LFL_QT */

    Shader *shader;
    int enabled_array, enabled_indexarray, matrix_target;
    vector<m44> modelview_matrix, projection_matrix;
    bool dirty_matrix, dirty_color;
    int cubemap_on, normals_on, texture_on, colorverts_on, lighting_on;
    LFL::Material material;
    LFL::Light light[4];

    struct VertexAttribPointer {
        int m, t, w, o;
        VertexAttribPointer()                           : m(0), t(0), w(0), o(0) {}
        VertexAttribPointer(int M, int T, int W, int O) : m(M), t(T), w(W), o(O) {}
    } position_ptr, tex_ptr, color_ptr, normal_ptr;

    OpenGLES2() : QT_init(0), QT_grabbed(0), shader(0), enabled_array(-1), enabled_indexarray(-1), matrix_target(-1), dirty_matrix(1), dirty_color(1), cubemap_on(0), normals_on(0), texture_on(0), colorverts_on(0), lighting_on(0) {
        modelview_matrix.push_back(m44::Identity());
        projection_matrix.push_back(m44::Identity());
        default_color.push_back(Color(1.0, 1.0, 1.0, 1.0));
    }

    void Init() {
        // GetIntegerv(GL_FRAMEBUFFER_BINDING, &oldFBO);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        string vertex_shader = LocalFile::FileContents(StrCat(ASSETS_DIR, "lfapp_vertex.glsl"));
        string pixel_shader  = LocalFile::FileContents(StrCat(ASSETS_DIR, "lfapp_pixel.glsl"));
        Shader::Create("lfapp",          vertex_shader, pixel_shader, "#define TEX2D  \r\n#define VERTEXCOLOR\r\n", &app->video.shader_default);
        Shader::Create("lfapp_cubemap",  vertex_shader, pixel_shader, "#define TEXCUBE\r\n#define VERTEXCOLOR\r\n", &app->video.shader_cubemap);
        Shader::Create("lfapp_normals",  vertex_shader, pixel_shader, "#define TEX2D  \r\n#define NORMALS\r\n",     &app->video.shader_normals);
        Shader::Create("lfapp_cubenorm", vertex_shader, pixel_shader, "#define TEXCUBE\r\n#define NORMALS\r\n",     &app->video.shader_cubenorm);
        UseShader(0);
    }

    vector<m44> *TargetMatrix() {
        if      (matrix_target == 1) return &modelview_matrix;
        else if (matrix_target == 2) return &projection_matrix;
        else FATAL("uknown matrix ", matrix_target);
    }
    void UpdateMatrix() { dirty_matrix = true; }
    void UpdateColor() { dirty_color = true; }
    void UpdateShader() {
        if (cubemap_on && normals_on) UseShader(&app->video.shader_cubenorm);
        else if          (cubemap_on) UseShader(&app->video.shader_cubemap);
        else if          (normals_on) UseShader(&app->video.shader_normals);
        else                          UseShader(&app->video.shader_default);
    }
    void UpdatePosition() {
        glEnableVertexAttribArray(shader->slot_position);
        SetVertexAttribPointer(shader->slot_position, position_ptr);
    }
    void UpdateTexture() {
        bool supports = shader->slot_tex >= 0;
        glUniform1i(shader->uniform_texon, texture_on && supports);
        if (supports) {
            if (texture_on) {  glEnableVertexAttribArray(shader->slot_tex); SetVertexAttribPointer(shader->slot_tex, tex_ptr); }
            else            { glDisableVertexAttribArray(shader->slot_tex); }
        } else if (texture_on && FLAGS_gd_debug) ERROR("shader doesnt support texture");
    }
    void UpdateColorVerts() {
        bool supports = shader->slot_color >= 0;
        glUniform1i(shader->uniform_coloron, colorverts_on && supports);
        if (supports) {
            if (colorverts_on) {  glEnableVertexAttribArray(shader->slot_color); SetVertexAttribPointer(shader->slot_color, color_ptr); }
            else               { glDisableVertexAttribArray(shader->slot_color); }
        } else if (colorverts_on) ERROR("shader doesnt support vertex color");
    }
    void UpdateNormals() {
        bool supports = shader->slot_normal >= 0;
        glUniform1i(shader->uniform_normalon, normals_on && supports);
        if (supports) {
            if (normals_on) {  glEnableVertexAttribArray(shader->slot_normal); SetVertexAttribPointer(shader->slot_normal, normal_ptr); }
            else            { glDisableVertexAttribArray(shader->slot_normal); }
        } else if (normals_on) ERROR("shader doesnt support normals");
    }

    bool ShaderSupport() { return true; }
    bool LightingSupport() { return false; }
    void EnableTexture()  { GDDebug("Texture=1"); if (Typed::Changed(&texture_on, 1)) UpdateTexture(); }
    void DisableTexture() { GDDebug("Texture=0"); if (Typed::Changed(&texture_on, 0)) UpdateTexture(); }
    void EnableLighting()  { GDDebug("Lighting=1"); lighting_on=1; }
    void DisableLighting() { GDDebug("Lighting=0"); lighting_on=0; }
    void EnableVertexColor()  { GDDebug("VertexColor=1"); if (Typed::Changed(&colorverts_on, 1)) UpdateColorVerts(); }
    void DisableVertexColor() { GDDebug("VertexColor=0"); if (Typed::Changed(&colorverts_on, 0)) UpdateColorVerts(); }
    void EnableNormals()  { GDDebug("Normals=1"); if (Typed::Changed(&normals_on, 1)) { UpdateShader(); UpdateNormals(); } }
    void DisableNormals() { GDDebug("Normals=0"); if (Typed::Changed(&normals_on, 0)) { UpdateShader(); UpdateNormals(); } }
    void EnableLight(int n) {}
    void DisableLight(int n) {}
    void Material(int t, float *v) {
        if      (t == GL_AMBIENT)             material.ambient  = Color(v);
        else if (t == GL_DIFFUSE)             material.diffuse  = Color(v);
        else if (t == GL_SPECULAR)            material.specular = Color(v);
        else if (t == GL_EMISSION)            material.emissive = Color(v);
        else if (t == GL_AMBIENT_AND_DIFFUSE) material.ambient = material.diffuse = Color(v);
        shader->dirty_material = app->video.shader_cubenorm.dirty_material = app->video.shader_normals.dirty_material = 1;
    }
    void Light(int n, int t, float *v) {
        bool light_pos = 0, light_color = 0;
        if (n != 0) { ERROR("ignoring Light(", n, ")"); return; }

        if      (t == GL_POSITION) { light_pos=1;   light[n].pos = modelview_matrix.back().Transform(v4(v)); }
        else if (t == GL_AMBIENT)  { light_color=1; light[n].color.ambient  = Color(v); }
        else if (t == GL_DIFFUSE)  { light_color=1; light[n].color.diffuse  = Color(v); }
        else if (t == GL_SPECULAR) { light_color=1; light[n].color.specular = Color(v); }

        if (light_pos)   { shader->dirty_light_pos  [n] = app->video.shader_cubenorm.dirty_light_pos  [n] = app->video.shader_normals.dirty_light_pos  [n] = 1; }
        if (light_color) { shader->dirty_light_color[n] = app->video.shader_cubenorm.dirty_light_color[n] = app->video.shader_normals.dirty_light_color[n] = 1; }
    }
    void DisableCubeMap()   { GDDebug("CubeMap=", 0); if (Typed::Changed(&cubemap_on, 0)) { UpdateShader(); glUniform1i(shader->uniform_cubeon, 0); } }
    void BindCubeMap(int n) { GDDebug("CubeMap=", n); if (Typed::Changed(&cubemap_on, 1)) { UpdateShader(); glUniform1i(shader->uniform_cubeon, 1); } glUniform1i(shader->uniform_cubetex, 0); glBindTexture(GL_TEXTURE_CUBE_MAP, n); }
    void TextureGenLinear() {}
    void TextureGenReflection() {}
    void ActiveTexture(int n) { glActiveTexture(n ? GL_TEXTURE1 : GL_TEXTURE0); }
    void BindTexture(int t, int n) {
        glActiveTexture(GL_TEXTURE0); 
        glBindTexture(t, n);
        glUniform1i(shader->uniform_tex, 0);
        GDDebug("BindTexture=", t, ",", n);
    }
    void SetVertexAttribPointer(int slot, const VertexAttribPointer &ptr) { 
        glVertexAttribPointer(slot, ptr.m, ptr.t, GL_FALSE, ptr.w, (GLvoid*)(long)ptr.o);
    }
    void VertexPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {
        bool input_dirty = dirty, first = (*out == -1);
        if (first) { glGenBuffers(1, (GLuint*)out); dirty = true; }
        if (*out != enabled_array) {
            CHECK(shader);
            CHECK((!o && !w) || o < w);
            enabled_array = *out;
            glBindBuffer(GL_ARRAY_BUFFER, *out);
            position_ptr = VertexAttribPointer(m, t, w, o);
            UpdatePosition();
        }
        if (first || dirty) {
            if (first) glBufferData(GL_ARRAY_BUFFER, l, verts, input_dirty ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
            else       glBufferSubData(GL_ARRAY_BUFFER, 0, l, verts);
        }
    }
    void TexPointer(int m, int t, int w, int o, float *tex, int l, int *out, bool dirty) {
        CHECK_LT(o, w);
        CHECK(*out == enabled_array);
        tex_ptr = VertexAttribPointer(m, t, w, o);
        if (!texture_on) EnableTexture();
        else SetVertexAttribPointer(shader->slot_tex, tex_ptr);
    }
    void ColorPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {
        CHECK_LT(o, w);
        CHECK(*out == enabled_array);
        color_ptr = VertexAttribPointer(m, t, w, o);
        if (!colorverts_on) EnableVertexColor();
        else SetVertexAttribPointer(shader->slot_color, color_ptr);
    }
    void NormalPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {
        CHECK_LT(o, w);
        CHECK(*out == enabled_array);
        normal_ptr = VertexAttribPointer(m, t, w, o);
        if (!normals_on) EnableNormals();
        else SetVertexAttribPointer(shader->slot_normal, normal_ptr);
    }
    void MatrixModelview()  { matrix_target=1; }
    void MatrixProjection() { matrix_target=2; }
    void GetMatrix(m44 *out) { *out = TargetMatrix()->back(); }
    void PopMatrix() {
        vector<m44> *target = TargetMatrix();
        if      (target->size() >= 1) target->pop_back();
        else if (target->size() == 1) target->back().Assign(m44::Identity());
        UpdateMatrix();
    }
    void PrintMatrix()        { TargetMatrix()->back().Print(StrCat("mt", matrix_target)); }
    void PushMatrix()         { TargetMatrix()->push_back(TargetMatrix()->back()); UpdateMatrix(); }
    void LoadIdentity()       { TargetMatrix()->back().Assign(m44::Identity());    UpdateMatrix(); }
    void Mult(const float *m) { TargetMatrix()->back().Mult(m44(m));               UpdateMatrix(); }
    void Scalef(float x, float y, float z) {
#if 0
        TargetMatrix()->back().mult(m44::scale(x, y, z));
#else
        m44 &m = TargetMatrix()->back();
        m[0].x *= x; m[0].y *= x; m[0].z *= x;
        m[1].x *= y; m[1].y *= y; m[1].z *= y;
        m[2].x *= z; m[2].y *= z; m[2].z *= z;
#endif
        UpdateMatrix();
    }
    void Rotatef(float angle, float x, float y, float z) { TargetMatrix()->back().Mult(m44::Rotate(DegreeToRadian(angle), x, y, z)); UpdateMatrix(); }
    void Ortho  (float l, float r, float b, float t, float nv, float fv) { TargetMatrix()->back().Mult(m44::Ortho  (l, r, b, t, nv, fv)); UpdateMatrix(); }
    void Frustum(float l, float r, float b, float t, float nv, float fv) { TargetMatrix()->back().Mult(m44::Frustum(l, r, b, t, nv, fv)); UpdateMatrix(); }
    void Translate(float x, float y, float z) { 
#if 0
        TargetMatrix()->back().mult(m44::translate(x, y, z));
#else
        m44 &m = TargetMatrix()->back();
        m[3].x += m[0].x * x + m[1].x * y + m[2].x * z;
        m[3].y += m[0].y * x + m[1].y * y + m[2].y * z;
        m[3].z += m[0].z * x + m[1].z * y + m[2].z * z;
        m[3].w += m[0].w * x + m[1].w * y + m[2].w * z;
#endif
        UpdateMatrix();
    }
    void Color4f(float r, float g, float b, float a) {
        if (lighting_on) {
            float c[] = { r, g, b, a };
            Material(GL_AMBIENT_AND_DIFFUSE, c);
        } else {
            default_color.back() = Color(r,g,b,a);
            UpdateColor();
        }
    }
    void Draw(int pt, int np, int it, int o, void *index, int l, int *out, bool dirty) {
        GDDebug("Draw(", pt, ", ", np, ", ", it, ", ", o, ", ", index, ", ", l, ", ", dirty, ")");

        bool input_dirty = dirty;
        if (*out == -1) { glGenBuffers(1, (GLuint*)out); dirty = true; }
        if (*out != enabled_indexarray) { 
            enabled_indexarray = *out;
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *out);
        }
#if 0
        int bound_buf, bound_array_buf;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &bound_buf);
        glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &bound_array_buf);
        INFO("bb=", bound_buf, " bab=", bound_array_buf);
#endif
        if (dirty) glBufferData(GL_ELEMENT_ARRAY_BUFFER, l, index, input_dirty ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);

        PushDirtyState();
        glDrawElements(pt, np, it, (GLvoid*)(long)o);
    }
    void DrawArrays(int type, int o, int n) {
        GDDebug("DrawArrays(", type, ", ", o, ", ", n, ")");

        //glBindBuffer(GL_ARRAY_BUFFER, 0);
        //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        PushDirtyState();
        glDrawArrays(type, o, n);
    }
    void PushDirtyState() {
        if (dirty_matrix) {
            dirty_matrix = false;
            m44 m = projection_matrix.back();
            m.Mult(modelview_matrix.back());
            glUniformMatrix4fv(shader->uniform_modelviewproj, 1, 0, m[0]);
            glUniformMatrix4fv(shader->uniform_modelview,     1, 0, modelview_matrix.back()[0]);
        }
        if (dirty_color && shader->uniform_colordefault >= 0) {
            dirty_color = false;
            glUniform4fv(shader->uniform_colordefault, 1, default_color.back().x);
        }
        if (shader->dirty_material) {
            glUniform4fv(shader->uniform_material_ambient,  1, material.ambient.x);
            glUniform4fv(shader->uniform_material_diffuse,  1, material.diffuse.x);
            glUniform4fv(shader->uniform_material_specular, 1, material.specular.x);
            glUniform4fv(shader->uniform_material_emission, 1, material.emissive.x);
        }
        for (int i=0; i<sizeofarray(light) && i<sizeofarray(shader->dirty_light_pos); i++) {
            if (shader->dirty_light_pos[i]) {
                shader->dirty_light_pos[i] = 0;
                glUniform4fv(shader->uniform_light0_pos, 1, light[i].pos);
            }
            if (shader->dirty_light_color[i]) {
                shader->dirty_light_color[i] = 0;
                glUniform4fv(shader->uniform_light0_ambient,  1, light[i].color.ambient.x);
                glUniform4fv(shader->uniform_light0_diffuse,  1, light[i].color.diffuse.x);
                glUniform4fv(shader->uniform_light0_specular, 1, light[i].color.specular.x);
            }
        }
    }
    void UseShader(Shader *S) {
        if (!S) return UpdateShader();
        if (shader == S || !S->ID) return;
        shader = S;
        glUseProgram(shader->ID);
        GDDebug("Shader=", S->name);
        dirty_matrix = dirty_color = true;
        for (int i=0, s; i<S->MaxVertexAttrib; i++) {
            if ((s = S->unused_attrib_slot[i]) < 0) break;
            glDisableVertexAttribArray(s);
        }
        UpdatePosition();
        UpdateNormals();
        UpdateColorVerts();
        UpdateTexture();
    }
};
#endif // LFL_GLES2

// Shader interaface
int GraphicsDevice::CreateProgram() { return glCreateProgram(); }
int GraphicsDevice::CreateShader(int t) { return glCreateShader(t); }
void GraphicsDevice::ShaderSource(int shader, int count, const char **source, int *len) { glShaderSource(shader, count, source, len); }
void GraphicsDevice::CompileShader(int shader) {
    char buf[1024] = {0}; int l=0;
    glCompileShader(shader);
    glGetShaderInfoLog(shader, sizeof(buf), &l, buf);
    if (l) INFO(buf);
}
void GraphicsDevice::AttachShader(int prog, int shader) { glAttachShader(prog, shader); }
void GraphicsDevice::BindAttribLocation(int prog, int loc, const string &name) { glBindAttribLocation(prog, loc, name.c_str()); }
void GraphicsDevice::LinkProgram(int prog) {
    char buf[1024] = {0}; int l=0;
    glLinkProgram(prog);
    glGetProgramInfoLog(prog, sizeof(buf), &l, buf);
    if (l) INFO(buf);
    GLint link_status;
    glGetProgramiv(prog, GL_LINK_STATUS, &link_status);
    if (link_status != GL_TRUE) FATAL("link failed");
}
void GraphicsDevice::GetProgramiv(int p, int t, int *out) { glGetProgramiv(p, t, out); }
void GraphicsDevice::GetIntegerv(int t, int *out) { glGetIntegerv(t, out); }
int GraphicsDevice::GetAttribLocation (int prog, const string &name) { return glGetAttribLocation (prog, name.c_str()); }
int GraphicsDevice::GetUniformLocation(int prog, const string &name) { return glGetUniformLocation(prog, name.c_str()); }
void GraphicsDevice::Uniform1i(int u, int v) { glUniform1i(u, v); }
void GraphicsDevice::Uniform1f(int u, float v) { glUniform1f(u, v); }
void GraphicsDevice::Uniform2f(int u, float v1, float v2) { glUniform2f(u, v1, v2); }
void GraphicsDevice::Uniform3fv(int u, int n, const float *v) { glUniform3fv(u, n, v); }

// Common layer
void GraphicsDevice::Flush() { glFlush(); }
void GraphicsDevice::Clear() { glClear(GL_COLOR_BUFFER_BIT | (draw_mode = DrawMode::_3D ? GL_DEPTH_BUFFER_BIT : 0)); }
void GraphicsDevice::ClearColor(const Color &c) { glClearColor(c.r(), c.g(), c.b(), c.a()); }
void GraphicsDevice::PushColor() { default_color.push_back(default_color.back()); UpdateColor();  }
void GraphicsDevice::PopColor() {
    if      (default_color.size() >= 1) default_color.pop_back();
    else if (default_color.size() == 1) default_color.back() = Color(1.0, 1.0, 1.0, 1.0);
    UpdateColor();
}
void GraphicsDevice::PointSize(float n) { glPointSize(n); }
void GraphicsDevice::LineWidth(float n) { glLineWidth(n); }
void GraphicsDevice::DelTextures(int n, const unsigned *id) { glDeleteTextures(n, id); }
void GraphicsDevice::GenTextures(int t, int n, unsigned *out) {
    for (int i=0; i<n; i++) CHECK_EQ(0, out[i]);
    if (t == GL_TEXTURE_CUBE_MAP) glEnable(GL_TEXTURE_CUBE_MAP);
    glGenTextures(n, out);
    for (int i=0; i<n; i++) {
        glBindTexture(t, out[i]);
        glTexParameteri(t, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(t, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
}

void GraphicsDevice::EnableDepthTest()  { GDDebug("DepthTest=1");  glEnable(GL_DEPTH_TEST); glDepthMask(GL_TRUE);  }
void GraphicsDevice::DisableDepthTest() { GDDebug("DepthTest=0"); glDisable(GL_DEPTH_TEST); glDepthMask(GL_FALSE); }
void GraphicsDevice::DisableBlend() { GDDebug("Blend=0"); glDisable(GL_BLEND); }
void GraphicsDevice::EnableBlend()  { GDDebug("Blend=1");  glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); }
void GraphicsDevice::BlendMode(int sm, int dm) { GDDebug("BlendMode=", sm, ",", dm); glBlendFunc(sm, dm); }
void GraphicsDevice::RestoreViewport(int dm) { ViewPort(screen->Box()); DrawMode(dm); }
void GraphicsDevice::DrawMode(int dm, bool flush) { return DrawMode(dm, screen->width, screen->height, flush); }
void GraphicsDevice::DrawMode(int dm, int W, int H, bool flush) {
    if (draw_mode == dm && !flush) return;
    draw_mode = dm;
    Color4f(1,1,1,1);
    MatrixProjection();
    LoadIdentity();
    if (FLAGS_rotate_view) Rotatef(FLAGS_rotate_view,0,0,1);

    bool _2D = draw_mode == DrawMode::_2D;
    if (_2D) Ortho(0, W, 0, H, 0, 100);
    else {
        float aspect=(float)W/H;
        double top = tan(FLAGS_field_of_view * M_PI/360.0) * FLAGS_near_plane;
        screen->gd->Frustum(aspect*-top, aspect*top, -top, top, FLAGS_near_plane, FLAGS_far_plane);
    }

    if (_2D) DisableDepthTest();
    else     EnableDepthTest();

    MatrixModelview();
    LoadIdentity();
    Scene::Select();
    if (_2D) EnableLayering();
}

void GraphicsDevice::LookAt(const v3 &pos, const v3 &targ, const v3 &up) {
    v3 Z = pos - targ;       Z.Norm();
    v3 X = v3::Cross(up, Z); X.Norm();
    v3 Y = v3::Cross(Z,  X); Y.Norm();
    float m[16] = {
        X.x, Y.x, Z.x, 0.0,
        X.y, Y.y, Z.y, 0.0,
        X.z, Y.z, Z.z, 0.0,
        0.0, 0.0, 0.0, 1.0
    };
    Mult(m);
    Translate(-pos.x, -pos.y, -pos.z);
}

void GraphicsDevice::ViewPort(Box w) {
    if (FLAGS_swap_axis) w.swapaxis(screen->width, screen->height);
    glViewport(w.x, w.y, w.w, w.h);
}

void GraphicsDevice::Scissor(Box w) {
    if (FLAGS_swap_axis) w.swapaxis(screen->width, screen->height);
    glEnable(GL_SCISSOR_TEST);
    glScissor(w.x, w.y, w.w, w.h);
}

void GraphicsDevice::PushScissor(Box w) {
    auto &ss = scissor_stack.back();
    if (ss.empty()) ss.push_back(w);
    else ss.push_back(w.Intersect(ss.back()));
    screen->gd->Scissor(ss.back());
}

void GraphicsDevice::PopScissor() {
    auto &ss = scissor_stack.back();
    if (ss.size()) ss.pop_back();
    if (ss.size()) screen->gd->Scissor(ss.back());
    else           glDisable(GL_SCISSOR_TEST);
}

void GraphicsDevice::PushScissorStack() {
    scissor_stack.push_back(vector<Box>());
    glDisable(GL_SCISSOR_TEST);
}

void GraphicsDevice::PopScissorStack() {
    CHECK_GT(scissor_stack.size(), 1);
    scissor_stack.pop_back();
    screen->gd->Scissor(scissor_stack.back().back());
}

int GraphicsDevice::VertsPerPrimitive(int primtype) {
    if (primtype == GL_TRIANGLES) return 3;
    return 0;
}

#else // LFL_HEADLESS
struct FakeGraphicsDevice : public GraphicsDevice {
    virtual void Init() {}
    virtual bool ShaderSupport() { return 0; }
    virtual void EnableTexture() {}
    virtual void DisableTexture() {}
    virtual void EnableLighting() {}
    virtual void DisableLighting() {}
    virtual void EnableNormals() {}
    virtual void DisableNormals() {}
    virtual void EnableVertexColor() {}
    virtual void DisableVertexColor() {}
    virtual void EnableLight(int n) {}
    virtual void DisableLight(int n) {}
    virtual void DisableCubeMap() {}
    virtual void BindCubeMap(int n) {}
    virtual void TextureGenLinear() {}
    virtual void TextureGenReflection() {}
    virtual void Material(int t, float *color) {}
    virtual void Light(int n, int t, float *color) {}
    virtual void BindTexture(int t, int n) {}
    virtual void ActiveTexture(int n) {}
    virtual void TexPointer(int m, int t, int w, int o, float *tex, int l, int *out, bool dirty) {}
    virtual void VertexPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {}
    virtual void ColorPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {}
    virtual void NormalPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {}
    virtual void Color4f(float r, float g, float b, float a) {}
    virtual void UpdateColor() {}
    virtual void MatrixProjection() {}
    virtual void MatrixModelview() {}
    virtual void LoadIdentity() {}
    virtual void PushMatrix() {}
    virtual void PopMatrix() {}
    virtual void GetMatrix(m44 *out) {}
    virtual void PrintMatrix() {}
    virtual void Scalef(float x, float y, float z) {}
    virtual void Rotatef(float angle, float x, float y, float z) {}
    virtual void Ortho(float l, float r, float b, float t, float nv, float fv) {}
    virtual void Frustum(float l, float r, float b, float t, float nv, float fv) {}
    virtual void Mult(const float *m) {}
    virtual void Translate(float x, float y, float z) {}
    virtual void UseShader(Shader *shader) {}
    virtual void Draw(int pt, int np, int it, int o, void *index, int l, int *out, bool dirty) {}
    virtual void DrawArrays(int t, int o, int n) {}
};

const int GraphicsDevice::Float = 0;
const int GraphicsDevice::Points = 0;
const int GraphicsDevice::Lines = 0;
const int GraphicsDevice::LineLoop = 0;
const int GraphicsDevice::Triangles = 0;
const int GraphicsDevice::TriangleStrip = 0;
const int GraphicsDevice::Texture2D = 0;
const int GraphicsDevice::UnsignedInt = 0;
const int GraphicsDevice::Ambient = 0;
const int GraphicsDevice::Diffuse = 0;
const int GraphicsDevice::Specular = 0;
const int GraphicsDevice::Position = 0;
const int GraphicsDevice::Emission = 0;
const int GraphicsDevice::One = 0;
const int GraphicsDevice::SrcAlpha = 0;
const int GraphicsDevice::OneMinusSrcAlpha = 0;
const int GraphicsDevice::OneMinusDstColor = 0;
const int GraphicsDevice::Fill = 0;
const int GraphicsDevice::Line = 0;
const int GraphicsDevice::Point = 0;

int GraphicsDevice::CreateProgram() { return 0; }
int GraphicsDevice::CreateShader(int t) { return 0; }
int GraphicsDevice::GetAttribLocation (int prog, const string &name) { return 0; }
int GraphicsDevice::GetUniformLocation(int prog, const string &name) { return 0; }
void GraphicsDevice::ShaderSource(int shader, int count, const char **source, int *len) {}
void GraphicsDevice::CompileShader(int shader) {}
void GraphicsDevice::AttachShader(int prog, int shader) {}
void GraphicsDevice::BindAttribLocation(int prog, int loc, const string &name) {}
void GraphicsDevice::LinkProgram(int prog) {}
void GraphicsDevice::GetProgramiv(int p, int t, int *out) {}
void GraphicsDevice::GetIntegerv(int t, int *out) {}
void GraphicsDevice::Uniform1i(int u, int v) {}
void GraphicsDevice::Uniform1f(int u, float v) {}
void GraphicsDevice::Uniform2f(int u, float v1, float v2) {}
void GraphicsDevice::Uniform3fv(int u, int n, const float *v) {}
void GraphicsDevice::Clear() {}
void GraphicsDevice::ClearColor(const Color &c) {}
void GraphicsDevice::PushColor() {}
void GraphicsDevice::PopColor() {}
void GraphicsDevice::PointSize(float n) {}
void GraphicsDevice::LineWidth(float n) {}
void GraphicsDevice::DelTextures(int n, const unsigned *id) {}
void GraphicsDevice::GenTextures(int t, int n, unsigned *out) {}
void GraphicsDevice::EnableDepthTest() {}
void GraphicsDevice::DisableDepthTest() {}
void GraphicsDevice::DisableBlend() {}
void GraphicsDevice::EnableBlend() {}
void GraphicsDevice::BlendMode(int sm, int dm) {}
void GraphicsDevice::RestoreViewport(int dm) {}
void GraphicsDevice::DrawMode(int dm, bool flush) {}
void GraphicsDevice::DrawMode(int dm, int W, int H, bool flush) {}
void GraphicsDevice::LookAt(const v3 &pos, const v3 &targ, const v3 &up) {}
void GraphicsDevice::ViewPort(Box w) {}
void GraphicsDevice::Scissor(Box w) {}
void GraphicsDevice::PushScissor(Box w) {}
void GraphicsDevice::PopScissor() {}
void GraphicsDevice::PushScissorStack() {}
void GraphicsDevice::PopScissorStack() {}
int GraphicsDevice::VertsPerPrimitive(int primtype) { return 0; }

bool Window::Create(Window *W) { screen->gd = new FakeGraphicsDevice(); Window::active[W->id] = W; return true; }
void Window::Close(Window *W) {}
void Window::MakeCurrent(Window *W) {}
#endif // LFL_HEADLESS

#ifdef LFL_ANDROIDVIDEO
struct AndroidVideoModule : public Module {
    int Init() {
        INFO("AndroidVideoModule::Init()");
        if (AndroidVideoInit(FLAGS_request_gles_version)) return -1;
        return 0;
    }
};
#endif

#ifdef LFL_IPHONEVIDEO
extern "C" void iPhoneVideoSwap();
struct IPhoneVideoModule : public Module {
    int Init() {
        INFO("IPhoneVideoModule::Init()");
        NativeWindowInit();
        NativeWindowSize(&screen->width, &screen->height);
        return 0;
    }
};
#endif

#ifdef LFL_OSXVIDEO
extern "C" void OSXVideoSwap(void*);
extern "C" void OSXSetWindowSize(void*, int W, int H);
struct OSXVideoModule : public Module {
    int Init() {
        INFO("OSXVideoModule::Init()");
        NativeWindowInit();
        NativeWindowSize(&screen->width, &screen->height);
        CHECK(Window::Create(screen));
        return 0;
    }
};
bool Window::Create(Window *W) { 
    CHECK(W->id);
    Window::active[W->id] = W;
    return true; 
}
void Window::MakeCurrent(Window *W) { screen=W; }
void Window::Close(Window *W) {}
#endif

#ifdef LFL_QT
struct QTVideoModule : public Module {
    int Init() {
        INFO("QTVideoModule::Init()");
        screen->Reshape(screen->width, screen->height);
        return 0;
    }
};
bool Window::Create(Window *W) {
    CHECK(!W->id && !W->gd);
    OpenGLES2 *gd = new OpenGLES2();
    QWindow *qwin = (QWindow*)gd;
    W->id = qwin;
    W->gd = gd;
    W->opengles_version = 2;

    QSurfaceFormat format;
    format.setSamples(16);
    gd->setFormat(format);
    gd->setSurfaceType(QWindow::OpenGLSurface);
    gd->resize(screen->width, screen->height);
    gd->show();
    gd->render_request();

    Window::active[W->id] = W;
    return true;
}
void Window::Close(Window *W) {
    Window::active.erase(W->id);
    if (Window::active.empty()) app->run = false;
    screen = 0;
}
void Window::MakeCurrent(Window *W) {
    screen = W; 
    ((QOpenGLContext*)screen->gl)->makeCurrent((QWindow*)screen->id);
}
void Mouse::GrabFocus()    { ((OpenGLES2*)screen->gd)->QT_grabbed=1; ((QWindow*)screen->id)->setCursor(Qt::BlankCursor); app->grab_mode.On();  screen->cursor_grabbed=true;  }
void Mouse::ReleaseFocus() { ((OpenGLES2*)screen->gd)->QT_grabbed=0; ((QWindow*)screen->id)->unsetCursor();              app->grab_mode.Off(); screen->cursor_grabbed=false; }
#endif

#ifdef LFL_GLFWVIDEO
/* struct NativeWindow { GLFWwindow *id; }; */
struct GLFWVideoModule : public Module {
    int Init() {
        INFO("GLFWVideoModule::Init");
        CHECK(Window::Create(screen));
        Window::MakeCurrent(screen);
        glfwSwapInterval(1);
        return 0;
    }
    int Free() {
        glfwTerminate();
        return 0;
    }
};
bool Window::Create(Window *W) {
    GLFWwindow *share = Window::active.empty() ? 0 : (GLFWwindow*)Window::active.begin()->second->id;
    if (!(W->id = glfwCreateWindow(W->width, W->height, W->caption.c_str(), 0, share))) { ERROR("glfwCreateWindow"); return false; }
    Window::active[W->id] = W;
    return true;
}
void Window::MakeCurrent(Window *W) {
    glfwMakeContextCurrent((GLFWwindow*)W->id);
    screen = W;
}
void Window::Close(Window *W) {
    Window::active.erase(W->id);
    bool done = Window::active.empty();
    if (done) app->shell.quit(vector<string>());
    if (!done) glfwDestroyWindow((GLFWwindow*)W->id);
    if (app->window_closed_cb) app->window_closed_cb();
    screen = 0;
}
#endif

#ifdef LFL_SDLVIDEO
/* struct NativeWindow { SDL_Window* id; SDL_GLContext gl; SDL_Surface *surface; }; */
struct SDLVideoModule : public Module {
    int Init() {
        INFO("SFLVideoModule::Init");
        CHECK(Window::Create(screen));
        Window::MakeCurrent(screen);
        SDL_GL_SetSwapInterval(1);
        return 0;
    }
    int Free() {
        SDL_Quit();
        return 0;
    }
};
bool Window::Create(Window *W) {
    int createFlag = SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN;
#if defined(LFL_IPHONE) || defined(LFL_ANDROID)
    createFlag |= SDL_WINDOW_BORDERLESS;
    int bitdepth[] = { 5, 6, 5 };
#else
    int bitdepth[] = { 8, 8, 8 };
#endif
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, bitdepth[0]);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, bitdepth[1]);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, bitdepth[2]);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    if (!(W->id = SDL_CreateWindow(W->caption.c_str(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, W->width, W->height, createFlag)))
    { ERROR("SDL_CreateWindow: ",     SDL_GetError()); return false; }

    if (!Window::active.empty()) W->gl = Window::active.begin()->second->gl;
    else if (!(W->gl = SDL_GL_CreateContext((SDL_Window*)W->id)))
    { ERROR("SDL_GL_CreateContext: ", SDL_GetError()); return false; } 

    SDL_Surface* icon = SDL_LoadBMP(StrCat(ASSETS_DIR, "icon.bmp").c_str());
    SDL_SetWindowIcon((SDL_Window*)W->id, icon);

    Window::active[(void*)(long)SDL_GetWindowID((SDL_Window*)W->id)] = W;
    return true;
}
void Window::MakeCurrent(Window *W) {
    if (SDL_GL_MakeCurrent((SDL_Window*)W->id, W->gl) < 0) ERROR("SDL_GL_MakeCurrent: ", SDL_GetError());
    screen = W; 
}
void Window::Close(Window *W) {
    SDL_GL_MakeCurrent(NULL, NULL);
    Window::active.erase((void*)(long)SDL_GetWindowID((SDL_Window*)W->id));
    if (Window::active.empty()) {
        app->run = false;
        SDL_GL_DeleteContext(W->gl);
    }
    SDL_DestroyWindow((SDL_Window*)W->id);
    if (app->window_closed_cb) app->window_closed_cb();
    screen = 0;
}
#endif /* LFL_SDLVIDEO */

int Video::Init() {
#if defined(LFL_QT)
    impl = new QTVideoModule();
#elif defined(LFL_GLFWVIDEO)
    impl = new GLFWVideoModule();
#elif defined(LFL_SDLVIDEO)
    impl = new SDLVideoModule();
#elif defined(LFL_ANDROIDVIDEO)
    impl = new AndroidVideoModule();
#elif defined(LFL_IPHONEVIDEO)
    impl = new IPhoneVideoModule();
#elif defined(LFL_OSXVIDEO)
    impl = new OSXVideoModule();
#endif
    if (impl) if (impl->Init()) return -1;

#if defined(LFL_GLEW) && !defined(LFL_HEADLESS)
#ifdef GLEW_MX
    screen.glew_context = new GLEWContext();
#endif
    GLenum glew_err;
    if ((glew_err = glewInit()) != GLEW_OK) { ERROR("glewInit: ", glewGetErrorString(glew_err)); return -1; }
#endif

    if (!screen->gd) CreateGraphicsDevice();
    InitGraphicsDevice();

    INFO("OpenGL Version: ", SpellNull((const char *)glGetString(GL_VERSION)));
    INFO("OpenGL Vendor: ",  SpellNull((const char *)glGetString(GL_VENDOR)));
#ifdef LFL_GLSL_SHADERS
    INFO("GL_SHADING_LANGUAGE_VERSION: ", SpellNull((const char *)glGetString(GL_SHADING_LANGUAGE_VERSION)));
#endif
    const char *glexts = SpellNull((const char *)glGetString(GL_EXTENSIONS));
    INFO("GL_EXTENSIONS: ", glexts);
#if defined(LFL_ANDROID) || defined(LFL_IPHONE)
    screen->opengles_cubemap = strstr(glexts, "GL_EXT_texture_cube_map") != 0;
#else
    screen->opengles_cubemap = strstr(glexts, "GL_ARB_texture_cube_map") != 0;
#endif
    INFO("lfapp_opengles_cubemap = ", screen->opengles_cubemap ? "true" : "false");

    vector<string> atlas_font_size;
    Split(FLAGS_atlas_font_sizes, iscomma, &atlas_font_size);
    for (int i=0; i<atlas_font_size.size(); i++) {
        int size = atoi(atlas_font_size[i].c_str());
        if      (FLAGS_font_engine == "atlas")    { CHECK(Fonts::InsertAtlas   (FLAGS_default_font, FLAGS_default_font_family, size, Color::white, FLAGS_default_font_flag)); CHECK(!FLAGS_atlas_dump); }
        else if (FLAGS_font_engine == "freetype") { CHECK(Fonts::InsertFreetype(FLAGS_default_font, FLAGS_default_font_family, size, Color::white, FLAGS_default_font_flag)); }
        else if (FLAGS_font_engine == "coretext") { CHECK(Fonts::InsertCoreText(FLAGS_default_font, FLAGS_default_font_family, size, Color::white, FLAGS_default_font_flag)); }
    }

    Fonts::InsertAtlas("MenuAtlas1", "", 0, Color::black, 0); 
    Fonts::InsertAtlas("MenuAtlas2", "", 0, Color::black, 0); 

    if (!screen->console) screen->InitConsole();
    return 0;
}

void Video::CreateGraphicsDevice() {
    CHECK(!screen->gd);
#ifndef LFL_HEADLESS
#ifdef LFL_GLES2
#if !defined(LFL_IPHONE) && !defined(LFL_ANDROID)
    screen->opengles_version = FLAGS_request_gles_version;
#endif
    if (screen->opengles_version == 2) screen->gd = new OpenGLES2();
    else
#endif /* LFL_GLES2 */
    screen->gd = new OpenGLES1();
#endif /* LFL_HEADLESS */
}

void Video::InitGraphicsDevice() {
    screen->gd->Init();
    screen->gd->ViewPort(screen->Box());
    screen->gd->DrawMode(DrawMode::_3D);

    float pos[]={-.5,1,-.3f,0}, grey20[]={.2f,.2f,.2f,1}, white[]={1,1,1,1}, black[]={0,0,0,1};
    screen->gd->EnableLight(0);
    screen->gd->Light(0, GraphicsDevice::Position, pos);
    screen->gd->Light(0, GraphicsDevice::Ambient,  grey20);
    screen->gd->Light(0, GraphicsDevice::Diffuse,  white);
    screen->gd->Light(0, GraphicsDevice::Specular, white);
    screen->gd->Material(GraphicsDevice::Emission, black);
    screen->gd->Material(GraphicsDevice::Specular, grey20);

    INFO("opengl_init: width=", screen->width, ", height=", screen->height,
         ", opengles_version: ", screen->opengles_version);
}

int Video::Swap() {
#ifndef LFL_QT
    screen->gd->Flush();
#endif

#if defined(LFL_QT)
    ((QOpenGLContext*)screen->gl)->swapBuffers((QWindow*)screen->id);
#elif defined(LFL_ANDROIDVIDEO)
    AndroidVideoSwap();
#elif defined(LFL_GLFWVIDEO)
    glfwSwapBuffers((GLFWwindow*)screen->id);
#elif defined(LFL_SDLVIDEO)
    SDL_GL_SwapWindow((SDL_Window*)screen->id);
#elif defined(LFL_IPHONEVIDEO)
    iPhoneVideoSwap();
#elif defined(LFL_OSXVIDEO)
    OSXVideoSwap(screen->id);
#endif

    GLint gl_error=0, gl_validate_status=0;
#ifdef LFL_GLSL_SHADERS
#if 0
    if (screen->opengles_version == 2) {
        glGetProgramiv(GraphicsDevice::shader, GL_VALIDATE_STATUS, &gl_validate_status);
        if (gl_validate_status != GL_TRUE) ERROR("gl validate status ", gl_validate_status);

        char buf[1024]; int len;
        glGetProgramInfoLog(GraphicsDevice::shader, sizeof(buf), &len, buf);
        if (len) INFO(buf);
    }
#endif
#endif
    if ((gl_error = glGetError())) ERROR("gl error ", gl_error);
    return 0;
}

int Video::Free() {
    if (impl) impl->Free();
    return 0;
}

void Window::Reshape(int w, int h) {
#if defined(LFL_QT)
    ((QWindow*)id)->resize(w, h);
    Window::MakeCurrent(screen);
#elif defined(LFL_GLFWVIDEO)
    glfwSetWindowSize((GLFWwindow*)id, w, h);
#elif defined(LFL_SDLVIDEO)
    SDL_SetWindowSize((SDL_Window*)id, w, h);
#elif defined(LFL_OSXVIDEO)
    OSXSetWindowSize(id, w, h);
#endif
}

void Window::Reshaped(int w, int h) {
    width = w;
    height = h;
    if (!gd) return;
    gd->ViewPort(LFL::Box(width, height));
    gd->DrawMode(DrawMode::_3D);
    for (auto g = screen->mouse_gui.begin(); g != screen->mouse_gui.end(); ++g) (*g)->Layout();
    if (app->reshaped_cb) app->reshaped_cb();
}

void Window::SwapAxis() {
    FLAGS_rotate_view = FLAGS_rotate_view ? 0 : -90;
    FLAGS_swap_axis = FLAGS_rotate_view != 0;
    Reshaped(height, width);
}

int Depth::OpenGLID(int id) {
    switch(id) {
        case _16: return GL_DEPTH_COMPONENT16;
    } return 0;
}

int CubeMap::OpenGLID(int id) {
    switch (id) {
        case NX: return GL_TEXTURE_CUBE_MAP_NEGATIVE_X;    case PX: return GL_TEXTURE_CUBE_MAP_POSITIVE_X;
        case NY: return GL_TEXTURE_CUBE_MAP_NEGATIVE_Y;    case PY: return GL_TEXTURE_CUBE_MAP_POSITIVE_Y;
        case NZ: return GL_TEXTURE_CUBE_MAP_NEGATIVE_Z;    case PZ: return GL_TEXTURE_CUBE_MAP_POSITIVE_Z;
    } return GL_TEXTURE_2D;
}

int ColorChannel::PixelOffset(int c) {
    switch (c) {
        case Red:  return 0;    case Green: return 1;
        case Blue: return 2;    case Alpha: return 3;
    }
    return 0;
}

const char *Pixel::Name(int p) {
    switch (p) {
        case RGB32: return "RGB32";    case RGB555:  return "RGB555";
        case BGR32: return "BGR32";    case BGR555:  return "BGR555";
        case RGB24: return "RGB24";    case RGB565:  return "RGB565";
        case BGR24: return "BGR24";    case BGR565:  return "BGR565";
        case RGBA:  return "RGBA";     case YUYV422: return "YUYV422";
        case GRAY8: return "GRAY8";    case GRAYA8:  return "GRAYA8";
        case LCD:   return "LCD";
    }; return 0; 
}

int Pixel::size(int p) {
    switch (p) {
        case RGB32:   case BGR32:  case RGBA:                             return 4;
        case RGB24:   case BGR24:  case LCD:                              return 3;
        case RGB555:  case BGR555: case RGB565: case BGR565: case GRAYA8: return 2;
        case YUYV422: case GRAY8:                                         return 1;
        default:                                                          return 0;
    }
}

int Pixel::OpenGLID(int p) {
    switch (p) {
        case RGBA:   case RGB32: case BGR32: return GL_RGBA;
        case RGB24:  case BGR24:             return GL_RGB;
        case GRAYA8:                         return GL_LUMINANCE_ALPHA;
        case GRAY8:                          return GL_LUMINANCE;
        default:                             return -1;
    }
}

string FloatContainer::DebugString() const {
    string ret = StrCat(Box::DebugString(), " fl{");
    for (int i=0; i<float_left.size(); i++) StrAppend(&ret, i?",":"", i, "=", float_left[i].DebugString());
    StrAppend(&ret, "} fr{");
    for (int i=0; i<float_right.size(); i++) StrAppend(&ret, i?",":"", i, "=", float_right[i].DebugString());
    return ret + "}";
}

string Box::DebugString() const { return StringPrintf("Box = { %d, %d, %d, %d }", x, y, w, h); }

void Box::Draw(const float *texcoord) const {
#if defined(LFL_GLES2) || defined(LFL_MOBILE)
    static int verts_ind=-1;
    static const float default_texcoord[4] = {0, 0, 1, 1};
    const float *tc = X_or_Y(texcoord, default_texcoord);
    float verts[] = { (float)x,   (float)y,   tc[Texture::CoordMinX], tc[Texture::CoordMinY],
                      (float)x,   (float)y+h, tc[Texture::CoordMinX], tc[Texture::CoordMaxY],
                      (float)x+w, (float)y,   tc[Texture::CoordMaxX], tc[Texture::CoordMinY],
                      (float)x+w, (float)y+h, tc[Texture::CoordMaxX], tc[Texture::CoordMaxY] };
    if (1)        screen->gd->VertexPointer(2, GraphicsDevice::Float, sizeof(float)*4, 0,               verts, sizeof(verts), &verts_ind, true);
    if (texcoord) screen->gd->TexPointer   (2, GraphicsDevice::Float, sizeof(float)*4, sizeof(float)*2, verts, sizeof(verts), &verts_ind, false);
    if (1)        screen->gd->DrawArrays(GraphicsDevice::TriangleStrip, 0, 4);
#else
    float rx=w.x*2.0/screen->width-1, ry=w.y*2.0/screen->height-1;
    glRectf(rx, ry, rx+(w.w-2)*2.0/screen->width, ry+(w.h-2)*2.0/screen->height);
#endif
}

void Box::DrawCrimped(const float *texcoord, int orientation, float scrollX, float scrollY) const {
    float left=x, right=x+w, top=y, bottom=y+h;
    float texMinX, texMinY, texMaxX, texMaxY, texMidX1, texMidX2, texMidY1, texMidY2;

    scrollX *= (texcoord[2] - texcoord[0]);
    scrollY *= (texcoord[3] - texcoord[1]);
    scrollX = ScrollCrimped(texcoord[0], texcoord[2], scrollX, &texMinX, &texMidX1, &texMidX2, &texMaxX);
    scrollY = ScrollCrimped(texcoord[1], texcoord[3], scrollY, &texMinY, &texMidY1, &texMidY2, &texMaxY);

#   define DrawCrimpedBoxTriangleStrip() \
    screen->gd->VertexPointer(2, GraphicsDevice::Float, 4*sizeof(float), 0,               verts, sizeof(verts), &vind, true); \
    screen->gd->TexPointer   (2, GraphicsDevice::Float, 4*sizeof(float), 2*sizeof(float), verts, sizeof(verts), &vind, false); \
    screen->gd->DrawArrays(GraphicsDevice::TriangleStrip, 0, 4); \
    screen->gd->DrawArrays(GraphicsDevice::TriangleStrip, 4, 4); \
    screen->gd->DrawArrays(GraphicsDevice::TriangleStrip, 8, 4); \
    screen->gd->DrawArrays(GraphicsDevice::TriangleStrip, 12, 4);

    switch (orientation) {
        case 0: {
            static int vind = -1;
            float xmid = x + w * scrollX, ymid = y + h * scrollY, verts[] = {
                /*02*/ xmid,  top,  texMidX1, texMaxY,  /*01*/ left, top,  texMinX,  texMaxY,  /*03*/ xmid,  ymid,   texMidX1, texMidY1, /*04*/ left, ymid,   texMinX,  texMidY1,
                /*06*/ right, top,  texMaxX,  texMaxY,  /*05*/ xmid, top,  texMidX2, texMaxY,  /*07*/ right, ymid,   texMaxX,  texMidY1, /*08*/ xmid, ymid,   texMidX2, texMidY1,
                /*10*/ right, ymid, texMaxX,  texMidY2, /*09*/ xmid, ymid, texMidX2, texMidY2, /*11*/ right, bottom, texMaxX,  texMinY,  /*12*/ xmid, bottom, texMidX2, texMinY,
                /*14*/ xmid,  ymid, texMidX1, texMidY2, /*13*/ left, ymid, texMinX,  texMidY2, /*15*/ xmid,  bottom, texMidX1, texMinY,  /*16*/ left, bottom, texMinX,  texMinY 
            };
            DrawCrimpedBoxTriangleStrip();
        } break;
        case 1: {
            static int vind = -1;
            float xmid = x + w * scrollX, ymid = y + h * (1-scrollY), verts[] = {
                /*02*/ xmid,  top,  texMidX1, texMinY,  /*01*/ left,  top,  texMinX,  texMinY,  /*03*/ xmid, ymid,    texMidX1, texMidY2, /*04*/ left, ymid,   texMinX,  texMidY2,
                /*06*/ right, top,  texMaxX,  texMinY,  /*05*/ xmid,  top,  texMidX2, texMinY,  /*07*/ right, ymid,   texMaxX,  texMidY2, /*08*/ xmid, ymid,   texMidX2, texMidY2,
                /*10*/ right, ymid, texMaxX,  texMidY1, /*09*/ xmid,  ymid, texMidX2, texMidY1, /*11*/ right, bottom, texMaxX,  texMaxY,  /*12*/ xmid, bottom, texMidX2, texMaxY,
                /*14*/ xmid,  ymid, texMidX1, texMidY1, /*13*/ left,  ymid, texMinX,  texMidY1, /*15*/ xmid, bottom,  texMidX1, texMaxY,  /*16*/ left, bottom, texMinX,  texMaxY 
            };
            DrawCrimpedBoxTriangleStrip();
        } break;
        case 2: {
            static int vind = -1;
            float xmid = x + w * (1-scrollX), ymid = y + h * scrollY, verts[] = {
                /*02*/ xmid,  top,  texMidX2, texMaxY,  /*01*/ left,  top,  texMaxX,  texMaxY,  /*03*/ xmid, ymid,    texMidX2, texMidY1, /*04*/ left, ymid,   texMaxX,  texMidY1,
                /*06*/ right, top,  texMinX,  texMaxY,  /*05*/ xmid,  top,  texMidX1, texMaxY,  /*07*/ right, ymid,   texMinX,  texMidY1, /*08*/ xmid, ymid,   texMidX1, texMidY1,
                /*10*/ right, ymid, texMinX,  texMidY2, /*09*/ xmid,  ymid, texMidX1, texMidY2, /*11*/ right, bottom, texMinX,  texMinY,  /*12*/ xmid, bottom, texMidX1, texMinY,
                /*14*/ xmid,  ymid, texMidX2, texMidY2, /*13*/ left,  ymid, texMaxX,  texMidY2, /*15*/ xmid, bottom,  texMidX2, texMinY,  /*16*/ left, bottom, texMaxX,  texMinY 
            };
            DrawCrimpedBoxTriangleStrip();
        } break;
        case 3: {
            static int vind = -1;
            float xmid = x + w * (1-scrollX), ymid = y + h * (1-scrollY), verts[] = {
                /*02*/ xmid,  top,  texMidX2, texMinY,  /*01*/ left,  top,   texMaxX,  texMinY,  /*03*/ xmid, ymid,    texMidX2, texMidY2, /*04*/ left, ymid,   texMaxX,  texMidY2,
                /*06*/ right, top,  texMinX,  texMinY,  /*05*/ xmid,  top,   texMidX1, texMinY,  /*07*/ right, ymid,   texMinX,  texMidY2, /*08*/ xmid, ymid,   texMidX1, texMidY2,
                /*10*/ right, ymid, texMinX,  texMidY1, /*09*/ xmid,  ymid,  texMidX1, texMidY1, /*11*/ right, bottom, texMinX,  texMaxY,  /*12*/ xmid, bottom, texMidX1, texMaxY,
                /*14*/ xmid,  ymid, texMidX2, texMidY1, /*13*/ left,  ymid,  texMaxX,  texMidY1, /*15*/ xmid, bottom,  texMidX2, texMaxY,  /*16*/ left, bottom, texMaxX,  texMaxY 
            };
            DrawCrimpedBoxTriangleStrip();
        } break;
        case 4: {
            static int vind = -1;
            float xmid = x + w * (1-scrollY), ymid = y + h * scrollX, verts[] = {
                /*13*/ xmid,  top,  texMinX,  texMidY2, /*16*/ left,  top,  texMinX,  texMaxY,  /*14*/ xmid, ymid,    texMidX1, texMidY2, /*15*/ left, ymid,   texMidX1, texMaxY, 
                /*01*/ right, top,  texMinX,  texMinY,  /*04*/ xmid,  top,  texMinX,  texMidY1, /*02*/ right, ymid,   texMidX1, texMinY,  /*03*/ xmid, ymid,   texMidX1, texMidY1,
                /*05*/ right, ymid, texMidX2, texMinY,  /*08*/ xmid,  ymid, texMidX2, texMidY1, /*06*/ right, bottom, texMaxX,  texMinY,  /*07*/ xmid, bottom, texMaxX,  texMidY1,
                /*09*/ xmid,  ymid, texMidX2, texMidY2, /*12*/ left,  ymid, texMidX2, texMaxY,  /*10*/ xmid, bottom,  texMaxX,  texMidY2, /*11*/ left, bottom, texMaxX,  texMaxY 
            };
            DrawCrimpedBoxTriangleStrip();
        } break;
        case 5: {
            static int vind = -1;
            float xmid = x + w * scrollY, ymid = y + h * scrollX, verts[] = {
                /*13*/ xmid,  top,  texMinX,  texMidY1, /*16*/ left,  top,  texMinX,  texMinY,  /*14*/ xmid, ymid,    texMidX1, texMidY1, /*15*/ left, ymid,   texMidX1, texMinY, 
                /*01*/ right, top,  texMinX,  texMaxY,  /*04*/ xmid,  top,  texMinX,  texMidY2, /*02*/ right, ymid,   texMidX1, texMaxY,  /*03*/ xmid, ymid,   texMidX1, texMidY2,
                /*05*/ right, ymid, texMidX2, texMaxY,  /*08*/ xmid,  ymid, texMidX2, texMidY2, /*06*/ right, bottom, texMaxX,  texMaxY,  /*07*/ xmid, bottom, texMaxX,  texMidY2,
                /*09*/ xmid,  ymid, texMidX2, texMidY1, /*12*/ left,  ymid, texMidX2, texMinY,  /*10*/ xmid, bottom,  texMaxX,  texMidY1, /*11*/ left, bottom, texMaxX,  texMinY 
            };
            DrawCrimpedBoxTriangleStrip();
        } break;
        case 6: {
            static int vind = -1;
            float xmid = x + w * (1-scrollY), ymid = y + h * (1-scrollX), verts[] = {
                /*13*/ xmid,  top,  texMaxX,  texMidY2, /*16*/ left,  top,  texMaxX,  texMaxY,  /*14*/ xmid, ymid,    texMidX2, texMidY2, /*15*/ left, ymid,   texMidX2, texMaxY, 
                /*01*/ right, top,  texMaxX,  texMinY,  /*04*/ xmid,  top,  texMaxX,  texMidY1, /*02*/ right, ymid,   texMidX2, texMinY,  /*03*/ xmid, ymid,   texMidX2, texMidY1,
                /*05*/ right, ymid, texMidX1, texMinY,  /*08*/ xmid,  ymid, texMidX1, texMidY1, /*06*/ right, bottom, texMinX,  texMinY,  /*07*/ xmid, bottom, texMinX,  texMidY1,
                /*09*/ xmid,  ymid, texMidX1, texMidY2, /*12*/ left,  ymid, texMidX1, texMaxY,  /*10*/ xmid, bottom,  texMinX,  texMidY2, /*11*/ left, bottom, texMinX,  texMaxY 
            };
            DrawCrimpedBoxTriangleStrip();
        } break;
        case 7: {
            static int vind = -1;
            float xmid = x + w * scrollY, ymid = y + h * (1-scrollX), verts[] = {
                /*13*/ xmid,  top,  texMaxX,  texMidY1, /*16*/ left,  top,  texMaxX,  texMinY,  /*14*/ xmid, ymid,    texMidX2, texMidY1, /*15*/ left, ymid,   texMidX2, texMinY, 
                /*01*/ right, top,  texMaxX,  texMaxY,  /*04*/ xmid,  top,  texMaxX,  texMidY2, /*02*/ right, ymid,   texMidX2, texMaxY,  /*03*/ xmid, ymid,   texMidX2, texMidY2,
                /*05*/ right, ymid, texMidX1, texMaxY,  /*08*/ xmid,  ymid, texMidX1, texMidY2, /*06*/ right, bottom, texMinX,  texMaxY,  /*07*/ xmid, bottom, texMinX,  texMidY2,
                /*09*/ xmid,  ymid, texMidX1, texMidY1, /*12*/ left,  ymid, texMidX1, texMinY,  /*10*/ xmid, bottom,  texMinX,  texMidY1, /*11*/ left, bottom, texMinX,  texMinY 
            };
            DrawCrimpedBoxTriangleStrip();
        } break;
    }
}

float Box::ScrollCrimped(float tex0, float tex1, float scroll, float *min, float *mid1, float *mid2, float *max) {
    if (tex1 < 1.0 && tex0 == 0.0) {
        *mid1=tex1; *mid2=0;
        if (scroll > 0) *min = *max = tex1 - scroll;
        else            *min = *max = tex0 - scroll;
    } else if (tex0 > 0.0 && tex1 == 1.0) {
        *mid1=1; *mid2=tex0;
        if (scroll > 0) *min = *max = tex0 + scroll;
        else            *min = *max = tex1 + scroll;
    } else if (tex0 == 0 && tex1 == 1) {
        *min = *max = 1;
        *mid1 = tex1; *mid2 = tex0;
    } else {
        return 0;
    }
    return (*mid1 - *min) / (tex1 - tex0); 
}

void Box3::Draw(const Color *c) const {
    if (c) screen->gd->SetColor(*c);
    for (int i=0; i<3; i++) if (v[i].h) v[i].Draw();
}

void SimpleVideoResampler::RGB2BGRCopyPixels(unsigned char *dst, const unsigned char *src, int l, int bpp) {
    for (int k = 0; k < l; k++) for (int i = 0; i < bpp; i++) dst[k*bpp+(!i?2:(i==2?0:i))] = src[k*bpp+i];
}

bool SimpleVideoResampler::Supports(int f) { return f == Pixel::RGB24 || f == Pixel::BGR24 || f == Pixel::RGB32 || f == Pixel::BGR32 || f == Pixel::RGBA; }

bool SimpleVideoResampler::Opened() { return s_fmt && d_fmt && s_width && d_width && s_height && d_height; }

void SimpleVideoResampler::Open(int sw, int sh, int sf, int dw, int dh, int df) {
    s_fmt = sf; s_width = sw; s_height = sh;
    d_fmt = df; d_width = dw; d_height = dh;
    // INFO("resample ", BlankNull(Pixel::Name(s_fmt)), " -> ", BlankNull(Pixel::Name(d_fmt)), " : (", sw, ",", sh, ") -> (", dw, ",", dh, ")");
}

void SimpleVideoResampler::Resample(const unsigned char *sb, int sls, unsigned char *db, int dls, bool flip_x, bool flip_y) {
    if (!Opened()) { ERROR("resample not opened()"); return; }

    int sw = Pixel::size(s_fmt), dw = Pixel::size(d_fmt);
    if (sw * s_width > sls) { ERROR(sw * s_width, " > ", sls); return; }
    if (dw * d_width > dls) { ERROR(dw * d_width, " > ", dls); return; }
    
    if (s_width == d_width && s_height == d_height) {
        for (int y=0; y<d_height; y++) {
            for (int x=0; x<d_width; x++) {
                const unsigned char *sp = (sb + sls * y                           + x                          * sw);
                /**/  unsigned char *dp = (db + dls * (flip_y ? d_height-1-y : y) + (flip_x ? d_width-1-x : x) * dw);
                CopyPixel(s_fmt, d_fmt, sp, dp, x == 0, x == d_width-1);
            }
        }
    } else {
        for (int po=0; po<sw && po<dw; po++) {
            Matrix M(s_height, s_width);
            ColorChannelToMatrix(sb, s_width, s_height, sw, sls, 0, 0, &M, po);
            for (int y=0; y<d_height; y++) {
                for (int x=0; x<d_width; x++) {
                    unsigned char *dp = (db + dls * (flip_y ? d_height-1-y : y) + (flip_x ? d_width-1-x : x) * dw);
                    *(dp + po) = MatrixAsFunc(&M, x?(float)x/(d_width-1):0, y?(float)y/(d_height-1):0) * 255;
                }
            }
        }
    }
}

void SimpleVideoResampler::CopyPixel(int s_fmt, int d_fmt, const unsigned char *sp, unsigned char *dp, bool sxb, bool sxe, int f) {
    unsigned char r, g, b, a;
    switch (s_fmt) {
        case Pixel::RGB24: r = *sp++; g = *sp++; b = *sp++; a = ((f & Flag::TransparentBlack) && !r && !g && !b) ? 0 : 255; break;
        case Pixel::BGR24: r = *sp++; g = *sp++; b = *sp++; a = ((f & Flag::TransparentBlack) && !r && !g && !b) ? 0 : 255; break;
        case Pixel::RGB32: r = *sp++; g = *sp++; b = *sp++; a=*sp++; break;
        case Pixel::BGR32: r = *sp++; g = *sp++; b = *sp++; a=*sp++; break;
        case Pixel::RGBA:  r = *sp++; g = *sp++; b = *sp++; a=*sp++; break;
        case Pixel::GRAY8: r = 255;   g = 255;   b = 255;   a=*sp++; break;
        case Pixel::LCD: 
            r = (sxb ? 0 : *(sp-1)) / 3.0 + *sp / 3.0 + (          *(sp+1)) / 3.0; sp++; 
            g = (          *(sp-1)) / 3.0 + *sp / 3.0 + (          *(sp+1)) / 3.0; sp++; 
            b = (          *(sp-1)) / 3.0 + *sp / 3.0 + (sxe ? 0 : *(sp+1)) / 3.0; sp++;
            a = ((f & Flag::TransparentBlack) && !r && !g && !b) ? 0 : 255;
            break;
        default: ERROR("s_fmt ", s_fmt, " not supported"); return;
    }
    switch (d_fmt) {
        case Pixel::RGB24: *dp++ = r; *dp++ = g; *dp++ = b; break;
        case Pixel::BGR24: *dp++ = r; *dp++ = g; *dp++ = b; break;
        case Pixel::RGB32: *dp++ = r; *dp++ = g; *dp++ = b; *dp++ = a; break;
        case Pixel::BGR32: *dp++ = r; *dp++ = g; *dp++ = b; *dp++ = a; break;
        case Pixel::RGBA:  *dp++ = r; *dp++ = g; *dp++ = b; *dp++ = a; break;
        default: ERROR("d_fmt ", d_fmt, " not supported"); return;
    }
}

void SimpleVideoResampler::Blit(const unsigned char *src, unsigned char *dst, int w, int h,
                                int sf, int sls, int sx, int sy,
                                int df, int dls, int dx, int dy, int flag) {
    bool flip_y = flag & Flag::FlipY;
    int sw = Pixel::size(sf), dw = Pixel::size(df); 
    for (int yi = 0; yi < h; ++yi) {
        for (int xi = 0; xi < w; ++xi) {
            int sind = flip_y ? sy + h - yi - 1 : sy + yi;
            const unsigned char *sp = src + (sls*(sind)    + (sx + xi)*sw);
            unsigned       char *dp = dst + (dls*(dy + yi) + (dx + xi)*dw);
            CopyPixel(sf, df, sp, dp, xi == 0, xi == w-1, flag);
        }
    }
}

void SimpleVideoResampler::Filter(unsigned char *buf, int w, int h,
                                  int pf, int ls, int x, int y,
                                  Matrix *kernel, int channel, int flag) {
    Matrix M(h, w), out(h, w);
    int pw = Pixel::size(pf);
    ColorChannelToMatrix(buf, w, h, pw, ls, x, y, &M, ColorChannel::PixelOffset(channel));
    Matrix::Convolve(&M, kernel, &out, (flag & Flag::ZeroOnly) ? mZeroOnly : 0);
    MatrixToColorChannel(&out, w, h, pw, ls, x, y, buf, ColorChannel::PixelOffset(channel));
}

void SimpleVideoResampler::ColorChannelToMatrix(const unsigned char *buf, int w, int h,
                                                int pw, int ls, int x, int y,
                                                Matrix *out, int po) {
    MatrixIter(out) { 
        const unsigned char *p = buf + (ls*(y + i) + (x + j)*pw);
        out->row(i)[j] = *(p + po) / 255.0;
    }
}

void SimpleVideoResampler::MatrixToColorChannel(const Matrix *M, int w, int h,
                                                int pw, int ls, int x, int y,
                                                unsigned char *out, int po) {
    MatrixIter(M) { 
        unsigned char *p = out + (ls*(y + i) + (x + j)*pw);
        *(p + po) = M->row(i)[j] * 255.0;
    }
}

#ifdef LFL_FFMPEG
FFMPEGVideoResampler::~FFMPEGVideoResampler() { if (conv) sws_freeContext((SwsContext*)conv); }
bool FFMPEGVideoResampler::Opened() { return conv || simple_resampler_passthru; }

void FFMPEGVideoResampler::Open(int sw, int sh, int sf, int dw, int dh, int df) {
    s_fmt = sf; s_width = sw; s_height = sh;
    d_fmt = df; d_width = dw; d_height = dh;
    // INFO("resample ", BlankNull(Pixel::Name(s_fmt)), " -> ", BlankNull(Pixel::Name(d_fmt)), " : (", sw, ",", sh, ") -> (", dw, ",", dh, ")");

    if (SimpleVideoResampler::Supports(s_fmt) && SimpleVideoResampler::Supports(d_fmt) && sw == dw && sh == dh)
    { simple_resampler_passthru = 1; return; }

    conv = sws_getContext(sw, sh, (PixelFormat)Pixel::ToFFMpegId(sf),
                          dw, dh, (PixelFormat)Pixel::ToFFMpegId(df), SWS_BICUBIC, 0, 0, 0);
}

void FFMPEGVideoResampler::Resample(const unsigned char *s, int sls, unsigned char *d, int dls, bool flip_x, bool flip_y) {
    if (simple_resampler_passthru) return SimpleVideoResampler::Resample(s, sls, d, dls, flip_x, flip_y);
    uint8_t *source  [4] = { (uint8_t*)s, 0, 0, 0 }, *dest[  4] = { (uint8_t*)d, 0, 0, 0 };
    int      sourcels[4] = {         sls, 0, 0, 0 },  destls[4] = {         dls, 0, 0, 0 };
    if (flip_y) {
        source[0] += sls * (s_height - 1);
        sourcels[0] *= -1;
    }
    sws_scale((SwsContext*)conv,
        flip_y ? source   : source,
        flip_y ? sourcels : sourcels, 0, s_height, dest, destls);
}

int Pixel::FromFFMpegId(int fmt) {
    switch (fmt) {
        case AV_PIX_FMT_RGB32:    return Pixel::RGB32;
        case AV_PIX_FMT_BGR32:    return Pixel::BGR32;
        case AV_PIX_FMT_RGB24:    return Pixel::RGB24;
        case AV_PIX_FMT_BGR24:    return Pixel::BGR24;
        case AV_PIX_FMT_GRAY8:    return Pixel::GRAY8;
        case AV_PIX_FMT_YUV410P:  return Pixel::YUV410P;
        case AV_PIX_FMT_YUV420P:  return Pixel::YUV420P;
        case AV_PIX_FMT_YUYV422:  return Pixel::YUYV422;
        case AV_PIX_FMT_YUVJ420P: return Pixel::YUVJ420P;
        case AV_PIX_FMT_YUVJ422P: return Pixel::YUVJ422P;
        case AV_PIX_FMT_YUVJ444P: return Pixel::YUVJ444P;
        default: ERROR("unknown pixel fmt: ", fmt); return 0;
    }
}

int Pixel::ToFFMpegId(int fmt) {
    switch (fmt) {
        case Pixel::RGB32:    return AV_PIX_FMT_RGB32;
        case Pixel::BGR32:    return AV_PIX_FMT_BGR32;
        case Pixel::RGB24:    return AV_PIX_FMT_RGB24;
        case Pixel::BGR24:    return AV_PIX_FMT_BGR24;
        case Pixel::RGBA:     return AV_PIX_FMT_RGBA;
        case Pixel::GRAY8:    return AV_PIX_FMT_GRAY8;
        case Pixel::YUV410P:  return AV_PIX_FMT_YUV410P;
        case Pixel::YUV420P:  return AV_PIX_FMT_YUV420P;
        case Pixel::YUYV422:  return AV_PIX_FMT_YUYV422;
        case Pixel::YUVJ420P: return AV_PIX_FMT_YUVJ420P;
        case Pixel::YUVJ422P: return AV_PIX_FMT_YUVJ422P;
        case Pixel::YUVJ444P: return AV_PIX_FMT_YUVJ444P;
        default: ERROR("unknown pixel fmt: ", fmt); return 0;
    }
}
#endif /* LFL_FFMPEG */

/* Texture */

void Texture::Coordinates(float *texcoord, int w, int h, int wd, int hd) {
    texcoord[CoordMinX] = texcoord[CoordMinY] = 0;
    texcoord[CoordMaxX] = (float)w / wd;
    texcoord[CoordMaxY] = (float)h / hd;
}

void Texture::Resize(int W, int H, int PF, int flag) {
    if (PF) pf = PF;
    width=W; height=H;
    if (buf || (flag & Flag::CreateBuf)) RenewBuffer();
    if (!ID && (flag & Flag::CreateGL)) {
        if (!cubemap) {
            screen->gd->DisableCubeMap();
            screen->gd->GenTextures(GL_TEXTURE_2D, 1, &ID);
        } else if (cubemap == CubeMap::PX) {
            screen->gd->ActiveTexture(0);
            screen->gd->GenTextures(GL_TEXTURE_CUBE_MAP, 1, &ID);
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        }
    }
    int opengl_width = NextPowerOfTwo(width), opengl_height = NextPowerOfTwo(height);
    if (ID || cubemap) {
        int gl_tt = GLTexType(), gl_pt = GLPixelType();
        if (ID) screen->gd->BindTexture(gl_tt, ID);
        glTexImage2D(gl_tt, 0, gl_pt, opengl_width, opengl_height, 0, gl_pt, GL_UNSIGNED_BYTE, 0);
    }
    Coordinates(coord, width, height, opengl_width, opengl_height);
}

void Texture::LoadBuffer(const unsigned char *B, int W, int H, int PF, int linesize, int flag) {
    Resize(W, H, pf, Flag::CreateBuf);
    SimpleVideoResampler::Blit(B, buf, width, height,
                               PF, linesize,   0, 0,
                               pf, LineSize(), 0, 0, flag);
}

void Texture::UpdateBuffer(const unsigned char *B, int W, int H, int PF, int linesize, int flag) {
    bool resample = flag & Flag::Resample;
    VideoResampler conv;
    conv.Open(W, H, PF, resample?width:W, resample?height:H, pf);
    conv.Resample(B, linesize, buf, LineSize(), 0, flag & Flag::FlipY);
}

void Texture::UpdateBuffer(const unsigned char *B, int X, int Y, int W, int H, int PF, int linesize, int blit_flag) {
    SimpleVideoResampler::Blit(B, buf, W, H, PF, linesize, 0, 0, pf, LineSize(), X, Y, blit_flag);
}

void Texture::Bind() const { screen->gd->BindTexture(GLTexType(), ID); }
void Texture::ClearGL() { if (ID) screen->gd->DelTextures(1, &ID); ID=0; }

void Texture::LoadGL(const unsigned char *B, int W, int H, int PF, int linesize, int flag) {
    Texture temp;
    temp .Resize(W, H, Pixel::RGBA, Flag::CreateBuf);
    temp .UpdateBuffer(B, W, H, PF, linesize, Flag::FlipY);
    this->Resize(W, H, Pixel::RGBA, Flag::CreateGL);
    this->UpdateGL(temp.buf, 0, 0, W, H, flag);
}

void Texture::UpdateGL(const unsigned char *B, int X, int Y, int W, int H, int flag) {
    int gl_tt = GLTexType(), gl_y = (flag & Flag::FlipY) ? height-Y-H : Y;
    screen->gd->BindTexture(gl_tt, ID);
    glTexSubImage2D(gl_tt, 0, X, gl_y, W, H, GLPixelType(), GL_UNSIGNED_BYTE, B);
}

void Texture::ToIplImage(_IplImage *out) {
#ifdef LFL_OPENCV
    memset(out, 0, sizeof(IplImage));
    out->nSize = sizeof(IplImage);
    out->nChannels = Pixel::size(pf);
    out->depth = IPL_DEPTH_8U;
    out->origin = 1;
    out->width = width;
    out->height = height;
    out->widthStep = out->width * out->nChannels;
    out->imageSize = out->widthStep * out->height;
    out->imageData = (char*)buf;
    out->imageDataOrigin = out->imageData;
#else
    ERROR("ToIplImage not implemented");
#endif
}

void Texture::Screenshot() {
    Resize(screen->width, screen->height, Pixel::RGBA, Flag::CreateBuf);
    unsigned char *pixels = NewBuffer();
    glReadPixels(0, 0, screen->width, screen->height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    UpdateBuffer(pixels, screen->width, screen->height, Pixel::RGBA, screen->width*4, Flag::FlipY);
    delete [] pixels;
}

/* DepthTexture */

void DepthTexture::Resize(int W, int H, int DF, int flag) {
    if (DF) df = DF;
    width=W; height=H;
    if (!ID && (flag & Flag::CreateGL)) glGenRenderbuffers(1, &ID);
    int opengl_width = NextPowerOfTwo(width), opengl_height = NextPowerOfTwo(height);
    if (ID) {
        glBindRenderbuffer(GL_RENDERBUFFER, ID);
        glRenderbufferStorage(GL_RENDERBUFFER, Depth::OpenGLID(df), opengl_width, opengl_height);
    }
}

/* FrameBuffer */

void FrameBuffer::Resize(int W, int H, int flag) {
    width=W; height=H;
    if (!ID && (flag & Flag::CreateGL)) {
        glGenFramebuffers(1, &ID);
        if (flag & Flag::CreateTexture)      AllocTexture(&tex);
        if (flag & Flag::CreateDepthTexture) AllocDepthTexture(&depth);
    } else {
        tex.Resize(width, height);
        depth.Resize(width, height);
    }
    Attach(tex.ID, depth.ID);
    int status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) ERROR("FrameBuffer status ", status);
    if (flag & Flag::ReleaseFB) Release();
}

void FrameBuffer::AllocDepthTexture(DepthTexture *out) { CHECK_EQ(out->ID, 0); out->Create(width, height); }
void FrameBuffer::AllocTexture     (     Texture *out) { CHECK_EQ(out->ID, 0); out->Create(width, height); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
}

void FrameBuffer::Release() { glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); }
void FrameBuffer::Attach(int ct, int dt) {
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, ID);
    if (ct) {
        tex.ID = ct;
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex.ID, 0);
    }
    if (dt) {
        depth.ID = dt;
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth.ID);
    }
}

void FrameBuffer::Render(FrameCB cb) {
    Attach();
    screen->gd->ViewPort(Box(0, 0, tex.width, tex.height));
    screen->gd->Clear();
    cb(0, 0, 0, 0, 0);
    Release();
    screen->gd->RestoreViewport(DrawMode::_3D);
}

/* Shader */

void Shader::SetGlobalUniform1f(const string &name, float v) {
    screen->gd->UseShader(&app->video.shader_default);  app->video.shader_default .SetUniform1f(name, v);
    screen->gd->UseShader(&app->video.shader_normals);  app->video.shader_normals .SetUniform1f(name, v);
    screen->gd->UseShader(&app->video.shader_cubemap);  app->video.shader_cubemap .SetUniform1f(name, v);
    screen->gd->UseShader(&app->video.shader_cubenorm); app->video.shader_cubenorm.SetUniform1f(name, v);
}

void Shader::SetGlobalUniform2f(const string &name, float v1, float v2){ 
    screen->gd->UseShader(&app->video.shader_default);  app->video.shader_default .SetUniform2f(name, v1, v2);
    screen->gd->UseShader(&app->video.shader_normals);  app->video.shader_normals .SetUniform2f(name, v1, v2);
    screen->gd->UseShader(&app->video.shader_cubemap);  app->video.shader_cubemap .SetUniform2f(name, v1, v2);
    screen->gd->UseShader(&app->video.shader_cubenorm); app->video.shader_cubenorm.SetUniform2f(name, v1, v2);
}

#ifdef LFL_GLSL_SHADERS
int Shader::Create(const string &name, const string &vertex_shader, const string &fragment_shader, const string &defines, Shader *out) {
    GLuint p = screen->gd->CreateProgram();

    string hdr; 
#ifdef LFL_GLES2
    if (screen->opengles_version == 2) hdr += "#define LFL_GLES2\r\n";
#endif
    hdr += defines + string("\r\n");

    if (vertex_shader.size()) {
        GLuint vs = screen->gd->CreateShader(GL_VERTEX_SHADER);
        const char *vss[] = { hdr.c_str(), vertex_shader.c_str(), 0 };
        screen->gd->ShaderSource(vs, 2, vss, 0);
        screen->gd->CompileShader(vs);
        screen->gd->AttachShader(p, vs);
    }

    if (fragment_shader.size()) {
        GLuint fs = screen->gd->CreateShader(GL_FRAGMENT_SHADER);
        const char *fss[] = { hdr.c_str(), fragment_shader.c_str(), 0 };
        screen->gd->ShaderSource(fs, 2, fss, 0);
        screen->gd->CompileShader(fs);
        screen->gd->AttachShader(p, fs);
    }

    screen->gd->BindAttribLocation(p, 0, "Position"   );
    screen->gd->BindAttribLocation(p, 1, "Normal"     );
    screen->gd->BindAttribLocation(p, 2, "VertexColor");
    screen->gd->BindAttribLocation(p, 3, "TexCoordIn" );

    screen->gd->LinkProgram(p);

    int active_uniforms=0, max_uniform_components=0, active_attributes=0, max_attributes=0;
    screen->gd->GetProgramiv(p, GL_ACTIVE_UNIFORMS, &active_uniforms);
    screen->gd->GetProgramiv(p, GL_ACTIVE_ATTRIBUTES, &active_attributes);
#if !defined(LFL_ANDROID) && !defined(LFL_IPHONE)
    screen->gd->GetIntegerv(GL_MAX_VERTEX_UNIFORM_COMPONENTS, &max_uniform_components);
#endif
    screen->gd->GetIntegerv(GL_MAX_VERTEX_ATTRIBS, &max_attributes);
    INFO("shader mu=", active_uniforms, " avg_comps/", max_uniform_components, ", ma=", active_attributes, "/", max_attributes);

    bool log_missing_attrib = false;
    if (out) {
        *out = Shader();
        out->ID = p;
        out->name = name;
        if ((out->slot_position             = screen->gd->GetAttribLocation (p, "Position"))            < 0 && log_missing_attrib) INFO("shader ", name, " missing Position");
        if ((out->slot_normal               = screen->gd->GetAttribLocation (p, "Normal"))              < 0 && log_missing_attrib) INFO("shader ", name, " missing Normal");
        if ((out->slot_color                = screen->gd->GetAttribLocation (p, "VertexColor"))         < 0 && log_missing_attrib) INFO("shader ", name, " missing VertexColor");
        if ((out->slot_tex                  = screen->gd->GetAttribLocation (p, "TexCoordIn"))          < 0 && log_missing_attrib) INFO("shader ", name, " missing TexCoordIn");
        if ((out->uniform_modelview         = screen->gd->GetUniformLocation(p, "Modelview"))           < 0 && log_missing_attrib) INFO("shader ", name, " missing Modelview");
        if ((out->uniform_modelviewproj     = screen->gd->GetUniformLocation(p, "ModelviewProjection")) < 0 && log_missing_attrib) INFO("shader ", name, " missing ModelviewProjection");
        if ((out->uniform_tex               = screen->gd->GetUniformLocation(p, "Texture"))             < 0 && log_missing_attrib) INFO("shader ", name, " missing Texture");
        if ((out->uniform_cubetex           = screen->gd->GetUniformLocation(p, "CubeTexture"))         < 0 && log_missing_attrib) INFO("shader ", name, " missing CubeTexture");
        if ((out->uniform_normalon          = screen->gd->GetUniformLocation(p, "NormalEnabled"))       < 0 && log_missing_attrib) INFO("shader ", name, " missing NormalEnabled");
        if ((out->uniform_texon             = screen->gd->GetUniformLocation(p, "TexCoordEnabled"))     < 0 && log_missing_attrib) INFO("shader ", name, " missing TexCoordEnabled");
        if ((out->uniform_coloron           = screen->gd->GetUniformLocation(p, "VertexColorEnabled"))  < 0 && log_missing_attrib) INFO("shader ", name, " missing VertexColorEnabled");
        if ((out->uniform_cubeon            = screen->gd->GetUniformLocation(p, "CubeMapEnabled"))      < 0 && log_missing_attrib) INFO("shader ", name, " missing CubeMapEnabled");
        if ((out->uniform_colordefault      = screen->gd->GetUniformLocation(p, "DefaultColor"))        < 0 && log_missing_attrib) INFO("shader ", name, " missing DefaultColor");
        if ((out->uniform_material_ambient  = screen->gd->GetUniformLocation(p, "MaterialAmbient"))     < 0 && log_missing_attrib) INFO("shader ", name, " missing MaterialAmbient");
        if ((out->uniform_material_diffuse  = screen->gd->GetUniformLocation(p, "MaterialDiffuse"))     < 0 && log_missing_attrib) INFO("shader ", name, " missing MaterialDiffuse");
        if ((out->uniform_material_specular = screen->gd->GetUniformLocation(p, "MaterialSpecular"))    < 0 && log_missing_attrib) INFO("shader ", name, " missing MaterialSpecular");
        if ((out->uniform_material_emission = screen->gd->GetUniformLocation(p, "MaterialEmission"))    < 0 && log_missing_attrib) INFO("shader ", name, " missing MaterialEmission");
        if ((out->uniform_light0_pos        = screen->gd->GetUniformLocation(p, "LightZeroPosition"))   < 0 && log_missing_attrib) INFO("shader ", name, " missing LightZeroPosition");
        if ((out->uniform_light0_ambient    = screen->gd->GetUniformLocation(p, "LightZeroAmbient"))    < 0 && log_missing_attrib) INFO("shader ", name, " missing LightZeroAmbient");
        if ((out->uniform_light0_diffuse    = screen->gd->GetUniformLocation(p, "LightZeroDiffuse"))    < 0 && log_missing_attrib) INFO("shader ", name, " missing LightZeroDiffuse");
        if ((out->uniform_light0_specular   = screen->gd->GetUniformLocation(p, "LightZeroSpecular"))   < 0 && log_missing_attrib) INFO("shader ", name, " missing LightZeroSpecular");

        int unused_attrib = 0;
        memset(out->unused_attrib_slot, -1, sizeof(out->unused_attrib_slot));
        for (int i=0; i<MaxVertexAttrib; i++) {
            if (out->slot_position == i || out->slot_normal == i || out->slot_color == i || out->slot_tex == i) continue;
            out->unused_attrib_slot[unused_attrib++] = i;
        }
    }

    return p;
}

int Shader::GetUniformIndex(const string &name) { return screen->gd->GetUniformLocation(ID, name); }
void Shader::SetUniform1i(const string &name, float v) { screen->gd->Uniform1i(GetUniformIndex(name), v); }
void Shader::SetUniform1f(const string &name, float v) { screen->gd->Uniform1f(GetUniformIndex(name), v); }
void Shader::SetUniform2f(const string &name, float v1, float v2) { screen->gd->Uniform2f(GetUniformIndex(name), v1, v2); }
void Shader::SetUniform3fv(const string &name, const float *v) { screen->gd->Uniform3fv(GetUniformIndex(name), 1, v); }
void Shader::SetUniform3fv(const string &name, int n, const float *v) { screen->gd->Uniform3fv(GetUniformIndex(name), n, v); }

#else /* LFL_GLSL_SHADERS */

int Shader::Create(const string &vertex_shader, const string &fragment_shader, const string &defines, Shader *out) { return -1; }
int Shader::GetUniformIndex(const string &name) { return -1; }
void Shader::SetUniform1i(const string &name, float v) {}
void Shader::SetUniform1f(const string &name, float v) {}
void Shader::SetUniform2f(const string &name, float v1, float v2) {}
void Shader::SetUniform3fv(const string &name, const float *v) {}
void Shader::SetUniform3fv(const string &name, int n, const float *v) {}
void Shader::ActiveTexture(int n) {}
#endif /* LFL_GLSL_SHADERS */

/* Atlas */

bool Atlas::Add(int *x_out, int *y_out, float *texcoord, int w, int h, int max_height) {
    Box box;
    flow.SetMinimumAscent(max_height);
    flow.AppendBox(w, h, Border(2,2,2,2), &box);

    *x_out = box.x;
    *y_out = box.y + flow.container->top();
    if (*x_out < 0 || *x_out + w > tex.width ||
        *y_out < 0 || *y_out + h > tex.height) return false;

    texcoord[Texture::CoordMinX] =     (float)(*x_out    ) / tex.width;
    texcoord[Texture::CoordMinY] = 1 - (float)(*y_out + h) / tex.height;
    texcoord[Texture::CoordMaxX] =     (float)(*x_out + w) / tex.width;
    texcoord[Texture::CoordMaxY] = 1 - (float)(*y_out    ) / tex.height;
    return true;
}

void Atlas::Update(const string &name, Font *f, bool dump) {
    if (dump) {
        LocalFile lf(ASSETS_DIR + name + "00.png", "w");
        PngWriter::Write(&lf, tex);
        INFO("wrote ", lf.Filename());
        WriteGlyphFile(name, f);
    }
    if (1) { /* complete atlas */
        tex.LoadGL();
        GlyphTableIter(f) if (i->       tex.width) i->       tex.ID = tex.ID;
        GlyphIndexIter(f) if (i->second.tex.width) i->second.tex.ID = tex.ID;
    }
}

void Atlas::WriteGlyphFile(const string &name, Font *f) {
    int glyph_count = 0, glyph_out = 0;
    GlyphTableIter(f) if (i->       tex.width) glyph_count++;
    GlyphIndexIter(f) if (i->second.tex.width) glyph_count++;

    Matrix *gm = new Matrix(glyph_count, 10);
    GlyphTableIter(f) if (i->       tex.width) i->       ToArray(gm->row(glyph_out++), gm->N);
    GlyphIndexIter(f) if (i->second.tex.width) i->second.ToArray(gm->row(glyph_out++), gm->N);
    MatrixFile(gm, "").WriteVersioned(VersionedFileName(ASSETS_DIR, name.c_str(), "glyphs"), 0);
}

void Atlas::MakeFromPNGFiles(const string &name, const vector<string> &png, int atlas_dim, Font **glyphs_out) {
    Font *ret = new Font();
    ret->fg = Color(1.0,1.0,1.0,1.0);
    ret->glyph = shared_ptr<Font::Glyphs>(new Font::Glyphs(ret, new Atlas(0, atlas_dim)));
    EnsureSize(ret->glyph->table, png.size());

    Atlas *atlas = ret->glyph->atlas.back().get();
    atlas->tex.RenewBuffer();

    for (int i = 0, skipped = 0; i < png.size(); ++i) {
        LocalFile in(png[i], "r");
        if (!in.Opened()) { INFO("Skipped: ", png[i]); skipped++; continue; }
        Font::Glyph *out = &ret->glyph->table[i - skipped];
        out->id = i - skipped;

        if (PngReader::Read(&in, &out->tex)) { skipped++; continue; }
        Typed::Max(&ret->height, out->tex.height);

        int atlas_x, atlas_y;
        CHECK(atlas->Add(&atlas_x, &atlas_y, out->tex.coord, out->tex.width, out->tex.height, ret->height));
        SimpleVideoResampler::Blit(out->tex.buf, atlas->tex.buf, out->tex.width, out->tex.height,
                                   out->tex.pf,   out->tex.LineSize(),   0,       0,
                                   atlas->tex.pf, atlas->tex.LineSize(), atlas_x, atlas_y);
        out->tex.ClearBuffer();
    }

    atlas->Update(name, ret, true);
    atlas->tex.ClearBuffer();

    if (glyphs_out) *glyphs_out = ret;
    else delete ret;
}

void Atlas::SplitIntoPNGFiles(const string &input_png_fn, const map<int, v4> &glyphs, const string &dir_out) {
    LocalFile in(input_png_fn, "r");
    if (!in.Opened()) { ERROR("open: ", input_png_fn); return; }

    Texture png;
    if (PngReader::Read(&in, &png)) { ERROR("read: ", input_png_fn); return; }

    for (map<int, v4>::const_iterator i = glyphs.begin(); i != glyphs.end(); ++i) {
        unsigned gx1 = RoundF(i->second.x * png.width), gy1 = RoundF((1 - i->second.y) * png.height);
        unsigned gx2 = RoundF(i->second.z * png.width), gy2 = RoundF((1 - i->second.w) * png.height);
        unsigned gw = gx2 - gx1, gh = gy1 - gy2;
        CHECK(gw > 0 && gh > 0);

        Texture glyph;
        glyph.Resize(gw, gh, Pixel::RGBA, Texture::Flag::CreateBuf);
        SimpleVideoResampler::Blit(png.buf, glyph.buf, glyph.width, glyph.height,
                                   png  .pf, png  .LineSize(), gx1, gy2,
                                   glyph.pf, glyph.LineSize(), 0,   0);

        LocalFile lf(dir_out + StringPrintf("glyph%03d.png", i->first), "w");
        CHECK(lf.Opened());
        PngWriter::Write(&lf, glyph);
    }
}

/* BoxRun */

point BoxRun::Draw(point p, DrawCB cb) {
    Box w;
    DrawBackground(p);
    if (attr.tex) attr.tex->Bind();
    if (attr.tex || attr.font) screen->gd-> SetColor(attr.fg ? *attr.fg : Color::white);
    else                       screen->gd->FillColor(attr.fg ? *attr.fg : Color::white);
    if (attr.font) attr.font->Select();
    else if (attr.tex) screen->gd->EnableLayering();
    if (attr.scissor) screen->gd->PushScissor(*attr.scissor + p);
    for (auto i = data.buf, e = data.end(); i != e; ++i) if (i->drawable) cb(i->drawable, (w = i->box + p));
    if (attr.scissor) screen->gd->PopScissor();
    return point(w.x + w.w, w.y);
}

void BoxRun::DrawBackground(point p, DrawBackgroundCB cb) {
    if (attr.bg) screen->gd->FillColor(*attr.bg);
    if (!attr.bg) return;
    int line_height = line ? line->h : (attr.font ? attr.font->height : 0);
    if (!line_height) return;
    HorizontalExtentTracker extent;
    for (int i=0; i<data.size(); i++) extent.AddDrawableBox(data.data()[i]);
    cb(extent.Get(-line_height, line_height) + p);
}

/* Font */

void Font::Glyph::FromArray(const double *in, int l) {
    CHECK_GE(l, 10);
    id   = (int)in[0];        /* 0  = in[1]*/ tex.width    = (int)in[2];  tex.height   = (int)in[3]; top          = (int)in[4];
    left = (int)in[5]; tex.coord[0] = in[6];  tex.coord[1] =      in[7];  tex.coord[2] =      in[8]; tex.coord[3] =      in[9];
}

int Font::Glyph::ToArray(double *out, int l) {
    CHECK_GE(l, 10);
    out[0] = id;   out[1] = 0;            out[2] = tex.width;    out[3] = tex.height;   out[4] = top;
    out[5] = left; out[6] = tex.coord[0]; out[7] = tex.coord[1]; out[8] = tex.coord[2]; out[9] = tex.coord[3];
    return sizeof(float)*8;
}

Font *Font::Clone(int pointsize, Color Fg, int Flag) {
    Font *ret = new Font(*this);
    ret->fg = Fg;
    ret->flag = Flag;
    ret->Scale(pointsize);
    return ret;
}

Font *Font::OpenAtlas(const string &name, int size, Color c, int flag) {
    Texture tex;
    Asset::LoadTexture(StrCat(ASSETS_DIR, name, "00.png"), &tex);
    if (!tex.ID) { ERROR("load ", name, "00.png failed"); return 0; }

    MatrixFile gm;
    gm.ReadVersioned(VersionedFileName(ASSETS_DIR, name.c_str(), "glyphs"), 0);
    if (!gm.F) { ERROR("load ", name, ".0000.glyphs.matrix failed"); return 0; }

    Font *ret = new Font();
    ret->fg = c;
    ret->size = size;
    ret->flag = flag;
    ret->glyph = shared_ptr<Font::Glyphs>(new Font::Glyphs(ret, new Atlas(tex.ID, tex.width, tex.height)));
    Atlas *atlas = ret->glyph->atlas.back().get();
    float max_t = 0, max_u = 0;

    MatrixRowIter(gm.F) {
        int glyph_ind = (int)gm.F->row(i)[0];
        Glyph *g = ret->FindOrInsertGlyph(glyph_ind);
        g->FromArray(gm.F->row(i), gm.F->N);
        g->tex.ID = tex.ID;
        if (g->tex.height > ret->height)    ret->height    = g->tex.height;
        if (g->tex.width  > ret->max_width) ret->max_width = g->tex.width;
        if (g->top        > ret->max_top)   ret->max_top   = g->top;
        if (g->tex.width && !ret->fixed_width)                 ret->fixed_width = g->tex.width;
        if (g->tex.width &&  ret->fixed_width != g->tex.width) ret->fixed_width = 0;

        for (int j=0; j<4; j++) {
            max_t = max(max(max_t, g->tex.coord[0]), g->tex.coord[2]);
            max_u = max(max(max_u, g->tex.coord[1]), g->tex.coord[3]);
        }
    }
    atlas->flow.SetMinimumAscent(ret->height);
    atlas->flow.p.x =  max_t * atlas->tex.width;
    atlas->flow.p.y = -max_u * atlas->tex.height;

    INFO("OpenAtlas ", name, ", texID=", tex.ID);
    tex.ID = 0;
    return ret;
}

Font::Glyph *Font::FindOrInsertGlyph(unsigned gind) {
    return gind < glyph->table.size() ? &glyph->table[gind] : &glyph->index[gind];
}

Font::Glyph *Font::FindGlyph(unsigned gind) {
    if (gind < glyph->table.size()) return &glyph->table[gind];
    map<int, Glyph>::iterator i = glyph->index.find(gind);
    return i != glyph->index.end() ? &i->second : glyph->primary->LoadGlyph(gind);
}

void Font::Select() {
    screen->gd->EnableLayering();
    if (mix_fg) screen->gd->SetColor(fg);
}

void Font::Scale(int new_size) {
    Font *primary = glyph->primary;
    CHECK_NE(primary, this);

    size        = new_size;
    scale       = (float)size / primary->size;
    height      = RoundF(primary->height      * scale);
    max_top     = RoundF(primary->max_top     * scale);
    max_width   = RoundF(primary->max_width   * scale);
    fixed_width = RoundF(primary->fixed_width * scale);
}

template <class X> void Font::Size(const StringPieceT<X> &text, Box *out, int maxwidth, int *lines_out) {
    vector<Box> line_box;
    int lines = Draw(text, Box(0,0,maxwidth,0), &line_box, Flag::Clipped);
    if (lines_out) *lines_out = lines;
    *out = Box(0, 0, 0, lines * height);
    for (int i=0; i<line_box.size(); i++) out->w = max(out->w, line_box[i].w);
}

template <class X> void Font::Encode(const StringPieceT<X> &text, const Box &box, BoxArray *out, int draw_flag, int attr_id) {
    Flow flow(&box, this, out);
    if (draw_flag & Flag::AssignFlowX) flow.p.x = box.x;
    flow.layout.wrap_lines   = !(draw_flag & Flag::NoWrap) && box.w;
    flow.layout.word_break   = !(draw_flag & Flag::GlyphBreak);
    flow.layout.align_center =  (draw_flag & Flag::AlignCenter);
    flow.layout.align_right  =  (draw_flag & Flag::AlignRight);
    if (!attr_id) {
        flow.cur_attr.underline  =  (draw_flag & Flag::Underline);
        flow.cur_attr.overline   =  (draw_flag & Flag::Overline);
        flow.cur_attr.midline    =  (draw_flag & Flag::Midline);
        flow.cur_attr.blink      =  (draw_flag & Flag::Blink);
    }
    if      (draw_flag & Flag::Uppercase)  flow.layout.char_tf = ::toupper;
    else if (draw_flag & Flag::Lowercase)  flow.layout.char_tf = ::tolower;
    if      (draw_flag & Flag::Capitalize) flow.layout.word_start_char_tf = ::toupper;
    flow.AppendText(text, attr_id);
    flow.Complete();
}

template <class X> int Font::Draw(const StringPieceT<X> &text, const Box &box, vector<Box> *lb, int draw_flag) {
    BoxArray out;
    Encode(text, box, &out, draw_flag);
    if (lb) *lb = out.line;
    if (!(draw_flag & Flag::Clipped)) out.Draw(box.TopLeft());
    return out.line.size();
}

template void Font::Size  <char> (const StringPiece   &text, Box *out, int maxwidth, int *lines_out);
template void Font::Size  <short>(const String16Piece &text, Box *out, int maxwidth, int *lines_out);
template void Font::Encode<char> (const StringPiece   &text, const Box &box, BoxArray *out, int draw_flag, int attr_id);
template void Font::Encode<short>(const String16Piece &text, const Box &box, BoxArray *out, int draw_flag, int attr_id);
template int  Font::Draw  <char> (const StringPiece   &text, const Box &box, vector<Box> *lb, int draw_flag);
template int  Font::Draw  <short>(const String16Piece &text, const Box &box, vector<Box> *lb, int draw_flag);

/* FreeType */

#ifdef LFL_FREETYPE
TTFFont::Resource::~Resource() { if (face) FT_Done_Face(face); }

void TTFFont::Init() {
    static bool init = false;
    if (init) return;
    init = true;
    int error;
    if ((error = FT_Init_FreeType(&ft_library))) ERROR("FT_Init_FreeType: ", error);
}

Font *TTFFont::Clone(int pointsize, Color Fg, int Flag) {
    if (!FLAGS_font_dont_reopen) return Open(resource, pointsize, Fg, Flag);
    Font *ret = new TTFFont(*this);
    ret->fg = Fg;
    ret->flag = Flag;
    ret->Scale(pointsize);
    return ret;
}

Font *TTFFont::OpenFile(const string &fn, const string &name, int size, Color c, int flag, int ttf_flag) {
    Init();
    FT_FaceRec_ *face = 0; int error;
    if ((error = FT_New_Face(ft_library, fn.c_str(), 0, &face))) { ERROR("FT_New_Face: ",       error); return 0; }
    if ((error = FT_Select_Charmap(face, FT_ENCODING_UNICODE)))  { ERROR("FT_Select_Charmap: ", error); return 0; }
    FT_Library_SetLcdFilter(ft_library, FLAGS_subpixel_fonts ? FT_LCD_FILTER_LIGHT : FT_LCD_FILTER_NONE);
    shared_ptr<TTFFont::Resource> res(new Resource(face, name, ttf_flag));
    return Open(res, size, c, flag);
}

Font *TTFFont::OpenBuffer(const shared_ptr<TTFFont::Resource> &res, int size, Color c, int flag) {
    Init();
    int error;
    if ((error = FT_New_Memory_Face(ft_library, (const FT_Byte*)res->content.data(), res->content.size(), 0, &res->face))) { ERROR("FT_New_Memory_Face: ", error); return 0; }
    if ((error = FT_Select_Charmap(res->face, FT_ENCODING_UNICODE)))                                                       { ERROR("FT_Select_Charmap: ",  error); return 0; }
    FT_Library_SetLcdFilter(ft_library, FLAGS_subpixel_fonts ? FT_LCD_FILTER_LIGHT : FT_LCD_FILTER_NONE);
    return Open(res, size, c, flag);
}

static bool TTFFontLoadGlyph(FT_FaceRec_ *face, int glyph_index, Font *ret, Font::Glyph *out, bool subpixel) {
    int error; FT_Int32 flags = FT_LOAD_RENDER | (subpixel ? FT_LOAD_TARGET_LCD : 0);
    if ((error = FT_Load_Glyph(face, glyph_index, flags))) { ERROR("FT_Load_Glyph(", glyph_index, ") = ", error); return false; }
    if (( subpixel && face->glyph->bitmap.pixel_mode != FT_PIXEL_MODE_LCD) ||
        (!subpixel && face->glyph->bitmap.pixel_mode != FT_PIXEL_MODE_GRAY))
    { ERROR("glyph bitmap pixel_mode ", face->glyph->bitmap.pixel_mode); return false; }

    out->top = face->glyph->bitmap_top;
    out->left = face->glyph->bitmap_left;
    out->tex.width = face->glyph->advance.x/64;
    out->tex.height = face->glyph->bitmap.rows;
    if (out->tex.width && !ret->fixed_width)                   ret->fixed_width = out->tex.width;
    if (out->tex.width &&  ret->fixed_width != out->tex.width) ret->fixed_width = 0;

    return true;
}

static void TTFFontFilter(unsigned char *buf, int wd, int ht, int pf, int linesize, int x, int y) {
    if (FLAGS_subpixel_fonts) {
        Matrix kernel(3, 3, 1/8.0); 
        SimpleVideoResampler::Filter(buf, wd, ht, pf, linesize, x, y, &kernel, ColorChannel::Alpha, SimpleVideoResampler::Flag::ZeroOnly);
    }
}

Font::Glyph *TTFFont::LoadGlyph(unsigned glyph_id) {
    Atlas *atlas_cur = glyph->atlas.back().get();
    Glyph *out = &glyph->index[glyph_id]; 
    if (out->id) { ERROR((void*)this, " glyph ", glyph_id, " already loaded"); return out; }
    *out = *Font::LoadGlyph(glyph_id);
    out->id = glyph_id;

    FT_FaceRec_ *face = resource->face;
    int error, glyph_index = FT_Get_Char_Index(face, glyph_id);
    if (!glyph_index) { INFOf("missing U+%06x", glyph_id); return out; }
    if ((error = FT_Set_Pixel_Sizes(face, 0, size))) { ERROR("FT_Set_Pixel_Sizes(", size, ") = ", error); return out; }

    if (!TTFFontLoadGlyph(face, glyph_index, this, out, FLAGS_subpixel_fonts)) return out;
    int ht = face->glyph->bitmap.rows, wd = face->glyph->bitmap.width / (FLAGS_subpixel_fonts ? 3 : 1);
    int spf = FLAGS_subpixel_fonts ? Pixel::LCD : Pixel::GRAY8, atlas_x, atlas_y;
    if (!atlas_cur->Add(&atlas_x, &atlas_y, out->tex.coord, out->tex.width, out->tex.height, height)) FATAL("atlas full");
    out->tex.ID = atlas_cur->tex.ID;

    Texture glyph(0, 0, atlas_cur->tex.pf);
    glyph.LoadBuffer(face->glyph->bitmap.buffer, wd, ht, spf, face->glyph->bitmap.pitch,
                     SimpleVideoResampler::Flag::TransparentBlack | SimpleVideoResampler::Flag::FlipY);
    TTFFontFilter(glyph.buf, wd, ht, glyph.pf, glyph.LineSize(), 0, 0);

    // PngWriter::Write(StringPrintf("glyph%06x.png", glyph_id), glyph);
    atlas_cur->tex.UpdateGL(glyph.buf, atlas_x + out->left, atlas_y, wd, ht, Texture::Flag::FlipY); 
    return out;
}

Font *TTFFont::Open(const shared_ptr<TTFFont::Resource> &resource, int size, Color c, int flag) {
    FT_FaceRec_ *face = resource->face; int count = 0, error;
    bool fixed_width = FT_IS_FIXED_WIDTH(face), write_atlas = resource->flag & Flag::WriteAtlas, outline = resource->flag & Flag::Outline;
    if ((error = FT_Set_Pixel_Sizes(face, 0, size))) { ERROR("FT_Set_Pixel_Sizes(", size, ") = ", error); return 0; }

    TTFFont *ret = 0; Atlas *atlas = 0;
    if (!ret) {
        ret = new TTFFont(resource, size, c, flag);
        ret->glyph = shared_ptr<Font::Glyphs>(new Font::Glyphs(ret));
        EnsureSize(ret->glyph->table, FLAGS_glyph_table_size);
        for (int i=0; i<FLAGS_glyph_table_size; i++)
            if ((ret->glyph->table[i].id = FT_Get_Char_Index(face, i))) count++;
#if 0
        FT_UInt gindex;
        for (FT_ULong charcode = FT_Get_First_Char(face,           &gindex); gindex;
             /**/     charcode = FT_Get_Next_Char (face, charcode, &gindex)) {
            INFOf("U+%06x index %d", charcode, gindex);
        }                                                                
#endif
        // determine maximum glyph dimensions
        int max_advance=0, max_top=0, max_bottom=0;
        FT_Int32 ttf_flags = FT_LOAD_RENDER | (FLAGS_subpixel_fonts ? FT_LOAD_TARGET_LCD : 0);
#if 0
        for (int glyph_index=0; glyph_index<face->num_glyphs; glyph_index++) {
#else
        for (int i=0, glyph_index; i<ret->glyph->table.size(); i++) {
            if (!(glyph_index = ret->glyph->table[i].id)) continue;
#endif
            if ((error = FT_Load_Glyph(face, glyph_index, ttf_flags))) { ERROR("FT_Load_Glyph(", glyph_index, ") = ", error); delete ret; continue; }
            int bottom = face->glyph->bitmap.rows - face->glyph->bitmap_top, advance = face->glyph->advance.x/64;
            if (bottom                   > max_bottom)  max_bottom  = bottom;
            if (advance                  > max_advance) max_advance = advance;
            if (face->glyph->bitmap_top  > max_top)     max_top     = face->glyph->bitmap_top;
        }

        ret->max_top = max_top + (FLAGS_atlas_pad_top ? RoundF(size * FLAGS_atlas_pad_top) : 0);
        ret->height = ret->max_top + max_bottom;
        ret->max_width = max_advance;
        if (ret->mono) ret->fixed_width = ret->max_width;
        ret->glyph->atlas.push_back(shared_ptr<Atlas>(new Atlas(0, Atlas::Dimension(ret->max_width, ret->height, count))));
        atlas = ret->glyph->atlas.back().get();
        atlas->tex.RenewBuffer();
    }

    // fill atlas
    for (int glyph_index, i=0; i<ret->glyph->table.size(); i++) {
        if (!(glyph_index = ret->glyph->table[i].id)) continue;
        Glyph *out = &ret->glyph->table[i];
        out->id = i;

        if (!TTFFontLoadGlyph(face, glyph_index, ret, out, FLAGS_subpixel_fonts)) continue;
        int ht = face->glyph->bitmap.rows, wd = face->glyph->bitmap.width / (FLAGS_subpixel_fonts ? 3 : 1);
        int spf = FLAGS_subpixel_fonts ? Pixel::LCD : Pixel::GRAY8, atlas_x, atlas_y;
        CHECK(atlas->Add(&atlas_x, &atlas_y, out->tex.coord, out->tex.width, out->tex.height, ret->height));
        atlas->tex.UpdateBuffer(face->glyph->bitmap.buffer, atlas_x + out->left, atlas_y,
                                wd, ht, spf, face->glyph->bitmap.pitch, SimpleVideoResampler::Flag::TransparentBlack);
        TTFFontFilter(atlas->tex.buf, wd, ht, atlas->tex.pf, atlas->tex.LineSize(), atlas_x + out->left, atlas_y);
    }

    atlas->Update(resource->name, ret, write_atlas);
    atlas->tex.ClearBuffer();
    INFO("TTTFont(", SpellNull(face->family_name), "), FW=", fixed_width, ", texID=", atlas->tex.ID);
    return ret;
}
#else /* LFL_FREETYPE */
TTFFont::Resource::~Resource() {}
Font::Glyph *TTFFont::LoadGlyph(unsigned gind) { return 0; }
void  TTFFont::Init() {}
Font *TTFFont::Clone(int pointsize, Color fg, int flag) { return 0; }
Font *TTFFont::OpenFile(const string &filename, const string &name, int size, Color c, int flag, int ttf_flag) { return 0; }
Font *TTFFont::OpenBuffer(const shared_ptr<TTFFont::Resource> &res, int size, Color c, int flag) { return 0; }
Font *TTFFont::Open      (const shared_ptr<TTFFont::Resource> &res, int size, Color c, int flag) { return 0; }
#endif /* LFL_FREETYPE */

#ifdef __APPLE__
CFStringRef ToCFStr(const string &n) { return CFStringCreateWithCString(0, n.c_str(), kCFStringEncodingUTF8); }
string FromCFStr(CFStringRef in) {
    string ret(CFStringGetMaximumSizeForEncoding(CFStringGetLength(in), kCFStringEncodingUTF8), 0);
    if (!CFStringGetCString(in, (char*)ret.data(), ret.size(), kCFStringEncodingUTF8)) return string();
    for (int i=0; i<ret.size(); i++) if (!ret[i]) { ret.resize(i); break; }
    return ret;
}

CGContextRef Texture::CGBitMap() { return CGBitMap(0, 0, width, height); }
CGContextRef Texture::CGBitMap(int X, int Y, int W, int H) {
    int linesize = LineSize(); CGImageAlphaInfo alpha_info;
    if      (pf == Pixel::RGBA)                        alpha_info = kCGImageAlphaPremultipliedLast;
    else if (pf == Pixel::RGB32 || pf == Pixel::BGR32) alpha_info = kCGImageAlphaNoneSkipLast;
    else { ERROR("unsupported pixel format: ", Pixel::Name(pf)); return 0; }
    CGColorSpaceRef colors = CGColorSpaceCreateDeviceRGB();
    CGContextRef ret = CGBitmapContextCreate(buf + Y*linesize + X*PixelSize(), W, H, 8, linesize, colors, alpha_info);
    CGColorSpaceRelease(colors);
    return ret;
}

int NSFMreadfonts(void *Iter, void (*IterAdd)(void *iter, const char *name, const char *family, int flag));
void FontsIterAdd(void *iter, const char *name, const char *family, int flag) { INFO("font ", name); }
void CoreTextFont::Init() { if (0) NSFMreadfonts(0, FontsIterAdd); }

Font *CoreTextFont::Clone(int pointsize, Color Fg, int Flag) {
    if (!FLAGS_font_dont_reopen) return Open(resource, pointsize, Fg, Flag);
    Font *ret = new CoreTextFont(*this);
    ret->fg = Fg;
    ret->flag = Flag;
    ret->Scale(pointsize);
    return ret;
}

Font *CoreTextFont::Open(const string &name, int size, Color c, int flag, int ct_flag) {
    CFStringRef cfsname = ToCFStr(name);
    CGFontRef cgfont = CGFontCreateWithFontName(cfsname);
    CFRelease(cfsname);
    if (!cgfont) return 0;
    shared_ptr<Resource> res(new Resource(name.c_str(), cgfont, ct_flag));
    return Open(res, size, c, flag);
}

static bool CoreTextFontLoadGlyph(CGRect *bounds, Font *ret, Font::Glyph *out, bool subpixel) {
    out->left       = -RoundF(bounds->origin.x);
    out->tex.width  =  RoundF(bounds->size.width);
    out->tex.height =  RoundF(bounds->size.height);
    out->top        =  RoundF(bounds->origin.y) + out->tex.height;
    if (out->tex.width && !ret->fixed_width)                   ret->fixed_width = out->tex.width;
    if (out->tex.width &&  ret->fixed_width != out->tex.width) ret->fixed_width = 0;
    return true;
}

Font::Glyph *CoreTextFont::LoadGlyph(unsigned glyph_id) {
    Atlas *atlas_cur = glyph->atlas.back().get();
    Glyph *out = &glyph->index[glyph_id]; 
    if (out->id) { ERROR((void*)this, " glyph ", glyph_id, " already loaded"); return out; }
    *out = *Font::LoadGlyph(glyph_id);
    out->id = glyph_id;
    out->tex.ID = atlas_cur->tex.ID;

    CGGlyph cgglyph; CGRect bounds; UniChar uc = glyph_id;
    CTFontRef ctfont = CTFontCreateWithGraphicsFont(resource->cgfont, size, 0, 0);
    CTFontGetGlyphsForCharacters(ctfont, &uc, &cgglyph, 1);
    CTFontGetBoundingRectsForGlyphs(ctfont, kCTFontDefaultOrientation, &cgglyph, &bounds, 1);
    CoreTextFontLoadGlyph(&bounds, this, out, FLAGS_subpixel_fonts);
    CFRelease(ctfont);

    Texture glyph(out->tex.width, out->tex.height, atlas_cur->tex.pf);
    glyph.RenewBuffer();
    CGContextRef context = glyph.CGBitMap();
    CGContextSetRGBFillColor(context, 1, 1, 1, 1);
    CGContextSetFont(context, resource->cgfont);
    CGContextSetFontSize(context, size);
    CGPoint point = CGPointMake(0, 0);
    CGContextShowGlyphsAtPositions(context, &cgglyph, &point, 1);
    CGContextRelease(context);

    int atlas_x=0, atlas_y=0;
    if (!atlas_cur->Add(&atlas_x, &atlas_y, out->tex.coord, out->tex.width, out->tex.height, height)) FATAL("atlas full");
    atlas_cur->tex.UpdateGL(glyph.buf, atlas_x + out->left, atlas_y, out->tex.width, out->tex.height, Texture::Flag::FlipY); 
    INFOf("LoadGlyph U+%06x texID=%d", glyph_id, atlas_cur->tex.ID);
    return out;
}

Font *CoreTextFont::Open(const shared_ptr<Resource> &resource, int size, Color c, int flag) {
    CoreTextFont *ret = 0; Atlas *atlas = 0;
    if (!ret) {
        ret = new CoreTextFont(resource, size, c, flag);
        ret->glyph = shared_ptr<Font::Glyphs>(new Font::Glyphs(ret));

        static const int ascii_glyph_start = 32, ascii_glyph_end = 127, ascii_glyphs = ascii_glyph_end - ascii_glyph_start;
        CHECK_LE(ascii_glyph_end, ret->glyph->table.size());
        vector<UniChar> ascii(ascii_glyphs);
        for (int i=0; i<ascii.size(); i++) ascii[i] = ascii_glyph_start + i;

        vector<CGGlyph> glyphs(ascii.size());
        vector<CGRect> bounds(glyphs.size());
        CTFontRef ctfont = CTFontCreateWithGraphicsFont(resource->cgfont, size, 0, 0);
        CTFontGetGlyphsForCharacters(ctfont, &ascii[0], &glyphs[0], glyphs.size());
        CTFontGetBoundingRectsForGlyphs(ctfont, kCTFontDefaultOrientation, &glyphs[0], &bounds[0], glyphs.size());
        CFRelease(ctfont);

        int max_bottom = 0;
        for (int i=0; i<glyphs.size(); i++) {
            int ind = ascii_glyph_start + i;
            Glyph *g = &ret->glyph->table[ind];
            g->id = glyphs[i];

            CoreTextFontLoadGlyph(&bounds[i], ret, g, FLAGS_subpixel_fonts);
            Typed::Max(&ret->max_top,   g->top);
            Typed::Max(&ret->max_width, g->tex.width);
            Typed::Max(&max_bottom,     g->tex.height - g->top);
        }
        ret->height = ret->max_top + max_bottom;
        if (ret->mono) ret->fixed_width = ret->max_width;
        ret->glyph->atlas.push_back(shared_ptr<Atlas>(new Atlas(0, Atlas::Dimension(ret->max_width, ret->height, ascii_glyphs))));
        atlas = ret->glyph->atlas.back().get();
        atlas->tex.RenewBuffer();
    }

    CGContextRef context = atlas->tex.CGBitMap();
    CGContextSetRGBFillColor(context, 1, 1, 1, 1);
    CGContextSetFont(context, resource->cgfont);
    CGContextSetFontSize(context, size);
    for (CGGlyph glyph_index, i=0; i<ret->glyph->table.size(); i++) {
        if (!(glyph_index = ret->glyph->table[i].id)) continue;
        Glyph *out = &ret->glyph->table[i];
        out->id = i;

        int atlas_x, atlas_y;
        CHECK(atlas->Add(&atlas_x, &atlas_y, out->tex.coord, out->tex.width, out->tex.height, ret->height));
        CGPoint point = CGPointMake(atlas_x + out->left, atlas->tex.height - atlas_y - out->top);
        CGContextShowGlyphsAtPositions(context, &glyph_index, &point, 1);
    }
    CGContextRelease(context);

    atlas->Update(resource->name, ret, resource->flag & Flag::WriteAtlas);
    atlas->tex.ClearBuffer();

    CFStringRef font_name = CGFontCopyFullName(resource->cgfont);
    INFO("CoreTextFont(", FromCFStr(font_name), "), texID=", atlas->tex.ID);
    CFRelease(font_name);
    return ret;
}
#endif /* __APPLE__ */

/* Fonts */

string Fonts::FontName(const string &filename, int pointsize, Color fg, int flag) {
    return StringPrintf("%s,%d,%d,%d,%d,%d", filename.c_str(), pointsize, fg.R(), fg.G(), fg.B(), flag);
}

unsigned Fonts::FontColor(Color fg, int flag) {
    unsigned char r = (unsigned char)fg.R(), g = (unsigned char)fg.G(), b = (unsigned char)fg.B(), f = (unsigned char)flag;
    return (r<<24) | (g<<16) | (b<<8) | f;
}

int Fonts::ScaledFontSize(int pointsize) {
    if (FLAGS_scale_font_height) {
        float ratio = (float)screen->height / FLAGS_scale_font_height;
        pointsize = (int)(pointsize * ratio);
    }
    return pointsize + FLAGS_add_font_size;
}

Font *Fonts::Fake() { return Singleton<FakeFont>::Get(); }
Font *Fonts::Default() {
    static Font *default_font = 0;
    if (!default_font) default_font = Get(FLAGS_default_font, FLAGS_default_font_size, Color::white);
    return default_font;
}

Font *Fonts::Insert(Font *font, const string &filename, const string &family, int pointsize, const Color &fg, int flag) {
    if (!font) return 0;
    Fonts *inst = Singleton<Fonts>::Get();
    FontSizeMap *m = &inst->font_map[filename][FontColor(fg, flag)];
    Font *old_font = FindOrNull(*m, pointsize);
    if (old_font) { ERROR("deleting duplicate font: ", (void*)old_font); delete old_font; }
    (*m)[pointsize] = font;

    if (!family.empty()) {
        Family *fam = &inst->family_map[family];
        bool bold = flag & FontDesc::Bold, italic = flag & FontDesc::Italic;
        if (bold && italic) fam->bold_italic.insert(filename);
        else if (bold)      fam->bold       .insert(filename);
        else if (italic)    fam->italic     .insert(filename);
        else                fam->normal     .insert(filename);
    }
    return font;
}

Font *Fonts::InsertAtlas(const string &filename, const string &family, int pointsize, const Color &fg, int flag) {
    string name = FontName(filename, pointsize, fg, flag);
    return Insert(Font::OpenAtlas(name, pointsize, fg, flag), filename, family, pointsize, fg, flag);
}

Font *Fonts::InsertFreetype(const string &filename, const string &family, int pointsize, const Color &fg, int flag) {
    TTFFont::Resource *resource = new TTFFont::Resource(LocalFile::FileContents(StrCat(ASSETS_DIR, filename)),
                                                        FontName(filename, pointsize, fg, flag),
                                                        (FLAGS_atlas_dump ? TTFFont::Flag::WriteAtlas : 0));
    if (resource->content.empty()) { ERROR("InsertFretype ", filename); delete resource; return 0; }
    return Insert(TTFFont::OpenBuffer(shared_ptr<TTFFont::Resource>(resource), pointsize, fg, flag),
                  filename, family, pointsize, fg, flag);
}

Font *Fonts::InsertCoreText(const string &filename, const string &family, int pointsize, const Color &fg, int flag) {
#ifdef __APPLE__
    Font *font = CoreTextFont::Open(filename, pointsize, fg, flag,
                                    (FLAGS_atlas_dump ? CoreTextFont::Flag::WriteAtlas : 0));
    if (!font) { ERROR("InsertCoreText ", filename); return 0; }
    return Insert(font, filename, family, pointsize, fg, flag);
#else
    return 0;
#endif
}

Font *Fonts::Get(Fonts::FontColorMap *colors, int pointsize, Color fg, int flag) {
    if (flag == -1) flag = FLAGS_default_font_flag;
    FontColorMap::iterator i = colors->find(FontColor(fg, flag));
    if (i == colors->end() || !i->second.size()) return 0;

    FontSizeMap::iterator j = i->second.lower_bound(pointsize);
    if (j != i->second.end()) return j->second;

    FontSizeMap::reverse_iterator k = i->second.rbegin();
    if (k != i->second.rend()) return k->second;

    return 0;
}

Font *Fonts::Get(const string &filename, const string &family, int pointsize, Color fg, int flag) {
    if (flag == -1) flag = FLAGS_default_font_flag;
    if (!filename.empty()) if (Font *ret = Get(filename, pointsize, fg, flag)) return ret;
    if (family.empty()) return 0;

    Fonts *inst = Singleton<Fonts>::Get();
    FamilyMap::iterator fi = inst->family_map.find(family);
    if (fi == inst->family_map.end()) return 0;

    bool bold = flag & FontDesc::Bold, italic = flag & FontDesc::Italic;
    if (bold && italic && fi->second.bold_italic.size()) return Get(*fi->second.bold_italic.begin(), pointsize, fg, flag);
    if (bold &&           fi->second.bold       .size()) return Get(*fi->second.bold       .begin(), pointsize, fg, flag);
    if (italic &&         fi->second.italic     .size()) return Get(*fi->second.italic     .begin(), pointsize, fg, flag);
    if (fi->second.normal.empty()) return 0;
    return Get(*fi->second.normal.begin(), pointsize, fg, flag);
}

Font *Fonts::Get(const string &filename, int pointsize, Color fg, int flag) {
    int origPointSize = pointsize;
    pointsize = ScaledFontSize(pointsize);
    if (flag == -1) flag = FLAGS_default_font_flag;

    Fonts *inst = Singleton<Fonts>::Get();
    FontMap::iterator fi = inst->font_map.find(filename);
    if (fi == inst->font_map.end() || !fi->second.size()) {
        if (filename == FakeFont::Filename()) return Fonts::Fake();
        ERROR("missing font: ", filename);
        return NULL;
    }

    bool is_fg_white = (fg.r() == 1 && fg.g() == 1 && fg.b() == 1);
    int max_ci = 2 - (is_fg_white || flag);
    for (int ci = 0; ci < max_ci; ++ci) {
        bool last_ci = ci == (max_ci - 1);
        Color *c = ci ? &Color::white : &fg;
        Font *f = Get(&fi->second, pointsize, *c, flag);
        if (!f) continue;
        if (!ci && f->size == pointsize) return f;
        if (!ci && !last_ci) {
            Font *second_match = Get(&fi->second, pointsize, Color::white, flag);
            if (second_match->size > f->size) f = second_match;
        }

        Font *font = f->Clone(pointsize, fg, flag);
        font->mix_fg = ci;

        fi->second[FontColor(fg, flag)][pointsize] = font;
        return font;
    }

    ERROR("missing font: ", filename, " ", pointsize, " ", fg.DebugString(), " ", flag);
    return NULL;
}

}; // namespace LFL
