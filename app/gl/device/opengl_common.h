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

void Finish() { ClearDeferred(); glFinish(); GDDebug("Finish"); }
void Flush() { ClearDeferred(); glFlush(); GDDebug("Flush"); }
void Clear() { glClear(GL_COLOR_BUFFER_BIT | (draw_mode == DrawMode::_3D ? GL_DEPTH_BUFFER_BIT : 0)); }
void ClearDepth() { glClear(GL_DEPTH_BUFFER_BIT); }
void ClearColor(const Color &c) { clear_color=c; glClearColor(c.r(), c.g(), c.b(), c.a()); GDDebug("ClearColor=", c.DebugString()); }
void PointSize(float n) { glPointSize(n); }
void LineWidth(float n) { glLineWidth(n); }
void DelTexture(TextureRef id) {
  ClearDeferred();
  glDeleteTextures(1, &id.v);
  if (FLAGS_gd_debug) GDLogRef("DelTexture ", id.v);
}

void UpdateTexture(const TextureRef &id, int targ, int l, int fi, int b, int t, const void *data) { glTexImage2D(GetCubeMap(targ), l, fi, id.w, id.h, b, id.f, t, data); } 
void UpdateSubTexture(const TextureRef &id, int targ, int l, int xo, int yo, int w, int h, int t, const void *data) { glTexSubImage2D(GetCubeMap(targ), l, xo, yo, w, h, id.f, t, data); }

void CopyTexImage2D(int targ, int l, int fi, int x, int y, int w, int h, int b) { glCopyTexImage2D(targ, l, fi, x, y, w, h, b); }
void CopyTexSubImage2D(int targ, int l, int xo, int yo, int x, int y, int w, int h) { glCopyTexSubImage2D(targ, l, xo, yo, x, y, w, h); }

TextureRef CreateTexture(int t, unsigned w, unsigned h, int f) {
  TextureRef ret = {0, w, h, GetCubeMap(t), GetPixel(f) };
  ClearDeferred();
  if (ret.t == GL_TEXTURE_CUBE_MAP) glEnable(GL_TEXTURE_CUBE_MAP);
  glGenTextures(1, &ret.v);
  glBindTexture(ret.t, ret.v);
  glTexParameteri(ret.t, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(ret.t, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  GDLogRef("GenTexture ", t, ",", ret.v);
  return ret;
}

void CheckForError(const char *file, int line) {
  GLint gl_error=0, gl_validate_status=0;
  if ((gl_error = glGetError())) {
    int framebuffer_id = 0;
    GetIntegerv(c.FramebufferBinding, &framebuffer_id);

    ERROR(file, ":", line, " gl error: ", gl_error, ", framebuffer = ", framebuffer_id);
    BreakHook();
    if (version == 2) {
      glValidateProgram(shader->ID);
      glGetProgramiv(shader->ID, GL_VALIDATE_STATUS, &gl_validate_status);
      if (gl_validate_status != GL_TRUE) ERROR(shader->name, ": gl validate status ", gl_validate_status);

      char buf[1024]; int len;
      glGetProgramInfoLog(shader->ID, sizeof(buf), &len, buf);
      if (len) INFO(buf);
    }
  }
}

bool GetEnabled(int v) { return glIsEnabled(v); }
void EnableScissor() { glEnable(GL_SCISSOR_TEST); }
void DisableScissor() { glDisable(GL_SCISSOR_TEST); }
void EnableDepthTest()  {  glEnable(GL_DEPTH_TEST); glDepthMask(GL_TRUE);  GDDebug("DepthTest=1"); }
void DisableDepthTest() { glDisable(GL_DEPTH_TEST); glDepthMask(GL_FALSE); GDDebug("DepthTest=0"); }
void DisableBlend() { if (Changed(&blend_enabled, false)) { ClearDeferred(); glDisable(GL_BLEND);                                                    GDDebug("Blend=0"); } }
void EnableBlend()  { if (Changed(&blend_enabled, true )) { ClearDeferred();  glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); GDDebug("Blend=1"); } }
void BlendMode(int sm, int dm) { ClearDeferred(); glBlendFunc(sm, dm); GDDebug("BlendMode=", sm, ",", dm); }
void ViewPort(Box w) {
  if (FLAGS_swap_axis) w.swapaxis(parent->gl_w, parent->gl_h);
  ClearDeferred();
  glViewport(w.x, w.y, w.w, w.h);
  GDDebug("Viewport(", w.x, ", ", w.y, ", ", w.w, ", ", w.h, ")");
}

void Scissor(Box w) {
  if (FLAGS_swap_axis) w.swapaxis(parent->gl_w, parent->gl_h);
  ClearDeferred();
  EnableScissor();
  glScissor(w.x, w.y, w.w, w.h);
  GDDebug("Scissor(", w.x, ", ", w.y, ", ", w.w, ", ", w.h, ")");
}

void TexParameter(int t, int p, int v) {
  glTexParameteri(t, p, v);
  GDDebug("TexParameter ", t, " ", p, " ", v);
}

DepthRef CreateDepthBuffer(unsigned w, unsigned h, int f) {
  DepthRef ret = {0, w, h};
  glGenRenderbuffersEXT(1, &ret.v);
  glBindRenderbufferEXT(GL_RENDERBUFFER, ret.v);
  glRenderbufferStorageEXT(GL_RENDERBUFFER, GetDepth(f), w, h);
  if (FLAGS_gd_debug) GDDebug("GenRenderBuffer ", ret.v);
  return ret;
}

void DelDepthBuffer(DepthRef id) { 
  if (FLAGS_gd_debug) GDDebug("DelRenderBuffer ", id[i]);
  glDeleteRenderbuffersEXT(1, &id.v);
}

void BindDepthBuffer(const DepthRef &id) { glBindRenderbufferEXT(GL_RENDERBUFFER, id); }

FrameBufRef CreateFrameBuffer(unsigned w, unsigned h) {
  FrameBufRef ret = {0, w, h};
  glGenFramebuffersEXT(1, &ret.v);
  if (FLAGS_gd_debug) GDLogRef("GenFrameBuffer ", ret.v);
  return ret;
}

void DelFrameBuffer(FrameBufRef id) {
  if (FLAGS_gd_debug) GDLogRef("DelFrameBuffer ", ret.v);
  glDeleteFramebuffersEXT(1, &id.v);
}

void BindFrameBuffer(const FrameBufRef &id) { ClearDeferred(); glBindFramebufferEXT(GL_FRAMEBUFFER, id); GDDebug("BindFrameBuffer ", id, " default_fb=", default_framebuffer); }
void FrameBufferTexture(const TextureRef &id) { glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, id, 0); }
void FrameBufferDepthTexture(const DepthRef &id) { glFramebufferRenderbufferEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, id); }
int CheckFrameBufferStatus() { return glCheckFramebufferStatusEXT(GL_FRAMEBUFFER); }

void Screenshot(Texture *out) { ScreenshotBox(out, Box(parent->gl_w, parent->gl_h), Texture::Flag::FlipY); }
void ScreenshotBox(Texture *out, const Box &b, int flag) {
  ClearDeferred();
  out->Resize(b.w, b.h, Texture::preferred_pf, Texture::Flag::CreateBuf);
  auto pixels = out->NewBuffer();
  glReadPixels(b.x, b.y, b.w, b.h, GetPixel(out->pf), out->GDBufferType(this), pixels.get());
  out->UpdateBuffer(pixels.get(), point(b.w, b.h), out->pf, b.w*4, flag);
  GDDebug("ScreenshotBox");
}

void DumpTexture(Texture *out, const TextureRef &tex_id) {
#if !defined(LFL_MOBILE) && !defined(LFL_EMSCRIPTEN)
  if (tex_id) {
    GLint gl_tt = GetCubeMap(out->cubemap), tex_w = 0, tex_h = 0;
    BindTexture(tex_id);
    glGetTexLevelParameteriv(gl_tt, 0, GL_TEXTURE_WIDTH, &tex_w);
    glGetTexLevelParameteriv(gl_tt, 0, GL_TEXTURE_WIDTH, &tex_h);
    CHECK_GT((out->width  = tex_w), 0);
    CHECK_GT((out->height = tex_h), 0);
  }
  out->RenewBuffer();
  glGetTexImage(GetCubeMap(out->cubemap), 0, GetPixel(out->pf), out->GDBufferType(this), out->buf);
#endif
}

const char *GetString(int t) { return SpellNull(MakeSigned(glGetString(t))); }
const char *GetGLEWString(int t) {
#ifdef LFL_GLEW
  return SpellNull(MakeSigned(glewGetString(t)));
#else
  return "";
#endif
}

ProgramRef CreateProgram() { ProgramRef p = { glCreateProgram() }; GDDebug("CreateProgram ", p); return p; }
void DelProgram(ProgramRef p) { glDeleteProgram(p); GDDebug("DelProgram ", p); }

ShaderRef CreateShader(int t) { ShaderRef p = { glCreateShader(t) }; return p; }
void CompileShader(const ShaderRef &shader, vector<const char*> source) {
  GLint success = 0;
  glShaderSource(shader, source.size(), &source[0], nullptr);
  glCompileShader(shader);
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (success == GL_FALSE) {
    int l = 0;
    char buf[1024] = {0};
    glGetShaderInfoLog(shader, sizeof(buf), &l, buf);
    INFO("CompileShader failed: ", buf);
    INFO("Source:");
    int line_no = 1;
    for (auto &s : source) {
      StringLineIter lines(s, StringLineIter::Flag::BlankLines);
      for (const char *line = lines.Next(); line; line = lines.Next())
        INFO(line_no++, ": ", string(line, lines.CurrentLength()));
    }
  }
}

void DelShader(const ShaderRef &shader) { glDeleteShader(shader); }
void AttachShader(const ProgramRef &prog, const ShaderRef &shader) { glAttachShader(prog, shader); }
void BindAttribLocation(const ProgramRef &prog, int loc, const string &name) { glBindAttribLocation(prog, loc, name.c_str()); }
bool LinkProgram(const ProgramRef &prog) {
  char buf[1024] = {0}; int l=0;
  glLinkProgram(prog);
  glGetProgramInfoLog(prog, sizeof(buf), &l, buf);
  if (l) INFO(buf);
  GLint link_status;
  glGetProgramiv(prog, GL_LINK_STATUS, &link_status);
  return link_status == GL_TRUE;
}
void GetProgramiv(const ProgramRef &p, int t, int *out) { glGetProgramiv(p, t, out); }
void GetIntegerv(int t, int *out) { glGetIntegerv(t, out); }

AttribRef  GetAttribLocation (const ProgramRef &prog, const string &name) { AttribRef  l = { glGetAttribLocation (prog, name.c_str()) }; return l; }
UniformRef GetUniformLocation(const ProgramRef &prog, const string &name) { UniformRef l = { glGetUniformLocation(prog, name.c_str()) }; return l; }

void Uniform1i(const UniformRef &u, int v) { glUniform1i(u, v); }
void Uniform1f(const UniformRef &u, float v) { glUniform1f(u, v); }
void Uniform2f(const UniformRef &u, float v1, float v2) { glUniform2f(u, v1, v2); }
void Uniform3f(const UniformRef &u, float v1, float v2, float v3) { glUniform3f(u, v1, v2, v3); }
void Uniform4f(const UniformRef &u, float v1, float v2, float v3, float v4) { glUniform4f(u, v1, v2, v3, v4); }
void Uniform3fv(const UniformRef &u, int n, const float *v) { glUniform3fv(u, n, v); }
void Uniform4fv(const UniformRef &u, int n, const float *v) { glUniform4fv(u, n, v); }
void UniformMatrix4fv(const UniformRef &u, int n, const float *v) { glUniformMatrix4fv(u, n, 0, v); }

void LogVersion() {
  const char *glslver = GetString(c.ShaderVersion);
  const char *glexts = GetString(c.Extensions);
  INFO("OpenGL Version: ", GetString(c.Version));
  INFO("OpenGL Vendor: ", GetString(c.Vendor));
  INFO("GLEW Version: ", GetGLEWString(c.GLEWVersion));
  INFO("GL_SHADING_LANGUAGE_VERSION: ", glslver);
  INFO("GL_EXTENSIONS: ", glexts);

#ifdef LFL_MOBILE
  have_cubemap = strstr(glexts, "GL_EXT_texture_cube_map") != 0;
#else
  have_cubemap = strstr(glexts, "GL_ARB_texture_cube_map") != 0;
#endif
  int depth_bits=0;
  GetIntegerv(c.DepthBits, &depth_bits);
  INFO("opengles_version = ", version, ", depth_bits = ", depth_bits);
  INFO("have_cubemap = ", have_cubemap ? "true" : "false");

#if 0
  int dim[2] = { 0, 0 };
  screen->gd->GetIntegerv(GraphicsDevice::MaxViewportDims, dim);
  INFO("max_viewport_dims = ", dim[0], ", ", dim[1]);
#endif
}

unsigned GetDepth(int id) {
  switch (id) {
  case Depth::_16: return GL_DEPTH_COMPONENT16;
  default: return 0;
  }
}

unsigned GetCubeMap(int id) {
  switch (id) {
  case CubeMap::NX: return GL_TEXTURE_CUBE_MAP_NEGATIVE_X;
  case CubeMap::PX: return GL_TEXTURE_CUBE_MAP_POSITIVE_X;
  case CubeMap::NY: return GL_TEXTURE_CUBE_MAP_NEGATIVE_Y;
  case CubeMap::PY: return GL_TEXTURE_CUBE_MAP_POSITIVE_Y;
  case CubeMap::NZ: return GL_TEXTURE_CUBE_MAP_NEGATIVE_Z;
  case CubeMap::PZ: return GL_TEXTURE_CUBE_MAP_POSITIVE_Z;
  default:          return GL_TEXTURE_2D;
  }
}

unsigned GetPixel(int p) {
  switch (p) {
  case Pixel::RGBA:   case Pixel::RGB32: return GL_RGBA;
  case Pixel::RGB24:                     return GL_RGB;
#if !defined(LFL_MOBILE) && !defined(LFL_EMSCRIPTEN)
  case Pixel::BGRA:   case Pixel::BGR32: return GL_BGRA;
  case Pixel::BGR24:                     return GL_BGR;
#endif
  case Pixel::GRAYA8:                    return GL_LUMINANCE_ALPHA;
  case Pixel::GRAY8:                     return GL_LUMINANCE;
  default:                               return -1;
  }
}
