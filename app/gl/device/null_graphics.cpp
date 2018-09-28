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

namespace LFL {
struct NullGraphicsDeviceConstants : public GraphicsDevice::Constants {
  NullGraphicsDeviceConstants() {
    Float = 0;
    Points = 0;
    Lines = 0;
    LineLoop = 0;
    Triangles = 0;
    TriangleStrip = 0;
    Polygon = 0;
    Texture2D = 0;
    TextureCubeMap = 0;
    UnsignedByte = 0;
    UnsignedInt = 0;
    FramebufferComplete = 0;
    FramebufferUndefined = 0;
    FramebufferBinding = 0;
    Ambient = 0;
    Diffuse = 0;
    Specular = 0;
    Position = 0;
    Emission = 0;
    AmbientAndDiffuse = 0;
    One = 0;
    SrcAlpha = 0;
    OneMinusSrcAlpha = 0;
    OneMinusDstColor = 0;
    TextureWrapS = 0;
    TextureWrapT = 0;
    ClampToEdge = 0;
    VertexShader = 0;
    FragmentShader = 0;
    ShaderVersion = 0;
    Extensions = 0;
    GLEWVersion = 0;
    Version = 0;
    Vendor = 0;
    DepthBits = 0;
    ScissorTest = 0;
    ActiveUniforms = 0;
    ActiveAttributes = 0;
    MaxVertexAttributes = 0;
    MaxVertexUniformComp = 0;
    MaxViewportDims = 0;
    ViewportBox = 0;
    ScissorBox = 0;
    Fill = 0;
    Line = 0;
    Point = 0;
    GLPreferredBuffer = 0;
    GLInternalFormat = 0;
  }
};

struct NullGraphicsDevice : public GraphicsDevice {
  NullGraphicsDevice(Window *w, Shaders *s) : GraphicsDevice(0, NullGraphicsDeviceConstants(), w, 0, s) {}

  void Init(AssetLoading*, const Box&) {}
  void Finish() {}
  void MarkDirty() {}
  void Clear() {}
  void ClearDepth() {}
  void Flush() {}

  bool GetEnabled(int) { return 0; }
  bool GetShaderSupport() { return 0; }
  void GetIntegerv(int t, int *out) {}
  const char *GetString(int t) { return ""; }
  const char *GetGLEWString(int t) { return ""; }
  void CheckForError(const char *file, int line) {}

  void EnableTexture() {}
  void DisableTexture() {}
  void EnableLighting() {}
  void DisableLighting() {}
  void EnableNormals() {}
  void DisableNormals() {}
  void EnableVertexColor() {}
  void DisableVertexColor() {}
  void EnableLight(int n) {}
  void DisableLight(int n) {}
  void DisableCubeMap() {}
  void EnableScissor() {}
  void DisableScissor() {}
  void EnableDepthTest() {}
  void DisableDepthTest() {}
  void DisableBlend() {}
  void EnableBlend() {}
  void BlendMode(int sm, int dm) {}

  void MatrixProjection() {}
  void MatrixModelview() {}
  void LoadIdentity() {}
  void PushMatrix() {}
  void PopMatrix() {}
  void GetMatrix(m44 *out) {}
  void PrintMatrix() {}
  void Scalef(float x, float y, float z) {}
  void Rotatef(float angle, float x, float y, float z) {}
  void Ortho(float l, float r, float b, float t, float nv, float fv) {}
  void Frustum(float l, float r, float b, float t, float nv, float fv) {}
  void Mult(const float *m) {}
  void Translate(float x, float y, float z) {}

  void ViewPort(Box w) {}
  void Scissor(Box w) {}
  void ClearColor(const Color &c) {}
  void Color4f(float r, float g, float b, float a) {}
  void Light(int n, int t, float *color) {}
  void Material(int t, float *color) {}
  void PointSize(float n) {}
  void LineWidth(float n) {}

  TextureRef CreateTexture(int t, unsigned w, unsigned h, int f) { return TextureRef(); }
  void DelTexture(TextureRef id) {}
  void UpdateTexture(const TextureRef&, int targ, int l, int fi, int b, int t, const void*) {}
  void UpdateSubTexture(const TextureRef&, int targ, int l, int xo, int yo, int w, int h, int t, const void*) {}
  void CopyTexImage2D(int targ, int l, int fi, int x, int y, int w, int h, int b) {}
  void CopyTexSubImage2D(int targ, int l, int xo, int yo, int x, int y, int w, int h) {}
  void BindTexture(const TextureRef &n) {}
  void BindCubeMap(const TextureRef &n) {}
  void ActiveTexture(int n) {}
  void TextureGenLinear() {}
  void TextureGenReflection() {}
  void TexParameter(int, int, int) {}

  FrameBufRef CreateFrameBuffer(unsigned w, unsigned h) { return FrameBufRef(); }
  void DelFrameBuffer(FrameBufRef id) {}
  void BindFrameBuffer(const FrameBufRef &id) {}
  void FrameBufferTexture(const TextureRef &id) {}
  void FrameBufferDepthTexture(const DepthRef &id) {}
  int CheckFrameBufferStatus() { return 0; }

  DepthRef CreateDepthBuffer(unsigned w, unsigned h, int f) { return DepthRef(); }
  void DelDepthBuffer(DepthRef id) {}
  void BindDepthBuffer(const DepthRef &id) {}

  ProgramRef CreateProgram() { return ProgramRef(); }
  void DelProgram(ProgramRef p) {}
  ShaderRef CreateShader(int t) { return ShaderRef(); }
  void CompileShader(const ShaderRef &shader, vector<const char*> source) {}
  void AttachShader(const ProgramRef &prog, const ShaderRef &shader) {}
  void DelShader(const ShaderRef &shader) {}
  void BindAttribLocation(const ProgramRef &prog, int loc, const string &name) {}
  bool LinkProgram(const ProgramRef &prog) { return false; }
  void GetProgramiv(const ProgramRef &p, int t, int *out) {}
  AttribRef  GetAttribLocation (const ProgramRef &prog, const string &name) { return AttribRef(); }
  UniformRef GetUniformLocation(const ProgramRef &prog, const string &name) { return UniformRef(); }
  void Uniform1i(const UniformRef &u, int v) {}
  void Uniform1f(const UniformRef &u, float v) {}
  void Uniform2f(const UniformRef &u, float v1, float v2) {}
  void Uniform3f(const UniformRef &u, float v1, float v2, float v3) {}
  void Uniform4f(const UniformRef &u, float v1, float v2, float v3, float v4) {}
  void Uniform3fv(const UniformRef &u, int n, const float *v) {}
  void Uniform4fv(const UniformRef &u, int n, const float *v) {}
  void UniformMatrix4fv(const UniformRef &u, int n, const float *v) {}

  void UseShader(Shader *shader) {}
  void Screenshot(Texture *out) {}
  void ScreenshotBox(Texture *out, const Box &b, int flag) {}
  void DumpTexture(Texture *out, const TextureRef &tex_id) {}

  bool VertexPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty, int) { return true; }
  void TexPointer(int m, int t, int w, int o, float *tex, int l, int *out, bool dirty) {}
  void ColorPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {}
  void NormalPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {}
  void DrawElements(int pt, int np, int it, int o, void *index, int l, int *out, bool dirty) {}
  void DrawArrays(int t, int o, int n) {}
  void DeferDrawArrays(int t, int o, int n) {}
};

unique_ptr<GraphicsDevice> GraphicsDevice::Create(Window *w, Shaders *s) { return make_unique<NullGraphicsDevice>(w, s); }

}; // namespace LFL
