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
    Emission = 0;
    Position = 0;
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
  bool GetEnabled(int) { return 0; }
  bool ShaderSupport() { return 0; }
  void MarkDirty() {}
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
  void BindCubeMap(int n) {}
  void TextureGenLinear() {}
  void TextureGenReflection() {}
  void Material(int t, float *color) {}
  void Light(int n, int t, float *color) {}
  void BindTexture(int t, int n) {}
  void ActiveTexture(int n) {}
  bool VertexPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty, int) { return true; }
  void TexPointer(int m, int t, int w, int o, float *tex, int l, int *out, bool dirty) {}
  void ColorPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {}
  void NormalPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {}
  void Color4f(float r, float g, float b, float a) {}
  void UpdateColor() {}
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
  void UseShader(Shader *shader) {}
  void DrawElements(int pt, int np, int it, int o, void *index, int l, int *out, bool dirty) {}
  void DrawArrays(int t, int o, int n) {}
  void DeferDrawArrays(int t, int o, int n) {}

  void Finish() {}
  void Flush() {}
  void Clear() {}
  void ClearDepth() {}
  void ClearColor(const Color &c) {}
  void PointSize(float n) {}
  void LineWidth(float n) {}
  void GenTextures(int t, int n, unsigned *out) {}
  void DelTextures(int n, const unsigned *id) {}
  void TexImage2D(int targ, int l, int fi, int w, int h, int b, int f, int t, const void*) {}
  void TexSubImage2D(int targ, int l, int xo, int yo, int w, int h, int f, int t, const void*) {}
  void CopyTexImage2D(int targ, int l, int fi, int x, int y, int w, int h, int b) {}
  void CopyTexSubImage2D(int targ, int l, int xo, int yo, int x, int y, int w, int h) {}
  void CheckForError(const char *file, int line) {}
  void EnableScissor() {}
  void DisableScissor() {}
  void EnableDepthTest() {}
  void DisableDepthTest() {}
  void DisableBlend() {}
  void EnableBlend() {}
  void BlendMode(int sm, int dm) {}
  void ViewPort(Box w) {}
  void Scissor(Box w) {}
  void TexParameter(int, int, int) {}
  void GenRenderBuffers(int n, unsigned *out) {}
  void DelRenderBuffers(int n, const unsigned *id) {}
  void BindRenderBuffer(int id) {}
  void RenderBufferStorage(int d, int w, int h) {}
  void GenFrameBuffers(int n, unsigned *out) {}
  void DelFrameBuffers(int n, const unsigned *id) {}
  void BindFrameBuffer(int id) {}
  void FrameBufferTexture(int id) {}
  void FrameBufferDepthTexture(int id) {}
  int CheckFrameBufferStatus() { return 0; }
  void Screenshot(Texture *out) {}
  void ScreenshotBox(Texture *out, const Box &b, int flag) {}
  void DumpTexture(Texture *out, unsigned tex_id) {}
  const char *GetString(int t) { return ""; }
  const char *GetGLEWString(int t) { return ""; }

  int CreateProgram() { return 0; }
  void DelProgram(int p) {}
  int CreateShader(int t) { return 0; }
  void CompileShader(int shader, vector<const char*> source) {}
  void AttachShader(int prog, int shader) {}
  void DelShader(int shader) {}
  void BindAttribLocation(int prog, int loc, const string &name) {}
  bool LinkProgram(int prog) { return false; }
  void GetProgramiv(int p, int t, int *out) {}
  void GetIntegerv(int t, int *out) {}
  int GetAttribLocation (int prog, const string &name) { return 0; }
  int GetUniformLocation(int prog, const string &name) { return 0; }
  void Uniform1i(int u, int v) {}
  void Uniform1f(int u, float v) {}
  void Uniform2f(int u, float v1, float v2) {}
  void Uniform3f(int u, float v1, float v2, float v3) {}
  void Uniform4f(int u, float v1, float v2, float v3, float v4) {}
  void Uniform3fv(int u, int n, const float *v) {}
  int GetDepth(int id) { return 0; }
  int GetCubeMap(int id) { return 0; }
  int GetPixel(int p) { return 0; }
};

unique_ptr<GraphicsDevice> GraphicsDevice::Create(Window *w, Shaders *s) { return make_unique<NullGraphicsDevice>(w, s); }

}; // namespace LFL
