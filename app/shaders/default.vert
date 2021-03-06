attribute vec4 Position;
varying vec4 DestinationColor;

uniform vec4 DefaultColor;
uniform mat4 InverseView;
uniform mat4 Modelview;
uniform mat4 ModelviewProjection;

#ifdef VERTEXCOLOR
attribute vec4 VertexColor;
uniform bool VertexColorEnabled;
#endif

#ifdef NORMALS
attribute vec4 Normal;
#ifndef TEXCUBE
uniform vec4 MaterialAmbient;
uniform vec4 MaterialDiffuse;
uniform vec4 MaterialSpecular;
uniform vec4 MaterialEmission;
uniform vec4 LightZeroPosition;
uniform vec4 LightZeroAmbient;
uniform vec4 LightZeroDiffuse;
uniform vec4 LightZeroSpecular;
#endif
#endif

#ifdef TEX2D
attribute vec2 TexCoordIn;
varying vec2 TexCoordOut;
uniform bool TexCoordEnabled;
#endif

#ifdef TEXCUBE
varying vec3 CubeCoordOut;
#endif

uniform float iGlobalTime;
uniform vec3  iResolution;

void main(void) {
  vec4 position = Position;
  // LFLPositionShaderMarker
  gl_Position = ModelviewProjection * position;

#ifdef TEX2D
  if (TexCoordEnabled) TexCoordOut = TexCoordIn;
#endif

#ifdef TEXCUBE
  CubeCoordOut = normalize(-Position.xyz);
  DestinationColor = DefaultColor;
#ifdef NORMALS
  vec3 N = Normal.xyz;
#endif
#else // TEXCUBE

#ifdef NORMALS
  vec3 eyespace_position = (Modelview * Position).xyz;
  vec3 N = normalize((Modelview * vec4(Normal.xyz, 0)).xyz);
  vec3 L = normalize((Modelview * LightZeroPosition).xyz - eyespace_position.xyz);
  float L_dot_N = dot(L, N);
  vec3 ambient = LightZeroAmbient.xyz * MaterialAmbient.xyz;
  vec3 diffuse = LightZeroDiffuse.xyz * MaterialDiffuse.xyz * max(0.0, L_dot_N);
  DestinationColor = vec4(ambient + diffuse, 1.0);
#else // NORMALS

#ifdef VERTEXCOLOR
  DestinationColor = VertexColorEnabled ? VertexColor : DefaultColor;
#else // VERTEXCOLOR
  DestinationColor = DefaultColor;
#endif // VERTEXCOLOR

#endif // NORMALS
#endif // TEXCUBE
}
